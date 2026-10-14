"""Pixel calibration and extraction; no network, OCR, or generative models."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image


def quarter_number(value: str) -> int:
    match = re.fullmatch(r"(\d{4})\s*[_-]?\s*[Qq]([1-4])", value.strip())
    if not match:
        raise ValueError(f"Invalid quarter {value!r}; use 2025Q3")
    return int(match[1]) * 4 + int(match[2]) - 1


def quarter_label(value: int) -> str:
    return f"{value // 4}Q{value % 4 + 1}"


def month_number(value: str) -> int:
    match = re.fullmatch(r"(\d{4})-(\d{2})", value.strip())
    if not match or not 1 <= int(match[2]) <= 12:
        raise ValueError(f"Invalid month {value!r}; use 2025-03")
    return int(match[1]) * 12 + int(match[2]) - 1


def month_label(value: int) -> str:
    return f"{value // 12}-{value % 12 + 1:02d}"


def sampling_mode(c: dict | None) -> str:
    mode = (c or {}).get('sampling', 'quarterly')
    if mode not in ('quarterly', 'monthly'):
        raise ValueError('sampling must be "quarterly" or "monthly"')
    return mode


def period_number(value: str, sampling: str) -> int:
    return month_number(value) if sampling == 'monthly' else quarter_number(value)


def period_label(value: int, sampling: str) -> str:
    return month_label(value) if sampling == 'monthly' else quarter_label(value)


def month_to_quarter(value: str) -> str:
    number = month_number(value)
    return quarter_label((number // 3))


def report_quarter(filename):
    """Read a report quarter from either YEAR_Qn or YEAR_Mon filenames."""
    stem = Path(filename).stem
    match = re.search(r'(?:^|[A-Za-z\s_-])(\d{4})[\s_-]?Q([1-4])(?:$|[\s_-])', stem, re.IGNORECASE)
    if match:
        return f'{match[1]}Q{match[2]}'
    match = re.search(r'(?:^|[\s_-])(\d{4})[\s_-](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$', stem, re.IGNORECASE)
    if match:
        month = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'].index(match[2].lower())
        return f'{match[1]}Q{month//3+1}'
    return stem


def report_sort_key(filename):
    try:
        return (quarter_number(report_quarter(filename)), str(filename))
    except ValueError:
        return (0, str(filename))



@dataclass
class Result:
    rows: list[dict]
    warnings: list[str]
    calibration: dict


def validate_calibration(c: dict, size: tuple[int, int]) -> None:
    sampling = sampling_mode(c)
    if list(size) != c["image_size"]:
        raise ValueError(f"Image size {size} differs from calibrated {c['image_size']}; recalibrate.")
    l, t, r, b = c["plot_box"]
    if not (0 <= l < r < size[0] and 0 <= t < b < size[1]):
        raise ValueError("Plot box must be inside image: left, top, right, bottom.")
    if not c["y_min"] < c["y_max"]:
        raise ValueError("y_min must be below y_max")
    start, end, last = [period_number(c[k], sampling) for k in ("x_start", "x_end", "last_historical")]
    if not start <= last <= end or start == end:
        raise ValueError("Require x_start <= last_historical <= x_end, with a positive time span")
    for region in c.get('ignore_regions', []):
        if len(region) != 4 or not (0 <= region[0] < region[2] <= size[0] and 0 <= region[1] < region[3] <= size[1]):
            raise ValueError('Ignored annotation rectangles must be inside the image')
    if 'historical_rgb' in c:
        historical_rgb = c['historical_rgb']
        if len(historical_rgb) != 3 or any(not 0 <= v <= 255 for v in historical_rgb):
            raise ValueError('historical_rgb must contain three values in 0..255')
        if not 0 < c.get('historical_color_tolerance', 45) <= 100:
            raise ValueError('historical_color_tolerance must be in (0, 100]')
    rgb = c["curve_rgb"]
    if len(rgb) != 3 or any(not 0 <= v <= 255 for v in rgb):
        raise ValueError("curve_rgb must contain three values in 0..255")
    if not 0 < c.get("color_tolerance", 22) <= 100:
        raise ValueError("color_tolerance must be in (0, 100]")
    start, end = [period_number(c[k], sampling) for k in ("x_start", "x_end")]
    for period, override in c.get('manual_point_overrides', {}).items():
        q = period_number(period, sampling)
        if not start <= q <= end:
            raise ValueError('Manual point override period must be inside the calibrated time range')
        if not isinstance(override, dict) or 'pixel_y' not in override:
            raise ValueError('Manual point overrides must map quarters to pixel_y records')
        pixel_y = float(override['pixel_y'])
        if not np.isfinite(pixel_y) or not t <= pixel_y <= b:
            raise ValueError('Manual point override pixel_y must be inside the plot box')


def extract(path: Path, calibration: dict, country: str) -> Result:
    c = dict(calibration)
    with Image.open(path) as opened:
        im = opened.convert("RGB")
    validate_calibration(c, im.size)
    rgb = np.asarray(im).astype(np.float32)
    l, t, r, b = c["plot_box"]
    color = np.array(c["curve_rgb"], dtype=float)
    # Float arithmetic avoids uint8 overflow in squared colour distances.
    distance = np.sqrt(np.sum((rgb - color) ** 2, axis=2))
    mask = distance <= c.get("color_tolerance", 22)
    mask[:int(np.ceil(t)), :] = False
    mask[int(np.floor(b)) + 1:, :] = False
    historical_mask = None
    if 'historical_rgb' in c:
        historical_mask = np.linalg.norm(rgb-np.array(c['historical_rgb']),axis=2) <= c.get('historical_color_tolerance',45)
        historical_mask[:int(np.ceil(t)),:] = False
        historical_mask[int(np.floor(b))+1:,:] = False
        # Remove horizontal axes/grid rules of the same black colour.
        left_inside,right_inside=int(np.ceil(l))+2,int(np.floor(r))-2
        rules=historical_mask[:,left_inside:right_inside].mean(axis=1)>.65
        historical_mask[rules,:]=False
    for x1,y1,x2,y2 in c.get('ignore_regions', []):
        mask[int(y1):int(y2),int(x1):int(x2)]=False
        if historical_mask is not None:
            historical_mask[int(y1):int(y2),int(x1):int(x2)]=False
    sampling = sampling_mode(c)
    start, end, last = [period_number(c[k], sampling) for k in ("x_start", "x_end", "last_historical")]
    warnings = []
    rows = []
    recovered_history = []
    recovered_forecast = []
    target_chroma = color-color.mean()
    target_norm = float(np.linalg.norm(target_chroma))

    def historical_edges(xx):
        ys=np.flatnonzero(historical_mask[:,xx])
        if not len(ys):
            return None
        groups=np.split(ys,np.where(np.diff(ys)>5)[0]+1)
        groups=[g for g in groups if len(g)>=2]
        if len(groups)!=1 or groups[0][-1]-groups[0][0] > max(20,.15*(b-t)):
            return None
        return float(groups[0][0]),float(groups[0][-1])

    def column_edges(xx, historical_fallback=False, bridge_ink=False):
        ys = np.flatnonzero(mask[:, xx])
        if historical_fallback:
            column = rgb[:, xx]
            chroma = column-column.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(chroma, axis=1)
            similarity = (chroma @ target_chroma) / np.maximum(norms*target_norm, 1e-8)
            # A thin historical stroke can be much greyer/lighter than the fan.
            # Reject neutral grid/text pixels and very pale compression noise.
            candidates = ((column.max(axis=1)-column.min(axis=1) >= 8)
                          & (column.mean(axis=1) < 220) & (similarity >= .25))
            candidates[:int(np.ceil(t))] = False
            candidates[int(np.floor(b))+1:] = False
            distances = distance[:, xx]
            if not candidates.any() or distances[candidates].min() > 140:
                return None
            cutoff = distances[candidates].min()+12
            ys = np.flatnonzero(candidates & (distances <= cutoff))
        if not ys.size:
            return None
        groups = np.split(ys, np.where(np.diff(ys) > 5)[0]+1)
        if bridge_ink and len(groups) > 1:
            merged = [groups[0]]
            for group in groups[1:]:
                gap = np.arange(merged[-1][-1]+1, group[0])
                # Join only small gaps filled by visible overprinted ink;
                # blank separation between distinct curves stays ambiguous.
                if len(gap) <= 12 and np.all(rgb[gap, xx].mean(axis=1) < 220):
                    merged[-1] = np.concatenate((merged[-1],group))
                else:
                    merged.append(group)
            groups = merged
        substantial = [g for g in groups if len(g) >= 2]
        if len(substantial) > 1:
            return None
        if substantial:
            ys = substantial[0]
        if historical_fallback and ys[-1]-ys[0] > max(20, .12*(b-t)):
            return None
        return (float(ys[0]), float(ys[-1]))

    def occluded_centerline(xx):
        column = rgb[:, xx]
        chroma = column-column.mean(axis=1, keepdims=True)
        similarity = (chroma @ target_chroma)/np.maximum(np.linalg.norm(chroma,axis=1)*target_norm,1e-8)
        fan = ((column.max(axis=1)-column.min(axis=1)>8)
               & (column.mean(axis=1)<245) & (similarity>.75))
        fan[:int(np.ceil(t))] = False
        fan[int(np.floor(b))+1:] = False
        ys = np.flatnonzero(fan)
        if not len(ys) or ys[-1]-ys[0] > max(15, .10*(b-t)):
            return None
        # A narrow fan can be entirely covered by an explicit black centreline.
        # Require coloured fan pixels on both sides of a single dark stroke.
        inside = np.arange(ys[0]+1,ys[-1])
        dark = inside[(column[inside].mean(axis=1)<min(100,color.mean()-25))
                      & (column[inside].max(axis=1)-column[inside].min(axis=1)<30)]
        if len(dark)<2 or dark[-1]-dark[0]>10 or np.any(np.diff(dark)>2):
            return None
        return float(dark[0]),float(dark[-1])

    vintage = report_quarter(path.name)
    span = c["y_max"] - c["y_min"]
    def to_value(y):
        return c["y_max"] - (y - t) * span / (b - t)
    overrides = c.get('manual_point_overrides', {})
    for q in range(start, end + 1):
        label = period_label(q, sampling)
        x = l + (r-l)*(q-start)/(end-start)
        # Three nearby columns reduce antialiasing and grid-line interference.
        # Sample just inside vertical plot borders, which may cover the band.
        columns = sorted(set(max(int(np.ceil(l)) + 2, min(int(np.floor(r)) - 2, round(x) + dx)) for dx in (-1, 0, 1)))
        history = q <= last
        method = 'color_match'
        used_columns = columns
        if history and historical_mask is not None:
            edges = [edge for xx in columns if (edge := historical_edges(xx)) is not None]
            method = 'historical_color_match'
            if q in (start,end) and min(abs(x-l),abs(x-r)) < 1:
                # The black axis can cover the stroke's endpoint. Fit the local
                # line just inside the frame and evaluate at the true quarter x.
                max_inset=min(12,max(6,round((r-l)/(end-start)*.3)))
                samples=[]
                for offset in range(3,max_inset+1):
                    xx=round(l+offset if q==start else r-offset)
                    edge=historical_edges(xx)
                    if edge is not None:
                        samples.append((xx,sum(edge)/2))
                if len(samples)>=3:
                    coeff=np.polyfit([v[0] for v in samples],[v[1] for v in samples],1)
                    yy=float(np.polyval(coeff,x))
                    if t <= yy <= b:
                        edges=[(yy,yy)]
                        used_columns=[v[0] for v in samples]
                        method='historical_endpoint_fit'
        else:
            edges = [edge for xx in columns if (edge := column_edges(xx)) is not None]
        # Only recover a historical colour mismatch, never resolve competing
        # original-colour curves by arbitrarily choosing one of them.
        if not edges and history and historical_mask is None and not mask[:, columns].any():
            edges = [edge for xx in columns if (edge := column_edges(xx, True)) is not None]
            if edges:
                method = 'historical_color_recovery'
                recovered_history.append(label)
        if not edges and not history:
            edges = [edge for xx in columns if (edge := column_edges(xx, bridge_ink=True)) is not None]
            if edges:
                method = 'forecast_overprint_recovery'
                recovered_forecast.append(label)
            else:
                edges = [edge for xx in columns if (edge := occluded_centerline(xx)) is not None]
                if edges:
                    method = 'visible_forecast_centerline'
                    recovered_forecast.append(label)
        # Blurred/thick frames can obscure more than the two-pixel inset.
        # Retry a narrow strip farther inside the plot only at axis endpoints.
        if not edges and q in (start, end) and min(abs(x-l),abs(x-r)) < 1 and not (history and historical_mask is not None):
            inset = min(10, max(5, round(.008*(r-l))))
            anchor = l+inset if q == start else r-inset
            used_columns = [round(anchor)+dx for dx in (-1,0,1)]
            edges = [edge for xx in used_columns if (edge := column_edges(xx)) is not None]
            if edges:
                method = 'edge_inset_color_match'
            elif not history:
                edges = [edge for xx in used_columns if (edge := column_edges(xx, bridge_ink=True)) is not None]
                if edges:
                    method = 'edge_inset_forecast_overprint'
                    recovered_forecast.append(label)
            elif history and not mask[:, used_columns].any():
                edges = [edge for xx in used_columns if (edge := column_edges(xx, True)) is not None]
                if edges:
                    method = 'edge_inset_historical_recovery'
                    recovered_history.append(label)
        issue = ""
        if not edges:
            top_y = bottom_y = middle_y = None
            issue = "missing_or_ambiguous_curve"
            warnings.append(f"{label}: no unambiguous curve pixels")
        else:
            top_y, bottom_y = np.median(np.array(edges), axis=0).tolist()
            middle_y = (top_y + bottom_y) / 2
            if top_y <= t + 1 or bottom_y >= b - 1:
                issue = "curve_touches_axis_bound"
                warnings.append(f"{label}: curve touches a vertical-axis limit")
        if method.startswith('edge_inset') and edges:
            issue = (issue+';' if issue else '')+'edge_sample_inset'
            warnings.append(f"{label}: sampled {inset} pixels inside the plot to avoid the frame")
        if label in overrides:
            middle_y = float(overrides[label]['pixel_y'])
            issue = (issue+';' if issue else '')+'manual_override'
            method = 'manual_override'
        row_method = method if (edges or label in overrides) else 'missing'
        row = dict(country=country, vintage=vintage)
        if sampling == 'monthly':
            row['period'] = label
        else:
            row['quarter'] = label
        row.update(dict(
            series="historical" if history else "forecast_band_midpoint",
            inflation_pct_yoy=None if middle_y is None else round(to_value(middle_y), 4),
            # Bounds describe the visible darkest band, not estimation error.
            central_band_lower_pct=None if history or method == 'visible_forecast_centerline' or bottom_y is None else round(to_value(bottom_y), 4),
            central_band_upper_pct=None if history or method == 'visible_forecast_centerline' or top_y is None else round(to_value(top_y), 4),
            pixel_x=round(x, 3), pixel_y=middle_y,
            extraction_method=row_method,
            sampled_x_min=min(used_columns), sampled_x_max=max(used_columns),
            source_file=path.name, source_path=str(path.resolve()),
            calibration_status=c.get("review_status", "unreviewed"), quality_flag=issue))
        rows.append(row)
    if recovered_forecast:
        warnings.append(f"Forecast overprint recovery used for {len(recovered_forecast)} points; check the magenta overlay")
    if recovered_history:
        warnings.append(f"Historical colour recovery used for {len(recovered_history)} points; check the orange overlay")
    if c.get("review_status") not in ("reviewed", "seed_visually_checked"):
        warnings.append("Calibration has not been visually reviewed")
    return Result(rows, warnings, c)
