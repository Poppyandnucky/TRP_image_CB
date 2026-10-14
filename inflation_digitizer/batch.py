from __future__ import annotations
import argparse
import hashlib
import html
import json
from .configuration import load_working_config, input_directory
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import pandas as pd
from .core import extract, report_sort_key, month_to_quarter, quarter_number
from .overlay import save_overlay
from .forecast_export import convert_frame

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config/THA.json'


def discover(source: Path, timeout: int) -> list[str]:
    # Some cloud placeholder directories block during enumeration. Keep that
    # operation in a child process, which can be stopped on timeout.
    code = r"""
import os,json,sys
root=sys.argv[1]
names=[]
for current, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for name in files:
        if name.lower().endswith('.png') and not name.startswith('.'):
            names.append(os.path.relpath(os.path.join(current,name), root).replace(os.sep,'/'))
print(json.dumps(sorted(names)))
"""
    try:
        proc = subprocess.run([sys.executable, '-c', code, str(source)], capture_output=True,
                              text=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'Directory listing timed out after {timeout}s: {source}. Open it in Finder and make the PNGs available offline, or use --known-only / --image FILENAME.') from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip()) from exc
    return sorted(json.loads(proc.stdout), key=report_sort_key)


def resolve_output(config):
    output = Path(config['output_directory']).expanduser()
    return output if output.is_absolute() else ROOT / output


def sampling_from_name(name: str, calibration: dict | None = None) -> str:
    parts = {part.lower() for part in Path(name).parts}
    if 'monthly' in parts:
        return 'monthly'
    if 'quarterly' in parts:
        return 'quarterly'
    return (calibration or {}).get('sampling', 'quarterly')


def calibration_key(calibrations: dict, name: str, sampling: str) -> str:
    if name in calibrations:
        return name
    basename = Path(name).name
    if sampling == 'quarterly' and basename in calibrations:
        return basename
    return name


def safe_stem(name: str) -> str:
    return Path(name).with_suffix('').as_posix().replace('/', '__')


def validate_image_name(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('--image must be a PNG filename or relative path inside the configured input directory')


QUARTERLY_COLUMNS = ['country', 'vintage', 'quarter', 'series', 'inflation_pct_yoy',
                     'central_band_lower_pct', 'central_band_upper_pct', 'pixel_x', 'pixel_y',
                     'extraction_method', 'sampled_x_min', 'sampled_x_max', 'source_file',
                     'source_path', 'calibration_status', 'quality_flag']


def aggregate_monthly_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(columns=QUARTERLY_COLUMNS), pd.DataFrame(columns=['source_file', 'quarter', 'reason'])
    rows, warnings = [], []
    monthly = frame.copy()
    monthly['quarter'] = monthly['period'].map(month_to_quarter)
    monthly['month_in_quarter'] = monthly['period'].map(lambda value: int(str(value)[5:7]) % 3)
    for (source_file, vintage, quarter), group in monthly.groupby(['source_file', 'vintage', 'quarter'], sort=True):
        present = set(group['month_in_quarter'])
        if present != {0, 1, 2} or len(group) != 3:
            warnings.append(dict(source_file=source_file, quarter=quarter, reason='incomplete_monthly_quarter'))
            continue
        if group['series'].nunique(dropna=False) != 1:
            warnings.append(dict(source_file=source_file, quarter=quarter, reason='mixed_historical_forecast_months'))
            continue
        values = pd.to_numeric(group['inflation_pct_yoy'], errors='coerce')
        if values.isna().any():
            warnings.append(dict(source_file=source_file, quarter=quarter, reason='missing_monthly_value'))
            continue
        first = group.iloc[0].to_dict()
        row = {column: first.get(column, '') for column in QUARTERLY_COLUMNS}
        row['quarter'] = quarter
        row['series'] = group['series'].iloc[0]
        row['inflation_pct_yoy'] = round(float(values.mean()), 4)
        row['central_band_lower_pct'] = ''
        row['central_band_upper_pct'] = ''
        row['pixel_x'] = ''
        row['pixel_y'] = ''
        row['extraction_method'] = 'monthly_average'
        row['sampled_x_min'] = ''
        row['sampled_x_max'] = ''
        flags = [flag for flag in ';'.join(group['quality_flag'].fillna('').astype(str)).split(';') if flag]
        flags.append('aggregated_from_monthly')
        row['quality_flag'] = ';'.join(dict.fromkeys(flags))
        rows.append(row)
    return (pd.DataFrame(rows, columns=QUARTERLY_COLUMNS),
            pd.DataFrame(warnings, columns=['source_file', 'quarter', 'reason']))


def _has_invalid_quarter_labels(frame: pd.DataFrame) -> bool:
    try:
        for column in ('vintage', 'quarter'):
            if column not in frame.columns:
                return True
            for value in frame[column].dropna().astype(str).unique():
                quarter_number(value)
    except ValueError:
        return True
    return False


def refresh_combined_outputs(out: Path, variable='CPI') -> pd.DataFrame:
    quarterly_frames = []
    warnings = pd.DataFrame(columns=['source_file', 'quarter', 'reason'])
    tables = out/'tables'
    if tables.is_dir():
        for path in sorted(tables.glob('*.csv')):
            frame = pd.read_csv(path)
            frame['_source_sampling'] = 'quarterly'
            frame['_raw_table'] = path.name
            if 'source_path' in frame.columns:
                frame['_source_path_exists'] = frame['source_path'].map(lambda value: Path(str(value)).exists())
            else:
                frame['_source_path_exists'] = False
            if _has_invalid_quarter_labels(frame) and not frame['_source_path_exists'].any():
                warning = pd.DataFrame([{
                    'source_file': frame['source_file'].iloc[0] if 'source_file' in frame.columns and len(frame) else path.name,
                    'quarter': '',
                    'reason': 'stale_invalid_raw_table_missing_source_path',
                }])
                warnings = pd.concat([warnings, warning], ignore_index=True)
                continue
            quarterly_frames.append(frame)
    monthly_frames = []
    monthly_tables = out/'monthly_tables'
    if monthly_tables.is_dir():
        for path in sorted(monthly_tables.glob('*.csv')):
            monthly_frames.append(pd.read_csv(path))
    if monthly_frames:
        monthly_quarterly, monthly_warnings = aggregate_monthly_frame(pd.concat(monthly_frames, ignore_index=True))
        warnings = pd.concat([warnings, monthly_warnings], ignore_index=True)
        if not monthly_quarterly.empty:
            monthly_quarterly['_source_sampling'] = 'monthly'
            quarterly_frames.append(monthly_quarterly)
    combined = (pd.concat(quarterly_frames, ignore_index=True)
                if quarterly_frames else pd.DataFrame(columns=QUARTERLY_COLUMNS))
    if not combined.empty:
        duplicate_keys = ['country', 'vintage', 'quarter']
        duplicates = combined.duplicated(duplicate_keys, keep=False)
        if duplicates.any():
            duplicate_frame = combined.loc[duplicates].copy()
            mixed_sampling = duplicate_frame.groupby(duplicate_keys)['_source_sampling'].nunique()
            if (mixed_sampling > 1).all():
                replaced = duplicate_frame.loc[duplicate_frame['_source_sampling'] == 'quarterly',
                                               duplicate_keys + ['source_file']]
                replacement_warnings = replaced.rename(columns={'source_file': 'replaced_source_file'})
                replacement_warnings['reason'] = 'monthly_replaced_quarterly_duplicate'
                warnings = pd.concat([warnings, replacement_warnings.rename(columns={'replaced_source_file': 'source_file'})[
                    ['source_file', 'quarter', 'reason']]], ignore_index=True)
                combined['_priority'] = combined['_source_sampling'].map({'quarterly': 0, 'monthly': 1}).fillna(0)
                combined = (combined.sort_values(duplicate_keys + ['_priority'], kind='stable')
                            .drop_duplicates(duplicate_keys, keep='last')
                            .drop(columns=['_priority']))
            else:
                # Moving screenshots into quarterly/ can leave an old raw CSV
                # whose source_path no longer exists plus a new raw CSV for the
                # same report. Prefer the live source without deleting the
                # user's preserved raw files.
                duplicate_groups = duplicate_frame.groupby(duplicate_keys, sort=False)
                keep_indexes = []
                stale_indexes = []
                can_resolve = True
                for _, group in duplicate_groups:
                    live = group[group['_source_path_exists']]
                    if len(live) == 1 and len(group) == 2:
                        keep_indexes.append(live.index[0])
                        stale_indexes.extend(group.index.difference(live.index).tolist())
                    else:
                        can_resolve = False
                        break
                if can_resolve:
                    stale = combined.loc[stale_indexes, duplicate_keys + ['source_file']]
                    stale_warnings = stale.copy()
                    stale_warnings['reason'] = 'stale_raw_duplicate_replaced_by_existing_source_path'
                    warnings = pd.concat([warnings, stale_warnings[['source_file', 'quarter', 'reason']]],
                                         ignore_index=True)
                    combined = combined.drop(index=stale_indexes)
                else:
                    examples = combined.loc[duplicates, duplicate_keys + ['source_file', '_raw_table']].head(6).to_dict('records')
                    raise ValueError(f'Duplicate quarterly rows after combining raw outputs: {examples}')
        combined = combined.drop(columns=['_source_sampling', '_raw_table', '_source_path_exists'], errors='ignore')
    combined.to_csv(out/'quarterly_estimates.csv', index=False)
    with pd.ExcelWriter(out/'quarterly_estimates.xlsx', engine='openpyxl') as writer:
        combined.to_excel(writer, sheet_name='Estimates', index=False)
        manifest = pd.read_csv(out/'manifest.csv') if (out/'manifest.csv').exists() else pd.DataFrame()
        manifest.to_excel(writer, sheet_name='Review', index=False)
        if not warnings.empty:
            warnings.to_excel(writer, sheet_name='Monthly aggregation warnings', index=False)
    warnings.to_csv(out/'monthly_aggregation_warnings.csv', index=False)
    if not combined.empty:
        converted = convert_frame(combined, variable=variable)
        converted.to_csv(out/'forecasts.csv', index=False)
        formatted = out/'formatted_tables'
        formatted.mkdir(exist_ok=True)
        for report, group in converted.groupby('report', sort=True):
            country = group['country'].iloc[0]
            group.to_csv(formatted/f'{country}_{report.replace("Q", "_Q")}.csv', index=False)
    return combined


def write_review(out, entries):
    cards = []
    for entry in entries:
        title = html.escape(entry['source_file'])
        status = html.escape(entry['status'])
        msg = html.escape(entry.get('message', ''))
        image = f'<a href="{html.escape(entry["overlay"])}"><img loading="lazy" src="{html.escape(entry["overlay"])}" alt="{title}"></a>' if entry.get('overlay') else ''
        cards.append(f'<article><h2>{title}</h2><p><b>{status}</b> {msg}</p>{image}</article>')
    content = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Inflation digitization review</title>
<style>body{font:16px system-ui,sans-serif;max-width:1250px;margin:30px auto;padding:0 20px;color:#172c43}article{border-top:1px solid #ccd4dc;padding:18px 0}img{max-width:100%;height:auto}h2{font-size:20px}p{line-height:1.5}</style>
<h1>Inflation digitization review</h1><p>Check axis dates, vertical limits, the historical/forecast boundary, and the pink curve against the darkest band. Unknown images have no extracted values until calibrated. Midpoints are not verified statistical medians.</p>'''
    (out/'review.html').write_text(content+'\n'.join(cards)+'</html>', encoding='utf-8')


def run(config_path: Path, known_only=False, images=None):
    config = load_working_config(config_path)
    source = input_directory(config)
    calibrations = config.get('images', {})
    if images:
        names = images
    elif known_only:
        names = sorted(calibrations, key=report_sort_key)
    else:
        names = discover(source, config.get('directory_timeout_seconds', 20))
    if not names:
        raise RuntimeError('No PNG files found.')
    out = resolve_output(config)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, entries = [], []
    for name in names:
        validate_image_name(name)
        path = source/name
        inferred_sampling = sampling_from_name(name)
        key = calibration_key(calibrations, name, inferred_sampling)
        c = calibrations.get(key)
        if not c:
            entries.append(dict(source_file=name, status='needs_calibration', message=f'Run calibrate.py --image "{name}"'))
            print(f'NEEDS CALIBRATION: {name}', flush=True)
            continue
        if known_only and not path.exists():
            entries.append(dict(source_file=name, status='skipped_missing_image',
                                message='Saved calibration exists, but screenshot file is no longer at this path.'))
            print(f'SKIPPED MISSING IMAGE: {name}', flush=True)
            continue
        sampling = sampling_from_name(name, c)
        c = dict(c)
        c.setdefault('sampling', sampling)
        try:
            result = extract(path, c, config['country'])
            rel_overlay = f'overlays/{safe_stem(name)}_overlay.png'
            save_overlay(path, result, out/rel_overlay)
            table_dir = out/('monthly_tables' if sampling == 'monthly' else 'tables')
            table_dir.mkdir(exist_ok=True)
            frame = pd.DataFrame(result.rows)
            frame.to_csv(table_dir/f'{safe_stem(name)}.csv', index=False)
            (out/'calibrations').mkdir(exist_ok=True)
            audit = dict(calibration=c, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         generated_utc=datetime.now(timezone.utc).isoformat(), warnings=result.warnings)
            (out/'calibrations'/f'{safe_stem(name)}.json').write_text(json.dumps(audit, indent=2))
            all_rows.extend(result.rows)
            entries.append(dict(source_file=name, status='review_flags' if result.warnings else 'extracted',
                                message='; '.join(result.warnings), overlay=rel_overlay))
            unit = 'months' if sampling == 'monthly' else 'quarters'
            print(f'EXTRACTED: {name} ({len(result.rows)} {unit}, {len(result.warnings)} flags)', flush=True)
        except Exception as exc:
            entries.append(dict(source_file=name, status='error', message=str(exc)))
            print(f'ERROR: {name}: {exc}', flush=True)
    manifest = pd.DataFrame(entries)
    manifest.to_csv(out/'manifest.csv', index=False)
    refresh_combined_outputs(out, variable=config.get('variable', 'CPI'))
    write_review(out, entries)
    print(f'\nReview: {out / "review.html"}\nData: {out / "quarterly_estimates.xlsx"}', flush=True)
    return entries


def main():
    parser = argparse.ArgumentParser(description='Extract calibrated inflation charts and create review overlays.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--known-only', action='store_true', help='Only process saved calibrations; do not list the input folder.')
    parser.add_argument('--image', action='append', help='Process one filename; can be repeated.')
    args = parser.parse_args()
    try:
        entries = run(args.config.resolve(), args.known_only, args.image)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Cannot run: {exc}', file=sys.stderr)
        return 1
    return 2 if any(e['status'] in ('needs_calibration', 'error') for e in entries) else 0
