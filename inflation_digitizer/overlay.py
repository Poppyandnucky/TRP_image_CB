"""Deterministic overlays preserving the original chart pixels beneath annotations."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .core import Result


def font(size, bold=False):
    # Matplotlib bundles DejaVu Sans on all supported platforms.
    from matplotlib import get_data_path
    filename = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(str(Path(get_data_path()) / 'fonts/ttf' / filename), size)


def save_overlay(source: Path, result: Result, destination: Path):
    im = Image.open(source).convert('RGB')
    width, height = im.size
    scale = max(1, width / 1100)
    footer = round(165 * scale)
    canvas = Image.new('RGB', (width, height + footer), 'white')
    canvas.paste(im, (0, 0))
    draw = ImageDraw.Draw(canvas)
    c = result.calibration
    l, t, r, b = c['plot_box']
    green, orange, pink = '#00816f', '#e65c00', '#d60078'
    # Dashed calibration rectangle.
    for y in (t, b):
        for x in range(round(l), round(r), 13):
            draw.line((x, y, min(x+8, r), y), fill=green, width=2)
    for x in (l, r):
        for y in range(round(t), round(b), 13):
            draw.line((x, y, x, min(y+8, b)), fill=green, width=2)
    previous = None
    for row in result.rows:
        point = None if row['pixel_y'] is None else (row['pixel_x'], row['pixel_y'])
        color = orange if row['series'] == 'historical' else pink
        if previous is not None and point is not None:
            draw.line([previous, point], fill=color, width=max(2, round(2.5 * scale)))
        previous = point
    for row in result.rows:
        if row['pixel_y'] is None:
            continue
        x, y = row['pixel_x'], row['pixel_y']
        period = row.get('period', row.get('quarter'))
        last_historical = period == c['last_historical']
        radius = (9 if last_historical else 6) * scale
        bounds = (x-radius, y-radius, x+radius, y+radius)
        shape = draw.rectangle if last_historical else draw.ellipse
        shape(bounds, fill=None, outline='black', width=max(2, round(2*scale)))

    # Labels sit wholly inside the calibrated box. Stack the lower-left Y
    # limit above the first quarter so the two labels do not cover each other.
    pad = max(3, round(6*scale))
    text_pad = max(2, round(3*scale))
    labels = [c['x_start'].replace('Q', ' Q'), c['x_end'].replace('Q', ' Q'),
              f"{c['y_min']:g}", f"{c['y_max']:g}"]
    label_font = font(round(17*scale), bold=True)
    while label_font.size > 8 and (
            max(draw.textlength(text, font=label_font) for text in labels) + 2*text_pad > (r-l-3*pad)/2
            or 3*(label_font.size+2*text_pad)+3*pad > b-t):
        label_font = font(label_font.size-1, bold=True)
    def label(text, left, top, right_align=False):
        bounds = draw.textbbox((0,0), text, font=label_font)
        box_width = bounds[2]-bounds[0]+2*text_pad
        box_height = bounds[3]-bounds[1]+2*text_pad
        if right_align:
            left -= box_width
        draw.rectangle((left,top,left+box_width,top+box_height), fill='white')
        draw.text((left+text_pad-bounds[0],top+text_pad-bounds[1]), text, font=label_font, fill=green)
        return box_height
    heights = [draw.textbbox((0,0),text,font=label_font)[3]-draw.textbbox((0,0),text,font=label_font)[1]+2*text_pad for text in labels]
    label(labels[0], l+pad, b-pad-heights[0])
    label(labels[1], r-pad, b-pad-heights[1], right_align=True)
    label(labels[2], l+pad, b-2*pad-heights[0]-heights[2])
    label(labels[3], l+pad, t+pad)
    # Footer text is outside the original PNG; fit long labels to width.
    lines = [
        (f'{source.stem} | Digitization check', '#172c43', 20),
        ('Orange: historical curve.  Magenta: darkest-band midpoint estimate.', '#172c43', 16),
        ('Open circles: sampled points.  Open square: last historical point.', '#172c43', 15),
        ('Darkest-band midpoint is a proxy; the statistical median is not verified.', '#536276', 15),
        (f"Review: {c.get('review_status', 'unreviewed')} | Flags: {len(result.warnings)} (see review report)", '#536276', 14),
    ]
    x, y = round(18*scale), height+round(15*scale)
    for text, color, size in lines:
        text_font = font(round(size*scale))
        while draw.textlength(text, font=text_font) > width-2*x and text_font.size > 9:
            text_font = font(text_font.size-1)
        draw.text((x, y), text, font=text_font, fill=color)
        y += round(28*scale)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
