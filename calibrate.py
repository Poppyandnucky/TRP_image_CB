"""Interactive per-image calibration. Run in a desktop PyCharm session."""
from __future__ import annotations
import argparse
import json
from inflation_digitizer.configuration import load_working_config, input_directory
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from inflation_digitizer.batch import DEFAULT_CONFIG, discover, run
from inflation_digitizer.core import extract, quarter_number, quarter_label, validate_calibration, report_sort_key
from inflation_digitizer.plot_box import detect_plot_box


def launch(config_path, filename, template=None, *, queue_position=None, advance_end=False):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox, Button
    config = load_working_config(config_path)
    if Path(filename).name != filename:
        raise ValueError('Use a filename, not a full path, for --image')
    path = input_directory(config)/filename
    im = Image.open(path).convert('RGB')
    pixels = np.asarray(im)
    w, h = im.size
    saved = config.get('images', {}).get(filename)
    seed = dict(config.get('calibration_defaults', {}))
    seed.update(config.get('images', {}).get(template, {}))
    seed.update(config.get('image_defaults', {}).get(filename, {}))
    seed.update(saved or {})
    if template and template not in config.get('images', {}):
        raise ValueError(f'Template {template!r} is not in this config')
    advanced = bool(advance_end and not saved and seed.get('x_end'))
    if advanced:
        seed['x_end'] = quarter_label(quarter_number(seed['x_end']) + 1)
    suggested = (list(saved['plot_box']) if saved and saved.get('image_size') == [w,h]
                 and saved.get('plot_box') else detect_plot_box(im, allow_open_axes=seed.get('plot_box_style') == 'open_axes'))
    fig = plt.figure(figsize=(18, 9))
    fig.canvas.manager.set_window_title(f'Calibrate {filename}')
    ax = fig.add_axes([0.035, 0.15, 0.56, 0.79])
    ax.imshow(im)
    ax.set_title(filename if queue_position is None else f'{filename} — {queue_position}')
    ax.set_axis_off()
    cal_x, manual_x, col_w = 0.62, 0.805, 0.165
    cal_panel = fig.add_axes([cal_x - 0.015, 0.135, col_w + 0.03, 0.815])
    manual_panel = fig.add_axes([manual_x - 0.015, 0.135, col_w + 0.03, 0.815])
    for panel in (cal_panel, manual_panel):
        panel.set_axis_off()
        panel.set_zorder(-1)
        panel.patch.set_visible(True)
        panel.patch.set_facecolor('#f7f8fa')
        panel.patch.set_edgecolor('#cfd6df')
        panel.patch.set_linewidth(1)
    fig.text(cal_x, 0.925, 'Calibration', fontsize=12, weight='bold')
    instruction = fig.text(cal_x, 0.89, '1. Plot TOP LEFT corner', fontsize=11, color='#00816f')
    fig.text(manual_x, 0.925, 'Manual Adjustment', fontsize=12, weight='bold')
    fig.text(manual_x, 0.89, 'Enabled after preview.', fontsize=9, color='#7b8794')
    fig.text(cal_x, 0.84, 'Accept the orange box, or\ncorrect its corners below.', fontsize=10, va='top')
    fields = {}
    quarter_buttons = []  # Keep widget objects alive for their callbacks.
    field_specs = [
        ('x_start', 'First quarter', '2022Q1'),
        ('x_end', 'Last quarter', '2027Q3'),
        ('y_min', 'Y minimum', '-4'), ('y_max', 'Y maximum', '10'),
        ('color_tolerance', 'Forecast tolerance', '22'),
        ('historical_color_tolerance', 'Historical tolerance', '45'),
    ]
    field_ys = [0.68, 0.59, 0.50, 0.41, 0.325, 0.25]
    for i, (key, label, default) in enumerate(field_specs):
        y = field_ys[i]
        fig.text(cal_x, y+0.042, label + (' (+1 quarter)' if key == 'x_end' and advanced else ''), fontsize=10)
        is_quarter = key in ('x_start', 'x_end')
        is_axis_limit = key in ('y_min', 'y_max')
        bounds = [cal_x + 0.036, y, col_w - 0.072, 0.038] if is_quarter or is_axis_limit else [cal_x, y, col_w, 0.038]
        fields[key] = TextBox(fig.add_axes(bounds), '', initial=str(seed.get(key, default)))
        if is_quarter or is_axis_limit:
            for left, delta, label in [(cal_x, -1, '◀'), (cal_x + col_w - 0.03, 1, '▶')]:
                button = Button(fig.add_axes([left, y, 0.03, 0.038]), label if is_quarter else ('−' if delta < 0 else '+'))
                def increment(event, field=key, amount=delta):
                    try:
                        if field in ('y_min', 'y_max'):
                            value = float(fields[field].text) + amount
                            if not np.isfinite(value):
                                raise ValueError('Enter a finite Y-axis value.')
                            fields[field].set_val(f'{value:g}')
                            fig.canvas.draw_idle()
                            return
                        value = quarter_number(fields[field].text) + amount
                        if not 4000 <= value <= 39999:
                            raise ValueError('Quarter must use a four-digit year.')
                        fields[field].set_val(quarter_label(value))
                    except ValueError as exc:
                        note.set_text(str(exc))
                        note.set_color('#a00000')
                    fig.canvas.draw_idle()
                button.on_clicked(increment)
                quarter_buttons.append(button)
    note = fig.text(0.035, 0.055, 'Accept the box or select two corners; then select the historical endpoint and darkest band.', fontsize=10)
    picks, markers = [], []
    adjustment = [None]
    proposal, = ax.plot([], [], '--o', color='#e69500', lw=1.5, ms=5, visible=False)
    def show_proposal():
        if suggested is not None:
            l,t,r,b = suggested
            proposal.set_data([l,r,r,l,l], [t,t,b,b,t])
            proposal.set_visible(True)
        accept_button.set_active(suggested is not None)
        accept_button.ax.set_alpha(1 if suggested is not None else .4)

    def show_selected_box():
        (l,t),(r,b) = picks[:2]
        markers.extend(ax.plot([l,r,r,l,l], [t,t,b,b,t], '--', color='#00816f', lw=1))
        quarter_readout.set_text('Hover over the endpoint.')
        note.set_text('The vertical guide snaps to quarters. Click the final historical point to select its quarter.')
        instruction.set_text('3. LAST HISTORICAL point')

    # Span the whole displayed image, including the printed quarter labels.
    quarter_guide = ax.axvline(0, color='#d60078', linestyle='--', lw=1,
                              visible=False, zorder=5)
    quarter_readout = fig.text(cal_x, 0.235, '', fontsize=9, color='#d60078', va='top')
    missing_readout = fig.text(manual_x, 0.735, '', fontsize=9, color='#a00000', va='top')
    adjustment_readout = fig.text(manual_x, 0.835, 'Selected Adjustment Point: none\nCurrent Value: --',
                                  fontsize=10, color='#536276', va='top')

    # A white outline keeps the moving crosshairs visible over dark fan bands.
    import matplotlib.patheffects as path_effects
    crosshair_style = dict(color='black', lw=.8, visible=False, zorder=10,
                           path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
    cursor_x = ax.axvline(0, gid='cursor-crosshair-x', **crosshair_style)
    cursor_y = ax.axhline(0, gid='cursor-crosshair-y', **crosshair_style)

    def hide_crosshairs():
        cursor_x.set_visible(False)
        cursor_y.set_visible(False)

    def nearest_quarter(x):
        (l, t), (r, b) = picks[:2]
        start = quarter_number(fields['x_start'].text)
        end = quarter_number(fields['x_end'].text)
        if r <= l or b <= t or end <= start:
            raise ValueError('Check plot corners and first/last quarters.')
        q = max(start, min(end, round(start + (x-l)*(end-start)/(r-l))))
        return l + (r-l)*(q-start)/(end-start), quarter_label(q)

    def guide_at(x, selected=False):
        try:
            snapped_x, label = nearest_quarter(x)
        except ValueError:
            quarter_guide.set_visible(False)
            quarter_readout.set_text('Check the axis values.')
            return None
        quarter_guide.set_xdata([snapped_x, snapped_x])
        quarter_guide.set_visible(True)
        quarter_readout.set_text(('Selected Historical Endpoint: ' if selected else 'Quarter: ') + display_quarter(label))
        return snapped_x

    def move(event):
        x, y = getattr(event, 'xdata', None), getattr(event, 'ydata', None)
        if workflow['state'] == PREVIEW:
            return
        if workflow['state'] == MANUAL_ADJUSTMENT:
            if preview['dragging'] and getattr(event, 'inaxes', None) == ax and y is not None:
                move_selected_point(y)
            return
        on_image = (getattr(event, 'inaxes', None) == ax and
                    x is not None and y is not None and
                    not getattr(fig.canvas.manager.toolbar, 'mode', ''))
        if on_image and len(picks) < 4:
            cursor_x.set_xdata([x, x])
            cursor_y.set_ydata([y, y])
            cursor_x.set_visible(True)
            cursor_y.set_visible(True)
        else:
            hide_crosshairs()
        if len(picks) == 2:
            if on_image and picks[0][0] <= x <= picks[1][0]:
                guide_at(x)
            else:
                quarter_guide.set_visible(False)
                quarter_readout.set_text('Hover over the endpoint.')
        fig.canvas.draw_idle()

    def leave(event):
        preview['dragging'] = False
        hide_crosshairs()
        if len(picks) == 2:
            quarter_guide.set_visible(False)
            quarter_readout.set_text('Hover over the endpoint.')
        fig.canvas.draw_idle()

    def release(event):
        preview['dragging'] = False

    def refresh_guide(text):
        if len(picks) >= 3:
            guide_at(picks[2][0], selected=True)
        elif len(picks) == 2:
            quarter_guide.set_visible(False)
            quarter_readout.set_text('Hover over the endpoint.')
        fig.canvas.draw_idle()

    for key in ('x_start', 'x_end'):
        fields[key].on_text_change(refresh_guide)
    hints = ['1. Plot TOP LEFT corner', '2. Plot BOTTOM RIGHT corner',
             '3. LAST HISTORICAL point', '4. HISTORICAL line colour',
             '5. INSIDE darkest forecast band', 'Review detected points']
    CALIBRATION, PREVIEW, MANUAL_ADJUSTMENT = 'calibration', 'preview', 'manual_adjustment'
    workflow = {'state': CALIBRATION}

    preview = dict(rows=[], automatic_rows=[], overrides={},
                   artists=[], selected=None, dragging=False, calibration=None,
                   adding_missing=None, missing_cycle=0)

    def display_quarter(label):
        return label.replace('Q', ' Q')

    def pixel_color(x, y):
        return pixels[min(h-1, max(0, round(y))), min(w-1, max(0, round(x)))].tolist()

    def y_value(calibration, y):
        l, t, r, b = calibration['plot_box']
        span = calibration['y_max'] - calibration['y_min']
        return calibration['y_max'] - (y - t) * span / (b - t)

    def build_calibration(include_overrides=False):
        if len(picks) != 5:
            raise ValueError('Accept or select the box, then click the endpoint, historical line, and forecast band.')
        (l,t),(r,b),(hx,hy),(histx,histy),(cx,cy) = picks
        start, end = (quarter_number(fields['x_start'].text),quarter_number(fields['x_end'].text))
        if r <= l or b <= t:
            raise ValueError('Select top left, then bottom right in that order.')
        for label, x, y in [('historical endpoint', hx, hy), ('historical colour sample', histx, histy),
                            ('forecast colour sample', cx, cy)]:
            if not l <= x <= r or not t <= y <= b:
                raise ValueError(f'The {label} must be inside the plot.')
        hist = round(start+(hx-l)*(end-start)/(r-l))
        forecast_color = pixel_color(cx, cy)
        if max(forecast_color)-min(forecast_color) < 20:
            raise ValueError('Selected forecast colour is nearly grey. Click inside the coloured darkest band, away from grid lines.')
        c = dict(image_size=[w,h], plot_box=[round(v,2) for v in (l,t,r,b)],
                 y_min=float(fields['y_min'].text), y_max=float(fields['y_max'].text),
                 x_start=quarter_label(start), x_end=quarter_label(end), last_historical=quarter_label(hist),
                 historical_rgb=pixel_color(histx, histy),
                 historical_color_tolerance=float(fields['historical_color_tolerance'].text),
                 curve_rgb=forecast_color, color_tolerance=float(fields['color_tolerance'].text),
                 review_status='manually_calibrated',
                 notes='Review generated overlay before changing review_status to reviewed.')
        for key in ('plot_box_style', 'ignore_regions'):
            if key in seed:
                c[key] = seed[key]
        if include_overrides and preview['overrides']:
            c['manual_point_overrides'] = {
                quarter: {'pixel_y': round(float(values['pixel_y']), 3)}
                for quarter, values in sorted(preview['overrides'].items())
            }
        validate_calibration(c, (w,h))
        return c

    def clear_preview_artists():
        for artist in preview['artists']:
            artist.remove()
        preview['artists'].clear()

    def draw_preview():
        clear_preview_artists()
        rows = preview['rows']
        selected = preview['selected']
        previous = None
        for row in rows:
            if row['pixel_y'] is None:
                previous = None
                continue
            point = (row['pixel_x'], row['pixel_y'])
            color = '#e65c00' if row['series'] == 'historical' else '#d60078'
            if previous is not None:
                preview['artists'].extend(ax.plot([previous[0], point[0]], [previous[1], point[1]],
                                                  color=color, lw=1.5, alpha=.75, zorder=20))
            previous = point
        for index, row in enumerate(rows):
            if row['pixel_y'] is None:
                continue
            selected_row = index == selected
            color = '#ffbf00' if selected_row else ('#e65c00' if row['series'] == 'historical' else '#d60078')
            size = 10 if selected_row else 6
            preview['artists'].extend(ax.plot(row['pixel_x'], row['pixel_y'], 'o',
                                              ms=size, mfc='none', mec=color, mew=2.2,
                                              zorder=25))
        fig.canvas.draw_idle()

    def selected_message():
        index = preview['selected']
        if index is None:
            return 'Selected Adjustment Point: none\nCurrent Value: --'
        row = preview['rows'][index]
        value = row['inflation_pct_yoy']
        value_text = 'missing' if value is None else f'{value:.4g}%'
        return f"Selected Adjustment Point: {display_quarter(row['quarter'])}\nCurrent Value: {value_text}"

    def update_preview_note():
        adjustment_readout.set_text(selected_message())
        if workflow['state'] == MANUAL_ADJUSTMENT:
            adjustment_readout.set_color('black' if preview['selected'] is not None else '#536276')
        else:
            adjustment_readout.set_color('#9aa3ad')

    def missing_rows():
        return [row for row in preview['rows'] if row['pixel_y'] is None]

    def update_missing_readout():
        missing = [display_quarter(row['quarter']) for row in missing_rows()]
        if not missing:
            missing_readout.set_text('')
        elif len(missing) == 1:
            missing_readout.set_text(f'Missing data point: {missing[0]}')
        else:
            missing_readout.set_text('Missing data points:\n' + ', '.join(missing))
        missing_readout.set_color('#a00000' if workflow['state'] == MANUAL_ADJUSTMENT else '#b7b7b7')

    def set_calibration_markers_visible(visible):
        # Keep the accepted plot box visible; hide only temporary pick markers.
        for marker in markers:
            if marker.get_linestyle() == '--':
                marker.set_visible(True)
            else:
                marker.set_visible(visible)
        proposal.set_visible(visible and suggested is not None and len(picks) < 2)

    def style_button_enabled(button, enabled):
        button.set_active(enabled)
        button.ax.set_facecolor('#e0e0e0' if enabled else '#f4f5f6')
        for text in button.ax.texts:
            text.set_color('black' if enabled else '#9aa3ad')
        for spine in button.ax.spines.values():
            spine.set_edgecolor('black' if enabled else '#c7ced6')

    def set_manual_controls_enabled(enabled):
        for button in (add_missing_button, reset_manual_button):
            style_button_enabled(button, enabled)
        adjustment_readout.set_color('#536276' if enabled else '#9aa3ad')
        missing_readout.set_color('#a00000' if enabled else '#b7b7b7')

    def set_stage(stage):
        workflow['state'] = stage
        next_manual_button.set_active(stage == PREVIEW)
        next_manual_button.ax.set_alpha(1 if stage == PREVIEW else .35)
        set_manual_controls_enabled(stage == MANUAL_ADJUSTMENT)
        if stage == PREVIEW:
            set_calibration_markers_visible(False)
            preview['selected'] = None
            preview['dragging'] = False
            preview['adding_missing'] = None
            instruction.set_text('Preview detected points')
            note.set_text('Preview only. Use Reset All to recalibrate, or continue to manual adjustment.')
            note.set_color('black')
            update_preview_note()
        elif stage == MANUAL_ADJUSTMENT:
            set_calibration_markers_visible(False)
            instruction.set_text('Manual point adjustment')
            note.set_text('Select and drag a point vertically. Add Missing Point fixes one missing quarter at a time.')
            note.set_color('black')
            update_preview_note()
        fig.canvas.draw_idle()

    def show_preview(reset_overrides=False, stage=PREVIEW):
        try:
            if reset_overrides:
                preview['overrides'].clear()
            calibration = build_calibration(include_overrides=False)
            result = extract(path, calibration, config.get('country', 'TST'))
            preview['calibration'] = calibration
            preview['automatic_rows'] = [dict(row) for row in result.rows]
            rows = [dict(row) for row in result.rows]
            for row in rows:
                override = preview['overrides'].get(row['quarter'])
                if override is not None:
                    row['pixel_y'] = float(override['pixel_y'])
                    row['inflation_pct_yoy'] = round(y_value(calibration, row['pixel_y']), 4)
                    row['extraction_method'] = 'manual_override'
                    flags = [flag for flag in (row.get('quality_flag') or '').split(';') if flag]
                    if 'manual_override' not in flags:
                        flags.append('manual_override')
                    row['quality_flag'] = ';'.join(flags)
            preview['rows'] = rows
            if preview['selected'] is not None and preview['selected'] >= len(rows):
                preview['selected'] = None
            draw_preview()
            update_missing_readout()
            set_stage(stage)
        except (ValueError, OSError) as exc:
            note.set_text(str(exc))
            note.set_color('#a00000')
            fig.canvas.draw_idle()

    def select_preview_point(x, y):
        candidates = [(index, (row['pixel_x']-x)**2 + (row['pixel_y']-y)**2)
                      for index, row in enumerate(preview['rows']) if row['pixel_y'] is not None]
        if not candidates:
            return False
        index, distance = min(candidates, key=lambda item: item[1])
        if distance > 20**2:
            return False
        preview['selected'] = index
        preview['dragging'] = True
        preview['adding_missing'] = None
        update_preview_note()
        draw_preview()
        return True

    def mark_manual_override(row, y):
        calibration = preview['calibration']
        top, bottom = calibration['plot_box'][1], calibration['plot_box'][3]
        y = max(top, min(bottom, y))
        row['pixel_y'] = float(y)
        row['inflation_pct_yoy'] = round(y_value(calibration, y), 4)
        row['extraction_method'] = 'manual_override'
        flags = [flag for flag in (row.get('quality_flag') or '').split(';') if flag]
        if 'manual_override' not in flags:
            flags.append('manual_override')
        row['quality_flag'] = ';'.join(flags)
        preview['overrides'][row['quarter']] = {'pixel_y': float(y)}

    def move_selected_point(y):
        index = preview['selected']
        if index is None:
            return
        row = preview['rows'][index]
        mark_manual_override(row, y)
        update_missing_readout()
        update_preview_note()
        draw_preview()

    def choose_missing_point(event):
        if workflow['state'] != MANUAL_ADJUSTMENT:
            return
        missing = missing_rows()
        if not missing:
            note.set_text('No missing points remain.')
            note.set_color('black')
            fig.canvas.draw_idle()
            return
        row = missing[preview['missing_cycle'] % len(missing)]
        preview['missing_cycle'] += 1
        preview['selected'] = preview['rows'].index(row)
        preview['adding_missing'] = row['quarter']
        update_preview_note()
        note.set_text(f"Click the plot to set the Y value for {display_quarter(row['quarter'])}. X stays fixed.")
        note.set_color('black')
        draw_preview()

    def reset_calibration():
        workflow['state'] = CALIBRATION
        preview['selected'] = None
        preview['dragging'] = False
        preview['overrides'].clear()
        preview['rows'].clear()
        preview['automatic_rows'].clear()
        preview['calibration'] = None
        preview['adding_missing'] = None
        clear_preview_artists()
        hide_crosshairs()
        picks.clear()
        adjustment[0] = None
        quarter_guide.set_visible(False)
        quarter_readout.set_text('')
        missing_readout.set_text('')
        adjustment_readout.set_text('Selected Adjustment Point: none\nCurrent Value: --')
        adjustment_readout.set_color('#9aa3ad')
        note.set_text('Accept the box or select two corners; then select the historical endpoint and darkest band.')
        note.set_color('black')
        for marker in markers:
            marker.remove()
        markers.clear()
        show_proposal()
        instruction.set_text('Review suggested box' if suggested is not None else hints[0])
        next_manual_button.set_active(False)
        next_manual_button.ax.set_alpha(.35)
        set_manual_controls_enabled(False)
        fig.canvas.draw_idle()

    def click(event):
        if workflow['state'] == PREVIEW:
            return
        if workflow['state'] == MANUAL_ADJUSTMENT:
            if event.inaxes == ax and event.button == 1 and preview['adding_missing']:
                index = next((i for i, row in enumerate(preview['rows'])
                              if row['quarter'] == preview['adding_missing']), None)
                if index is not None and event.ydata is not None:
                    preview['selected'] = index
                    mark_manual_override(preview['rows'][index], event.ydata)
                    preview['adding_missing'] = None
                    update_missing_readout()
                    update_preview_note()
                    draw_preview()
                return
            if event.inaxes == ax and event.button == 1:
                select_preview_point(event.xdata, event.ydata)
            return
        if event.inaxes != ax or event.button != 1 or len(picks) >= 5:
            return
        if getattr(fig.canvas.manager.toolbar, 'mode', ''):
            return
        x, y = event.xdata, event.ydata
        if adjustment[0] is not None:
            offset = adjustment[0] * 2
            suggested[offset:offset+2] = [x,y]
            adjustment[0] = None
            show_proposal()
            instruction.set_text('Review corrected box')
            note.set_text('Click Accept box if the orange rectangle follows the plot axes.')
            fig.canvas.draw_idle()
            return
        if len(picks) == 0:
            proposal.set_visible(False)
            accept_button.set_active(False)
            accept_button.ax.set_alpha(.4)
        if len(picks) == 2:
            (l, t), (r, b) = picks
            if not l <= x <= r or not t <= y <= b:
                note.set_text('Click the historical endpoint inside the plot.')
                fig.canvas.draw_idle()
                return
            x = guide_at(x, selected=True)
            if x is None:
                fig.canvas.draw_idle()
                return
        picks.append((x, y))
        markers.extend(ax.plot(x, y, 'o', ms=7, mfc='none', mec='#d60078', mew=2))
        instruction.set_text(hints[len(picks)])
        if len(picks) == 5:
            hide_crosshairs()
            show_preview(reset_overrides=True, stage=PREVIEW)
        if len(picks) == 2:
            show_selected_box()
        fig.canvas.draw_idle()
    def reset(event):
        reset_calibration()

    def reset_manual_adjustments(event):
        if workflow['state'] != MANUAL_ADJUSTMENT:
            return
        preview['selected'] = None
        preview['adding_missing'] = None
        show_preview(reset_overrides=True, stage=MANUAL_ADJUSTMENT)
        note.set_text('Manual adjustments reset to the current automatic extraction.')
        note.set_color('black')
        fig.canvas.draw_idle()

    def accept_box(event):
        if suggested is None:
            return
        l,t,r,b = suggested
        if not (0 <= l < r < w and 0 <= t < b < h):
            note.set_text('Correct the corners: top left must be above and left of bottom right.')
            fig.canvas.draw_idle()
            return
        reset(None)
        picks.extend([(l,t),(r,b)])
        proposal.set_visible(False)
        accept_button.set_active(False)
        accept_button.ax.set_alpha(.4)
        show_selected_box()
        fig.canvas.draw_idle()

    def adjust_corner(index):
        if suggested is None:
            return
        reset(None)
        adjustment[0] = index
        instruction.set_text('Click TOP LEFT correction' if index == 0 else 'Click BOTTOM RIGHT correction')
        note.set_text('Click the corrected corner; then accept the updated orange rectangle.')
        fig.canvas.draw_idle()

    outcome = ['stopped']
    def save(event):
        try:
            if len(picks) != 5:
                raise ValueError('Accept or select the box, then click the endpoint, historical line, and forecast band.')
            if workflow['state'] == CALIBRATION:
                show_preview(reset_overrides=False, stage=PREVIEW)
                raise ValueError('Review the detected points, then click Save again.')
            c = build_calibration(include_overrides=True)
            validate_calibration(c, (w,h))
            latest_config = load_working_config(config_path)
            if filename in latest_config.get('images', {}):
                backup = config_path.with_suffix('.json.bak')
                backup.write_text(config_path.read_text())
            latest_config.setdefault('images', {})[filename] = c
            latest_config['last_calibrated_image'] = filename
            temporary = config_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(latest_config, indent=2)+'\n')
            temporary.replace(config_path)
            outcome[0] = 'saved'
            plt.close(fig)
        except (ValueError, OSError) as exc:
            note.set_text(str(exc))
            note.set_color('#a00000')
            fig.canvas.draw_idle()
    def enter_manual_adjustment(event):
        if workflow['state'] != PREVIEW:
            return
        set_stage(MANUAL_ADJUSTMENT)
    accept_button = Button(fig.add_axes([cal_x,0.755,col_w,0.04]), 'Accept box')
    accept_button.on_clicked(accept_box)
    corner_buttons = []
    for index, label in enumerate(('Adjust top left', 'Adjust bottom right')):
        button = Button(fig.add_axes([0.04+index*.19,0.095,0.18,0.04]), label)
        button.on_clicked(lambda event, i=index: adjust_corner(i))
        button.set_active(suggested is not None)
        button.ax.set_alpha(1 if suggested is not None else .4)
        corner_buttons.append(button)
    show_proposal()
    if suggested is not None:
        instruction.set_text('Review suggested box')
    else:
        note.set_text('No clear plot box detected. Click the top-left and bottom-right corners.')
    reset_button = Button(fig.add_axes([cal_x,0.15,col_w,0.045]), 'Reset All')
    next_manual_button = Button(fig.add_axes([cal_x,0.095,col_w,0.045]), 'Next: Manual Adjustment')
    add_missing_button = Button(fig.add_axes([manual_x,0.56,col_w,0.05]), 'Add Missing Point')
    reset_manual_button = Button(fig.add_axes([manual_x,0.495,col_w,0.05]), 'Reset Manual Adjustments')
    save_button = Button(fig.add_axes([cal_x,0.045,col_w * 2 + (manual_x - cal_x - col_w),0.04]),
                         'Save and next' if queue_position is not None else 'Save and extract')
    next_manual_button.set_active(False)
    next_manual_button.ax.set_alpha(.35)
    set_manual_controls_enabled(False)
    navigation_buttons = []
    if queue_position is not None:
        def skip(event):
            outcome[0] = 'skipped'
            plt.close(fig)
        def stop(event):
            plt.close(fig)
        skip_button = Button(fig.add_axes([cal_x,0.0,0.175,0.035]), 'Skip')
        stop_button = Button(fig.add_axes([manual_x,0.0,0.165,0.035]), 'Stop')
        skip_button.on_clicked(skip)
        stop_button.on_clicked(stop)
        navigation_buttons.extend([skip_button, stop_button])
    reset_button.on_clicked(reset)
    reset_manual_button.on_clicked(reset_manual_adjustments)
    add_missing_button.on_clicked(choose_missing_point)
    next_manual_button.on_clicked(enter_manual_adjustment)
    save_button.on_clicked(save)
    fig.canvas.mpl_connect('button_press_event', click)
    fig.canvas.mpl_connect('motion_notify_event', move)
    fig.canvas.mpl_connect('button_release_event', release)
    fig.canvas.mpl_connect('figure_leave_event', leave)
    fig.canvas.mpl_connect('axes_leave_event', leave)
    plt.show(block=True)
    if outcome[0] == 'saved':
        run(config_path, images=[filename])
    elif outcome[0] == 'skipped':
        print(f'Skipped for this session: {filename}')
    else:
        print('Closed without saving the current image. Earlier calibrations are retained.')
    return outcome[0]


def launch_queue(config_path, template=None, start_at=None, include_calibrated=False):
    """Process remaining images in filename order; save each result immediately."""
    config = load_working_config(config_path)
    names = discover(input_directory(config),
                     config.get('directory_timeout_seconds', 20))
    if not names:
        raise ValueError('No PNG files found in the configured input folder.')
    if config.get('calibration_end_before'):
        cutoff = config['calibration_end_before']
        names = [name for name in names if report_sort_key(name) < report_sort_key(cutoff)]
        print(f'Queue limited to reports before {cutoff}: see configuration notes.', flush=True)
    if start_at:
        if start_at not in names:
            raise ValueError(f'Start image not found in folder: {start_at}')
        names = names[names.index(start_at):]
    known = config.get('images', {})
    if template and template not in known:
        raise ValueError(f'Template {template!r} is not in this config')
    pending = [name for name in names if include_calibrated or name not in known]
    if not pending:
        print('All images in this range already have saved calibrations. Use --include-calibrated to revisit them.')
        return
    # Resume with the most recently completed image when possible. For the
    # first session, choose the closest preceding calibrated filename.
    previous = template or config.get('last_calibrated_image')
    if previous not in known:
        before = [name for name in sorted(known, key=report_sort_key) if report_sort_key(name) < report_sort_key(pending[0])]
        previous = before[-1] if before else next(iter(sorted(known, key=report_sort_key)), None)
    print(f'Calibration queue: {len(pending)} images. Save and next advances; closing the window stops.', flush=True)
    completed = 0
    try:
        for index, name in enumerate(pending, 1):
            print(f'[{index}/{len(pending)}] {name}; template: {previous or "defaults"}', flush=True)
            result = launch(config_path, name, previous,
                            queue_position=f'{index} of {len(pending)} remaining',
                            advance_end=bool(previous and report_sort_key(name) > report_sort_key(previous)))
            if result == 'saved':
                previous = name
                completed += 1
            elif result == 'stopped':
                break
    finally:
        # Individual artifacts are saved after every completion; refresh the
        # combined report for all saved charts when stopping or finishing.
        current = load_working_config(config_path)
        if current.get('images'):
            print('Refreshing combined results for all saved calibrations...', flush=True)
            run(config_path, known_only=True)
    print(f'Session finished: {completed} calibrations saved. Run again to resume unsaved images.', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Calibrate a folder continuously, or one selected image.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--image', help='Calibrate just this PNG filename instead of the folder queue')
    parser.add_argument('--template', help='Existing calibration to prefill the first chart')
    parser.add_argument('--start-at', help='Start the queue at this filename')
    parser.add_argument('--include-calibrated', action='store_true', help='Also revisit already saved calibrations')
    args = parser.parse_args()
    if args.image and (args.start_at or args.include_calibrated):
        parser.error('--start-at and --include-calibrated apply to folder mode, not --image')
    try:
        from inflation_digitizer.desktop import configure_desktop_backend
        configure_desktop_backend()
        if args.image:
            launch(args.config.resolve(), args.image, args.template)
        else:
            launch_queue(args.config.resolve(), args.template, args.start_at, args.include_calibrated)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Cannot calibrate: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Stopped. Completed calibrations have been saved.')
        return 130
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
