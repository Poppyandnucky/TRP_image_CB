"""Interactive per-image calibration. Run in a desktop PyCharm session."""
from __future__ import annotations
import argparse
import json
from inflation_digitizer.configuration import load_working_config, input_directory
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from inflation_digitizer.batch import DEFAULT_CONFIG, discover, run, sampling_from_name, calibration_key, validate_image_name
from inflation_digitizer.core import (extract, quarter_number, quarter_label, month_number, month_label,
                                      period_number, period_label, validate_calibration, report_sort_key)
from inflation_digitizer.plot_box import detect_plot_box


def _quarter_as_month(value: str, *, end: bool) -> str:
    quarter = quarter_number(value)
    year, index = divmod(quarter, 4)
    return month_label(year * 12 + index * 3 + (2 if end else 0))


def _coerce_seed_periods(seed: dict, sampling: str) -> None:
    """Convert inherited template dates when switching between folder modes."""
    converted = False
    if sampling == 'monthly':
        for key, end in (('x_start', False), ('x_end', True), ('last_historical', True)):
            if key not in seed:
                continue
            try:
                month_number(str(seed[key]))
            except ValueError:
                seed[key] = _quarter_as_month(str(seed[key]), end=end)
                converted = True
    else:
        for key in ('x_start', 'x_end', 'last_historical'):
            if key not in seed:
                continue
            try:
                quarter_number(str(seed[key]))
            except ValueError:
                seed[key] = quarter_label(month_number(str(seed[key])) // 3)
                converted = True
    if converted:
        # Pixel-y overrides are tied to the old grid labels. Keep the inherited
        # calibration geometry/colors, but require fresh manual edits per mode.
        seed.pop('manual_point_overrides', None)


def launch(config_path, filename, template=None, *, queue_position=None, advance_end=False):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox, Button
    config = load_working_config(config_path)
    validate_image_name(filename)
    path = input_directory(config)/filename
    im = Image.open(path).convert('RGB')
    pixels = np.asarray(im)
    w, h = im.size
    inferred_sampling = sampling_from_name(filename)
    key = calibration_key(config.get('images', {}), filename, inferred_sampling)
    saved = config.get('images', {}).get(key)
    sampling = sampling_from_name(filename, saved)
    seed = dict(config.get('calibration_defaults', {}))
    template_key = None
    if template:
        template_key = calibration_key(config.get('images', {}), template, sampling_from_name(template))
    template_seed = dict(config.get('images', {}).get(template_key, {}))
    # Ignore regions are tied to one image's pixel dimensions and should not
    # be inherited by the next image's template calibration.
    template_seed.pop('ignore_regions', None)
    seed.update(template_seed)
    seed.update(config.get('image_defaults', {}).get(filename, {}))
    seed.update(saved or {})
    _coerce_seed_periods(seed, sampling)
    if template and template_key not in config.get('images', {}):
        raise ValueError(f'Template {template!r} is not in this config')
    advanced = bool(advance_end and not saved and seed.get('x_end'))
    if advanced:
        amount = 3 if sampling == 'monthly' else 1
        seed['x_end'] = period_label(period_number(seed['x_end'], sampling) + amount, sampling)
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
    calibration_controls = []
    preview_controls = []
    manual_controls = []
    shared_controls = []
    widget_by_axes = {}
    def remember(group, *artists):
        group.extend(artists)
        return artists[-1] if len(artists) == 1 else artists

    cal_title = remember(calibration_controls, fig.text(cal_x, 0.925, 'Calibration', fontsize=12, weight='bold'))
    instruction = fig.text(cal_x, 0.885, '1. Plot TOP LEFT corner\n(Please select a point on the plot grid, not on the data line.)',
                           fontsize=9, color='#00816f')
    remember(calibration_controls, instruction)
    manual_title = remember(manual_controls, fig.text(manual_x, 0.925, 'Manual Adjustment', fontsize=12, weight='bold'))
    manual_hint = remember(manual_controls, fig.text(manual_x, 0.89, 'Edit extracted points only.', fontsize=9, color='#7b8794'))
    preview_title = remember(preview_controls, fig.text(manual_x, 0.925, 'Preview', fontsize=12, weight='bold'))
    preview_summary = remember(preview_controls, fig.text(manual_x, 0.89, 'No preview yet.', fontsize=9, color='#536276', va='top'))
    plot_help = remember(calibration_controls, fig.text(cal_x, 0.84, 'Accept the orange box, or\ncorrect its corners below.', fontsize=10, va='top'))
    fields = {}
    quarter_buttons = []  # Keep widget objects alive for their callbacks.
    period_name = 'month' if sampling == 'monthly' else 'quarter'
    period_name_title = period_name.title()
    field_specs = [
        ('x_start', f'First {period_name}', '2022-01' if sampling == 'monthly' else '2022Q1'),
        ('x_end', f'Last {period_name}', '2027-12' if sampling == 'monthly' else '2027Q3'),
        ('y_min', 'Y minimum', '-4'), ('y_max', 'Y maximum', '10'),
        ('color_tolerance', 'Forecast tolerance', '22'),
        ('historical_color_tolerance', 'Historical tolerance', '45'),
    ]
    field_ys = [0.68, 0.59, 0.50, 0.41, 0.325, 0.25]
    for i, (key, label, default) in enumerate(field_specs):
        y = field_ys[i]
        advance_note = f' (+{3 if sampling == "monthly" else 1} {period_name}{"s" if sampling == "monthly" else ""})'
        remember(calibration_controls, fig.text(cal_x, y+0.042, label + (advance_note if key == 'x_end' and advanced else ''), fontsize=10))
        is_period = key in ('x_start', 'x_end')
        is_axis_limit = key in ('y_min', 'y_max')
        bounds = [cal_x + 0.036, y, col_w - 0.072, 0.038] if is_period or is_axis_limit else [cal_x, y, col_w, 0.038]
        fields[key] = TextBox(fig.add_axes(bounds), '', initial=str(seed.get(key, default)))
        remember(calibration_controls, fields[key].ax)
        if is_period or is_axis_limit:
            for left, delta, label in [(cal_x, -1, '◀'), (cal_x + col_w - 0.03, 1, '▶')]:
                button = Button(fig.add_axes([left, y, 0.03, 0.038]), label if is_period else ('−' if delta < 0 else '+'))
                def increment(event, field=key, amount=delta):
                    try:
                        if field in ('y_min', 'y_max'):
                            value = float(fields[field].text) + amount
                            if not np.isfinite(value):
                                raise ValueError('Enter a finite Y-axis value.')
                            fields[field].set_val(f'{value:g}')
                            fig.canvas.draw_idle()
                            return
                        value = period_number(fields[field].text, sampling) + amount
                        fields[field].set_val(period_label(value, sampling))
                    except ValueError as exc:
                        note.set_text(str(exc))
                        note.set_color('#a00000')
                    fig.canvas.draw_idle()
                button.on_clicked(increment)
                quarter_buttons.append(button)
                remember(calibration_controls, button.ax)
    note = fig.text(0.035, 0.055, 'Accept the box or select two grid corners; then select the historical endpoint and darkest band.', fontsize=10)
    picks, markers = [], []
    adjustment = [None]
    reference_line_tolerance = float(seed.get('reference_line_tolerance', config.get('reference_line_tolerance', 4)))
    excluded_ref_ys = []
    ref_line_artists = []
    ref_mode = {'active': False}
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
        note.set_text(f'The vertical guide snaps to {period_name}s. Click the final historical point to select its {period_name}.')
        instruction.set_text('3. LAST HISTORICAL point')

    # Span the whole displayed image, including the printed quarter labels.
    quarter_guide = ax.axvline(0, color='#d60078', linestyle='--', lw=1.4,
                              visible=False, zorder=15)
    quarter_readout = fig.text(cal_x, 0.235, '', fontsize=9, color='#d60078', va='top')
    missing_readout = fig.text(manual_x, 0.735, '', fontsize=9, color='#a00000', va='top')
    adjustment_readout = fig.text(manual_x, 0.835, 'Selected Adjustment Point: none\nCurrent Value: --',
                                  fontsize=10, color='#536276', va='top')
    remember(preview_controls, missing_readout)
    remember(manual_controls, missing_readout, adjustment_readout)
    hist_swatch_label = remember(calibration_controls, fig.text(cal_x, 0.205, 'Historical colour', fontsize=8, color='#536276'))
    forecast_swatch_label = remember(calibration_controls, fig.text(cal_x, 0.18, 'Forecast colour', fontsize=8, color='#536276'))
    hist_swatch = fig.add_axes([cal_x + 0.105, 0.202, 0.028, 0.02])
    forecast_swatch = fig.add_axes([cal_x + 0.105, 0.177, 0.028, 0.02])
    remember(calibration_controls, hist_swatch, forecast_swatch)
    for swatch in (hist_swatch, forecast_swatch):
        swatch.set_xticks([])
        swatch.set_yticks([])
        swatch.patch.set_facecolor('#ffffff')
        swatch.patch.set_edgecolor('#c7ced6')
        swatch.patch.set_linewidth(1)
    ref_readout = remember(calibration_controls, fig.text(manual_x, 0.705, '0 reference lines selected', fontsize=8, color='#9aa3ad'))

    # A white outline keeps the moving crosshairs visible over dark fan bands.
    import matplotlib.patheffects as path_effects
    crosshair_style = dict(color='black', lw=.8, visible=False, zorder=10,
                           path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
    cursor_x = ax.axvline(0, gid='cursor-crosshair-x', **crosshair_style)
    cursor_y = ax.axhline(0, gid='cursor-crosshair-y', **crosshair_style)
    ref_guide = ax.axhline(0, color='#7f3fbf', linestyle='--', lw=1.2, visible=False, zorder=16)

    def hide_crosshairs():
        cursor_x.set_visible(False)
        cursor_y.set_visible(False)
        if not ref_mode['active']:
            ref_guide.set_visible(False)

    def plot_bounds():
        (l, t), (r, b) = picks[:2]
        return l, t, r, b

    def draw_ref_lines():
        for artist in ref_line_artists:
            artist.remove()
        ref_line_artists.clear()
        if len(picks) < 2:
            return
        l, t, r, b = plot_bounds()
        for y in excluded_ref_ys:
            ref_line_artists.extend(ax.plot([l, r], [y, y], '--', color='#7f3fbf',
                                            lw=1.2, alpha=.9, zorder=14))
        ref_readout.set_text(f'{len(excluded_ref_ys)} reference line{"s" if len(excluded_ref_ys) != 1 else ""} selected')
        ref_readout.set_color('#7f3fbf' if excluded_ref_ys else '#9aa3ad')

    def update_color_swatch(swatch, rgb):
        swatch.patch.set_facecolor(np.array(rgb) / 255)
        swatch.patch.set_edgecolor('#222222' if np.mean(rgb) > 190 else '#c7ced6')

    def selected_ref_regions():
        if len(picks) < 2:
            return []
        l, t, r, b = plot_bounds()
        regions = []
        for y in excluded_ref_ys:
            y1 = max(t, y - reference_line_tolerance)
            y2 = min(b, y + reference_line_tolerance)
            if y2 > y1:
                regions.append([round(l, 2), round(y1, 2), round(r, 2), round(y2, 2)])
        return regions

    def set_ref_buttons_enabled(enabled, selecting=False):
        for button, active in ((exclude_ref_button, enabled and not selecting),
                               (confirm_ref_button, selecting),
                               (reset_ref_button, selecting and bool(excluded_ref_ys)),
                               (skip_ref_button, selecting)):
            button.set_active(active)
            button.ax.set_alpha(1 if active else .35)

    def continue_after_ref_lines():
        ref_mode['active'] = False
        ref_guide.set_visible(False)
        set_ref_buttons_enabled(len(picks) == 3, False)
        instruction.set_text('4. HISTORICAL line colour')
        note.set_text('Click the historical line colour sample, away from axes and reference lines.')
        note.set_color('black')
        fig.canvas.draw_idle()

    def enable_ref_line_selection(event):
        if workflow['state'] != CALIBRATION or len(picks) != 3:
            return
        ref_mode['active'] = True
        set_ref_buttons_enabled(False, True)
        instruction.set_text('4. Exclude reference lines')
        note.set_text('Hover inside the plot; click each horizontal reference line. Confirm, reset, or skip when done.')
        note.set_color('black')
        fig.canvas.draw_idle()

    def confirm_ref_lines(event):
        if ref_mode['active']:
            continue_after_ref_lines()

    def reset_ref_lines(event):
        if not ref_mode['active']:
            return
        excluded_ref_ys.clear()
        draw_ref_lines()
        set_ref_buttons_enabled(False, True)
        note.set_text('Reference lines cleared. Click horizontal reference lines, or skip.')
        note.set_color('black')
        fig.canvas.draw_idle()

    def skip_ref_lines(event):
        if not ref_mode['active']:
            return
        excluded_ref_ys.clear()
        draw_ref_lines()
        continue_after_ref_lines()

    def nearest_period(x):
        (l, t), (r, b) = picks[:2]
        start = period_number(fields['x_start'].text, sampling)
        end = period_number(fields['x_end'].text, sampling)
        if r <= l or b <= t or end <= start:
            raise ValueError(f'Check plot corners and first/last {period_name}s.')
        q = max(start, min(end, round(start + (x-l)*(end-start)/(r-l))))
        return l + (r-l)*(q-start)/(end-start), period_label(q, sampling)

    def guide_at(x, selected=False):
        try:
            snapped_x, label = nearest_period(x)
        except ValueError:
            quarter_guide.set_visible(False)
            quarter_readout.set_text('Check the axis values.')
            return None
        quarter_guide.set_xdata([snapped_x, snapped_x])
        quarter_guide.set_visible(True)
        quarter_readout.set_text(('Selected Historical Endpoint: ' if selected else f'{period_name_title}: ') + display_period(label))
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
        if ref_mode['active']:
            hide_crosshairs()
            if on_image and len(picks) >= 2:
                l, t, r, b = plot_bounds()
                clamped_y = max(t, min(b, y))
                ref_guide.set_ydata([clamped_y, clamped_y])
                ref_guide.set_visible(True)
            else:
                ref_guide.set_visible(False)
            fig.canvas.draw_idle()
            return
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
        ref_guide.set_visible(False)
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
    hints = ['1. Plot TOP LEFT corner\n(Please select a point on the plot grid, not on the data line.)', '2. Plot BOTTOM RIGHT corner',
             '3. LAST HISTORICAL point', '4. HISTORICAL line colour',
             '5. INSIDE darkest forecast band', 'Review detected points']
    CALIBRATION, PREVIEW, MANUAL_ADJUSTMENT = 'calibration', 'preview', 'manual_adjustment'
    workflow = {'state': CALIBRATION}

    preview = dict(rows=[], automatic_rows=[], overrides={},
                   artists=[], selected=None, dragging=False, calibration=None,
                   adding_missing=None, missing_cycle=0)

    def display_period(label):
        return label.replace('Q', ' Q')

    def row_period(row):
        return row.get('period', row.get('quarter'))

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
        start, end = (period_number(fields['x_start'].text, sampling), period_number(fields['x_end'].text, sampling))
        if r <= l or b <= t:
            raise ValueError('Select top left, then bottom right in that order.')
        for label, x, y in [('historical endpoint', hx, hy), ('historical colour sample', histx, histy),
                            ('forecast colour sample', cx, cy)]:
            if not l <= x <= r or not t <= y <= b:
                raise ValueError(f'The {label} must be inside the plot.')
        hist = round(start+(hx-l)*(end-start)/(r-l))
        forecast_color = pixel_color(cx, cy)
        if np.linalg.norm(np.array(forecast_color, dtype=float) - 255) < 45:
            raise ValueError('Selected forecast colour is too close to the white background. Click inside the darkest band or forecast line, away from grid lines.')
        c = dict(image_size=[w,h], plot_box=[round(v,2) for v in (l,t,r,b)],
                 y_min=float(fields['y_min'].text), y_max=float(fields['y_max'].text),
                 x_start=period_label(start, sampling), x_end=period_label(end, sampling),
                 last_historical=period_label(hist, sampling), sampling=sampling,
                 historical_rgb=pixel_color(histx, histy),
                 historical_color_tolerance=float(fields['historical_color_tolerance'].text),
                 curve_rgb=forecast_color, color_tolerance=float(fields['color_tolerance'].text),
                 review_status='manually_calibrated',
                 notes='Review generated overlay before changing review_status to reviewed.')
        for key in ('plot_box_style',):
            if key in seed:
                c[key] = seed[key]
        ignore_regions = [list(region) for region in seed.get('ignore_regions', [])]
        ignore_regions.extend(selected_ref_regions())
        if ignore_regions:
            c['ignore_regions'] = ignore_regions
        if excluded_ref_ys or reference_line_tolerance != 10:
            c['reference_line_tolerance'] = reference_line_tolerance
        if include_overrides and preview['overrides']:
            c['manual_point_overrides'] = {
                period: {'pixel_y': round(float(values['pixel_y']), 3)}
                for period, values in sorted(preview['overrides'].items())
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
        return f"Selected Adjustment Point: {display_period(row_period(row))}\nCurrent Value: {value_text}"

    def update_preview_note():
        adjustment_readout.set_text(selected_message())
        if workflow['state'] == MANUAL_ADJUSTMENT:
            adjustment_readout.set_color('black' if preview['selected'] is not None else '#536276')
        else:
            adjustment_readout.set_color('#9aa3ad')

    def missing_rows():
        return [row for row in preview['rows'] if row['pixel_y'] is None]

    def update_missing_readout():
        missing = [display_period(row_period(row)) for row in missing_rows()]
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

    def set_artist_visible(artist, visible):
        artist.set_visible(visible)
        widget = widget_by_axes.get(artist)
        if widget is not None:
            widget.set_active(visible)

    def set_group_visible(group, visible):
        for artist in group:
            set_artist_visible(artist, visible)

    def update_stage_visibility():
        stage = workflow['state']
        set_group_visible(calibration_controls, stage == CALIBRATION)
        set_group_visible(preview_controls, stage == PREVIEW)
        set_group_visible(manual_controls, stage == MANUAL_ADJUSTMENT)
        set_group_visible(shared_controls, True)
        cal_panel.set_visible(stage == CALIBRATION)
        manual_panel.set_visible(stage in (PREVIEW, MANUAL_ADJUSTMENT))
        if stage == CALIBRATION:
            accept_button.set_active(suggested is not None and len(picks) < 2)
            accept_button.ax.set_alpha(1 if accept_button.active else .4)
            for button in corner_buttons:
                button.set_active(suggested is not None and len(picks) < 2)
                button.ax.set_alpha(1 if button.active else .4)
            set_ref_buttons_enabled(len(picks) == 3, ref_mode['active'])
        elif stage == PREVIEW:
            next_manual_button.set_active(True)
            next_manual_button.ax.set_alpha(1)
        elif stage == MANUAL_ADJUSTMENT:
            set_manual_controls_enabled(True)

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
            missing = missing_rows()
            preview_summary.set_text(f'Detected {len(preview["rows"])} points.\nMissing points: {len(missing)}')
        elif stage == MANUAL_ADJUSTMENT:
            set_calibration_markers_visible(False)
            instruction.set_text('Manual point adjustment')
            note.set_text(f'Select and drag a point vertically. Add Missing Point fixes one missing {period_name} at a time.')
            note.set_color('black')
            update_preview_note()
        update_stage_visibility()
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
                override = preview['overrides'].get(row_period(row))
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
        preview['overrides'][row_period(row)] = {'pixel_y': float(y)}

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
        preview['adding_missing'] = row_period(row)
        update_preview_note()
        note.set_text(f"Click the plot to set the Y value for {display_period(row_period(row))}. X stays fixed.")
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
        ref_guide.set_visible(False)
        excluded_ref_ys.clear()
        for artist in ref_line_artists:
            artist.remove()
        ref_line_artists.clear()
        ref_readout.set_text('0 reference lines selected')
        ref_readout.set_color('#9aa3ad')
        ref_mode['active'] = False
        hist_swatch.patch.set_facecolor('#ffffff')
        forecast_swatch.patch.set_facecolor('#ffffff')
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
        set_ref_buttons_enabled(False, False)
        update_stage_visibility()
        fig.canvas.draw_idle()

    def click(event):
        if workflow['state'] == PREVIEW:
            return
        if workflow['state'] == MANUAL_ADJUSTMENT:
            if event.inaxes == ax and event.button == 1 and preview['adding_missing']:
                index = next((i for i, row in enumerate(preview['rows'])
                              if row_period(row) == preview['adding_missing']), None)
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
        if ref_mode['active']:
            if len(picks) >= 2:
                l, t, r, b = plot_bounds()
                if not l <= x <= r or not t <= y <= b:
                    note.set_text('Click reference lines inside the plot box.')
                    note.set_color('#a00000')
                    fig.canvas.draw_idle()
                    return
                excluded_ref_ys.append(max(t, min(b, y)))
                draw_ref_lines()
                set_ref_buttons_enabled(False, True)
                note.set_text(f'{len(excluded_ref_ys)} reference line(s) selected. Add more, confirm, reset, or skip.')
                note.set_color('black')
                fig.canvas.draw_idle()
            return
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
        if len(picks) == 3:
            note.set_text('Optional: exclude horizontal reference lines, or click the historical line colour sample to continue.')
            note.set_color('black')
            set_ref_buttons_enabled(True, False)
        elif len(picks) > 3:
            set_ref_buttons_enabled(False, False)
        if len(picks) == 4:
            update_color_swatch(hist_swatch, pixel_color(x, y))
        elif len(picks) == 5:
            update_color_swatch(forecast_swatch, pixel_color(x, y))
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
            save_key = calibration_key(latest_config.get('images', {}), filename, sampling)
            if save_key in latest_config.get('images', {}):
                backup = config_path.with_suffix('.json.bak')
                backup.write_text(config_path.read_text())
            latest_config.setdefault('images', {})[save_key] = c
            latest_config['last_calibrated_image'] = save_key
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
    def back_to_preview(event):
        if workflow['state'] == MANUAL_ADJUSTMENT and preview['rows']:
            set_stage(PREVIEW)
    def back_to_calibration(event):
        if workflow['state'] in (PREVIEW, MANUAL_ADJUSTMENT):
            reset_calibration()
    accept_button = Button(fig.add_axes([cal_x,0.755,col_w,0.04]), 'Accept box')
    accept_button.on_clicked(accept_box)
    remember(calibration_controls, accept_button.ax)
    corner_buttons = []
    for index, label in enumerate(('Adjust top left', 'Adjust bottom right')):
        button = Button(fig.add_axes([0.04+index*.19,0.095,0.18,0.04]), label)
        button.on_clicked(lambda event, i=index: adjust_corner(i))
        button.set_active(suggested is not None)
        button.ax.set_alpha(1 if suggested is not None else .4)
        corner_buttons.append(button)
        remember(calibration_controls, button.ax)
    show_proposal()
    if suggested is not None:
        instruction.set_text('Review suggested box')
    else:
        note.set_text('No clear plot box detected. Click the top-left and bottom-right corners.')
    reset_button = Button(fig.add_axes([cal_x,0.15,col_w,0.045]), 'Reset All')
    next_manual_button = Button(fig.add_axes([cal_x,0.095,col_w,0.045]), 'Next: Manual Adjustment')
    back_calibration_button = Button(fig.add_axes([manual_x,0.15,col_w,0.045]), 'Back to Calibration')
    back_preview_button = Button(fig.add_axes([manual_x,0.15,col_w,0.045]), 'Back to Preview')
    ref_label = remember(calibration_controls, fig.text(manual_x, 0.68, 'Reference Lines', fontsize=9, color='#536276'))
    exclude_ref_button = Button(fig.add_axes([manual_x,0.635,col_w,0.04]), 'Exclude Ref Lines')
    confirm_ref_button = Button(fig.add_axes([manual_x,0.585,col_w,0.04]), 'Confirm Ref Lines')
    reset_ref_button = Button(fig.add_axes([manual_x,0.535,col_w,0.04]), 'Reset Ref Lines')
    skip_ref_button = Button(fig.add_axes([manual_x,0.485,col_w,0.04]), 'Skip Ref Lines')
    add_missing_button = Button(fig.add_axes([manual_x,0.42,col_w,0.05]), 'Add Missing Point')
    reset_manual_button = Button(fig.add_axes([manual_x,0.355,col_w,0.05]), 'Reset Manual Adjustments')
    save_button = Button(fig.add_axes([cal_x,0.045,col_w * 2 + (manual_x - cal_x - col_w),0.04]),
                         'Save and next' if queue_position is not None else 'Save and extract')
    remember(calibration_controls, reset_button.ax, exclude_ref_button.ax, confirm_ref_button.ax,
             reset_ref_button.ax, skip_ref_button.ax)
    remember(preview_controls, reset_button.ax, next_manual_button.ax, back_calibration_button.ax)
    remember(manual_controls, add_missing_button.ax, reset_manual_button.ax, back_preview_button.ax)
    remember(shared_controls, save_button.ax)
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
        remember(shared_controls, skip_button.ax, stop_button.ax)
    for button in (quarter_buttons + corner_buttons + navigation_buttons +
                   [accept_button, reset_button, next_manual_button, back_calibration_button,
                    back_preview_button, exclude_ref_button, confirm_ref_button, reset_ref_button,
                    skip_ref_button, add_missing_button, reset_manual_button, save_button]):
        widget_by_axes[button.ax] = button
    reset_button.on_clicked(reset)
    back_calibration_button.on_clicked(back_to_calibration)
    back_preview_button.on_clicked(back_to_preview)
    exclude_ref_button.on_clicked(enable_ref_line_selection)
    confirm_ref_button.on_clicked(confirm_ref_lines)
    reset_ref_button.on_clicked(reset_ref_lines)
    skip_ref_button.on_clicked(skip_ref_lines)
    reset_manual_button.on_clicked(reset_manual_adjustments)
    add_missing_button.on_clicked(choose_missing_point)
    next_manual_button.on_clicked(enter_manual_adjustment)
    save_button.on_clicked(save)
    set_ref_buttons_enabled(False, False)
    update_stage_visibility()
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
    if template and calibration_key(known, template, sampling_from_name(template)) not in known:
        raise ValueError(f'Template {template!r} is not in this config')
    pending = [name for name in names
               if include_calibrated or calibration_key(known, name, sampling_from_name(name)) not in known]
    if not pending:
        print('All images in this range already have saved calibrations. Use --include-calibrated to revisit them.')
        return
    # Resume with the most recently completed image when possible. For the
    # first session, choose the closest preceding calibrated filename.
    previous = template or config.get('last_calibrated_image')
    if previous not in names and previous not in known:
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
