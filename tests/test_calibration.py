import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, Event
from PIL import Image, ImageDraw
import calibrate


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config_path = self.root/'config.json'
        self.names = ['TST_2000_Q1.png', 'TST_2000_Q2.png', 'TST_2000_Q3.png']
        self.config = dict(input_directory=str(self.root), images={self.names[0]: {'x_start':'1999Q1'}})
        self.config_path.write_text(json.dumps(self.config))

    def tearDown(self):
        plt.close('all')
        self.temp.cleanup()

    def test_previous_completed_chart_becomes_next_template(self):
        with patch.object(calibrate, 'discover', return_value=self.names), \
             patch.object(calibrate, 'launch', return_value='saved') as launch, \
             patch.object(calibrate, 'run') as run:
            calibrate.launch_queue(self.config_path)
        self.assertEqual([c.args[1] for c in launch.call_args_list], self.names[1:])
        self.assertEqual([c.args[2] for c in launch.call_args_list], self.names[:2])
        self.assertTrue(all(c.kwargs['advance_end'] for c in launch.call_args_list))
        run.assert_called_once_with(self.config_path, known_only=True)

    def test_skip_does_not_become_template_and_stop_ends_queue(self):
        names = self.names + ['TST_2000_Q4.png']
        with patch.object(calibrate, 'discover', return_value=names), \
             patch.object(calibrate, 'launch', side_effect=['skipped','stopped']) as launch, \
             patch.object(calibrate, 'run'):
            calibrate.launch_queue(self.config_path)
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(launch.call_args_list[1].args[2], self.names[0])

    def test_resume_uses_last_completed_calibration(self):
        self.config['images'][self.names[1]] = {'x_start':'1999Q2'}
        self.config['last_calibrated_image'] = self.names[1]
        self.config_path.write_text(json.dumps(self.config))
        with patch.object(calibrate, 'discover', return_value=self.names), \
             patch.object(calibrate, 'launch', return_value='stopped') as launch, \
             patch.object(calibrate, 'run'):
            calibrate.launch_queue(self.config_path)
        self.assertEqual(launch.call_args.args[1:3], (self.names[2],self.names[1]))

    def test_configured_queue_cutoff(self):
        self.config['calibration_end_before']=self.names[2]
        self.config_path.write_text(json.dumps(self.config))
        with patch.object(calibrate,'discover',return_value=self.names), \
             patch.object(calibrate,'launch',return_value='saved') as launch, \
             patch.object(calibrate,'run'):
            calibrate.launch_queue(self.config_path)
        self.assertEqual([call.args[1] for call in launch.call_args_list],[self.names[1]])

    def test_start_at_and_include_calibrated(self):
        with patch.object(calibrate, 'discover', return_value=self.names), \
             patch.object(calibrate, 'launch', return_value='stopped') as launch, \
             patch.object(calibrate, 'run'):
            calibrate.launch_queue(self.config_path, start_at=self.names[0], include_calibrated=True)
        self.assertEqual(launch.call_args.args[1], self.names[0])

    def test_quarter_template_dates_are_converted_for_monthly_mode(self):
        seed = {
            'x_start': '2024Q1',
            'x_end': '2026Q4',
            'last_historical': '2025Q2',
            'manual_point_overrides': {'2025Q3': {'pixel_y': 50}},
        }
        calibrate._coerce_seed_periods(seed, 'monthly')
        self.assertEqual(seed['x_start'], '2024-01')
        self.assertEqual(seed['x_end'], '2026-12')
        self.assertEqual(seed['last_historical'], '2025-06')
        self.assertNotIn('manual_point_overrides', seed)

    def test_quarter_arrows_guide_and_save_next(self):
        name = self.names[0]
        image = Image.new('RGB',(140,140),'white')
        draw = ImageDraw.Draw(image)
        draw.line([(20,70),(45,70)], fill=(35,68,134), width=3)
        draw.polygon([(45,70),(70,50),(120,40),(120,60),(70,70)], fill=(35,68,134))
        image.save(self.root/name)
        self.config['calibration_defaults'] = dict(historical_rgb=[0,0,0], historical_color_tolerance=45, plot_box_style='open_axes')
        self.config['images'][name] = dict(x_start='2025Q1',x_end='2026Q1',y_min=-5,y_max=5)
        self.config_path.write_text(json.dumps(self.config))
        def interact(*args, **kwargs):
            fig = plt.gcf()
            fig.canvas.draw()
            ax = fig.axes[0]
            def event(kind, x, y, axes=ax):
                transform = axes.transData if axes is ax else axes.transAxes
                sx,sy=transform.transform((x,y))
                fig.canvas.callbacks.process(kind,MouseEvent(kind,fig.canvas,sx,sy,button=1))
            def button(label, index=0):
                axes = [a for a in fig.axes if any(t.get_text()==label for t in a.texts)]
                target=axes[index]
                event('button_press_event',.5,.5,target)
                event('button_release_event',.5,.5,target)
            def field(y):
                target=next(a for a in fig.axes if abs(a.get_position().y0-y)<.001 and a.get_position().width > .08)
                return target.texts[-1].get_text()
            button('\u25c0',1)
            self.assertEqual(field(.59),'2025Q4')
            button('\u25b6',1)
            self.assertEqual(field(.59),'2026Q1')
            button('\u25c0',0)
            self.assertEqual(field(.68),'2024Q4')
            button('\u25b6',0)
            self.assertEqual(field(.68),'2025Q1')
            button('−',0)
            self.assertEqual(field(.50),'-6')
            button('+',0)
            self.assertEqual(field(.50),'-5')
            button('−',0)
            button('+',1)
            self.assertEqual(field(.41),'6')
            button('−',1)
            self.assertEqual(field(.41),'5')
            button('+',1)
            cross_x = next(line for line in ax.lines if line.get_gid() == 'cursor-crosshair-x')
            cross_y = next(line for line in ax.lines if line.get_gid() == 'cursor-crosshair-y')
            def check_crosshairs(x, y):
                event('motion_notify_event',x,y)
                self.assertTrue(cross_x.get_visible())
                self.assertTrue(cross_y.get_visible())
                self.assertAlmostEqual(cross_x.get_xdata()[0],x)
                self.assertAlmostEqual(cross_y.get_ydata()[0],y)
            check_crosshairs(20,20)
            fig.canvas.callbacks.process('figure_leave_event', Event('figure_leave_event',fig.canvas))
            self.assertFalse(cross_x.get_visible())
            self.assertFalse(cross_y.get_visible())
            event('button_press_event',20,20)
            check_crosshairs(120,120)
            event('button_press_event',120,120)
            check_crosshairs(46,70)
            self.assertTrue(any(t.get_text()=='Quarter: 2025 Q2' for t in fig.texts))
            event('button_press_event',46,70)
            check_crosshairs(30,70)
            event('button_press_event',30,70)
            event('motion_notify_event',.5,.5,fig.axes[1])
            self.assertFalse(cross_x.get_visible())
            self.assertFalse(cross_y.get_visible())
            event('button_press_event',100,50)
            self.assertFalse(cross_x.get_visible())
            self.assertFalse(cross_y.get_visible())
            event('motion_notify_event',90,60)
            self.assertFalse(cross_x.get_visible())
            button('Save and next')
        with patch.object(plt,'show',side_effect=interact), patch.object(calibrate,'run') as run:
            result=calibrate.launch(self.config_path,name,queue_position='1 of 2 remaining',advance_end=True)
        self.assertEqual(result,'saved')
        c=json.loads(self.config_path.read_text())
        self.assertEqual(c['last_calibrated_image'],name)
        self.assertEqual(c['images'][name]['y_min'],-6)
        self.assertEqual(c['images'][name]['y_max'],6)
        self.assertEqual(c['images'][name]['historical_rgb'],[35,68,134])
        self.assertEqual(c['images'][name]['historical_color_tolerance'],45)
        self.assertEqual(c['images'][name]['plot_box_style'],'open_axes')
        self.assertNotIn('ignore_regions', c['images'][name])
        self.assertEqual(c['images'][name]['last_historical'],'2025Q2')
        run.assert_called_once_with(self.config_path,images=[name])

    def test_horizontal_reference_lines_are_saved_as_ignore_regions(self):
        name = self.names[0]
        image = Image.new('RGB',(140,140),'white')
        draw = ImageDraw.Draw(image)
        draw.line([(20,70),(45,70)], fill=(35,68,134), width=3)
        draw.line([(20,50),(120,50)], fill=(35,68,134), width=2)
        draw.line([(20,90),(120,90)], fill=(35,68,134), width=2)
        draw.polygon([(45,70),(70,50),(120,40),(120,60),(70,70)], fill=(35,68,134))
        image.save(self.root/name)
        self.config['images'][name] = dict(x_start='2025Q1',x_end='2026Q1',y_min=-5,y_max=5,
                                           historical_rgb=[35,68,134], historical_color_tolerance=45)
        self.config_path.write_text(json.dumps(self.config))
        def interact(*args, **kwargs):
            fig = plt.gcf()
            fig.canvas.draw()
            ax = fig.axes[0]
            def event(kind, x, y, axes=ax):
                sx,sy=(axes.transData if axes is ax else axes.transAxes).transform((x,y))
                fig.canvas.callbacks.process(kind,MouseEvent(kind,fig.canvas,sx,sy,button=1))
            def button(label):
                target=next(a for a in fig.axes if any(t.get_text()==label for t in a.texts))
                event('button_press_event',.5,.5,target)
                event('button_release_event',.5,.5,target)
            event('button_press_event',20,20)
            event('button_press_event',120,120)
            event('button_press_event',46,70)
            button('Exclude Ref Lines')
            event('motion_notify_event',50,50)
            guide=next(line for line in ax.lines if line.get_color() == '#7f3fbf' and line.get_visible())
            self.assertAlmostEqual(guide.get_ydata()[0],50)
            event('button_press_event',50,50)
            self.assertTrue(any(t.get_text() == '1 reference line selected' for t in fig.texts))
            event('button_press_event',60,90)
            self.assertTrue(any(t.get_text() == '2 reference lines selected' for t in fig.texts))
            button('Confirm Ref Lines')
            self.assertTrue(any('HISTORICAL line colour' in t.get_text() for t in fig.texts))
            event('button_press_event',30,70)
            event('button_press_event',100,50)
            button('Save and next')
        with patch.object(plt,'show',side_effect=interact), patch.object(calibrate,'run'):
            result=calibrate.launch(self.config_path,name,queue_position='1 of 1')
        self.assertEqual(result,'saved')
        saved=json.loads(self.config_path.read_text())['images'][name]
        self.assertEqual(saved['ignore_regions'], [[20.0, 46.0, 120.0, 54.0], [20.0, 86.0, 120.0, 94.0]])
        self.assertEqual(saved['reference_line_tolerance'], 4.0)

    def test_black_forecast_colour_is_allowed_in_calibration(self):
        name = self.names[0]
        image = Image.new('RGB',(140,140),'white')
        draw = ImageDraw.Draw(image)
        draw.line([(20,70),(45,70)], fill=(0,0,0), width=3)
        draw.polygon([(45,70),(70,50),(120,40),(120,60),(70,70)], fill=(0,0,0))
        image.save(self.root/name)
        self.config['images'][name] = dict(x_start='2025Q1',x_end='2026Q1',y_min=-5,y_max=5,
                                           historical_rgb=[0,0,0], historical_color_tolerance=20)
        self.config_path.write_text(json.dumps(self.config))
        def interact(*args, **kwargs):
            fig = plt.gcf()
            fig.canvas.draw()
            ax = fig.axes[0]
            def event(kind, x, y, axes=ax):
                sx,sy=(axes.transData if axes is ax else axes.transAxes).transform((x,y))
                fig.canvas.callbacks.process(kind,MouseEvent(kind,fig.canvas,sx,sy,button=1))
            def button(label):
                target=next(a for a in fig.axes if any(t.get_text()==label for t in a.texts))
                event('button_press_event',.5,.5,target)
                event('button_release_event',.5,.5,target)
            event('button_press_event',20,20)
            event('button_press_event',120,120)
            event('button_press_event',46,70)
            event('button_press_event',30,70)
            event('button_press_event',100,50)
            self.assertFalse(any('nearly grey' in t.get_text() for t in fig.texts))
            button('Save and next')
        with patch.object(plt,'show',side_effect=interact), patch.object(calibrate,'run'):
            result=calibrate.launch(self.config_path,name,queue_position='1 of 1')
        self.assertEqual(result,'saved')
        saved=json.loads(self.config_path.read_text())['images'][name]
        self.assertEqual(saved['curve_rgb'], [0, 0, 0])

    def test_monthly_mode_snaps_endpoint_guide_and_clicks_like_quarterly(self):
        name = 'monthly/TST_2000_Q2.png'
        (self.root/'monthly').mkdir()
        image = Image.new('RGB',(140,140),'white')
        draw = ImageDraw.Draw(image)
        draw.line([(20,70),(45,70)], fill=(35,68,134), width=3)
        draw.polygon([(45,70),(70,50),(120,40),(120,60),(70,70)], fill=(35,68,134))
        image.save(self.root/name)
        self.config['images'][self.names[0]] = dict(x_start='2025Q1',x_end='2025Q4',
                                                     last_historical='2025Q2',
                                                     y_min=-5,y_max=5,
                                                     historical_rgb=[35,68,134],
                                                     curve_rgb=[35,68,134],
                                                     color_tolerance=10)
        self.config_path.write_text(json.dumps(self.config))
        def interact(*args, **kwargs):
            fig = plt.gcf()
            fig.canvas.draw()
            ax = fig.axes[0]
            def event(kind, x, y, axes=ax):
                sx, sy = (axes.transData if axes is ax else axes.transAxes).transform((x, y))
                fig.canvas.callbacks.process(kind, MouseEvent(kind, fig.canvas, sx, sy, button=1))
            event('button_press_event', 20, 20)
            event('button_press_event', 120, 120)
            guide = next(line for line in ax.lines if line.get_color() == '#d60078' and line.get_linestyle() == '--')
            event('motion_notify_event', 46, 70)
            self.assertTrue(guide.get_visible())
            self.assertGreater(guide.get_zorder(), max(line.get_zorder() for line in ax.lines
                                                       if (line.get_gid() or '').startswith('cursor-crosshair')))
            self.assertTrue(any('Month: 2025-04' == t.get_text() for t in fig.texts))
            event('button_press_event', 46, 70)
            self.assertTrue(any('Selected Historical Endpoint: 2025-04' == t.get_text() for t in fig.texts))
            event('button_press_event', 30, 70)
            event('button_press_event', 100, 50)
            target = next(a for a in fig.axes if any(t.get_text()=='Save and next' for t in a.texts))
            event('button_press_event', .5, .5, target)
            event('button_release_event', .5, .5, target)
        with patch.object(plt,'show',side_effect=interact), patch.object(calibrate,'run'):
            result=calibrate.launch(self.config_path,name,self.names[0],queue_position='1 of 1',advance_end=False)
        self.assertEqual(result,'saved')
        saved=json.loads(self.config_path.read_text())['images'][name]
        self.assertEqual(saved['sampling'],'monthly')
        self.assertEqual(saved['x_start'],'2025-01')
        self.assertEqual(saved['x_end'],'2025-12')
        self.assertEqual(saved['last_historical'],'2025-04')

    def test_detected_box_can_be_corrected_and_accepted(self):
        name = self.names[1]
        image = Image.new('RGB',(140,140),'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((20,20,120,120),outline=(100,100,100))
        draw.line([(20,70),(45,70)], fill=(35,68,134), width=3)
        draw.polygon([(45,70),(70,50),(120,40),(120,60),(70,70)], fill=(35,68,134))
        image.save(self.root/name)
        self.config['images'][self.names[0]] = dict(x_start='2025Q1',x_end='2025Q4',y_min=-5,y_max=5)
        self.config_path.write_text(json.dumps(self.config))
        def interact(*args, **kwargs):
            fig=plt.gcf();fig.canvas.draw();ax=fig.axes[0]
            def event(kind,x,y,axes=ax):
                sx,sy=(axes.transData if axes is ax else axes.transAxes).transform((x,y))
                fig.canvas.callbacks.process(kind,MouseEvent(kind,fig.canvas,sx,sy,button=1))
            def button(label):
                target=next(a for a in fig.axes if any(t.get_text()==label for t in a.texts))
                event('button_press_event',.5,.5,target)
                event('button_release_event',.5,.5,target)
            # The new image inherits 2025Q4 + 1 = 2026Q1 without modifying its template.
            end=next(a for a in fig.axes if abs(a.get_position().y0-.59)<.001 and a.get_position().width > .08)
            self.assertEqual(end.texts[-1].get_text(),'2026Q1')
            button('Adjust top left')
            event('motion_notify_event',21,21)
            crosshairs = [line for line in ax.lines if (line.get_gid() or '').startswith('cursor-crosshair')]
            self.assertEqual(len(crosshairs),2)
            self.assertTrue(all(line.get_visible() for line in crosshairs))
            event('button_press_event',21,21)
            button('Accept box')
            self.assertTrue(any(t.get_text()=='3. LAST HISTORICAL point' for t in fig.texts))
            event('button_press_event',46,70)
            event('button_press_event',30,70)
            event('button_press_event',100,50)
            button('Save and next')
        with patch.object(plt,'show',side_effect=interact), patch.object(calibrate,'run'):
            result=calibrate.launch(self.config_path,name,self.names[0],queue_position='1 of 2',advance_end=True)
        self.assertEqual(result,'saved')
        c=json.loads(self.config_path.read_text())
        self.assertEqual(c['images'][self.names[0]]['x_end'],'2025Q4')
        self.assertEqual(c['images'][name]['x_end'],'2026Q1')
        self.assertAlmostEqual(c['images'][name]['plot_box'][0],21,delta=.1)
        self.assertAlmostEqual(c['images'][name]['plot_box'][2],120,delta=.1)
        self.assertEqual(c['images'][name]['last_historical'],'2025Q2')

    def test_preview_manual_adjustment_can_be_reset_to_automatic(self):
        name = self.names[1]
        image = Image.new('RGB',(140,140),'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((20,20,120,120),outline=(100,100,100))
        draw.line([(20,70),(45,70)], fill=(35,68,134), width=3)
        draw.polygon([(45,70),(70,50),(120,40),(120,60),(70,70)], fill=(35,68,134))
        image.save(self.root/name)
        self.config['images'][self.names[0]] = dict(x_start='2025Q1',x_end='2025Q4',y_min=-5,y_max=5)
        self.config_path.write_text(json.dumps(self.config))
        def interact(*args, **kwargs):
            fig=plt.gcf();fig.canvas.draw();ax=fig.axes[0]
            def event(kind,x,y,axes=ax):
                sx,sy=(axes.transData if axes is ax else axes.transAxes).transform((x,y))
                fig.canvas.callbacks.process(kind,MouseEvent(kind,fig.canvas,sx,sy,button=1))
            def button(label):
                target=next(a for a in fig.axes if any(t.get_text()==label for t in a.texts))
                event('button_press_event',.5,.5,target)
                event('button_release_event',.5,.5,target)
            button('Accept box')
            event('button_press_event',46,70)
            event('button_press_event',30,70)
            event('button_press_event',100,50)
            button('Next: Manual Adjustment')
            event('button_press_event',70,60)
            event('motion_notify_event',70,85)
            fig.canvas.callbacks.process('button_release_event',MouseEvent('button_release_event',fig.canvas,*ax.transData.transform((70,85)),button=1))
            button('Reset Manual Adjustments')
            button('Save and next')
        with patch.object(plt,'show',side_effect=interact), patch.object(calibrate,'run'):
            result=calibrate.launch(self.config_path,name,self.names[0],queue_position='1 of 2',advance_end=True)
        self.assertEqual(result,'saved')
        c=json.loads(self.config_path.read_text())
        self.assertNotIn('manual_point_overrides',c['images'][name])

    def test_preview_reset_returns_to_calibration_before_saving(self):
        name = self.names[1]
        image = Image.new('RGB',(140,140),'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((20,20,120,120),outline=(100,100,100))
        draw.line([(20,70),(45,70)], fill=(35,68,134), width=3)
        draw.polygon([(45,70),(70,50),(120,40),(120,60),(70,70)], fill=(35,68,134))
        image.save(self.root/name)
        self.config['images'][self.names[0]] = dict(x_start='2025Q1',x_end='2025Q4',y_min=-5,y_max=5)
        self.config_path.write_text(json.dumps(self.config))
        def interact(*args, **kwargs):
            fig=plt.gcf();fig.canvas.draw();ax=fig.axes[0]
            def event(kind,x,y,axes=ax):
                sx,sy=(axes.transData if axes is ax else axes.transAxes).transform((x,y))
                fig.canvas.callbacks.process(kind,MouseEvent(kind,fig.canvas,sx,sy,button=1))
            def button(label):
                target=next(a for a in fig.axes if any(t.get_text()==label for t in a.texts))
                event('button_press_event',.5,.5,target)
                event('button_release_event',.5,.5,target)
            button('Accept box')
            event('button_press_event',46,70)
            event('button_press_event',30,70)
            event('button_press_event',100,50)
            button('Reset All')
            self.assertTrue(any(t.get_text()=='Review suggested box' for t in fig.texts))
            button('Accept box')
            event('button_press_event',46,70)
            event('button_press_event',30,70)
            event('button_press_event',100,50)
            button('Save and next')
        with patch.object(plt,'show',side_effect=interact), patch.object(calibrate,'run'):
            result=calibrate.launch(self.config_path,name,self.names[0],queue_position='1 of 2',advance_end=True)
        self.assertEqual(result,'saved')

    def test_manual_adjustment_hides_calibration_pick_markers(self):
        name = self.names[1]
        image = Image.new('RGB',(140,140),'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((20,20,120,120),outline=(100,100,100))
        draw.line([(20,70),(45,70)], fill=(35,68,134), width=3)
        draw.polygon([(45,70),(70,50),(120,40),(120,60),(70,70)], fill=(35,68,134))
        image.save(self.root/name)
        self.config['images'][self.names[0]] = dict(x_start='2025Q1',x_end='2025Q4',y_min=-5,y_max=5)
        self.config_path.write_text(json.dumps(self.config))
        def interact(*args, **kwargs):
            fig=plt.gcf();fig.canvas.draw();ax=fig.axes[0]
            def event(kind,x,y,axes=ax):
                sx,sy=(axes.transData if axes is ax else axes.transAxes).transform((x,y))
                fig.canvas.callbacks.process(kind,MouseEvent(kind,fig.canvas,sx,sy,button=1))
            def button(label):
                target=next(a for a in fig.axes if any(t.get_text()==label for t in a.texts))
                event('button_press_event',.5,.5,target)
                event('button_release_event',.5,.5,target)
            button('Accept box')
            event('button_press_event',46,70)
            event('button_press_event',30,70)
            event('button_press_event',100,50)
            self.assertTrue(any(line.get_visible() and line.get_marker() == 'o' for line in ax.lines))
            button('Next: Manual Adjustment')
            self.assertFalse(any(line.get_visible() and line.get_marker() == 'o' and line.get_markersize() == 7 for line in ax.lines))
            button('Save and next')
        with patch.object(plt,'show',side_effect=interact), patch.object(calibrate,'run'):
            result=calibrate.launch(self.config_path,name,self.names[0],queue_position='1 of 2',advance_end=True)
        self.assertEqual(result,'saved')

    def test_missing_point_can_be_added_at_fixed_quarter_x(self):
        name = self.names[1]
        image = Image.new('RGB',(140,140),'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((20,20,120,120),outline=(100,100,100))
        draw.line([(20,70),(45,70)], fill=(35,68,134), width=3)
        draw.polygon([(45,70),(70,50),(120,40),(120,60),(70,70)], fill=(35,68,134))
        draw.rectangle((93,20,97,120),fill='white')
        image.save(self.root/name)
        self.config['images'][self.names[0]] = dict(x_start='2025Q1',x_end='2025Q4',y_min=-5,y_max=5)
        self.config_path.write_text(json.dumps(self.config))
        def interact(*args, **kwargs):
            fig=plt.gcf();fig.canvas.draw();ax=fig.axes[0]
            def event(kind,x,y,axes=ax):
                sx,sy=(axes.transData if axes is ax else axes.transAxes).transform((x,y))
                fig.canvas.callbacks.process(kind,MouseEvent(kind,fig.canvas,sx,sy,button=1))
            def button(label):
                target=next(a for a in fig.axes if any(t.get_text()==label for t in a.texts))
                event('button_press_event',.5,.5,target)
                event('button_release_event',.5,.5,target)
            button('Accept box')
            event('button_press_event',46,70)
            event('button_press_event',30,70)
            event('button_press_event',100,50)
            self.assertTrue(any('Missing data point: 2025 Q4' in t.get_text() for t in fig.texts))
            button('Next: Manual Adjustment')
            button('Add Missing Point')
            event('button_press_event',30,85)
            button('Save and next')
        with patch.object(plt,'show',side_effect=interact), patch.object(calibrate,'run'):
            result=calibrate.launch(self.config_path,name,self.names[0],queue_position='1 of 2',advance_end=True)
        self.assertEqual(result,'saved')
        c=json.loads(self.config_path.read_text())
        self.assertAlmostEqual(c['images'][name]['manual_point_overrides']['2025Q4']['pixel_y'],85,delta=.1)

if __name__ == '__main__':
    unittest.main()
