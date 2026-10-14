import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
from inflation_digitizer.core import (extract, quarter_number, quarter_label, month_number, month_label,
                                      validate_calibration, report_quarter, report_sort_key)

class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'TST_2025_Q1.png'
        self.c = dict(image_size=[140,140], plot_box=[20,20,120,120],
                      y_min=-5,y_max=5,x_start='2025Q1',x_end='2026Q1',
                      last_historical='2025Q2',curve_rgb=[35,68,134],color_tolerance=10,
                      review_status='reviewed')
    def tearDown(self):
        self.temp.cleanup()
    def make_image(self, missing=False):
        image=Image.new('RGB',(140,140),'white')
        draw=ImageDraw.Draw(image)
        draw.line([(20,70),(45,70)], fill=(35,68,134), width=3)
        draw.polygon([(45,70),(70,50),(120,40),(120,60),(70,70)],fill=(35,68,134))
        # Occluding grey grid line through the fan must not shift its centre.
        draw.line((0,55,139,55),fill=(150,150,150))
        if missing:
            draw.rectangle((94,20,96,120),fill='white')
        image.save(self.path)
    def test_black_history_and_red_fan_with_black_axes(self):
        image=Image.new('RGB',(140,140),'white');d=ImageDraw.Draw(image)
        d.line((20,20,20,120,120,120),fill='black',width=2)
        d.line((20,90,120,90),fill='black')
        d.line((20,70,45,70),fill='black',width=3)
        d.polygon([(46,70),(70,50),(120,40),(120,60),(70,70)],fill=(145,47,45))
        image.save(self.path)
        c={**self.c,'curve_rgb':[145,47,45],'historical_rgb':[0,0,0]}
        rows=extract(self.path,c,'TST').rows
        self.assertAlmostEqual(rows[0]['inflation_pct_yoy'],0,delta=.1)
        self.assertEqual(rows[0]['extraction_method'],'historical_endpoint_fit')
        self.assertAlmostEqual(rows[1]['inflation_pct_yoy'],0,delta=.1)
        self.assertAlmostEqual(rows[2]['inflation_pct_yoy'],1,delta=.1)

    def test_month_named_report_vintages_and_order(self):
        self.assertEqual(report_quarter('BRA_2012_Sep.png'),'2012Q3')
        self.assertEqual(report_quarter('BRA_2012_Dec.png'),'2012Q4')
        self.assertEqual(report_quarter('CHL_2025_Q3.png'),'2025Q3')
        self.assertEqual(report_quarter('PER 2002 Q2.png'),'2002Q2')
        self.assertEqual(report_quarter('PER2002Q2.png'),'2002Q2')
        names=['BRA_2013_Dec.png','BRA_2013_Sep.png','BRA_2013_Mar.png','BRA_2013_Jun.png']
        self.assertEqual(sorted(names,key=report_sort_key),[names[2],names[3],names[1],names[0]])
        self.make_image()
        path=self.path.with_name('BRA_2025_Mar.png')
        path.write_bytes(self.path.read_bytes())
        self.assertEqual(extract(path,self.c,'BRA').rows[0]['vintage'],'2025Q1')

    def test_unsupported_sampling_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'sampling'):
            validate_calibration({**self.c,'sampling':'quarter_end_month'},(140,140))

    def test_quarters(self):
        self.assertEqual(quarter_label(quarter_number('2025_Q4')+1),'2026Q1')
        with self.assertRaises(ValueError):quarter_number('2025Q5')

    def test_monthly_sampling_uses_month_periods_and_overrides(self):
        self.make_image()
        c={**self.c,'sampling':'monthly','x_start':'2025-01','x_end':'2025-05',
           'last_historical':'2025-02','manual_point_overrides':{'2025-03':{'pixel_y':80}}}
        result=extract(self.path,c,'TST')
        self.assertEqual([row['period'] for row in result.rows],
                         ['2025-01','2025-02','2025-03','2025-04','2025-05'])
        self.assertEqual(result.rows[1]['series'],'historical')
        self.assertEqual(result.rows[2]['series'],'forecast_band_midpoint')
        self.assertEqual(result.rows[2]['extraction_method'],'manual_override')
        self.assertAlmostEqual(result.rows[2]['inflation_pct_yoy'],-1,delta=.01)
        self.assertEqual(month_label(month_number('2025-12')+1),'2026-01')
    def test_known_band_midpoints_and_history_split(self):
        self.make_image()
        result=extract(self.path,self.c,'TST')
        self.assertEqual(len(result.rows),5)
        self.assertAlmostEqual(result.rows[0]['inflation_pct_yoy'],0,delta=.1)
        self.assertAlmostEqual(result.rows[2]['inflation_pct_yoy'],1,delta=.1)
        self.assertAlmostEqual(result.rows[4]['inflation_pct_yoy'],2,delta=.1)
        self.assertEqual(result.rows[1]['series'],'historical')
        self.assertEqual(result.rows[2]['series'],'forecast_band_midpoint')
        self.assertAlmostEqual(result.rows[4]['central_band_lower_pct'],1,delta=.1)
        self.assertAlmostEqual(result.rows[4]['central_band_upper_pct'],3,delta=.101)
    def test_manual_point_override_replaces_extracted_y_value(self):
        self.make_image()
        c={**self.c,'manual_point_overrides':{'2025Q3':{'pixel_y':80}}}
        row=extract(self.path,c,'TST').rows[2]
        self.assertEqual(row['extraction_method'],'manual_override')
        self.assertEqual(row['quality_flag'],'manual_override')
        self.assertAlmostEqual(row['inflation_pct_yoy'],-1,delta=.01)
    def test_manual_point_override_can_fill_missing_auto_point(self):
        self.make_image(missing=True)
        c={**self.c,'manual_point_overrides':{'2025Q4':{'pixel_y':80}}}
        row=extract(self.path,c,'TST').rows[3]
        self.assertEqual(row['extraction_method'],'manual_override')
        self.assertIn('manual_override',row['quality_flag'])
        self.assertAlmostEqual(row['inflation_pct_yoy'],-1,delta=.01)
    def test_missing_values_are_flagged_not_fabricated(self):
        self.make_image(missing=True)
        result=extract(self.path,self.c,'TST')
        self.assertIsNone(result.rows[3]['inflation_pct_yoy'])
        self.assertTrue(result.warnings)
    def test_vertical_frame_does_not_erase_final_forecast(self):
        self.make_image()
        image = Image.open(self.path)
        ImageDraw.Draw(image).rectangle((119,20,121,120),fill=(150,150,150))
        image.save(self.path)
        result=extract(self.path,self.c,'TST')
        self.assertAlmostEqual(result.rows[-1]['inflation_pct_yoy'],2,delta=.1)
    def test_changed_dimensions_rejected(self):
        with self.assertRaises(ValueError):validate_calibration(self.c,(280,280))
    def test_invalid_historical_boundary_rejected(self):
        c={**self.c,'last_historical':'2027Q1'}
        with self.assertRaises(ValueError):validate_calibration(c,(140,140))

    def test_greyer_historical_line_is_recovered_without_changing_forecast(self):
        self.make_image()
        expected=extract(self.path,self.c,'TST')
        image=Image.open(self.path)
        draw=ImageDraw.Draw(image)
        draw.rectangle((20,65,43,75),fill='white')
        draw.line((20,70,43,70),fill=(130,120,145),width=3)
        # A neutral grid line should not be mistaken for historical data.
        draw.line((20,90,43,90),fill=(100,100,100),width=2)
        image.save(self.path)
        result=extract(self.path,self.c,'TST')
        self.assertAlmostEqual(result.rows[0]['inflation_pct_yoy'],0,delta=.1)
        self.assertEqual(result.rows[0]['extraction_method'],'historical_color_recovery')
        self.assertEqual(result.rows[2]['inflation_pct_yoy'],expected.rows[2]['inflation_pct_yoy'])

    def test_neutral_grid_is_not_fabricated_as_missing_history(self):
        self.make_image()
        image=Image.open(self.path);draw=ImageDraw.Draw(image)
        draw.rectangle((20,20,30,120),fill='white')
        draw.line((20,80,30,80),fill=(100,100,100),width=3)
        image.save(self.path)
        result=extract(self.path,self.c,'TST')
        self.assertIsNone(result.rows[0]['inflation_pct_yoy'])

    def test_ambiguous_historical_strokes_remain_flagged(self):
        self.make_image()
        image=Image.open(self.path);draw=ImageDraw.Draw(image)
        draw.line((40,90,49,90),fill=(35,68,134),width=3)
        image.save(self.path)
        result=extract(self.path,self.c,'TST')
        self.assertIsNone(result.rows[1]['inflation_pct_yoy'])

    def test_thick_frame_uses_an_auditable_interior_sample(self):
        self.make_image()
        image=Image.open(self.path)
        ImageDraw.Draw(image).rectangle((117,20,121,120),fill=(150,150,150))
        image.save(self.path)
        result=extract(self.path,self.c,'TST')
        row=result.rows[-1]
        self.assertAlmostEqual(row['inflation_pct_yoy'],2,delta=.2)
        self.assertEqual(row['extraction_method'],'edge_inset_color_match')
        self.assertIn('edge_sample_inset',row['quality_flag'])
        self.assertLess(row['sampled_x_max'],row['pixel_x'])

    def test_overprinted_forecast_line_does_not_split_one_band(self):
        self.make_image()
        expected=extract(self.path,self.c,'TST').rows[3]['inflation_pct_yoy']
        image=Image.open(self.path)
        ImageDraw.Draw(image).rectangle((94,51,96,59),fill=(20,20,20))
        image.save(self.path)
        row=extract(self.path,self.c,'TST').rows[3]
        self.assertAlmostEqual(row['inflation_pct_yoy'],expected,delta=.1)
        self.assertEqual(row['extraction_method'],'forecast_overprint_recovery')

    def test_white_gap_is_not_bridged_as_overprinted_ink(self):
        self.make_image()
        image=Image.open(self.path)
        ImageDraw.Draw(image).rectangle((94,51,96,59),fill='white')
        image.save(self.path)
        self.assertIsNone(extract(self.path,self.c,'TST').rows[3]['inflation_pct_yoy'])

    def test_explicit_centerline_inside_narrow_fan_is_auditable(self):
        self.make_image()
        image=Image.open(self.path);draw=ImageDraw.Draw(image)
        draw.rectangle((69,20,71,120),fill='white')
        draw.rectangle((69,54,71,66),fill=(140,160,200))
        draw.rectangle((69,58,71,62),fill=(13,13,13))
        image.save(self.path)
        row=extract(self.path,self.c,'TST').rows[2]
        self.assertAlmostEqual(row['inflation_pct_yoy'],1,delta=.1)
        self.assertEqual(row['extraction_method'],'visible_forecast_centerline')
        self.assertIsNone(row['central_band_lower_pct'])
        self.assertIsNone(row['central_band_upper_pct'])

if __name__=='__main__':unittest.main()
