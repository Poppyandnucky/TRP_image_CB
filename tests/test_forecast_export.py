import tempfile
import unittest
from pathlib import Path
import pandas as pd
from inflation_digitizer.batch import aggregate_monthly_frame, refresh_combined_outputs
from inflation_digitizer.forecast_export import COLUMNS, convert_frame, export_csv_files


class ForecastExportTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            'country':['COL']*4,
            'vintage':['2021Q4']*4,
            'quarter':['2022Q2','2021Q4','2020Q3','2022Q1'],
            'inflation_pct_yoy':[5.6,4.0,2.1,6.23],
            'series':['forecast_band_midpoint','historical','historical','forecast_band_midpoint'],
        })

    def test_exact_schema_signed_horizons_and_value_preservation(self):
        result=convert_frame(self.frame())
        self.assertEqual(list(result.columns),COLUMNS)
        self.assertEqual(result.target.tolist(),['2020Q3','2021Q4','2022Q1','2022Q2'])
        self.assertEqual(result.h.tolist(),[-5,0,1,2])
        self.assertEqual(result.forecast.tolist(),[2.1,4.0,6.23,5.6])
        self.assertEqual(result.variable.tolist(),['CPI']*4)
        self.assertEqual(result.h.dtype.kind,'i')

    def test_missing_estimates_remain_blank_not_zero(self):
        frame=self.frame();frame.loc[0,'inflation_pct_yoy']=None
        result=convert_frame(frame)
        self.assertEqual(len(result),4)
        self.assertTrue(pd.isna(result.loc[result.target=='2022Q2','forecast'].iloc[0]))

    def test_invalid_dates_and_duplicate_keys_are_rejected(self):
        frame=self.frame();frame.loc[0,'quarter']='2022Q5'
        with self.assertRaisesRegex(ValueError,'Invalid report or target'):
            convert_frame(frame)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            convert_frame(pd.concat([self.frame(),self.frame()],ignore_index=True))

    def test_all_per_report_files_are_combined_without_dropping_repeated_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'tables';source.mkdir()
            self.frame().to_csv(source/'a.csv',index=False)
            second=self.frame();second['vintage']='2022Q1';second.to_csv(source/'b.csv',index=False)
            result=export_csv_files(source,root/'forecasts.csv',per_report_directory=root/'formatted')
            self.assertEqual(len(result),8)
            self.assertEqual(len(list((root/'formatted').glob('*.csv'))),2)
            saved=pd.read_csv(root/'forecasts.csv')
            self.assertEqual(list(saved.columns),COLUMNS)
            self.assertEqual(sorted(saved.report.unique()),['2021Q4','2022Q1'])
            self.assertEqual(len(pd.read_csv(source/'a.csv')),4)
            self.assertEqual((root/'forecasts.csv').read_text().splitlines()[0],','.join(COLUMNS))

    def test_empty_input_with_headers_has_exact_output_headers(self):
        result=convert_frame(self.frame().iloc[:0])
        self.assertEqual(list(result.columns),COLUMNS)
        self.assertEqual(len(result),0)

    def test_monthly_rows_aggregate_only_complete_unmixed_quarters(self):
        frame=pd.DataFrame({
            'country':['PER']*8,
            'vintage':['2021Q1']*8,
            'period':['2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08'],
            'series':['historical']*3+['historical','forecast_band_midpoint','historical']+['forecast_band_midpoint']*2,
            'inflation_pct_yoy':[1,2,3,4,5,6,7,8],
            'central_band_lower_pct':['']*8,
            'central_band_upper_pct':['']*8,
            'pixel_x':['']*8,'pixel_y':['']*8,
            'extraction_method':['color_match']*8,
            'sampled_x_min':['']*8,'sampled_x_max':['']*8,
            'source_file':['chart.png']*8,
            'source_path':['/tmp/chart.png']*8,
            'calibration_status':['reviewed']*8,
            'quality_flag':['']*8,
        })
        quarterly,warnings=aggregate_monthly_frame(frame)
        self.assertEqual(quarterly.quarter.tolist(),['2021Q1'])
        self.assertEqual(quarterly.inflation_pct_yoy.tolist(),[2.0])
        self.assertEqual(quarterly.extraction_method.tolist(),['monthly_average'])
        self.assertEqual(set(warnings.reason),{'mixed_historical_forecast_months','incomplete_monthly_quarter'})

    def test_refresh_prefers_moved_raw_table_with_existing_source_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);tables=root/'tables';tables.mkdir()
            live_image=root/'screenshots'/'quarterly'/'PHL_2025_Q1.png'
            live_image.parent.mkdir(parents=True)
            live_image.write_bytes(b'placeholder')
            stale=self.frame().iloc[:1].copy()
            stale['country']='PHL';stale['vintage']='2025Q1';stale['quarter']='2025Q1'
            stale['source_file']='PHL_2025_Q1.png'
            stale['source_path']=str(root/'screenshots'/'PHL_2025_Q1.png')
            stale.to_csv(tables/'PHL_2025_Q1.csv',index=False)
            live=stale.copy()
            live['inflation_pct_yoy']=9.5
            live['source_path']=str(live_image)
            live.to_csv(tables/'quarterly__PHL_2025_Q1.csv',index=False)
            combined=refresh_combined_outputs(root)
            self.assertEqual(combined.inflation_pct_yoy.tolist(),[9.5])
            warnings=pd.read_csv(root/'monthly_aggregation_warnings.csv')
            self.assertEqual(warnings.reason.tolist(),
                             ['stale_raw_duplicate_replaced_by_existing_source_path'])

    def test_refresh_skips_stale_invalid_raw_table_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);tables=root/'tables';tables.mkdir()
            monthly=root/'monthly_tables';monthly.mkdir()
            stale=self.frame().iloc[:1].copy()
            stale['country']='PER';stale['vintage']='PER2002Q2';stale['quarter']='2001Q1'
            stale['source_file']='PER2002Q2.png'
            stale['source_path']=str(root/'screenshots'/'PER2002Q2.png')
            stale.to_csv(tables/'PER2002Q2.csv',index=False)
            monthly_frame=pd.DataFrame({
                'country':['PER']*3,
                'vintage':['2002Q2']*3,
                'period':['2001-01','2001-02','2001-03'],
                'series':['historical']*3,
                'inflation_pct_yoy':[1,2,3],
                'central_band_lower_pct':['']*3,
                'central_band_upper_pct':['']*3,
                'pixel_x':['']*3,'pixel_y':['']*3,
                'extraction_method':['color_match']*3,
                'sampled_x_min':['']*3,'sampled_x_max':['']*3,
                'source_file':['PER2002Q2.png']*3,
                'source_path':[str(root/'screenshots'/'monthly'/'PER2002Q2.png')]*3,
                'calibration_status':['reviewed']*3,
                'quality_flag':['']*3,
            })
            monthly_frame.to_csv(monthly/'monthly__PER2002Q2.csv',index=False)
            combined=refresh_combined_outputs(root)
            self.assertTrue((tables/'PER2002Q2.csv').exists())
            self.assertEqual(combined.vintage.tolist(),['2002Q2'])
            self.assertEqual(combined.quarter.tolist(),['2001Q1'])
            warnings=pd.read_csv(root/'monthly_aggregation_warnings.csv')
            self.assertIn('stale_invalid_raw_table_missing_source_path', warnings.reason.tolist())

if __name__=='__main__':unittest.main()
