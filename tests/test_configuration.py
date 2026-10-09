import json
import tempfile
import unittest
from pathlib import Path
from inflation_digitizer.configuration import initialize_country, publish_seed, load_working_config, input_directory


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_seed_copy_preserves_examples_and_refuses_overwrite(self):
        p=self.root/'config/seeds/PER.json';p.parent.mkdir(parents=True)
        seed=dict(country='PER',config_role='seed',input_directory='data/PHL/screenshots',
                  output_directory='output/PER',images={'PER_2020_Q1.png':{'review_status':'reviewed'}})
        p.write_text(json.dumps(seed))
        path=initialize_country('PER',root=self.root)
        working=load_working_config(path)
        self.assertEqual(working['images'],seed['images'])
        self.assertEqual(working['config_role'],'working')
        before=path.read_bytes()
        seed['images']={};p.write_text(json.dumps(seed))
        with self.assertRaisesRegex(ValueError,'progress was preserved'):
            initialize_country('PER',root=self.root)
        self.assertEqual(path.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'shared seed'):
            load_working_config(p)

    def test_new_country_and_relative_paths(self):
        path=initialize_country('per',root=self.root)
        c=load_working_config(path)
        self.assertEqual(c['images'],{})
        self.assertEqual(input_directory(c,self.root),self.root/'data/PHL/screenshots')
        self.assertFalse((self.root/'config/seeds/PER.json').exists())
        with self.assertRaises(ValueError):initialize_country('../bad',root=self.root)

    def test_publish_selected_examples_is_portable_and_leaves_progress(self):
        path=initialize_country('PER',source=self.root/'private-input',root=self.root)
        c=load_working_config(path)
        c['images']={'one.png':{'y_min':-2},'two.png':{'y_min':0}}
        c['last_calibrated_image']='two.png'
        c['output_directory']=str(self.root/'personal-results')
        path.write_text(json.dumps(c));before=path.read_bytes()
        seed_path=publish_seed('PER',['one.png'],root=self.root)
        seed=json.loads(seed_path.read_text())
        self.assertEqual(list(seed['images']),['one.png'])
        self.assertEqual(seed['input_directory'],'data/PHL/screenshots')
        self.assertEqual(seed['output_directory'],'output/PER')
        self.assertNotIn('last_calibrated_image',seed)
        self.assertEqual(path.read_bytes(),before)
        with self.assertRaises(ValueError):publish_seed('PER',['missing.png'],root=self.root)
        self.assertEqual(json.loads(seed_path.read_text()),seed)
