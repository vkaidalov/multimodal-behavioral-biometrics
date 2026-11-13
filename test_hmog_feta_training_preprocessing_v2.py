"""Run: python -m unittest -v test_hmog_feta_training_preprocessing_v2.py
No real HMOG/FETA archives are required.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import hmog_feta_gesture_dataset_prototype_v1 as v1
import hmog_feta_training_preprocessing_v2 as m


class TrainingPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.cache=self.root/'cache'
        self.cache.mkdir()
        entries=[]
        for scene,dataset,uid,acc in [('feta_7_swipe','FETA','F1',1.0),
                                      ('hmog_sitting','HMOG','H1',m.G0)]:
            rec=[]
            for j in range(4):
                length=90 if j==3 else 5+j
                elapsed=(np.arange(length,dtype=np.float32)*0.01)
                # one repeated timestamp, no forced regular grid in the converter
                elapsed[2]=elapsed[1]
                touch=np.column_stack((np.linspace(.1,.8,length),np.linspace(.3,.6,length),
                    np.linspace(0,.7,length),np.linspace(0,.3,length),elapsed,
                    np.r_[0,np.diff(elapsed)])).astype(np.float32)
                accel=np.column_stack((np.zeros(length),np.zeros(length),
                    np.full(length,acc),elapsed,np.r_[0,np.diff(elapsed)])).astype(np.float32)
                gyro=np.column_stack((np.zeros(length),np.zeros(length),
                    np.full(length,.05),elapsed,np.r_[0,np.diff(elapsed)])).astype(np.float32)
                rec.append({'touch':touch,'accel':accel,'gyro':gyro,
                            'meta':{'dataset':dataset,'model':'iPhone 7' if dataset=='FETA' else 'Samsung Galaxy S4',
                                    'user_id':uid,'session_id':'1','task':'swipe' if dataset=='FETA' else 'reading_sitting',
                                    'repetition':'','duration_ms':length*10}})
            file=f'{scene}/{uid}.npz'
            v1.save_shard(self.cache/'shards'/file,rec)
            entries.append({'scenario':scene,'dataset':dataset,'device_model':'iPhone 7' if dataset=='FETA' else 'Samsung Galaxy S4',
                            'task':'swipe' if dataset=='FETA' else 'reading_sitting',
                            'role':'train','user_id':uid,'file':'shards/'+file,'count':4})
        index={'version':v1.VERSION,'role':'train','source_splits_sha256':'synthetic-split-hash',
               'feature_names':{'touch':v1.TOUCH_FEATURES,'accel':v1.MOTION_FEATURES,'gyro':v1.MOTION_FEATURES},
               'shards':entries}
        (self.cache/'index.json').write_text(json.dumps(index))
        self.stats_file=self.root/'stats.json'
        self.au_file=self.root/'units.json'

    def tearDown(self):self.tmp.cleanup()

    def fit(self):
        m.fit(SimpleNamespace(cache=self.cache,seed=7,feta_accel_unit='g',hmog_accel_unit='mps2',
                              feta_gyro_unit='radps',hmog_gyro_unit='radps',gestures_per_user=0,
                              points_per_gesture=0,output=self.stats_file))

    def test_units_and_preflight(self):
        m.audit_units(SimpleNamespace(cache=self.cache,seed=7,max_users_per_scenario=0,
                     gestures_per_user=0,points_per_gesture=0,output=self.au_file))
        r=json.loads(self.au_file.read_text())
        self.assertAlmostEqual(r['scenarios']['feta_7_swipe']['accel']['norm_p50'],1.0,places=5)
        self.assertAlmostEqual(r['scenarios']['hmog_sitting']['accel']['norm_p50'],m.G0,places=5)
        with self.assertRaises(ValueError):
            m.unit_config(SimpleNamespace(feta_accel_unit=None,hmog_accel_unit='mps2',
                                         feta_gyro_unit='radps',hmog_gyro_unit='radps'))

    def test_fit_split_integrity_and_normalized_loader(self):
        self.fit()
        stats=json.loads(self.stats_file.read_text())
        self.assertEqual(stats['sensor_units_output']['accel'],'m/s^2')
        self.assertEqual(stats['source_index_sha256'],m.checksum(self.cache/'index.json'))
        self.assertTrue(stats['training_only'])
        self.assertEqual(stats['coverage']['feta_7_swipe']['users'],1)
        ds=m.TrainingGestureDataset(self.cache,self.stats_file,max_touch=8,max_motion=8)
        self.assertEqual(len(ds),8)
        self.assertEqual(len(ds.lengths),8)
        self.assertEqual(len(ds[3]['accel']),8)
        self.assertTrue(ds[3]['downsampled']['accel'])
        self.assertEqual(len(ds[3]['touch']),8)
        arr=ds[3]['accel'].numpy()
        self.assertAlmostEqual(float(arr[-1,-2]),.89,places=4)
        self.assertAlmostEqual(float(np.sum(arr[:,-1])),.89,places=4)
        self.assertTrue(np.all(arr[:,-1]>=0))
        self.assertEqual(float(arr[0,-1]),0)
        from torch.utils.data import DataLoader
        b=next(iter(DataLoader(ds,batch_sampler=m.LengthBucketBatchSampler(ds,batch_size=3),
                               collate_fn=v1.collate_gestures)))
        self.assertEqual(len(b['meta']),3)
        for k in m.KINDS:
            self.assertEqual(b[k].shape[:2],b[k+'_mask'].shape)
            self.assertTrue(bool((b[k+'_mask'].sum(1)==b[k+'_lengths']).all()))
        sampler=m.LengthBucketBatchSampler(ds,batch_size=3,seed=4)
        flattened=[x for batch in sampler for x in batch]
        self.assertEqual(sorted(flattened),list(range(len(ds))))
        # Input cache must remain untouched
        original=v1.GestureDataset(self.cache)[0]
        self.assertEqual(float(original['accel'][0,2]),1.0)
        self.assertTrue(np.isfinite(np.array(stats['global']['accel']['std'])).all())

    def test_inspect_and_hash_guard(self):
        self.fit()
        out=self.root/'inspect.json'
        m.inspect(SimpleNamespace(cache=self.cache,stats=self.stats_file,scenario=None,
                  scope='global',max_touch=8,max_motion=8,clip_z=0.0,batch_size=3,
                  bucket_multiplier=2,seed=8,inspect_batches=5,output=out))
        r=json.loads(out.read_text())
        self.assertEqual(r['gestures_inspected'],8)
        self.assertEqual(r['capped_count']['accel'],2)
        self.assertEqual(r['negative_dt'],0)
        st=json.loads(self.stats_file.read_text())
        st['source_index_sha256']='wrong'
        with self.assertRaises(ValueError):
            m.TrainingGestureDataset(self.cache,st)

if __name__=='__main__':unittest.main()
