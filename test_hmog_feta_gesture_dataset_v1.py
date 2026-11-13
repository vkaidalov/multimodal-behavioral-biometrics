"""Tests for gesture preprocessing; no real HMOG/FETA archives needed.

Run: python -m unittest -v test_hmog_feta_gesture_dataset_v1.py
"""
from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

import hmog_feta_gesture_dataset_prototype_v1 as pp
import analyze_hmog_feta_feta_criterion_v2_0 as v2


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.feta=self.root/'input_data'/'data_files'
        self.hmog=self.root/'input_data'/'hmog_unarchived'
        self.userdata=self.root/'input_data'/'tables'/'tables'/'userdata.csv'
        self.out=self.root/'out'
        self.split_path=self.root/'frozen.json'
        self.make_feta()
        self.make_hmog()
        doc={'status':'FROZEN_VERIFIED_AGAINST_V2_AUDIT',
             'protocol':{'count_mode':'strict','threshold_gestures_per_each_first_four_sessions':20},
             'splits':{'FETA':{'iPhone 7':{'train':['F1'],'validation':['F2'],'test':[]}},
                       'HMOG':{'reading_sitting':{'train':['H1'],'validation':['H2'],'test':[]}}}}
        self.split_path.write_text(json.dumps(doc),encoding='utf-8')

    def tearDown(self):self.temp.cleanup()

    def make_feta(self):
        write_csv(self.userdata,['uuid','phone_model'],[['F1','iPhone 7'],['F2','iPhone 7']])
        def rep(user,repetition,touch_rows,sensor_start=1004):
            d=self.feta/user/'100'/'swipe'/str(repetition)
            write_csv(d/'touch_data'/'touch.csv',['timestamp','type','x','y'],touch_rows)
            rows=[[sensor_start+i*8,0.1+i*0.01,0.2+i*0.01,0.3+i*0.01] for i in range(10)]
            write_csv(d/'accelerometer_data'/'sensor.csv',['timestamp','x','y','z'],rows)
            write_csv(d/'gyroscope_data'/'sensor.csv',['timestamp','x','y','z'],rows)
        # FETA accepted >=3 & >5 px, 32 ms duration
        rep('F1','0',[[1000,0,100,100],[1016,1,107,105],[1032,2,110,105]])
        # Exactly 5 px radial displacement: rejected.
        rep('F1','1',[[1000,0,100,100],[1016,1,103,104],[1032,2,100,100]])
        # This validation user exists on disk but MUST never get parsed.
        rep('F2','0',[[1000,0,100,100],[1016,1,120,100],[1032,2,130,100]])

    def make_hmog(self):
        session=self.hmog/'H1'/'1'
        act_rows=[['a1','H1','1',0,0,0,0,0,1,0]]
        write_csv(session/'Activity.csv',v2.H_SCHEMA['activity'],act_rows)
        touch_rows=[
            [0,1000,'a1',1,0,0,100,200,0,0,0],
            [0,1100,'a1',1,0,2,110,220,0,0,0],
            [0,1200,'a1',1,0,2,125,250,0,0,0],
            [0,1600,'a1',1,0,1,180,300,0,0,0],
            # 2nd contact contains a landscape point -> invalid
            [0,2000,'a1',1,0,0,100,200,0,0,0],
            [0,2020,'a1',1,0,2,112,215,0,0,1],
            [0,2040,'a1',1,0,1,125,240,0,0,0],
        ]
        write_csv(session/'TouchEvent.csv',v2.H_SCHEMA['touch'],touch_rows)
        sensor_rows=[]
        for t in list(range(1008,1600,18))+[1600]+[2004,2016,2028,2039]:
            sensor_rows.append([0,int(t*1e6),'a1',0.3,0.4,0.5,0])
        write_csv(session/'Accelerometer.csv',v2.H_SCHEMA['accel'],sensor_rows)
        write_csv(session/'Gyroscope.csv',v2.H_SCHEMA['gyro'],sensor_rows)
        # validation exists; never allowed in samples
        other=self.hmog/'H2'/'1'
        write_csv(other/'Activity.csv',v2.H_SCHEMA['activity'],act_rows)
        write_csv(other/'TouchEvent.csv',v2.H_SCHEMA['touch'],touch_rows)

    def test_v2_comparable_and_masked_batches(self):
        import torch
        from torch.utils.data import DataLoader
        # Reference v2 counts, real CSV parsing, not handpicked numbers.
        params={'min_sensor_samples':3,'edge_gap_ms':40.,'internal_gap_ms':120.,
                'pair_tolerance_ms':25.,'min_sensor_pairs':1,'save_gestures':False}
        fdir=self.feta/'F1'/'100'/'swipe'/'0'
        cnt,*_=v2.feta_rep(fdir,{'dataset':'FETA','device_model':'iPhone 7',
            'user_id':'F1','session_id':'100','task':'swipe'},params)
        self.assertEqual(cnt['gestures_strict_multimodal'],1)
        hrows,_,_=v2.hmog_session(self.hmog/'H1'/'1','H1',params)
        self.assertEqual([r['gestures_strict_multimodal'] for r in hrows if r['task']=='reading_sitting'],[1])
        argv=['prepare','--splits',str(self.split_path),'--feta-root',str(self.feta),
              '--hmog-root',str(self.hmog),'--userdata',str(self.userdata),'--output',str(self.out),
              '--scenarios','feta_7_swipe,hmog_sitting','--max-users-per-scenario','0',
              '--max-gestures-per-user','0']
        args=pp.parser().parse_args(argv)
        pp.prepare(args)
        dataset=pp.GestureDataset(self.out)
        self.assertEqual(len(dataset),2)
        d=[dataset[i] for i in range(2)]
        self.assertEqual({x['meta']['dataset'] for x in d},{'FETA','HMOG'})
        self.assertNotIn('F2',[x['meta']['user_id'] for x in d])
        self.assertNotIn('H2',[x['meta']['user_id'] for x in d])
        feta=next(x for x in d if x['meta']['dataset']=='FETA')
        hmog=next(x for x in d if x['meta']['dataset']=='HMOG')
        self.assertEqual(tuple(feta['touch'].shape),(3,6))
        self.assertGreater(hmog['accel'].shape[0],30)
        self.assertEqual(len(hmog['touch']),4)
        self.assertTrue((feta['touch'][:,-1]>=0).all().item())
        self.assertAlmostEqual(float(feta['touch'][-1,-2]),0.032,places=4)
        loader=DataLoader(dataset,batch_size=2,num_workers=0,collate_fn=pp.collate_gestures)
        b=next(iter(loader))
        self.assertEqual(tuple(b['touch'].shape),(2,4,6))
        self.assertTrue(bool((b['touch_mask'].sum(1)==b['touch_lengths']).all()))
        self.assertTrue(bool((b['gyro_mask'].sum(1)==b['gyro_lengths']).all()))
        pp.inspect(pp.parser().parse_args(['inspect','--cache',str(self.out),'--batch-size','2']))
        self.assertTrue((self.out/'sample_inspection.csv').exists())
        self.assertTrue((self.out/'inspection_summary.json').exists())

    def test_strict_requires_portrait_for_all_hmog_points(self):
        gestures=pp.parse_hmog_touch(v2.read_hmog(self.hmog/'H1'/'1'/'TouchEvent.csv','touch'),{'a1':'reading_sitting'})
        self.assertEqual(len(gestures),2)
        self.assertTrue(pp.valid_geometry(gestures[0]))
        self.assertFalse(pp.valid_geometry(gestures[1]))

    def test_feta_exact_threshold_and_timestamps(self):
        good=pp.parse_feta_touch(self.feta/'F1'/'100'/'swipe'/'0'/'touch_data'/'touch.csv')
        exact=pp.parse_feta_touch(self.feta/'F1'/'100'/'swipe'/'1'/'touch_data'/'touch.csv')
        self.assertTrue(pp.valid_geometry(good[0]))
        self.assertFalse(pp.valid_geometry(exact[0]))
        t,dt=pp.temporal(np.array([1000.,1007.,1007.,1029.]),1000.)
        self.assertTrue(np.array_equal(dt,np.array([0.,.007,0.,.022])))
        self.assertEqual(t[-1],.029)

if __name__=='__main__':unittest.main()
