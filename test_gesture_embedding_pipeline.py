"""Synthetic end-to-end checks; raw HMOG/FETA archives are NOT required."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

import hmog_feta_gesture_dataset_prototype_v1 as pp
from gesture_pipeline_data import GestureCacheDataset,IdentityScenarioPKSampler,file_hash,collate
from gesture_model import GestureEncoder,supervised_contrastive_loss
from gesture_evaluation_core import (save_embedding_table,load_embedding_table,
    build_role_scenarios,biometric_scores_for_scenario,roc_metrics,checkpoint_macro_auc)

SC='feta_7_swipe'


class GesturePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.splits=self.root/'splits.json'
        split={'status':'FROZEN_VERIFIED_AGAINST_V2_AUDIT',
               'protocol':{'count_mode':'strict','threshold_gestures_per_each_first_four_sessions':20},
               'splits':{'FETA':{'iPhone 7':{'train':['T1','T2','T3','T4'],
                           'validation':['V1','V2','V3'],'test':['E1','E2','E3']}},'HMOG':{}},
               'session_index':{'FETA':{'iPhone 7':{u:{'swipe':['1','2','3','4']}
                       for u in ['T1','T2','T3','T4','V1','V2','V3','E1','E2','E3']}},'HMOG':{}}}
        self.splits.write_text(json.dumps(split),encoding='utf-8')
        self.paths={}
        for role,users in [('train',['T1','T2','T3','T4']),('validation',['V1','V2','V3']),('test',['E1','E2','E3'])]:
            folder=self.root/role;folder.mkdir()
            entries=[]
            for user in users:
                records=[]
                for sess in range(1,5):
                    for g in range(4):
                        start=1_000_000+sess*100000+g*200
                        t=np.linspace(0,.1,5,dtype=np.float32)
                        touch=np.column_stack((np.linspace(.1,.2,5),np.linspace(.2,.5,5),
                            np.linspace(0,.1,5),np.linspace(0,.3,5),t,np.diff(t,prepend=0))).astype(np.float32)
                        tm=np.linspace(0,.1,7,dtype=np.float32)
                        motion=np.column_stack((np.ones(7),np.ones(7)*.2,np.ones(7)*.3,
                                                tm,np.diff(tm,prepend=0))).astype(np.float32)
                        records.append({'touch':touch,'accel':motion,'gyro':motion,
                          'meta':{'dataset':'FETA','model':'iPhone 7','task':'swipe','user_id':user,
                                  'session_id':str(sess),'repetition':'0','start_ms':start,
                                  'end_ms':start+100,'duration_ms':100.,'coord_outside_screen':False}})
                fname=f'shards/{SC}/{user}.npz';pp.save_shard(folder/fname,records)
                entries.append({'scenario':SC,'dataset':'FETA','device_model':'iPhone 7',
                                 'task':'swipe','role':role,'user_id':user,'file':fname,'count':len(records)})
            index={'role':role,'version':pp.VERSION,'source_splits_sha256':file_hash(self.splits),
                   'settings':{'session_scope':'first-four'},
                   'feature_names':{'touch':list(pp.TOUCH_FEATURES),'accel':list(pp.MOTION_FEATURES),
                                    'gyro':list(pp.MOTION_FEATURES)},'shards':entries}
            (folder/'index.json').write_text(json.dumps(index),encoding='utf-8')
            self.paths[role]=folder
        self.stats=self.root/'stats.json'
        norm={'accel':{'mean':[0,0,0],'std':[1,1,1]},'gyro':{'mean':[0,0,0],'std':[1,1,1]}}
        self.stats.write_text(json.dumps({'training_only':True,
               'source_index_sha256':file_hash(self.paths['train']/'index.json'),
               'splits_sha256':file_hash(self.splits),
               'sensor_units_input':{'FETA':{'accel':'g','gyro':'radps'},
                                     'HMOG':{'accel':'mps2','gyro':'radps'}},
               'global':norm,'by_scenario':{SC:norm}}),encoding='utf-8')
        self.splitdoc=split

    def tearDown(self):self.temp.cleanup()

    def ds(self,role):
        return GestureCacheDataset(self.paths[role],self.stats,expected_role=role,
                                   scope='scenario',splits=self.splits)

    def test_loader_sampler_and_model(self):
        train=self.ds('train');val=self.ds('validation')
        self.assertEqual(len(train),64);self.assertEqual(len(val),48)
        self.assertFalse(set(train.by_identity)&set(val.by_identity))
        sampler=IdentityScenarioPKSampler(train,p=2,k=4,steps=2)
        batch_ids=next(iter(sampler))
        self.assertEqual(len(batch_ids),8)
        for label in set(train.labels[i] for i in batch_ids):
            idx=[i for i in batch_ids if train.labels[i]==label]
            self.assertEqual(len(idx),4)
            self.assertGreaterEqual(len({train.metadata[i]['session_id'] for i in idx}),2)
        b=collate([train[i] for i in batch_ids]);model=GestureEncoder(width=16,embedding_dim=64)
        emb=model(b);self.assertEqual(tuple(emb.shape),(8,64))
        self.assertTrue(torch.allclose(torch.linalg.norm(emb,dim=1),torch.ones(8),atol=1e-5))
        loss=supervised_contrastive_loss(emb,b['labels']);self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_role_isolation(self):
        with self.assertRaises(ValueError):
            GestureCacheDataset(self.paths['test'],self.stats,expected_role='validation',splits=self.splits)
        with self.assertRaises(ValueError):
            GestureCacheDataset(self.paths['validation'],self.stats,expected_role='train',splits=self.splits)

    def test_validation_raw_parser_uses_frozen_four_sessions(self):
        # Integration with the actual FETA CSV parser from the corrected v1.1.
        import argparse
        import shutil
        from prepare_eval_gesture_cache import build
        from test_hmog_feta_gesture_dataset_v1 import PreprocessingTests
        fx=PreprocessingTests()
        fx.setUp()
        try:
            # Make four physical validation sessions, plus a fifth that MUST be excluded.
            for session in ('101','102','103','104'):
                shutil.copytree(fx.feta/'F2'/'100',fx.feta/'F2'/session)
            doc=json.loads(fx.split_path.read_text(encoding='utf-8'))
            doc['session_index']={'FETA':{'iPhone 7':{'F2':{'swipe':['100','101','102','103']}}}}
            fx.split_path.write_text(json.dumps(doc),encoding='utf-8')
            dest=fx.root/'eval_cache'
            args=argparse.Namespace(role='validation',splits=fx.split_path,
                hmog_root=fx.hmog,feta_root=fx.feta,userdata=fx.userdata,
                output=dest,scenarios='feta_7_swipe',feta_coordinates='points',fail_on_errors=True)
            build(args)
            idx=json.loads((dest/'index.json').read_text(encoding='utf-8'))
            self.assertEqual(idx['role'],'validation')
            self.assertEqual(idx['settings']['session_scope'],'first-four')
            self.assertEqual(idx['scenarios']['feta_7_swipe']['gestures'],4)
            self.assertEqual(len(idx['shards']),1)
            with np.load(dest/idx['shards'][0]['file'],allow_pickle=False) as z:
                self.assertEqual({json.loads(str(x))['session_id'] for x in z['meta_json']},
                                 {'100','101','102','103'})
        finally:
            fx.tearDown()

    def test_per_user_roc_and_windows(self):
        val=self.ds('validation')
        rng=np.random.default_rng(9)
        embeddings=[]
        for m in val.metadata:
            n=int(m['user_id'][-1]);v=np.zeros(64,dtype=float);v[n-1]=1
            v+=rng.normal(0,.015,64);v/=np.linalg.norm(v);embeddings.append(v)
        emb=np.asarray(embeddings,dtype=np.float32)
        tmp=self.root/'emb.npz'
        save_embedding_table(tmp,emb,val.metadata,{'role':'validation','model_id':SC})
        e,m,extra=load_embedding_table(tmp)
        self.assertEqual(extra['role'],'validation')
        groups=build_role_scenarios(e,m,self.splitdoc,'validation')
        rows=biometric_scores_for_scenario(e,m,groups[SC],self.splitdoc,SC,3,10)
        self.assertGreater(len(rows),0)
        self.assertTrue(all(x[8]>=0 for x in rows))
        # Four gestures/session, two verification sessions -> two windows per session.
        self.assertEqual(sum(r[2] for r in rows if r[0]=='V1'),4)
        r=[x for x in rows if x[0]=='V1']
        result=roc_metrics([x[2] for x in r],[x[3] for x in r])
        self.assertGreater(result['auc'],.9)
        auc,by=checkpoint_macro_auc(e,m,self.splitdoc,'validation',[SC],5)
        self.assertGreater(auc,.9)


if __name__=='__main__':unittest.main()
