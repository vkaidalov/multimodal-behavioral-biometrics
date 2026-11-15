"""Method comparison regression tests; CPU, synthetic data, no raw datasets."""
from __future__ import annotations
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gesture_enrollment_methods import PreparedScenario, nearest_k_score
from gesture_evaluation_core import (save_embedding_table,load_embedding_table,
    biometric_scores_for_scenario,build_role_scenarios)
from gesture_pipeline_data import file_hash

HERE=Path(__file__).resolve().parent
SC='feta_x_scroll'


def make_synthetic(root, role='validation', model_id=SC):
    out=Path(root)/'results'/model_id/'embeddings'
    out.mkdir(parents=True,exist_ok=True)
    names=['A','B','C','D'] if role=='validation' else ['X','Y','Z','W']
    all_names=['A','B','C','D','X','Y','Z','W']
    doc={'splits': {'FETA': {'iPhone X': {'validation':['A','B','C','D'],
          'test':['X','Y','Z','W'], 'train': []}}, 'HMOG': {}},
         'session_index': {'FETA': {'iPhone X': {
              u:{'scroll':['s1','s2','s3','s4']} for u in all_names}}, 'HMOG': {}}}
    splits=Path(root)/'splits.json';splits.write_text(json.dumps(doc),encoding='utf8')
    rng=np.random.default_rng(13)
    embedding=[];metadata=[]
    for j,uid in enumerate(names):
        for ses in range(1,5):
            for rep,n in [(0,23),(1,8)]:
                for i in range(n):
                    a=np.zeros(64,dtype=np.float32)
                    a[j]=0.8;a[(j+1)%4]=0.14
                    a+=rng.normal(0,0.07,size=64).astype('float32')
                    a/=np.linalg.norm(a)
                    embedding.append(a)
                    start=1e6*ses+2e5*rep+900*i
                    metadata.append({'scenario':SC,'dataset':'FETA', 'user_id':uid,
                       'session_id':f's{ses}','repetition':str(rep),
                       'start_ms':start,'end_ms':start+180})
    save_embedding_table(out/f'{role}.npz',np.array(embedding),metadata,
        {'role':role,'model_id':model_id,'splits_sha256':file_hash(splits),
         'normalizer_sha256':'test-stats','checkpoint_epoch':4})
    return splits,out/f'{role}.npz'


class EnrollmentMethodsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.splits,self.embpath=make_synthetic(self.root)
        self.emb,self.meta,self.extra=load_embedding_table(self.embpath)
        self.doc=json.loads(self.splits.read_text())
        self.users=build_role_scenarios(self.emb,self.meta,self.doc,'validation')[SC]

    def test_nearest_five_scores_normalized(self):
        probes=np.array([[2.,0.,0.],[0.,2.,0.],[-2.,0.,0.]])
        templates=np.array([[1.,0.,0.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.]])
        ans=nearest_k_score(probes,templates,k=5,chunk_size=1)
        np.testing.assert_allclose(ans,[.2,0.,-.2],atol=1e-12)

    def test_centroid_scores_match_legacy_sampling_and_values(self):
        prepared=PreparedScenario(self.emb,self.meta,self.users,self.doc,SC,methods=('centroid','top5'))
        for k in [1,2,5,20]:
            old=biometric_scores_for_scenario(self.emb,self.meta,self.users,self.doc,SC,k,
                                                 impostor_per_source=10,seed=20261112)
            new=prepared.score_records(k,'centroid',10,20261112)
            self.assertEqual(len(old),len(new))
            for x,y in zip(old,new):
                self.assertEqual(x[:3],y[:3])
                self.assertEqual(x[4:],y[4:])
                self.assertAlmostEqual(x[3],y[3],places=9)

    def test_both_methods_same_sampling_and_window_metadata(self):
        prepared=PreparedScenario(self.emb,self.meta,self.users,self.doc,SC)
        for k in [1,4,15,20,25]:
            a=prepared.score_records(k,'centroid',impostor_per_source=10)
            b=prepared.score_records(k,'top5',impostor_per_source=10)
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):
                self.assertEqual(x[:3]+x[4:], y[:3]+y[4:])
            # No 25-gesture window can cross the 23-gesture repetition.
            if k==25:self.assertEqual(a,[])

    def test_top5_window_score_equals_average_individual_scores(self):
        prepared=PreparedScenario(self.emb,self.meta,self.users,self.doc,SC)
        single=prepared.score_records(1,'top5',0)
        pair=prepared.score_records(2,'top5',0)
        # Genuine windows per target: sorted sessions, then FETA repetition.
        u=self.users.keys().__iter__().__next__()
        one=[r for r in single if r[0]==u and r[1]==u]
        two=[r for r in pair if r[0]==u and r[1]==u]
        for r in two:
            group=[x for x in one if x[6:8]==r[6:8] and
                   x[4]>=r[4] and x[5]<=r[5]]
            self.assertEqual(len(group),2)
            self.assertAlmostEqual(r[3],(group[0][3]+group[1][3])/2,places=10)

    def test_method_study_cli_validation_and_test_calibration(self):
        cmd=[sys.executable,str(HERE/'evaluate_authentication.py'),
             '--model-id',SC,'--splits',str(self.splits),
             '--results',str(self.root/'results'),
             '--methods','centroid,top5','--windows','1,5,20,25',
             '--bootstrap','20','--workers','1','--impostor-per-source','10']
        val=subprocess.run(cmd+['--role','validation'],cwd=HERE,text=True,capture_output=True,timeout=80)
        self.assertEqual(val.returncode,0,val.stderr+'\n'+val.stdout)
        output=self.root/'results'/SC/'evaluation'/'method_study'/'validation'
        self.assertTrue((output/'mean_scenario_comparison.csv').exists())
        for method in ('centroid','top5'):
            with (output/method/'scenario_metrics.csv').open(newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([int(r['k']) for r in rows],[1,5,20,25])
            self.assertEqual(rows[-1]['status'],'insufficient_windows')
            self.assertEqual(float(rows[-1]['coverage']),0.0)
            self.assertTrue(np.isfinite(float(rows[0]['macro_auc'])))
            with np.load(output/method/f'{SC}_k01_scores.npz',allow_pickle=False) as z:
                cols={key:z[key].copy() for key in z.files}
            if method=='centroid':ref=cols
            else:
                for field in ('target_user','source_user','label','session_id','repetition',
                              'start_ms','end_ms','collection_latency_s'):
                    np.testing.assert_array_equal(cols[field],ref[field])
        # Test threshold MUST be sourced from method-specific validation file.
        make_synthetic(self.root,'test')
        test=subprocess.run(cmd+['--role','test','--calibration-dir',str(output)],
                            cwd=HERE,text=True,capture_output=True,timeout=80)
        self.assertEqual(test.returncode,0,test.stderr+'\n'+test.stdout)
        outtest=self.root/'results'/SC/'evaluation'/'method_study'/'test'
        for method in ('centroid','top5'):
            self.assertTrue((outtest/method/'scenario_metrics.csv').exists())
            self.assertFalse((outtest/method/'thresholds_validation.json').exists())
        # Wrong method-specific calibration must be rejected.
        with (output/'centroid'/'thresholds_validation.json').open() as f: doc=json.load(f)
        doc['enrollment_method']='top5'
        wrong=self.root/'wrong.json';wrong.write_text(json.dumps(doc))
        bad=subprocess.run(cmd+['--methods','centroid','--role','test',
                             '--calibration',str(wrong)],cwd=HERE,text=True,capture_output=True,timeout=80)
        self.assertNotEqual(bad.returncode,0)


if __name__=='__main__':unittest.main()
