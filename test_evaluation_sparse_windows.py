"""Regression tests for long authentication windows with inactive users.

Run from the repository root after replacing evaluate_authentication.py:
    python -m unittest -v test_evaluation_sparse_windows.py
No raw HMOG/FETA archives, GPU, or trained checkpoints required.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gesture_evaluation_core import save_embedding_table
from gesture_pipeline_data import file_hash
from evaluate_authentication import main as evaluate_main

SCENARIO = 'feta_x_scroll'


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


class SparseWindowEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.results = self.root/'results'
        self.model_dir = self.results/SCENARIO
        (self.model_dir/'embeddings').mkdir(parents=True)
        self.splits = self.root/'splits.json'
        users = [f'{role}{i}' for role in ('V', 'E') for i in range(1, 4)]
        self.doc = {
            'splits': {'FETA': {'iPhone X': {
                'validation': ['V1', 'V2', 'V3'],
                'test': ['E1', 'E2', 'E3'],
                'train': []}}, 'HMOG': {}},
            'session_index': {'FETA': {'iPhone X': {
                u: {'scroll': ['1', '2', '3', '4']} for u in users
            }}, 'HMOG': {}},
        }
        self.splits.write_text(json.dumps(self.doc), encoding='utf-8')

    def make_embeddings(self, role, active_count=2):
        identifiers = ['V1','V2','V3'] if role == 'validation' else ['E1','E2','E3']
        embeddings=[]
        metadata=[]
        for uid_idx, user in enumerate(identifiers):
            for session in range(1, 5):
                # Every session has >=20 accepted gestures. For inactive
                # users the 24 gestures in verification sessions are split
                # across two FETA repetitions, 12 in each, so no k=20 window.
                if session<=2 or uid_idx<active_count:
                    parts = [(0, 22)]
                else:
                    parts = [(0,12), (1,12)]
                for rep, n in parts:
                    for gesture in range(n):
                        x=np.zeros(64,dtype=np.float32)
                        x[uid_idx]=1
                        embeddings.append(x)
                        start=100_000*session+10_000*rep+250*gesture
                        metadata.append({
                            'scenario': SCENARIO, 'dataset':'FETA',
                            'user_id':user, 'session_id':str(session),
                            'repetition':str(rep),
                            'start_ms':start, 'end_ms':start+90,
                        })
        path=self.model_dir/'embeddings'/f'{role}.npz'
        save_embedding_table(path, np.stack(embeddings), metadata, extra={
            'role':role, 'model_id': SCENARIO,
            'splits_sha256':file_hash(self.splits),
            'checkpoint_epoch':7,
            'normalizer_sha256':'synthetic-stats',
        })
        return path

    def evaluate(self, role, calibration=None):
        argv=[
            'evaluate_authentication.py', '--model-id',SCENARIO,
            '--role',role, '--splits',str(self.splits),
            '--results',str(self.results), '--windows','1,20',
            '--bootstrap','20',
        ]
        if calibration:
            argv+=['--calibration',str(calibration)]
        with patch.object(sys,'argv',argv), redirect_stdout(StringIO()):
            evaluate_main()
        return self.model_dir/'evaluation'/role

    def assert_sparse_metrics(self, output):
        per = read_rows(output/'per_user_metrics.csv')
        scenario = read_rows(output/'scenario_metrics.csv')
        k20=[r for r in per if int(r['k'])==20]
        self.assertEqual(len(k20),3)
        ok=[r for r in k20 if r['status']=='ok']
        skipped=[r for r in k20 if r['status']=='insufficient_windows']
        self.assertEqual(len(ok),2)
        self.assertEqual(len(skipped),1)
        self.assertEqual(skipped[0]['genuine_count'],'0')
        self.assertEqual(skipped[0]['impostor_count'],'0')
        self.assertTrue(all(float(r['auc'])>.99 for r in ok))
        s20=next(r for r in scenario if int(r['k'])==20)
        self.assertEqual(s20['status'],'ok')
        self.assertEqual(int(s20['users']),3)
        self.assertEqual(int(s20['evaluable_users']),2)
        self.assertAlmostEqual(float(s20['coverage']),2/3)
        self.assertTrue(np.isfinite(float(s20['macro_eer'])))
        self.assertTrue(np.isfinite(float(s20['macro_auc'])))
        self.assertTrue((output/f'{SCENARIO}_k20_scores.npz').exists())
        self.assertTrue((output/f'{SCENARIO}_k20_per_user_rocs.npz').exists())
        k1=[r for r in per if int(r['k'])==1]
        self.assertEqual(len(k1),3)
        self.assertTrue(all(r['status']=='ok' for r in k1))

    def test_validation_partial_k20_coverage(self):
        self.make_embeddings('validation',active_count=2)
        out=self.evaluate('validation')
        self.assert_sparse_metrics(out)
        cal=json.loads((out/'thresholds_validation.json').read_text(encoding='utf-8'))
        self.assertIsInstance(cal['thresholds'][f'{SCENARIO}/k20'],float)

    def test_test_partial_k20_coverage_with_validation_threshold(self):
        self.make_embeddings('validation',active_count=2)
        self.make_embeddings('test',active_count=2)
        val=self.evaluate('validation')
        out=self.evaluate('test',calibration=val/'thresholds_validation.json')
        self.assert_sparse_metrics(out)
        self.assertFalse((out/'thresholds_validation.json').exists())

    def test_test_auc_without_calibrated_operating_threshold(self):
        # Validation has no k=20 windows, but test does. Test EER/AUC are
        # meaningful; FAR/FRR at the prescribed validation threshold are not.
        self.make_embeddings('validation',active_count=0)
        self.make_embeddings('test',active_count=2)
        val=self.evaluate('validation')
        out=self.evaluate('test',calibration=val/'thresholds_validation.json')
        k20=next(r for r in read_rows(out/'scenario_metrics.csv') if int(r['k'])==20)
        self.assertEqual(k20['status'],'ok')
        self.assertEqual(k20['operating_threshold_status'],'unavailable')
        self.assertEqual(k20['threshold'],'')
        self.assertTrue(np.isfinite(float(k20['macro_eer'])))
        self.assertTrue(np.isfinite(float(k20['macro_auc'])))
        self.assertTrue(np.isnan(float(k20['macro_far'])))
        self.assertTrue(np.isnan(float(k20['macro_balanced_accuracy'])))

    def test_no_long_windows_reports_missing_values_without_crashing(self):
        self.make_embeddings('validation',active_count=0)
        out=self.evaluate('validation')
        rows=read_rows(out/'scenario_metrics.csv')
        s20=next(r for r in rows if int(r['k'])==20)
        self.assertEqual(s20['status'],'insufficient_windows')
        self.assertEqual(int(s20['evaluable_users']),0)
        per=read_rows(out/'per_user_metrics.csv')
        self.assertEqual(sum(r['status']=='insufficient_windows' for r in per if int(r['k'])==20),3)
        cal=json.loads((out/'thresholds_validation.json').read_text(encoding='utf-8'))
        self.assertIsNone(cal['thresholds'][f'{SCENARIO}/k20'])


if __name__=='__main__':
    unittest.main()
