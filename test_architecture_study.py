"""No raw datasets needed: model/gradient/safety and synthetic full-pipeline tests."""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from gesture_architectures import ARCHITECTURES,make_encoder,prototypical_cross_session_loss
from gesture_model import GestureEncoder
from compare_architecture_study import summarize

HERE=Path(__file__).resolve().parent


def batch(b=4,n=13):
    torch.manual_seed(112)
    data={}
    for kind,dim in [('touch',6),('accel',5),('gyro',5)]:
        lengths=[n,n-3,n-5,n-2][:b]
        x=torch.randn(b,n,dim)*.1
        mask=torch.arange(n)[None,:]<torch.tensor(lengths)[:,None]
        # Monotonic real elapsed times, unequal sampling rates.
        for i,l in enumerate(lengths):
            x[i,:,4 if kind=='touch' else 3]=0
            x[i,:l,4 if kind=='touch' else 3]=torch.linspace(0,.25+.1*i,l)
        x[~mask]=0
        data[kind]=x
        data[kind+'_mask']=mask
    data['labels']=torch.tensor([0,0,1,1][:b])
    data['meta']=[{'session_id':s} for s in ('1','2','1','2')[:b]]
    return data


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_all_architectures_forward_backward(self):
        for name in ARCHITECTURES:
            with self.subTest(name=name):
                model=make_encoder({'name':name,'width':12,'embedding_dim':16,
                                    'grid_points':12,'attention_heads':4})
                x=batch()
                emb=model(x)
                self.assertEqual(tuple(emb.shape),(4,16))
                self.assertTrue(torch.isfinite(emb).all())
                self.assertTrue(torch.allclose(emb.norm(dim=1),torch.ones(4),atol=1e-5))
                loss=(emb.square().sum()+emb[:,0].sum())
                loss.backward()
                self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_padding_does_not_change_embedding(self):
        for name in ARCHITECTURES:
            with self.subTest(name=name):
                model=make_encoder({'name':name,'width':12,'embedding_dim':16,
                                    'grid_points':12,'attention_heads':4}).eval()
                original=batch()
                padded={}
                for kind in ('touch','accel','gyro'):
                    x=original[kind];mask=original[kind+'_mask']
                    padded[kind]=torch.cat([x,torch.zeros(4,8,x.shape[-1])],dim=1)
                    padded[kind+'_mask']=torch.cat([mask,torch.zeros(4,8,dtype=torch.bool)],dim=1)
                with torch.no_grad():
                    a=model(original);b=model(padded)
                self.assertTrue(torch.allclose(a,b,atol=1e-5,rtol=1e-5),name)

    def test_checkpoint_roundtrip(self):
        arch={'name':'crossmodal','width':12,'embedding_dim':16,
              'grid_points':12,'attention_heads':4}
        model=make_encoder(arch).eval();x=batch()
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'model_best.pt'
            torch.save({'model':model.state_dict(),'architecture':arch},f)
            loaded=torch.load(f,weights_only=True)
            restored=make_encoder(loaded['architecture']).eval()
            restored.load_state_dict(loaded['model'])
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(x),restored(x)))

    def test_original_baseline_state_dict_is_compatible(self):
        original=GestureEncoder(width=12,embedding_dim=16)
        new=make_encoder({'name':'baseline','width':12,'embedding_dim':16})
        new.load_state_dict(original.state_dict())
        with torch.no_grad():
            original.eval();new.eval()
            self.assertTrue(torch.allclose(original(batch()),new(batch())))
        legacy=make_encoder({'width':12,'embedding_dim':16})
        legacy.load_state_dict(original.state_dict())

    def test_prototypical_loss_gradient(self):
        torch.manual_seed(23)
        vectors=torch.nn.functional.normalize(torch.randn(8,16),dim=1).requires_grad_()
        labels=torch.tensor([0,0,0,0,1,1,1,1])
        metadata=[{'session_id':s} for s in ('1','1','2','2','1','2','1','2')]
        loss=prototypical_cross_session_loss(vectors,labels,metadata,temperature=.05)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(vectors.grad).all())

    def test_temporal_alignment_time_sensitive(self):
        # Two sequences with the same positions but different physical times
        # must not be reduced to a naive sample-index pairing.
        from gesture_architectures import _resample_on_grid
        x=torch.tensor([[[0.],[10.],[20.]]])
        mask=torch.tensor([[True,True,True]])
        query=torch.tensor([[0.,.5,1.]])
        early=torch.tensor([[0.,.25,1.]])
        linear=torch.tensor([[0.,.5,1.]])
        a=_resample_on_grid(x,early,mask,query)
        b=_resample_on_grid(x,linear,mask,query)
        self.assertAlmostEqual(a[0,1,0].item(),13.333333,places=4)
        self.assertAlmostEqual(b[0,1,0].item(),10.,places=4)

    def test_summary_seed_aggregation(self):
        runs=[{'architecture':'tcn','objective':'supcon','method':'centroid',
               'macro_auc':x,'macro_eer':y}
              for x,y in [(0.7,0.3),(0.8,0.2)]]
        one=summarize(runs)[0]
        self.assertAlmostEqual(one['macro_auc_mean'],.75)
        self.assertAlmostEqual(one['macro_eer_mean'],.25)
        self.assertEqual(one['seeds'],2)


class SyntheticPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_train_extract_evaluate_and_overwrite_guard(self):
        # Reuses the existing synthetic fixture supplied by the v1 pipeline.
        from test_gesture_embedding_pipeline import GesturePipelineTests
        fx=GesturePipelineTests()
        fx.setUp()
        try:
            root=fx.root/'new_arch_study'
            common=[sys.executable,'-u',str(HERE/'train_architecture_experiment.py'),
                '--scenario','feta_7_swipe','--architecture','attention_pool',
                '--output-root',str(root),'--train-cache',str(fx.paths['train']),
                '--validation-cache',str(fx.paths['validation']),
                '--stats',str(fx.stats),'--splits',str(fx.splits),
                '--width','12','--embedding-dim','16','--epochs','1',
                '--steps-per-epoch','2','--p','2','--k','4','--validate-every','1',
                '--val-gestures-per-session','4','--val-impostors-per-source','4',
                '--log-every','1','--device','cpu']
            proc=subprocess.run(common,capture_output=True,text=True,timeout=120)
            self.assertEqual(proc.returncode,0,proc.stdout+'\n'+proc.stderr)
            best=root/'feta_7_swipe'/'model_best.pt'
            self.assertTrue(best.is_file())
            ck=torch.load(best,weights_only=True)
            self.assertEqual(ck['architecture']['name'],'attention_pool')
            again=subprocess.run(common,capture_output=True,text=True,timeout=120)
            self.assertNotEqual(again.returncode,0)
            self.assertIn('FileExistsError',again.stderr)
            extract=[sys.executable,'-u',str(HERE/'extract_architecture_embeddings.py'),
                '--model-id','feta_7_swipe','--role','validation',
                '--cache',str(fx.paths['validation']),
                '--stats',str(fx.stats),'--splits',str(fx.splits),
                '--results',str(root),'--device','cpu']
            out=subprocess.run(extract,capture_output=True,text=True,timeout=120)
            self.assertEqual(out.returncode,0,out.stdout+'\n'+out.stderr)
            result=root/'feta_7_swipe'/'embeddings'/'validation.npz'
            self.assertTrue(result.is_file())
            with np.load(result,allow_pickle=False) as z:
                self.assertEqual(z['embeddings'].shape,(48,16))
                self.assertEqual(json.loads(str(z['extra_json']))['architecture']['name'],'attention_pool')
            repeated=subprocess.run(extract,capture_output=True,text=True,timeout=120)
            self.assertNotEqual(repeated.returncode,0)
            self.assertIn('FileExistsError',repeated.stderr)
            # Existing evaluator (v2) consumes the newly extracted NPZ unchanged.
            evaluator=Path(__file__).parent/'evaluate_authentication.py'
            if evaluator.is_file():
                ev=[sys.executable,'-u',str(HERE/'evaluate_architecture_run.py'),
                    '--model-id','feta_7_swipe',
                    '--splits',str(fx.splits),'--results',str(root),
                    '--methods','centroid,top5','--windows','1,2,10',
                    '--workers','1','--bootstrap','20','--impostor-per-source','4']
                got=subprocess.run(ev,capture_output=True,text=True,timeout=120)
                self.assertEqual(got.returncode,0,got.stdout+'\n'+got.stderr)
                for method in ('centroid','top5'):
                    sc=root/'feta_7_swipe'/'evaluation'/'method_study'/'validation'/method/'scenario_metrics.csv'
                    self.assertTrue(sc.is_file())
                # Never silently overwrite the evaluation files when rerun.
                duplicate=subprocess.run(ev,capture_output=True,text=True,timeout=120)
                self.assertNotEqual(duplicate.returncode,0)
                self.assertIn('FileExistsError',duplicate.stderr)
        finally:fx.tearDown()


if __name__=='__main__':unittest.main()
