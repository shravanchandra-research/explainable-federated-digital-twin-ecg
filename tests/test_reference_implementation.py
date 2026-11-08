import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from efdt.data import PTBXLH5Dataset, fold_rows, read_metadata
from efdt.explainability import integrated_gradients, perturbation_faithfulness
from efdt.federated import partition_patients
from efdt.metrics import multilabel_metrics, tune_thresholds
from efdt.model import CompactECGCNN, count_parameters
from efdt.resource import analytical_resource_profile
from efdt.twin import PatientTwinState, replay_patient_observation

class ReferenceImplementationTests(unittest.TestCase):
    def test_model_shape_and_parameter_count(self):
        model=CompactECGCNN()
        self.assertEqual(count_parameters(model),14581)
        self.assertEqual(tuple(model(torch.zeros(2,12,1000)).shape),(2,5))

    def test_hdf5_contract_and_fold_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"toy.h5"
            with h5py.File(p,"w") as h:
                h.create_dataset("X",data=np.zeros((4,1000,12),dtype=np.float32))
                h.create_dataset("Y",data=np.zeros((4,5),dtype=np.int8))
                h.create_dataset("ecg_id",data=np.arange(4))
                h.create_dataset("patient_id",data=np.array([11,12,13,14]))
                h.create_dataset("fold",data=np.array([1,9,10,0]))
            ds=PTBXLH5Dataset(p,[0,2]); x,y=ds[0]
            self.assertEqual(tuple(x.shape),(12,1000)); self.assertEqual(tuple(y.shape),(5,)); ds.close()
            tr,va,te=fold_rows(read_metadata(p))
            self.assertEqual(tr.tolist(),[0]); self.assertEqual(va.tolist(),[1]); self.assertEqual(te.tolist(),[2])

    def test_patient_disjoint_iid_and_dirichlet(self):
        pids=np.repeat(np.arange(20),2); rows=np.arange(40); labels=np.zeros((40,5),dtype=np.int8)
        labels[::4,0]=1; labels[1::4,1]=1
        for mode in ("iid","non-iid"):
            clients=partition_patients(rows,pids,labels,num_clients=4,mode=mode,alpha=.3,seed=17)
            self.assertEqual(sorted(np.concatenate(clients).tolist()),rows.tolist())
            owners={}
            for i,cr in enumerate(clients):
                for pid in np.unique(pids[cr]):
                    self.assertNotIn(int(pid),owners); owners[int(pid)]=i
            self.assertEqual(len(owners),20)

    def test_multilabel_metrics_and_threshold_selection(self):
        y=np.array([[0,1],[1,0],[1,1],[0,0]])
        p=np.array([[.1,.9],[.8,.2],[.7,.8],[.2,.1]])
        m=multilabel_metrics(y,p,tune_thresholds(y,p))
        self.assertAlmostEqual(m["macro_f1"],1.0); self.assertAlmostEqual(m["macro_auroc"],1.0)

    def test_integrated_gradients_and_perturbation(self):
        class SumModel(nn.Module):
            def forward(self,x):
                s=x.sum(dim=(1,2)); return torch.stack([s,-s,s*.5,s*.2,-s*.3],dim=1)
        model=SumModel(); x=torch.ones(1,2,10)
        attr=integrated_gradients(model,x,0,n_steps=8)
        self.assertEqual(tuple(attr.shape),tuple(x.shape))
        self.assertAlmostEqual(float(attr.sum()),float(model(x)[0,0]),places=4)
        self.assertGreater(perturbation_faithfulness(model,x,0,attr,.5)["probability_drop"],0)

    def test_resource_equations(self):
        r=analytical_resource_profile(100,.5)
        self.assertEqual(r["trainable_parameters"],14581)
        self.assertEqual(r["macs_per_ecg"],1309760)
        self.assertAlmostEqual(r["scenario_latency_ms"],13.0976,places=4)
        self.assertAlmostEqual(r["scenario_energy_mj"],6.5488,places=4)

    def test_retrospective_patient_state_replay(self):
        model=CompactECGCNN(); state=PatientTwinState(patient_id=42)
        updated=replay_patient_observation(model,state,torch.zeros(12,1000),1001,"test-v0")
        self.assertEqual(updated.observation_index,1); self.assertEqual(updated.latest_ecg_id,1001)
        self.assertEqual(len(updated.embedding),64); self.assertEqual(len(updated.class_probabilities),5)
        self.assertEqual(updated.signal_quality["finite_fraction"],1.0)

if __name__=="__main__": unittest.main()
