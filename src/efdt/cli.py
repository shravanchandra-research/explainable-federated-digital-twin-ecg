"""Command-line entry points for the EFDT reference implementation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from .adaptation import evaluate_patient_adaptation
from .data import PTBXLH5Dataset, fold_rows, read_metadata
from .explainability import integrated_gradients, perturbation_faithfulness, attribution_stability
from .federated import train_federated
from .metrics import multilabel_metrics, tune_thresholds
from .model import CompactECGCNN, count_parameters
from .resource import analytical_resource_profile, measure_inference_latency
from .training import collect_predictions, fit_centralized, save_checkpoint, write_json

def _centralized(a):
    meta=read_metadata(a.h5); tr,va,te=fold_rows(meta)
    train=PTBXLH5Dataset(a.h5,tr); valid=PTBXLH5Dataset(a.h5,va); test=PTBXLH5Dataset(a.h5,te)
    tl=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0)
    vl=DataLoader(valid,batch_size=a.batch_size,shuffle=False,num_workers=0)
    xl=DataLoader(test,batch_size=a.batch_size,shuffle=False,num_workers=0)
    model,fit=fit_centralized(tl,vl,a.device,a.max_epochs,a.patience,a.seed)
    vy,vp=collect_predictions(model,vl,torch.device(a.device)); thresholds=tune_thresholds(vy,vp)
    ty,tp=collect_predictions(model,xl,torch.device(a.device)); metrics=multilabel_metrics(ty,tp,thresholds)
    result={"implementation":"new_reference_implementation","seed":a.seed,"train_records":len(train),
            "validation_records":len(valid),"test_records":len(test),"fit":fit,
            "thresholds_selected_on_fold_9":thresholds.tolist(),"test_metrics":metrics,
            "architecture_parameters":count_parameters(model)}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    save_checkpoint(model,out/f"centralized_seed{a.seed}.pt",result)
    write_json(result,out/f"centralized_seed{a.seed}.json")
    for ds in (train,valid,test): ds.close()
    print(json.dumps({"checkpoint":str(out/f"centralized_seed{a.seed}.pt"),"test_metrics":metrics},indent=2))

def _federated(a):
    result=train_federated(a.h5,a.output_dir,optimizer_name=a.optimizer,allocation=a.allocation,
        num_clients=a.num_clients,rounds=a.rounds,local_epochs=a.local_epochs,alpha=a.alpha,
        batch_size=a.batch_size,seed=a.seed,device=a.device)
    print(json.dumps({"test_metrics":result["test_metrics"],"client_record_counts":result["client_record_counts"]},indent=2))

def _adapt(a):
    dev=torch.device(a.device); payload=torch.load(a.checkpoint,map_location=dev,weights_only=False)
    model=CompactECGCNN().to(dev); model.load_state_dict(payload["state_dict"]); model.eval()
    thresholds=payload.get("metadata",{}).get("thresholds_selected_on_fold_9",[0.5]*5)
    result=evaluate_patient_adaptation(model,a.h5,a.output,thresholds=thresholds,
        support_records=a.support_records,adaptation_epochs=a.adaptation_epochs,
        learning_rate=a.learning_rate,device=a.device,evaluation_fold=a.fold,seed=a.seed)
    print(json.dumps({"eligible_patients":result["eligible_patients"],"mean_delta_f1":result["mean_delta_f1"]},indent=2))

def _resources(a):
    model=CompactECGCNN(); result=analytical_resource_profile(a.throughput,a.power)
    result["implemented_model_parameters"]=count_parameters(model)
    if a.measure:
        result["host_measurement"]=measure_inference_latency(model,device=a.device,warmup=a.warmup,repetitions=a.repetitions)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2))

def _explain(a):
    dev=torch.device(a.device); payload=torch.load(a.checkpoint,map_location=dev,weights_only=False)
    model=CompactECGCNN().to(dev); model.load_state_dict(payload["state_dict"]); model.eval()
    with h5py.File(a.h5,"r") as h5:
        if a.row<0 or a.row>=len(h5["X"]): raise IndexError("row is outside the HDF5 dataset")
        x=torch.from_numpy(np.asarray(h5["X"][a.row],dtype=np.float32).T.copy()).unsqueeze(0).to(dev)
    attr=integrated_gradients(model,x,a.target_class,n_steps=a.steps)
    faith=perturbation_faithfulness(model,x,a.target_class,attr,fraction=a.remove_fraction)
    stable=attribution_stability(model,x,a.target_class,n_repeats=3,n_steps=min(a.steps,32))
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); np.save(out.with_suffix(".npy"),attr.detach().cpu().numpy())
    report={"row_index":a.row,"target_class":a.target_class,"attribution_shape":list(attr.shape),
        "attribution_abs_sum":float(attr.abs().sum().item()),"perturbation_faithfulness":faith,
        "noise_stability_cosine":stable,"warning":"Attribution audits are not clinical validity evidence."}
    write_json(report,out.with_suffix(".json")); print(json.dumps(report,indent=2))

def build_parser():
    p=argparse.ArgumentParser(prog="efdt",description="EFDT PTB-XL research reference implementation")
    sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("centralized",help="Train and evaluate centralized CNN")
    c.add_argument("--h5",required=True); c.add_argument("--output-dir",default="outputs/centralized")
    c.add_argument("--device",default="cpu"); c.add_argument("--batch-size",type=int,default=64)
    c.add_argument("--max-epochs",type=int,default=100); c.add_argument("--patience",type=int,default=15)
    c.add_argument("--seed",type=int,choices=[17,42,2026],default=17); c.set_defaults(func=_centralized)
    f=sub.add_parser("federated",help="Run patient-disjoint simulated-client FL")
    f.add_argument("--h5",required=True); f.add_argument("--output-dir",default="outputs/federated")
    f.add_argument("--optimizer",choices=["fedavg","fedadam"],default="fedavg")
    f.add_argument("--allocation",choices=["iid","non-iid"],default="iid")
    f.add_argument("--num-clients",type=int,default=10); f.add_argument("--rounds",type=int,default=20)
    f.add_argument("--local-epochs",type=int,default=2); f.add_argument("--alpha",type=float,default=0.3)
    f.add_argument("--batch-size",type=int,default=64); f.add_argument("--device",default="cpu")
    f.add_argument("--seed",type=int,choices=[17,42,2026],default=17); f.set_defaults(func=_federated)
    a=sub.add_parser("adapt",help="Run patient-level support/query adaptation")
    a.add_argument("--h5",required=True); a.add_argument("--checkpoint",required=True)
    a.add_argument("--output",default="outputs/adaptation.json"); a.add_argument("--support-records",type=int,default=1)
    a.add_argument("--adaptation-epochs",type=int,default=5); a.add_argument("--learning-rate",type=float,default=1e-4)
    a.add_argument("--fold",type=int,default=10); a.add_argument("--device",default="cpu")
    a.add_argument("--seed",type=int,default=17); a.set_defaults(func=_adapt)
    r=sub.add_parser("resources",help="Compute analytical profile and optionally time host inference")
    r.add_argument("--output",default="outputs/resource_profile.json"); r.add_argument("--throughput",type=float,default=100.0)
    r.add_argument("--power",type=float,default=0.5); r.add_argument("--measure",action="store_true")
    r.add_argument("--device",default="cpu"); r.add_argument("--warmup",type=int,default=20)
    r.add_argument("--repetitions",type=int,default=100); r.set_defaults(func=_resources)
    x=sub.add_parser("explain",help="Generate IG attribution and perturbation/stability audits")
    x.add_argument("--h5",required=True); x.add_argument("--checkpoint",required=True)
    x.add_argument("--row",type=int,required=True); x.add_argument("--target-class",type=int,choices=range(5),required=True)
    x.add_argument("--steps",type=int,default=64); x.add_argument("--remove-fraction",type=float,default=0.1)
    x.add_argument("--output",default="outputs/explanation"); x.add_argument("--device",default="cpu")
    x.set_defaults(func=_explain)
    return p

def main():
    args=build_parser().parse_args(); args.func(args)

if __name__=="__main__": main()
