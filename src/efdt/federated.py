"""Patient-preserving simulated-client FL with FedAvg and FedAdam."""
from __future__ import annotations
import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import PTBXLH5Dataset,fold_rows,read_metadata
from .metrics import multilabel_metrics,tune_thresholds
from .model import CompactECGCNN
from .training import collect_predictions,save_checkpoint,set_seed,train_epochs,write_json

def partition_patients(row_indices,patient_ids,labels,num_clients=10,mode="iid",alpha=0.3,seed=17):
    """Assign complete patients to disjoint clients with IID or label-signature Dirichlet allocation."""
    if num_clients<2: raise ValueError("num_clients must be at least 2")
    if mode not in {"iid","non-iid"}: raise ValueError("mode must be iid or non-iid")
    if mode=="non-iid" and alpha<=0: raise ValueError("Dirichlet alpha must be positive")
    rng=np.random.default_rng(seed); rows=np.asarray(row_indices,dtype=np.int64); pids=np.asarray(patient_ids); y=np.asarray(labels).astype(int)
    patient_rows=defaultdict(list)
    for r in rows: patient_rows[int(pids[r])].append(int(r))
    patients=np.asarray(list(patient_rows),dtype=np.int64)
    if len(patients)<num_clients: raise ValueError("Fewer unique patients than clients")
    groups=[[] for _ in range(num_clients)]
    if mode=="iid":
        for i,chunk in enumerate(np.array_split(rng.permutation(patients),num_clients)): groups[i].extend(int(p) for p in chunk)
    else:
        strata=defaultdict(list)
        for p in patients:
            pr=patient_rows[int(p)]; sig=tuple((y[pr].max(axis=0)>0).astype(int).tolist()); strata[sig].append(int(p))
        for members in strata.values():
            members=rng.permutation(np.asarray(members)); counts=rng.multinomial(len(members),rng.dirichlet(np.full(num_clients,alpha)))
            offset=0
            for i,n in enumerate(counts): groups[i].extend(int(p) for p in members[offset:offset+n]); offset+=int(n)
        for empty in [i for i,g in enumerate(groups) if not g]:
            donor=max(range(num_clients),key=lambda i:len(groups[i]))
            if len(groups[donor])<=1: raise RuntimeError("Cannot populate clients; reduce client count or use IID")
            groups[empty].append(groups[donor].pop())
    clients=[np.asarray([r for p in g for r in patient_rows[p]],dtype=np.int64) for g in groups]
    merged=np.concatenate(clients)
    if len(merged)!=len(rows) or len(np.unique(merged))!=len(rows) or set(merged)!=set(rows): raise RuntimeError("Invalid row coverage")
    owners={}
    for i,g in enumerate(groups):
        for p in g:
            if p in owners: raise RuntimeError("Patient assigned to multiple clients")
            owners[p]=i
    return clients

def _weighted_average(states,weights):
    total=float(sum(weights))
    if total<=0: raise ValueError("Client weights must be positive")
    out={}
    for key in states[0]:
        out[key]=sum((s[key].detach().cpu()*(w/total) for s,w in zip(states,weights)),torch.zeros_like(states[0][key].detach().cpu()))
    return out

def train_federated(h5_path: str|Path,output_dir: str|Path,optimizer_name="fedavg",allocation="iid",
                    num_clients=10,rounds=20,local_epochs=2,alpha=0.3,batch_size=64,
                    learning_rate=1e-3,server_learning_rate=0.01,beta1=0.9,beta2=0.99,
                    tau=1e-3,seed=17,device="cpu"):
    optimizer_name=optimizer_name.lower()
    if optimizer_name not in {"fedavg","fedadam"}: raise ValueError("optimizer_name must be fedavg or fedadam")
    if rounds<1 or local_epochs<1: raise ValueError("rounds and local_epochs must be positive")
    set_seed(seed); dev=torch.device(device); meta=read_metadata(h5_path)
    tr,va,te=fold_rows(meta)
    clients=partition_patients(tr,meta["patient_id"],meta["Y"],num_clients,allocation,alpha,seed)
    global_model=CompactECGCNN().to(dev); state=copy.deepcopy(global_model.state_dict())
    m={k:torch.zeros_like(v,device="cpu") for k,v in state.items()}; v={k:torch.zeros_like(t,device="cpu") for k,t in state.items()}
    history=[]
    for rnd in range(1,rounds+1):
        local_states=[]; weights=[]; losses=[]
        for rows in clients:
            local=CompactECGCNN().to(dev); local.load_state_dict(state)
            ds=PTBXLH5Dataset(h5_path,rows); loader=DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0)
            losses.extend(train_epochs(local,loader,dev,local_epochs,learning_rate))
            local_states.append({k:t.detach().cpu().clone() for k,t in local.state_dict().items()})
            weights.append(len(rows)); ds.close()
        if optimizer_name=="fedavg": state=_weighted_average(local_states,weights)
        else:
            delta=_weighted_average([{k:local_states[i][k]-state[k].detach().cpu() for k in state} for i in range(len(clients))],weights)
            for key in state:
                m[key]=beta1*m[key]+(1-beta1)*delta[key]; v[key]=beta2*v[key]+(1-beta2)*torch.square(delta[key])
                mh=m[key]/(1-beta1**rnd); vh=v[key]/(1-beta2**rnd)
                state[key]=state[key].detach().cpu()+server_learning_rate*mh/(torch.sqrt(vh)+tau)
        global_model.load_state_dict(state)
        history.append({"round":rnd,"mean_local_bce":float(np.mean(losses)),"client_samples":weights})
    vds,tds=PTBXLH5Dataset(h5_path,va),PTBXLH5Dataset(h5_path,te)
    vl=DataLoader(vds,batch_size=batch_size,shuffle=False); tl=DataLoader(tds,batch_size=batch_size,shuffle=False)
    vy,vp=collect_predictions(global_model,vl,dev); thresholds=tune_thresholds(vy,vp)
    ty,tp=collect_predictions(global_model,tl,dev); metrics=multilabel_metrics(ty,tp,thresholds)
    result={"implementation":"new_reference_implementation","optimizer":optimizer_name,"allocation":allocation,
            "num_clients":num_clients,"rounds":rounds,"local_epochs":local_epochs,"dirichlet_alpha":alpha,
            "batch_size":batch_size,"learning_rate":learning_rate,"server_learning_rate":server_learning_rate,
            "beta1":beta1,"beta2":beta2,"tau":tau,"seed":seed,
            "client_patient_counts":[len(np.unique(meta["patient_id"][r])) for r in clients],
            "client_record_counts":[len(r) for r in clients],"thresholds_selected_on_fold_9":thresholds.tolist(),
            "test_fold":10,"history":history,"test_metrics":metrics}
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    save_checkpoint(global_model,out/f"{optimizer_name}_{allocation}_seed{seed}.pt",result)
    write_json(result,out/f"{optimizer_name}_{allocation}_seed{seed}.json")
    vds.close(); tds.close(); return result
