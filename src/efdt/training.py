"""Centralized training utilities and evaluation helpers."""
from __future__ import annotations
import copy, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .metrics import multilabel_metrics, tune_thresholds
from .model import CompactECGCNN

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval(); ys, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)); ys.append(y.cpu().numpy()); probs.append(torch.sigmoid(logits).cpu().numpy())
    if not ys: raise ValueError("The evaluation DataLoader is empty")
    return np.concatenate(ys), np.concatenate(probs)

def train_epochs(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4):
    model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(); losses=[]
    for _ in range(int(epochs)):
        model.train(); total=count=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True); loss=criterion(model(x),y)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss")
            loss.backward(); optimizer.step(); total+=float(loss.item())*len(x); count+=len(x)
        if not count: raise ValueError("The training DataLoader is empty")
        losses.append(total/count)
    return losses

def fit_centralized(train_loader: DataLoader, validation_loader: DataLoader, device="cpu",
                    max_epochs: int=100, patience: int=15, seed: int=17,
                    learning_rate: float=1e-3, weight_decay: float=1e-4):
    set_seed(seed); device=torch.device(device); model=CompactECGCNN().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=5)
    criterion=nn.BCEWithLogitsLoss(); best=float("inf"); best_state=None; best_epoch=stale=0; history=[]
    for epoch in range(1,int(max_epochs)+1):
        model.train(); total=count=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True); loss=criterion(model(x),y)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss")
            loss.backward(); optimizer.step(); total+=float(loss.item())*len(x); count+=len(x)
        if not count: raise ValueError("Training split is empty")
        model.eval(); vloss=vcount=0
        with torch.no_grad():
            for x,y in validation_loader:
                x,y=x.to(device),y.to(device); vloss+=float(criterion(model(x),y).item())*len(x); vcount+=len(x)
        if not vcount: raise ValueError("Validation split is empty")
        vloss/=vcount; scheduler.step(vloss)
        history.append({"epoch":epoch,"train_bce":total/count,"validation_bce":vloss,
                        "learning_rate":float(optimizer.param_groups[0]["lr"])})
        if vloss<best: best=vloss; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
        else: stale+=1
        if stale>=patience: break
    if best_state is None: raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    return model,{"best_epoch":best_epoch,"best_validation_bce":best,"epochs_run":len(history),"history":history}

def evaluate_with_validation_thresholds(model, validation_loader, test_loader, device="cpu"):
    device=torch.device(device); vy,vp=collect_predictions(model,validation_loader,device)
    thresholds=tune_thresholds(vy,vp); ty,tp=collect_predictions(model,test_loader,device)
    return multilabel_metrics(ty,tp,thresholds),thresholds

def save_checkpoint(model: nn.Module,path: str|Path,metadata: dict|None=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"metadata":metadata or {}},path)

def load_checkpoint(path: str|Path,device="cpu"):
    payload=torch.load(path,map_location=device,weights_only=False); model=CompactECGCNN().to(device)
    model.load_state_dict(payload["state_dict"]); model.eval(); return model,payload.get("metadata",{})

def write_json(data: dict,path: str|Path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=True)+"\n",encoding="utf-8")
