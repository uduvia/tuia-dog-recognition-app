"""Bucles de entrenamiento y evaluación según class_2b; no depende de FastAPI."""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models
from sklearn.metrics import confusion_matrix
from lib.tp2_data import cargar_splits, loader_split, fijar_semilla
from lib.models.cnn_custom import CNNCustom

logger = logging.getLogger(__name__)

def preparar_modelo(nombre, clases, dim=512, estrategia='finetune', dropout=0.3):
    if nombre == 'resnet18_finetuned':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if dim != model.fc.in_features:
            raise ValueError('ResNet18 requiere EMBEDDING_DIM=512')
        if estrategia == 'frozen':
            for p in model.parameters(): p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features,len(clases))
    elif nombre == 'cnn_custom':
        model = CNNCustom(len(clases), dim, dropout)
    else:
        raise ValueError(nombre)
    model.classes = list(clases)
    model.class_names = list(clases)
    return model

def entrenar_epoca(model, loader, criterion, optimizer, device, frozen=False):
    model.train()
    if frozen:
        model.eval()  # evita actualización de buffers BN del backbone
        model.fc.train()
    total_loss = correctas = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*len(labels)
        correctas += (logits.argmax(1)==labels).sum().item()
        total += len(labels)
    return total_loss/total, correctas/total

@torch.no_grad()
def evaluar(model, loader, criterion, device):
    model.eval()
    ys, preds, scores = [], [], []
    loss_sum = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss_sum += criterion(logits,labels).item()*len(labels)
        score, pred = logits.softmax(1).max(1)
        ys.extend(labels.cpu().tolist()); preds.extend(pred.cpu().tolist()); scores.extend(score.cpu().tolist())
    return loss_sum/len(ys), pd.DataFrame({'real':ys,'pred':preds,'score':scores})

def metricas(df, clases):
    cm = confusion_matrix(df.real, df.pred, labels=range(len(clases)))
    tp = cm.diagonal(); fn = cm.sum(1)-tp; fp = cm.sum(0)-tp; tn = cm.sum()-tp-fn-fp
    def div(a,b): return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)
    pr, re, sp = div(tp,tp+fp),div(tp,tp+fn),div(tn,tn+fp)
    tabla = pd.DataFrame({'breed':clases,'support':cm.sum(1),'precision':pr,'recall':re,
                          'specificity':sp,'f1':div(2*pr*re,pr+re)})
    # En independiente solo promediar razas con soporte real; reportar cobertura aparte.
    presentes = tabla.support>0
    resumen = {'accuracy':float(tp.sum()/cm.sum()),**{k:float(tabla.loc[presentes,k].mean()) for k in ['precision','recall','specificity','f1']}}
    return resumen, tabla, cm

def entrenar(nombre, dataset_path, checkpoint, output_path, image_size=224, dim=512,
             epochs=15, batch_size=32, lr=1e-3, backbone_lr=1e-4, weight_decay=1e-4,
             patience=4, estrategia='finetune', dropout=0.3, seed=42):
    fijar_semilla(seed)
    df = cargar_splits(dataset_path)
    clases = sorted(df.loc[df.split=='train','breed'].unique())
    model = preparar_modelo(nombre, clases, dim, estrategia, dropout)
    model.image_size = image_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train = loader_split(dataset_path,'train',clases,image_size,batch_size,seed)
    valid = loader_split(dataset_path,'valid',clases,image_size,batch_size,seed)
    frozen = nombre=='resnet18_finetuned' and estrategia=='frozen'
    if nombre=='resnet18_finetuned' and not frozen:
        params = [{'params':[p for n,p in model.named_parameters() if not n.startswith('fc.')],'lr':backbone_lr},
                  {'params':model.fc.parameters(),'lr':lr}]
    else:
        params = filter(lambda p:p.requires_grad,model.parameters())
    optimizer = torch.optim.Adam(params,lr=lr,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=2)
    criterion = nn.CrossEntropyLoss()
    checkpoint = Path(checkpoint); checkpoint.parent.mkdir(parents=True,exist_ok=True)
    output_path = Path(output_path); output_path.mkdir(parents=True,exist_ok=True)
    hist = []; best = -1.; espera = 0
    for ep in range(1,epochs+1):
        tl,ta = entrenar_epoca(model,train,criterion,optimizer,device,frozen)
        vl,pred = evaluar(model,valid,criterion,device)
        va = float((pred.real==pred.pred).mean())
        hist.append(dict(epoch=ep,train_loss=tl,train_acc=ta,val_loss=vl,val_acc=va,lr=optimizer.param_groups[0]['lr']))
        logger.info('%s época %s: train %.4f %.3f | valid %.4f %.3f',nombre,ep,tl,ta,vl,va)
        if va>best:
            best=va; espera=0
            # Modelo completo, clase importable, mapping y tamaño persistidos. map_location permite cargar en CPU.
            torch.save(model,checkpoint)
        else: espera+=1
        scheduler.step(va)
        (output_path/f'{nombre}_{estrategia}_hist.json').write_text(json.dumps(hist,indent=2))
        if espera>=patience: break
    return hist

def evaluar_checkpoint(checkpoint,dataset_path,split='test',image_size=224,batch_size=32):
    # Cargar únicamente checkpoints propios/confiables: weights_only=False deserializa Python.
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=torch.load(checkpoint,map_location=device,weights_only=False)
    if getattr(model,'image_size',image_size)!=image_size: raise ValueError('IMAGE_SIZE incompatible')
    loader=loader_split(dataset_path,split,model.classes,image_size,batch_size)
    _,pred=evaluar(model,loader,nn.CrossEntropyLoss(),device)
    pred['path']=loader.dataset.filas.path
    return (*metricas(pred,model.classes),pred)

@torch.no_grad()
def features(model, x):
    if isinstance(model,CNNCustom):
        z=model.forward_features(x)
    else:  # ResNet18, capa anterior a fc
        z=model.conv1(x); z=model.bn1(z); z=model.relu(z); z=model.maxpool(z)
        z=model.layer1(z); z=model.layer2(z); z=model.layer3(z); z=model.layer4(z)
        z=model.avgpool(z).flatten(1)
    return nn.functional.normalize(z,p=2,dim=1)
