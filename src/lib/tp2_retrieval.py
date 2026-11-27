"""Evaluación de retrieval en memoria: misma galería para todos los modelos."""
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from lib.tp2_data import dataset_filas
from lib.services.similarity_service import votar
from lib.evaluation.metrics import ndcg_at_k

def features(model, x):
    vectors = model.embed(x) if hasattr(model, "embed") else model(x)
    return nn.functional.normalize(vectors, dim=1)


@torch.no_grad()
def extraer_matriz(model, root, filas, clases, image_size=224, batch_size=32):
    device=next(model.parameters()).device
    model.eval()
    loader=DataLoader(dataset_filas(root,filas,clases,image_size,False),batch_size=batch_size,shuffle=False)
    return np.concatenate([features(model,x.to(device)).cpu().numpy() for x,_ in loader])

def baseline(device):
    # Al sustituir fc, el forward devuelve la salida del pooling global.
    model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc=nn.Identity()
    return model.eval().to(device)

def evaluar_busqueda(gallery, g_rows, queries, q_rows, metric='cosine', k=10, threshold=-1):
    if metric not in ('cosine','l2'): raise ValueError(metric)
    if k<=0 or not len(gallery): raise ValueError('k o galería inválidos')
    clases=g_rows.breed.to_numpy(); counts=Counter(clases)
    resultados=[]
    rows=list(q_rows.itertuples(index=False))
    for start in range(0,len(queries),64):
        matriz=np.clip(queries[start:start+64]@gallery.T,-1,1)
        scores_batch=matriz if metric=='cosine' else 1/(1+np.sqrt(np.maximum(0,2-2*matriz)))
        for scores,row in zip(scores_batch,rows[start:start+64]):
            order=np.argsort(-scores,kind='stable')[:max(k,10)]
            ids=order[:10]; rel=(clases[ids]==row.breed).astype(float)
            ndcg=ndcg_at_k(rel, k=10, n_relevant=counts[row.breed])
            voto=order[:k]; pred,score=votar(clases[voto],scores[voto],threshold)
            resultados.append(dict(path=row.path,breed=row.breed,pred=pred,score=score,
                                   best_score=float(scores[order[0]]),ndcg=ndcg,precision10=float(rel.sum()/10),
                                   correct=pred==row.breed,neighbors=ids.tolist(),neighbor_scores=scores[ids].tolist(),
                                   vote_neighbors=voto.tolist(),vote_scores=scores[voto].tolist()))
    return pd.DataFrame(resultados)

def resumen_busqueda(df):
    return {'ndcg@10':df.ndcg.mean(),'precision@10':df.precision10.mean(),
            'accuracy':df.correct.mean(),'coverage':(df.pred!='unknown').mean()}

def calibrar(gallery,g_rows,queries,q_rows,metric):
    """Elige K y umbral por accuracy entre aceptadas con cobertura mínima del 90 % en validación."""
    base=evaluar_busqueda(gallery,g_rows,queries,q_rows,metric,20,-1)
    thresholds=np.unique(np.r_[(-1 if metric=='cosine' else 0),np.quantile(base.best_score,np.linspace(0,1,11))])
    filas=[]
    for k in (1,3,5,10,20):
        preds=[votar(g_rows.iloc[row.vote_neighbors[:k]].breed.tolist(),row.vote_scores[:k],-1)[0]
               for row in base.itertuples()]
        correct=np.asarray(preds)==base.breed.to_numpy()
        for t in thresholds:
            accepted=base.best_score>=t
            filas.append(dict(k=k,threshold=float(t),accuracy=float((correct&accepted).mean()),coverage=float(accepted.mean()),accuracy_accepted=float(correct[accepted].mean()),ndcg=base.ndcg.mean()))
    tabla = pd.DataFrame(filas)
    # Con razas conocidas se admite hasta un 10 % de rechazos en validación.
    elegibles = tabla[tabla.coverage >= 0.90]
    best = elegibles.sort_values(["accuracy_accepted", "coverage", "k", "threshold"],
                                 ascending=[False, False, True, True]).iloc[0]
    return tabla, best.to_dict()
