"""Evaluación de retrieval en memoria: misma galería para todos los modelos."""
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from lib.tp2_data import dataset_filas
from lib.tp2_training import features

@torch.no_grad()
def extraer_matriz(model, root, filas, clases, image_size=224, batch_size=32):
    device=next(model.parameters()).device
    model.eval()
    loader=DataLoader(dataset_filas(root,filas,clases,image_size,False),batch_size=batch_size,shuffle=False)
    return np.concatenate([features(model,x.to(device)).cpu().numpy() for x,_ in loader])

def baseline(device):
    # features() toma avgpool antes de fc; se elimina fc también para representar el extractor.
    model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc=nn.Identity()
    return model.eval().to(device)

def votar(razas, scores, threshold):
    if not len(scores): return 'unknown',0.
    # Contrato de test_similarity.py: suma de scores y media de la raza ganadora.
    votos = Counter()
    cantidades = Counter()
    for raza, score in zip(razas, scores):
        votos[raza] += float(score)
        cantidades[raza] += 1
    raza = max(votos, key=votos.get)  # empate: primer vecino de las razas empatadas
    score = votos[raza] / cantidades[raza]
    return (raza if max(scores) >= threshold else 'unknown'), float(score)

def evaluar_busqueda(gallery, g_rows, queries, q_rows, metric='cosine', k=10, threshold=-1):
    if metric not in ('cosine','l2'): raise ValueError(metric)
    if k<=0 or not len(gallery): raise ValueError('k o galería inválidos')
    clases=g_rows.breed.to_numpy(); counts=Counter(clases)
    resultados=[]
    descuentos=1/np.log2(np.arange(2,12))
    rows=list(q_rows.itertuples(index=False))
    for start in range(0,len(queries),64):
        matriz=np.clip(queries[start:start+64]@gallery.T,-1,1)
        scores_batch=matriz if metric=='cosine' else 1/(1+np.sqrt(np.maximum(0,2-2*matriz)))
        for scores,row in zip(scores_batch,rows[start:start+64]):
            order=np.argsort(-scores,kind='stable')[:max(k,10)]
            ids=order[:10]; rel=(clases[ids]==row.breed).astype(float)
            ideal=descuentos[:min(10,counts[row.breed])].sum()
            ndcg=float((rel*descuentos[:len(ids)]).sum()/ideal) if ideal else 0.
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
    """Validación de razas conocidas: maximizar aciertos contando rechazo como error.
    Sin negativos de validación no es calibración de open-set. Se reporta cobertura.
    """
    base=evaluar_busqueda(gallery,g_rows,queries,q_rows,metric,20,-1)
    thresholds=np.unique(np.r_[(-1 if metric=='cosine' else 0),np.quantile(base.best_score,np.linspace(0,1,11))])
    filas=[]
    for k in (1,3,5,10,20):
        preds=[votar(g_rows.iloc[row.vote_neighbors[:k]].breed.tolist(),row.vote_scores[:k],-1)[0]
               for row in base.itertuples()]
        correct=np.asarray(preds)==base.breed.to_numpy()
        for t in thresholds:
            accepted=base.best_score>=t
            filas.append(dict(k=k,threshold=float(t),accuracy=float((correct&accepted).mean()),coverage=float(accepted.mean()),ndcg=base.ndcg.mean()))
    tabla=pd.DataFrame(filas).sort_values(['accuracy','coverage','threshold','k'],ascending=[False,False,False,True])
    return tabla,tabla.iloc[0].to_dict()
