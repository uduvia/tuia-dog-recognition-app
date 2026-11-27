"""Validación de metadata existente y exportación sin mover archivos originales."""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

COLS = ['path','breed','source_url','author','license','license_version','kind']

def revisar_independiente(data, folder, classes):
    data, folder = Path(data).resolve(), Path(folder).resolve()
    if not (folder/'metadata.csv').is_file():
        return pd.DataFrame(columns=COLS), pd.DataFrame([{'alerta':'Falta metadata.csv'}])
    meta = pd.read_csv(folder/'metadata.csv',dtype=str,keep_default_na=False)
    label_column = 'carpeta' if 'carpeta' in meta.columns else 'raza'
    required = {'seleccionada',label_column,'ruta_local','pagina_original','licencia'}
    if not required <= set(meta.columns):
        raise ValueError(f'Faltan columnas: {required-set(meta.columns)}')
    rows, alerts = [], []
    seen = set()
    for r in meta.to_dict('records'):
        if r['seleccionada'].strip().upper() not in ('SI','SÍ','TRUE','1'): continue
        source_label = r[label_column].strip()
        breed = 'Yorkie' if source_label == 'Yorkshire Terrier' and 'Yorkie' in classes else source_label
        filename = Path(r['ruta_local'].replace(chr(92), '/')).name
        image_root = folder / 'imagenes' if (folder / 'imagenes').is_dir() else folder
        p = (image_root / source_label / filename).resolve()
        if not p.is_relative_to(folder) or not p.is_relative_to(data):
            alerts.append({'path':str(p),'alerta':'Ruta inválida'}); continue
        if breed not in classes:
            seen.add(p)
            alerts.append({'path':str(p),'alerta':f'Raza fuera del clasificador: {source_label}'}); continue
        if p in seen:
            alerts.append({'path':str(p),'alerta':'Ruta repetida'}); continue
        if not p.is_file() or cv2.imread(str(p)) is None:
            alerts.append({'path':str(p),'alerta':'Imagen ausente o ilegible'}); continue
        if not r['pagina_original'] or not r['licencia']:
            alerts.append({'path':str(p),'alerta':'Falta fuente o licencia'}); continue
        if not r.get('autor'):
            alerts.append({'path':str(p),'alerta':'Autor sin registrar: verificar atribución original'})
        seen.add(p)
        rows.append(dict(path=p.relative_to(data).as_posix(),breed=breed,source_url=r['pagina_original'],author=r.get('autor',''),license=r['licencia'],license_version=r.get('version_licencia',''),kind='clean'))
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'} and p.resolve() not in seen:
            alerts.append({'path':str(p),'alerta':'Archivo no incorporado a selección válida'})
    return pd.DataFrame(rows,columns=COLS),pd.DataFrame(alerts,columns=['path','alerta'])

def dhash(path):
    g=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if g is None: raise ValueError(f'Imagen ilegible: {path}')
    return dhash_gray(g)


def dhash_gray(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int.from_bytes(np.packbits(bits.ravel()).tobytes(), 'big')


def filtrar_independiente(data, main, ind, hash_dist=4):
    data=Path(data).resolve()
    # Se comprueba contra los splits finales; las imágenes independientes no eliminan train.
    refs=[(str(r.path),int(r.dhash,16)) for r in main.itertuples() if isinstance(r.dhash,str) and r.dhash]
    # dHash coincide necesariamente para copias de píxeles idénticos; no garantiza detectar todo casi-duplicado.
    keep, pairs=[],[]
    for r in ind.itertuples():
        h=dhash(data/r.path)
        matches=[(path,(h^ref).bit_count()) for path,ref in refs if (h^ref).bit_count()<=hash_dist]
        if matches:
            pairs.extend({'independent':r.path,'reference':path,'distance':dist} for path,dist in matches)
        else:
            keep.append(r.Index); refs.append((r.path,h))
    return ind.loc[keep].copy(),pd.DataFrame(pairs,columns=['independent','reference','distance'])

def exportar_splits(data, main, independent, classes):
    data=Path(data).resolve()
    df=main[['path','split','breed']].copy()
    def rel(p):
        p=Path(p)
        # EDA guarda rutas relativas a la raíz del repo o absolutas.
        p=p.resolve()
        if not p.is_relative_to(data) or not p.is_file(): raise ValueError(f'Ruta inválida: {p}')
        return p.relative_to(data).as_posix()
    df.path=df.path.map(rel)
    extra=independent[['path','breed']].assign(split='independent')
    df=pd.concat([df,extra],ignore_index=True)[['path','split','breed']]
    assert not df.path.duplicated().any(), 'Rutas repetidas'
    assert set(df[df.split=='train'].breed)==set(classes), 'Faltan clases en train'
    for split in ('valid','test'):
        assert set(df[df.split==split].breed)==set(classes), f'Faltan clases en {split}'
    assert set(df.breed)<=set(classes), 'Etiquetas externas'
    df=df.sort_values(['split','breed','path']).reset_index(drop=True)
    df.to_csv(data/'splits_finales.csv',index=False)
    return df
