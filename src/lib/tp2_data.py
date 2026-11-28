"""Adaptador de los splits definitivos al dataset compartido."""
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lib.datasets import BreedDataset, basic_augment
from lib.preprocessing import preprocess_bgr

SEED = 42

def fijar_semilla(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_bgr(image, image_size=224):
    return torch.from_numpy(np.ascontiguousarray(preprocess_bgr(image, image_size))).unsqueeze(0)


def cargar_splits(dataset_path):
    root = Path(dataset_path).resolve()
    df = pd.read_csv(root / 'splits_finales.csv', dtype=str, keep_default_na=False)
    if not {'path', 'split', 'breed'} <= set(df.columns):
        raise ValueError('CSV requiere path, split y breed')
    if not set(df.split) <= {'train', 'valid', 'test', 'independent'}:
        raise ValueError('Split desconocido')
    classes = set(df.loc[df.split == 'train', 'breed'])
    if not classes or '' in classes:
        raise ValueError("El manifiesto requiere razas de entrenamiento")
    for split in ('valid', 'test'):
        if set(df.loc[df.split == split, 'breed']) != classes:
            raise ValueError(f"El split {split} debe cubrir todas las razas de train")
    if not set(df.breed) <= classes:
        raise ValueError("Hay razas fuera de train")
    paths = []
    for value in df.path:
        rel = Path(value)
        p = (root / rel).resolve()
        if rel.is_absolute() or not p.is_relative_to(root) or not p.is_file():
            raise ValueError(f'Ruta de imagen inválida: {value}')
        paths.append(p)
    if len(set(paths)) != len(paths):
        raise ValueError('Rutas duplicadas entre los splits')
    return df


def dataset_filas(dataset_path, filas, clases, image_size=224, train=False):
    filas = filas.reset_index(drop=True)
    ids = {c: i for i, c in enumerate(clases)}
    if filas.empty or not set(filas.breed) <= set(ids):
        raise ValueError('Split vacío o clases no incluidas en train')
    samples = [(Path(dataset_path) / r.path, ids[r.breed]) for r in filas.itertuples(index=False)]
    ds = BreedDataset(samples, image_size, basic_augment if train else None)
    ds.filas = filas  # conserva orden para relacionar predicciones con rutas
    return ds


def rutas_galeria(dataset_path):
    df = cargar_splits(dataset_path)
    return [(Path(dataset_path) / r.path, r.breed) for r in df[df.split == 'train'].itertuples(index=False)]
