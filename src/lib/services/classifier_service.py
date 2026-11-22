from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime

from lib.evaluation.metrics import precision_recall_f1, specificity
from lib.preprocessing import l2_normalize, preprocess_bgr

logger = logging.getLogger(__name__)

# Hiperparametros por defecto del entrenamiento. Se pueden sobreescribir desde la notebook:
#   classifier.train_config = {"epochs": 10, "strategy": "frozen"}
DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "strategy": "full",  # "frozen" (solo la cabeza) | "full" (fine-tuning completo)
    "epochs": 5,
    "batch_size": 32,
    "lr_head": 1e-3,
    "lr_backbone": 1e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "dropout": 0.2,
    "patience": 3,  # early stopping sobre accuracy de validacion
    "seed": 42,
    "num_workers": 0,  # 0 es lo mas seguro en Windows
    "max_images_per_class": None,  # para pruebas rapidas (ej. 20)
}


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 (o ResNet50) pre-entrenado.
        Modelo B (obligatorio): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en 1-pre-processing.ipynb.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        from lib.datasets import BreedDataset, basic_augment, list_class_names, list_samples
        from lib.models.resnet_classifier import BreedResNet

        cfg = self._train_cfg()
        self._seed_everything(cfg["seed"])
        if self.active_model_name != "resnet18_finetuned":
            raise ValueError(
                f"train_classifier aun no soporta '{self.active_model_name}' (la CNN propia se agrega en la Fase 4)"
            )

        class_names = list_class_names(self.dataset_path / "train")
        limit = cfg["max_images_per_class"]
        train_samples = list_samples(self.dataset_path / "train", class_names, limit, cfg["seed"])
        val_samples = list_samples(self.dataset_path / "valid", class_names, limit, cfg["seed"])
        logger.info("Clases=%d train=%d valid=%d", len(class_names), len(train_samples), len(val_samples))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader_kw = dict(num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
        train_loader = DataLoader(
            BreedDataset(train_samples, self.image_size, transform=basic_augment),
            batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **loader_kw,
        )
        val_loader = DataLoader(
            BreedDataset(val_samples, self.image_size), batch_size=cfg["batch_size"], shuffle=False, **loader_kw
        )

        model = BreedResNet(len(class_names), class_names, pretrained=True, dropout=cfg["dropout"]).to(device)
        frozen = cfg["strategy"] == "frozen"
        if frozen:
            for p in model.backbone.parameters():
                p.requires_grad = False
            groups = [{"params": model.classifier.parameters(), "lr": cfg["lr_head"]}]
        elif cfg["strategy"] == "full":
            groups = [
                {"params": model.backbone.parameters(), "lr": cfg["lr_backbone"]},
                {"params": model.classifier.parameters(), "lr": cfg["lr_head"]},
            ]
        else:
            raise ValueError(f"strategy invalida: {cfg['strategy']}")

        optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

        history: list[dict[str, float]] = []
        best_acc, best_state, bad_epochs = -1.0, None, 0
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            if frozen:
                model.backbone.eval()  # no actualizar las estadisticas de BatchNorm
            tr_loss, tr_correct, tr_n = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * y.size(0)
                tr_correct += (logits.argmax(1) == y).sum().item()
                tr_n += y.size(0)
            scheduler.step()

            model.eval()
            va_loss, va_correct, va_n = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    va_loss += criterion(logits, y).item() * y.size(0)
                    va_correct += (logits.argmax(1) == y).sum().item()
                    va_n += y.size(0)
            row = {
                "epoch": epoch,
                "train_loss": tr_loss / max(tr_n, 1),
                "train_acc": tr_correct / max(tr_n, 1),
                "val_loss": va_loss / max(va_n, 1),
                "val_acc": va_correct / max(va_n, 1),
            }
            history.append(row)
            logger.info("epoch %d | %s", epoch, {k: round(v, 4) for k, v in row.items()})

            if row["val_acc"] > best_acc:  # mejor checkpoint segun validacion
                best_acc, bad_epochs = row["val_acc"], 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= cfg["patience"]:
                    logger.info("Early stopping en la epoca %d", epoch)
                    break

        model.load_state_dict(best_state)
        model.to("cpu").eval()
        self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, self.active_checkpoint)  # modelo completo (incluye class_names)
        self._loaded.pop(self.active_model_name, None)  # invalida la cache
        self.output_path.mkdir(parents=True, exist_ok=True)
        hist_path = self.output_path / f"{self.active_model_name}_history.json"
        hist_path.write_text(
            json.dumps({"config": cfg, "best_val_acc": best_acc, "history": history}, indent=2),
            encoding="utf-8",
        )
        logger.info("Checkpoint guardado en %s (best val_acc=%.4f)", self.active_checkpoint, best_acc)

    def _train_cfg(self) -> dict[str, Any]:
        return {**DEFAULT_TRAIN_CONFIG, **getattr(self, "train_config", {})}

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _run_logits(self, model: Any, batch: np.ndarray) -> np.ndarray:
        """Logits (N, C) de un batch NCHW float32 para modelos .pth y .onnx."""
        if hasattr(model, "run"):
            return np.asarray(model.run(None, {model.get_inputs()[0].name: batch})[0])
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(batch)).cpu().numpy()

    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Debe reportar: accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. La matriz de confusion y las
        curvas de entrenamiento se documentan en 2-train.ipynb.

        Retorna un dict con las metricas, ej:
          {"accuracy": 0.91, "precision": 0.90, "recall": 0.89,
           "specificity": 0.99, "f1": 0.90}
        """
        from lib.datasets import BreedDataset, list_samples

        model = self.load_model()
        if hasattr(model, "run"):
            raw = model.get_modelmeta().custom_metadata_map.get("class_names", "")
            class_names = json.loads(raw) if raw else []
        else:
            class_names = list(getattr(model, "class_names", []))
        if not class_names:
            raise ValueError("El modelo no trae la lista de razas (class_names)")

        samples = list_samples(self.dataset_path / "test", class_names)
        dataset = BreedDataset(samples, self.image_size)
        y_true, y_pred = [], []
        batch_size = 32
        for start in range(0, len(dataset), batch_size):
            items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            batch = np.stack([x for x, _ in items]).astype(np.float32)
            y_true.extend(label for _, label in items)
            y_pred.extend(self._run_logits(model, batch).argmax(1).tolist())

        n = len(class_names)
        cm = np.zeros((n, n), dtype=np.int64)  # filas: real, columnas: predicho
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        self.last_eval = {"confusion_matrix": cm, "class_names": class_names,
                          "y_true": y_true, "y_pred": y_pred}  # para la notebook

        precisions, recalls, f1s, specs = [], [], [], []
        total = int(cm.sum())
        for c in range(n):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            tn = total - tp - fp - fn
            pr, rc, f1 = precision_recall_f1(tp, fp, fn)
            precisions.append(pr); recalls.append(rc); f1s.append(f1)
            specs.append(specificity(tn, fp))
        return {
            "accuracy": float(np.trace(cm) / max(total, 1)),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "specificity": float(np.mean(specs)),
            "f1": float(np.mean(f1s)),
        }

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.
        """
        model = self.load_model()
        tensor = preprocess_bgr(image, self.image_size)[None, ...]
        if hasattr(model, "run"):
            # Convencion ONNX: salidas = [logits, embedding]
            outputs = model.run(None, {model.get_inputs()[0].name: tensor})
            if len(outputs) < 2:
                raise ValueError("El ONNX debe exportar dos salidas: logits y embedding")
            emb = np.asarray(outputs[1])[0]
        else:
            model.eval()
            with torch.no_grad():
                emb = model.embed(torch.from_numpy(tensor))[0].cpu().numpy()
        return l2_normalize(emb).astype(float).tolist()
