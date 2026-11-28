"""Modelo A: ResNet18 preentrenada en ImageNet adaptada a las razas.

Interfaz comun a los modelos entrenados (tambien la va a cumplir la CNN propia):
  - forward(x)  -> logits (N, num_classes)
  - embed(x)    -> embedding (N, embedding_dim), capa previa al clasificador
  - class_names -> lista de razas en el orden de los indices (viaja dentro del .pth)

La clase vive en un modulo importable (lib.models.resnet_classifier) para que
torch.load del modelo completo funcione desde el backend.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class BreedResNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_names: list[str] | None = None,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.embedding_dim: int = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.embedding_dim, num_classes))
        self.class_names: list[str] = list(class_names or [])

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))
