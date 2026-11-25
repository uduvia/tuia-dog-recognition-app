"""CNN propia: bloques Conv-BN-ReLU-Pool, GAP y embedding explícito."""
import torch
from torch import nn

class CNNCustom(nn.Module):
    def __init__(self, num_clases, embedding_dim=512, dropout=0.3, canales=(32,64,128,256)):
        super().__init__()
        bloques = []
        entrada = 3
        for salida in canales:
            bloques += [nn.Conv2d(entrada, salida, 3, padding=1, bias=False),
                        nn.BatchNorm2d(salida), nn.ReLU(), nn.MaxPool2d(2)]
            entrada = salida
        self.features = nn.Sequential(*bloques)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(entrada, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_clases)
    def forward_features(self, x):
        return self.embedding(self.pool(self.features(x)).flatten(1))
    def forward(self, x):
        return self.classifier(self.dropout(self.forward_features(x)))
