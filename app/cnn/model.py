import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class DepthwiseSeparableConv(nn.Module):
    """
    Bloque de convolución separable en profundidad (Depthwise + Pointwise).
    Reduce enormemente el peso del modelo sin perder capacidades espaciales.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, 
            padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MelanomaNet(nn.Module):
    """
    CNN eficiente optimizada para la detección de melanoma.
    Limitada a ~150k parámetros (Objetivo: < 250k).
    """
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        
        # Extracción de características (Escalado para ~200k-220k parámetros)
        self.features = nn.Sequential(
            # Entrada: 160x160x3
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), 
            
            DepthwiseSeparableConv(16, 32, stride=2),  
            DepthwiseSeparableConv(32, 64, stride=2),  
            DepthwiseSeparableConv(64, 128, stride=2), 
            DepthwiseSeparableConv(128, 320, stride=2),
            DepthwiseSeparableConv(320, 320, stride=1),
            
            nn.AdaptiveAvgPool2d(1) 
        )
        
        # Clasificador denso ajustable
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate), # 👈 Configurable para búsqueda sistemática
            nn.Linear(320, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
        logger.info(f"Modelo MelanomaNet inicializado con {self.count_parameters():,} parámetros.")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        """Calcula el total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _initialize_weights(self):
        """Inicialización de Kaiming para capas convolucionales."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
