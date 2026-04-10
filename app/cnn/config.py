import json
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class CNNConfig:
    """Configuración optimizada (V2) para el ajuste de hiperparámetros."""
    input_size: int = 160        
    num_channels: int = 3
    num_classes: int = 2
    batch_size: int = 32
    learning_rate: float = 0.0005 
    weight_decay: float = 1e-4    
    dropout_rate: float = 0.5     # Nuevo: parametrizado para Optuna
    epochs: int = 30              
    model_dir: Path = Path("models/cnn")
    data_dir: Path = Path("data/train")

    def __post_init__(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_best_params(self):
        """Carga automáticamente los mejores parámetros si el archivo existe."""
        best_path = self.model_dir / "best_hyperparams.json"
        if best_path.exists():
            with open(best_path, 'r') as f:
                params = json.load(f)
                self.learning_rate = params.get('lr', self.learning_rate)
                self.weight_decay = params.get('wd', self.weight_decay)
                self.dropout_rate = params.get('dropout', self.dropout_rate)
                logger.info(f"✅ Hiperparámetros cargados desde {best_path}")
        else:
            logger.info("⚠️ No se encontró best_hyperparams.json. Usando valores por defecto.")
