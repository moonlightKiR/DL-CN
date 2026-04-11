import shutil
import logging
from abc import ABC, abstractmethod
from pathlib import Path

# Configuración de logging unificada con zip_manager
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Mover(ABC):
    @abstractmethod
    def move(self, source: str | Path, destination: str | Path) -> None:
        pass

    @abstractmethod
    def remove(self, path: str | Path) -> None:
        pass

class FileMover(Mover):
    """Implementación concreta para mover y borrar archivos/carpetas."""
    def move(self, source: str | Path, destination: str | Path) -> None:
        src = Path(source)
        dest = Path(destination)

        if not src.exists():
            logger.error(f"Source path not found: {src}")
            return

        if dest.exists():
            logger.warning(f"Destination already exists: {dest}. Skipping.")
            return

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            logger.info(f"Successfully moved: {src.name} -> {dest}")
        except Exception as e:
            logger.error(f"Failed to move {src}: {e}")

    def remove(self, path: str | Path) -> None:
        target = Path(path)
        if target.exists():
            try:
                shutil.rmtree(str(target))
                logger.info(f"Successfully removed residual directory: {target}")
            except Exception as e:
                logger.error(f"Failed to remove {target}: {e}")

class MelanomaDataService:
    """Servicio de alto nivel para organizar el dataset de Melanoma."""
    def __init__(self, mover: Mover):
        self._mover = mover

    def organize(self, root_path: str | Path) -> None:
        root = Path(root_path)
        data_dir = root / "data"

        # 1. Definir operaciones de movimiento
        moves = [
            (data_dir / "melanoma_test_data" / "test", data_dir / "test"),
            (data_dir / "melanoma_train_data_benign" / "Benign", data_dir / "train" / "benign"),
            (data_dir / "melanoma_train_data_malignant" / "Malignant", data_dir / "train" / "malignant")
        ]

        # 2. Ejecutar movimientos
        logger.info("Starting Melanoma data organization...")
        for src, dest in moves:
            self._mover.move(src, dest)

        # 3. Limpieza de carpetas vacías/residuales
        residuals = [
            data_dir / "melanoma_test_data",
            data_dir / "melanoma_train_data_benign",
            data_dir / "melanoma_train_data_malignant"
        ]
        
        logger.info("Cleaning up residual directories...")
        for path in residuals:
            self._mover.remove(path)
        
        logger.info("Organization and cleanup complete.")

if __name__ == "__main__":
    MOVER = FileMover()
    SERVICE = MelanomaDataService(MOVER)
    # Ejemplo de uso: SERVICE.organize(Path.cwd())
