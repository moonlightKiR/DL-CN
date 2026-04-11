import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Configuración de logging unificada
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageProvider(ABC):
    """
    Interface Segregation: 
    Define cómo se obtienen las imágenes para visualizar.
    """
    @abstractmethod
    def fetch_images(self, directory: Path, limit: int) -> List[Path]:
        pass

class LocalImageProvider(ImageProvider):
    """Implementación que busca imágenes en el disco local."""
    def fetch_images(self, directory: Path, limit: int) -> List[Path]:
        if not directory.exists():
            logger.error(f"El directorio no existe: {directory}")
            return []
        
        # Filtramos por extensiones comunes
        extensions = ['*.jpg', '*.jpeg', '*.png']
        images = []
        for ext in extensions:
            images.extend(list(directory.glob(ext)))
        
        # Devolvemos solo las primeras N imágenes
        return sorted(images)[:limit]

class GridVisualizer:
    """
    Single Responsibility: 
    Se encarga exclusivamente de renderizar las imágenes en una rejilla de Matplotlib.
    """
    @staticmethod
    def show_grid(image_paths: List[Path], title: str, cols: int = 5, save_path: Optional[Path] = None):
        if not image_paths:
            logger.warning(f"No hay imágenes para mostrar en {title}")
            return

        rows = (len(image_paths) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for i, img_path in enumerate(image_paths):
            try:
                with Image.open(img_path) as img:
                    axes[i].imshow(img)
                    axes[i].set_title(img_path.name, fontsize=8)
                    axes[i].axis('off')
            except Exception as e:
                logger.error(f"Error cargando {img_path.name}: {e}")

        # Limpiar ejes sobrantes si la rejilla no está llena
        for i in range(len(image_paths), len(axes)):
            axes[i].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Visualización guardada con éxito en: {save_path}")
        else:
            plt.show()
            
        plt.close(fig) # Liberamos recursos de Matplotlib

class DataViewerService:
    """
    Dependency Inversion: 
    Orquesta el proveedor de imágenes y el visualizador.
    """
    def __init__(self, provider: ImageProvider):
        self._provider = provider

    def visualize_category(self, category_path: str | Path, limit: int = 20, cols: int = 5, save_path: Optional[Path] = None):
        """Visualiza las primeras N imágenes de una categoría específica y opcionalmente guarda el resultado."""
        path = Path(category_path)
        images = self._provider.fetch_images(path, limit)
        GridVisualizer.show_grid(images, f"Exploración: {path.name.upper()}", cols=cols, save_path=save_path)

def get_default_viewer() -> DataViewerService:
    """Factory method para uso rápido en Notebooks."""
    return DataViewerService(LocalImageProvider())
