import logging
import random
from abc import ABC, abstractmethod
from typing import List, Tuple
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

# Configuración de logging unificada
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageTransform(ABC):
    """
    Interface Segregation / Open-Closed:
    Interfaz para cualquier tipo de transformación de imagen.
    """
    @abstractmethod
    def apply(self, image: Image.Image) -> Image.Image:
        pass

class RotationTransform(ImageTransform):
    """Gira la imagen en ángulos aleatorios (90, 180, 270)."""
    def apply(self, image: Image.Image) -> Image.Image:
        angles = [90, 180, 270]
        return image.rotate(random.choice(angles))

class FlipTransform(ImageTransform):
    """Realiza un volteo horizontal aleatorio."""
    def apply(self, image: Image.Image) -> Image.Image:
        if random.random() > 0.5:
            return ImageOps.mirror(image)
        return image

class BrightnessTransform(ImageTransform):
    """Aleatoriza el brillo para simular diferentes condiciones de luz dermoscópica."""
    def __init__(self, factor_range: Tuple[float, float] = (0.8, 1.2)):
        self.factor_range = factor_range

    def apply(self, image: Image.Image) -> Image.Image:
        factor = random.uniform(*self.factor_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

class AugmentationPipeline:
    """
    Single Responsibility / Composite:
    Gestiona una lista de transformaciones y las aplica en cadena.
    """
    def __init__(self, transforms: List[ImageTransform]):
        self._transforms = transforms

    def process(self, image: Image.Image) -> Image.Image:
        """Aplica todas las transformaciones registradas secuencialmente."""
        augmented_image = image.copy()
        for transform in self._transforms:
            augmented_image = transform.apply(augmented_image)
        return augmented_image

class AugmentationService:
    """
    Dependency Inversion: Orquesta el uso de la tubería para el usuario.
    Permite tanto previsualizar como persistir aumentos en disco.
    """
    def __init__(self, pipeline: AugmentationPipeline):
        self._pipeline = pipeline

    def preview_augmentation(self, image_path: str, num_variants: int = 4) -> List[Image.Image]:
        """Genera variantes aumentadas de una sola imagen para visualizar en el Notebook."""
        variants = []
        try:
            original = Image.open(image_path)
            variants.append(original.copy()) # Incluimos la original

            for _ in range(num_variants):
                variants.append(self._pipeline.process(original))
            
            logger.info(f"Generadas {num_variants} variantes aumentadas de {image_path}")
            return variants
        except Exception as e:
            logger.error(f"Error al procesar la imagen {image_path}: {e}")
            return []

    def augment_directory(self, source_dir: str, variants_per_image: int = 1) -> None:
        """
        Lee todas las imágenes de una carpeta y guarda versiones aumentadas físicamente.
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"La carpeta {source_dir} no existe.")
            return

        # Buscamos formatos comunes
        extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in extensions:
            image_files.extend(source_path.glob(ext))

        logger.info(f"Iniciando aumento de datos en {source_dir} ({len(image_files)} imágenes)...")
        
        for img_path in image_files:
            # Evitamos aumentar una imagen que ya es un aumento (para evitar bucles infinitos)
            if "_aug_" in img_path.name:
                continue

            try:
                with Image.open(img_path) as img:
                    for i in range(variants_per_image):
                        augmented = self._pipeline.process(img)
                        # Creamos un nombre descriptivo para identificar los aumentos
                        new_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                        save_path = img_path.parent / new_name
                        
                        augmented.save(save_path)
            except Exception as e:
                logger.error(f"Error procesando {img_path.name}: {e}")

        logger.info(f"Aumento completado con éxito en {source_dir}.")

# Configuración por defecto lista para ser importada
def get_default_augmentation_service() -> AugmentationService:
    """Factory method para instanciar rápidamente la configuración recomendada."""
    transforms = [
        RotationTransform(),
        FlipTransform(),
        BrightnessTransform(factor_range=(0.7, 1.3))
    ]
    pipeline = AugmentationPipeline(transforms)
    return AugmentationService(pipeline)
