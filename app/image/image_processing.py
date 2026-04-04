import logging
import cv2  # Requiere opencv-contrib-python
import concurrent.futures
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

# Configuración de logging unificada
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageAnalyzer(ABC):
    """Interfaz para diferentes tipos de análisis de imagen."""
    @abstractmethod
    def analyze(self, image_path: Path) -> Dict[str, Any]:
        pass

class GeometryAnalyzer(ImageAnalyzer):
    """Analiza dimensiones y proporciones de las imágenes."""
    def analyze(self, image_path: Path) -> Dict[str, Any]:
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                return {"width": w, "height": h, "aspect_ratio": w / h}
        except Exception as e:
            logger.error(f"Error analizando geometría en {image_path.name}: {e}")
            return {}

class ColorDistributionAnalyzer(ImageAnalyzer):
    """Analiza la intensidad media de color (RGB) y brillo."""
    def analyze(self, image_path: Path) -> Dict[str, Any]:
        try:
            with Image.open(image_path) as img:
                img_array = np.array(img.convert("RGB"))
                return {
                    "mean_red": float(np.mean(img_array[:, :, 0])),
                    "mean_green": float(np.mean(img_array[:, :, 1])),
                    "mean_blue": float(np.mean(img_array[:, :, 2])),
                    "brightness": float(np.mean(img_array))
                }
        except Exception as e:
            logger.error(f"Error analizando distribución de color en {image_path.name}: {e}")
            return {}

class ColorHeterogeneityAnalyzer(ImageAnalyzer):
    """Analiza la varianza de color (Heterogeneidad cromática - Regla ABCDE)."""
    def analyze(self, image_path: Path) -> Dict[str, Any]:
        try:
            with Image.open(image_path) as img:
                img_array = np.array(img.convert("RGB"))
                # La desviación estándar nos dice cuánto varían los colores entre sí
                return {
                    "std_red": float(np.std(img_array[:, :, 0])),
                    "std_green": float(np.std(img_array[:, :, 1])),
                    "std_blue": float(np.std(img_array[:, :, 2])),
                    "total_heterogeneity": float(np.std(img_array))
                }
        except Exception as e:
            logger.error(f"Error analizando heterogeneidad de color en {image_path.name}: {e}")
            return {}

class SimpleSymmetryAnalyzer(ImageAnalyzer):
    """Analiza la asimetría básica (A de ABCDE) comparando la imagen con sus espejos."""
    def analyze(self, image_path: Path) -> Dict[str, Any]:
        try:
            with Image.open(image_path).convert("L") as img: # Convertimos a gris para rapidez
                img_array = np.array(img)
                
                # Espejo horizontal y cálculo de diferencia absoluta media
                h_flipped = np.fliplr(img_array)
                h_asymmetry = np.mean(np.abs(img_array - h_flipped)) / 255.0
                
                # Espejo vertical
                v_flipped = np.flipud(img_array)
                v_asymmetry = np.mean(np.abs(img_array - v_flipped)) / 255.0

                return {
                    "h_asymmetry_score": float(h_asymmetry),
                    "v_asymmetry_score": float(v_asymmetry),
                    "avg_asymmetry_score": float((h_asymmetry + v_asymmetry) / 2)
                }
        except Exception as e:
            logger.error(f"Error analizando asimetría en {image_path.name}: {e}")
            return {}

class DatasetAnalyzerService:
    """Servicio que orquestas el análisis de múltiples categorías."""
    def __init__(self, analyzers: List[ImageAnalyzer]):
        self._analyzers = analyzers

    def analyze_category(self, category_path: Path) -> List[Dict[str, Any]]:
        results = []
        extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in extensions:
            image_files.extend(category_path.glob(ext))

        logger.info(f"Procesando {len(image_files)} imágenes en '{category_path.name}'...")
        
        for img_path in image_files:
            data = {"filename": img_path.name}
            for analyzer in self._analyzers:
                data.update(analyzer.analyze(img_path))
            results.append(data)
            
        return results

class MelanomaPatternExplorer:
    """Clase principal para comparar patrones entre Benigno y Maligno."""
    def __init__(self, service: DatasetAnalyzerService):
        self._service = service

    def compare_train_data(self, train_path: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Compara las categorías benign/malignant extrayendo todas sus características."""
        categories = ["benign", "malignant"]
        full_report = {}

        for cat in categories:
            cat_path = train_path / cat
            if cat_path.exists():
                full_report[cat] = self._service.analyze_category(cat_path)
            else:
                logger.warning(f"Categoría no encontrada: {cat_path}")
        
        return full_report

class SuperResolutionEnhancer:
    """
    Single Responsibility: Mejorar la resolución de imágenes usando 
    modelos de Deep Learning (Redes Neuronales Convolucionales).
    Versión Optimizada: Uso de FSRCNN (Rápido) y procesamiento paralelo.
    """
    def __init__(self, model_path: str = "models/FSRCNN_x2.pb"):
        self.model_path = Path(model_path)
        self._is_ready = self.model_path.exists()

    def _upscale_single_image(self, img_path: Path):
        """Tarea para un solo núcleo de CPU (Hilo)."""
        sr_path = img_path.with_name(f"{img_path.stem}_sr{img_path.suffix}")
        if sr_path.exists(): 
            return
        
        try:
            # Una instancia por hilo para evitar bloqueos internos del framework
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(str(self.model_path))
            sr.setModel("fsrcnn", 2) # FSRCNN x2
            
            img = cv2.imread(str(img_path))
            if img is not None:
                upscaled = sr.upsample(img)
                if cv2.imwrite(str(sr_path), upscaled):
                    img_path.unlink()
        except Exception:
            # Silenciamos errores individuales para no interrumpir el flujo masivo
            pass

    def upscale_directory(self, directory: Path):
        """Usa todos los núcleos de tu procesador para ir a máxima velocidad."""
        if not self._is_ready:
            logger.warning(f"No se encontró el modelo en {self.model_path}. Saltando mejora.")
            return

        image_files = [f for f in directory.glob("*") if f.suffix.lower() in ['.jpg', '.png'] and "_sr" not in f.name]
        logger.info(f"🚀 Iniciando Super-Resolución FSRCNN PARALELA en {directory.name} ({len(image_files)} archivos)...")
        
        # Procesamiento paralelo masivo usando todos los hilos (OpenCV libera el GIL)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self._upscale_single_image, image_files)
            
        logger.info(f"✅ Mejora de imágenes completada en {directory.name}.")
