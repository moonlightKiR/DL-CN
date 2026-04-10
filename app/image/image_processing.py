import logging
import cv2  # Requiere opencv-contrib-python
import concurrent.futures
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

# Importaciones de PyTorch para ESRGAN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Configuración de logging unificada (Gestionada por main.py)
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

# --- ARQUITECTURA FSRCNN (Fast Super-Resolution CNN) ---

class FSRCNN(nn.Module):
    """
    Arquitectura ultra-ligera diseñada para velocidad extrema.
    Procesa habitualmente 1 solo canal (Luminancia).
    """
    def __init__(self, scale_factor=4, num_channels=1):
        super(FSRCNN, self).__init__()
        # 1. Feature extraction
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 56, kernel_size=5, padding=2),
            nn.PReLU(56)
        )
        # 2. Shrinking, 3. Non-linear Mapping, 4. Expanding
        self.mid_part = nn.Sequential(
            nn.Conv2d(56, 12, kernel_size=1),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(12),
            nn.Conv2d(12, 56, kernel_size=1),
            nn.PReLU(56)
        )
        # 5. Deconvolution (Upsampling)
        self.last_part = nn.ConvTranspose2d(
            56, num_channels, kernel_size=9, stride=scale_factor, 
            padding=4, output_padding=scale_factor-1
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

# --- SERVICIO DE SUPER RESOLUCIÓN Y REFINAMIENTO ---

class SuperResolutionEnhancer:
    """
    Servicio híbrido: Super Resolución (FSRCNN Y-Channel) + Enfoque (Laplaciano).
    Soporta modelos de 1 canal (estándar de FSRCNN .pth).
    """
    def __init__(self, model_path: str = "models/FSRCNN_x4.pth"):
        self.model_path = Path(model_path)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.model = None
        self._is_ready = self.model_path.exists()

    def load_model(self):
        """Carga el modelo FSRCNN x4 (1 canal)."""
        if not self._is_ready:
            logger.warning(f"No se encontró el modelo FSRCNN en {self.model_path}")
            return

        try:
            self.model = FSRCNN(scale_factor=4, num_channels=1)
            state_dict = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
            
            if 'params' in state_dict: state_dict = state_dict['params']
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '').replace('model.', '')] = v
            
            self.model.load_state_dict(new_state_dict, strict=True)
            self.model.eval().to(self.device)
            logger.info(f"FSRCNN x4 (Luminancia) cargado con éxito en {self.device}")
        except Exception as e:
            logger.error(f"Error cargando FSRCNN: {e}")
            self._is_ready = False

    @torch.no_grad()
    def upscale_image(self, img_pil: Image.Image) -> Image.Image:
        """Escala el canal Y con PyTorch y CbCr con Bicubic."""
        # 1. Preparar canales YCbCr
        ycbcr = img_pil.convert('YCbCr')
        y, cb, cr = ycbcr.split()

        # 2. Inferencia en canal Y
        img_t = TF.to_tensor(y).unsqueeze(0).to(self.device)
        output_y = self.model(img_t).squeeze().cpu().clamp(0, 1)
        output_y_pil = TF.to_pil_image(output_y)

        # 3. Rescale Cb y Cr de forma tradicional (Bicubic)
        new_size = output_y_pil.size
        output_cb = cb.resize(new_size, resample=Image.BICUBIC)
        output_cr = cr.resize(new_size, resample=Image.BICUBIC)

        # 4. Recombinar y aplicar Sharpening Laplaciano
        sr_img_pil = Image.merge('YCbCr', (output_y_pil, output_cb, output_cr)).convert('RGB')
        img_np = np.array(sr_img_pil)
        
        laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
        sharpened = img_np - (0.3 * laplacian)
        
        return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))

    def _process_single_file(self, img_path: Path):
        """Procesa y limpia un archivo individual."""
        sr_path = img_path.with_name(f"{img_path.stem}_sr{img_path.suffix}")
        if sr_path.exists(): return

        try:
            with Image.open(img_path) as img_pil:
                sr_img = self.upscale_image(img_pil)
                sr_img.save(sr_path, quality=95)
                img_path.unlink() # Borrar original tras éxito
        except Exception:
            pass

    def upscale_directory(self, directory: Path):
        """Ejecución masiva en el directorio de entrenamiento."""
        if not self._is_ready or self.model is None:
            return

        image_files = [f for f in directory.glob("*") if f.suffix.lower() in ['.jpg', '.png'] and "_sr" not in f.name]
        logger.info(f"Iniciando Pipeline Rápido (FSRCNN + Laplaciano) en {directory.name}...")
        
        for i, img_path in enumerate(image_files):
            self._process_single_file(img_path)
            if i % 100 == 0:
                logger.info(f"Progreso {directory.name}: {i}/{len(image_files)}")

        logger.info(f"Pipeline completado en {directory.name}.")
