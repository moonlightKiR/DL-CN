import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

from data.zip_manager import ZipExtractor
from data.file_manager import FileMover, MelanomaDataService
from image.image_processing import (
    GeometryAnalyzer,
    ColorDistributionAnalyzer,
    ColorHeterogeneityAnalyzer,
    SimpleSymmetryAnalyzer,
    DatasetAnalyzerService,
    MelanomaPatternExplorer,
    SuperResolutionEnhancer
)
from eda.comparasion import DescriptiveStatsComparator, EDAComparisonService
from eda.data_augmentation import get_default_augmentation_service
from data.data_viewer import get_default_viewer

# Configuración de Logging y Directorios
log_dir = Path("log")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = log_dir / f"pipeline_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler() # Mantenemos salida por consola
    ],
    force=True # Forzamos la configuración para que el fichero de log funcione siempre
)
logger = logging.getLogger(__name__)

def run_pipeline():
    # Usamos Path.cwd() (Directorio de Trabajo Actual), que suele ser la raíz del proyecto
    base_path = Path.cwd()
    data_path = base_path / "data"

    logger.info(f"Ruta de ejecución detectada: {base_path}")

    # 1. SETUP: Extracción y Organización
    logger.info("=== STEP 1: SETUP DATA ===")
    extractor = ZipExtractor()
    zips = {
        "data_compressed/melanoma_test_data.zip": "data/melanoma_test_data",
        "data_compressed/melanoma_train_data_benign.zip": "data/melanoma_train_data_benign",
        "data_compressed/melanoma_train_data_malignant.zip": "data/melanoma_train_data_malignant"
    }
    for zip_src, dest in zips.items():
        if (base_path / zip_src).exists():
            extractor.extract(zip_src, dest)
        else:
            logger.warning(f"Zip no encontrado en: {zip_src}. Saltando extración.")

    # Organizamos en carpetas limpias
    organizer = MelanomaDataService(FileMover())
    organizer.organize(base_path)

    # 1.5 ENHANCEMENT: Super Resolución x4 (FSRCNN + Laplacian)
    logger.info("=== STEP 1.5: IMAGE ENHANCEMENT (FSRCNN PyTorch x4 + Laplacian) ===")
    enhancer = SuperResolutionEnhancer(model_path="models/FSRCNN_x4.pth")
    enhancer.load_model() # Cargamos pesos en GPU/MPS
    
    # Procesamos las carpetas de entrenamiento
    train_path = data_path / "train"
    if train_path.exists():
        enhancer.upscale_directory(train_path / "benign")
        enhancer.upscale_directory(train_path / "malignant")

    # 2. ANALYSIS
    logger.info("=== STEP 2: PATTERN ANALYSIS ===")
    analyzers = [
        GeometryAnalyzer(),
        ColorDistributionAnalyzer(),
        ColorHeterogeneityAnalyzer(),
        SimpleSymmetryAnalyzer()
    ]
    analyzer_service = DatasetAnalyzerService(analyzers)
    explorer = MelanomaPatternExplorer(analyzer_service)
    
    train_path = data_path / "train"
    if train_path.exists():
        results = explorer.compare_train_data(train_path)

        # Convertimos a DataFrame para el EDA
        df_benign = pd.DataFrame(results['benign'])
        df_malignant = pd.DataFrame(results['malignant'])

        # 3. EDA
        logger.info("=== STEP 3: EXPLORATORY DATA ANALYSIS ===")
        eda_service = EDAComparisonService(DescriptiveStatsComparator())
        comparison_report = eda_service.execute_analysis(df_benign, df_malignant)
        
        print("\n--- RESUMEN DE DIFERENCIAS (BENIGN vs MALIGNANT) ---")
        print(comparison_report[['benign_avg', 'malignant_avg', 'percentage_change']].head(10))

        # 3.1 VISUAL EDA: Guardar Boxplots
        from eda.comparasion import EDAVisualizer
        key_metrics = ['total_heterogeneity', 'avg_asymmetry_score', 'brightness']
        EDAVisualizer.save_comparison_plots(
            df_benign, 
            df_malignant, 
            metrics=key_metrics, 
            save_path=base_path / "images" / "eda_boxplots.png"
        )

        # 4. AUGMENTATION
        logger.info("=== STEP 4: DATA AUGMENTATION ===")
        aug_service = get_default_augmentation_service()
        aug_service.augment_directory(train_path / "benign", variants_per_image=3)
        aug_service.augment_directory(train_path / "malignant", variants_per_image=3)

        # 4.5 POST-AUGMENTATION EDA (Validación de impacto)
        logger.info("=== STEP 4.5: POST-AUGMENTATION ANALYSIS ===")
        results_post = explorer.compare_train_data(train_path)
        df_benign_post = pd.DataFrame(results_post['benign'])
        df_malignant_post = pd.DataFrame(results_post['malignant'])

        comparison_report_post = eda_service.execute_analysis(df_benign_post, df_malignant_post)
        print("\n--- RESUMEN DE DIFERENCIAS (POST-AUGMENTATION) ---")
        print(comparison_report_post[['benign_avg', 'malignant_avg', 'percentage_change']].head(10))

        # Guardamos la gráfica del estado final para comparar con el original
        EDAVisualizer.save_comparison_plots(
            df_benign_post, 
            df_malignant_post, 
            metrics=key_metrics, 
            save_path=base_path / "images" / "eda_boxplots_augmented.png"
        )

        # 5. VISUALIZATION: Guardado de resultados en disco
        logger.info("=== STEP 5: SAVING VISUALIZATIONS ===")
        output_folder = base_path / "images"
        viewer = get_default_viewer()
        
        # Guardamos el resumen de benignos (las primeras 20)
        viewer.visualize_category(
            train_path / "benign", 
            limit=20, 
            save_path=output_folder / "summary_benign.png"
        )
        
        # Guardamos el resumen de malignos (las primeras 20)
        viewer.visualize_category(
            train_path / "malignant", 
            limit=20, 
            save_path=output_folder / "summary_malignant.png"
        )
    else:
        logger.error(f"No se ha encontrado la carpeta de datos en: {train_path}")

if __name__ == "__main__":
    try:
        run_pipeline()
        logger.info("Pipeline ejecutado con éxito.")
    except Exception as e:
        logger.error(f"Error crítico en el pipeline: {e}")
