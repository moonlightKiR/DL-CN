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
from train_cnn import run_training
from hyperparam_search import run_study

# Configuración de Logging y Directorios
log_dir = Path("log")
log_dir.mkdir(exist_ok=True)
log_path = log_dir / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8', mode='a'),
        logging.StreamHandler() 
    ],
    force=True 
)
logger = logging.getLogger(__name__)

# --- FUNCIONES DE CADA PASO (STEPS) ---

def step_setup_data(base_path, data_path):
    """Extracción y organización inicial de los datos."""
    logger.info("=== STEP 1: SETUP DATA ===")
    train_path = data_path / "train"
    if train_path.exists():
        logger.info("✅ Datos ya organizados en data/train. Saltando extracción.")
        return train_path

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
            logger.warning(f"Zip no encontrado: {zip_src}")

    organizer = MelanomaDataService(FileMover())
    organizer.organize(base_path)
    return train_path

def step_image_enhancement(train_path):
    """Mejora de imagen con Super Resolución."""
    logger.info("=== STEP 1.5: IMAGE ENHANCEMENT (FSRCNN + Laplacian) ===")
    enhancer = SuperResolutionEnhancer(model_path="models/FSRCNN_x4.pth")
    enhancer.load_model()
    
    # Comprobar si hay trabajo pendiente
    for cat in ["benign", "malignant"]:
        cat_path = train_path / cat
        if not cat_path.exists(): continue
        
        pending = [f for f in cat_path.glob("*") if f.suffix.lower() in ['.jpg', '.png'] and "_sr" not in f.name]
        if not pending:
            logger.info(f"✅ Imágenes de '{cat}' ya procesadas por SR. Saltando.")
        else:
            enhancer.upscale_directory(cat_path)

def step_pattern_analysis_and_eda(train_path, base_path):
    """Análisis descriptivo y comparativo de las imágenes."""
    logger.info("=== STEP 2 & 3: PATTERN ANALYSIS & EDA ===")
    analyzers = [GeometryAnalyzer(), ColorDistributionAnalyzer(), ColorHeterogeneityAnalyzer(), SimpleSymmetryAnalyzer()]
    explorer = MelanomaPatternExplorer(DatasetAnalyzerService(analyzers))
    
    results = explorer.compare_train_data(train_path)
    df_benign = pd.DataFrame(results['benign'])
    df_malignant = pd.DataFrame(results['malignant'])

    eda_service = EDAComparisonService(DescriptiveStatsComparator())
    comparison_report = eda_service.execute_analysis(df_benign, df_malignant)
    
    print("\n--- RESUMEN DE DIFERENCIAS (ORIGINALES) ---")
    print(comparison_report[['benign_avg', 'malignant_avg', 'percentage_change']].head(10))

    from eda.comparasion import EDAVisualizer
    EDAVisualizer.save_comparison_plots(
        df_benign, df_malignant, 
        metrics=['total_heterogeneity', 'avg_asymmetry_score', 'brightness'], 
        save_path=base_path / "images" / "eda_boxplots.png"
    )
    return explorer, eda_service

def step_data_augmentation(train_path, base_path, explorer, eda_service):
    """Aumento de datos físico (en disco)."""
    logger.info("=== STEP 4: DATA AUGMENTATION (OFFLINE) ===")
    aug_service = get_default_augmentation_service()
    aug_service.augment_directory(train_path / "benign", variants_per_image=3)
    aug_service.augment_directory(train_path / "malignant", variants_per_image=3)

    logger.info("=== STEP 4.5: POST-AUGMENTATION ANALYSIS ===")
    results_post = explorer.compare_train_data(train_path)
    df_benign_post = pd.DataFrame(results_post['benign'])
    df_malignant_post = pd.DataFrame(results_post['malignant'])
    
    comparison_report_post = eda_service.execute_analysis(df_benign_post, df_malignant_post)
    print("\n--- RESUMEN DE DIFERENCIAS (POST-AUGMENTATION) ---")
    print(comparison_report_post[['benign_avg', 'malignant_avg', 'percentage_change']].head(10))

    from eda.comparasion import EDAVisualizer
    EDAVisualizer.save_comparison_plots(
        df_benign_post, df_malignant_post, 
        metrics=['total_heterogeneity', 'avg_asymmetry_score', 'brightness'], 
        save_path=base_path / "images" / "eda_boxplots_augmented.png"
    )

def step_visualizations(train_path, base_path):
    """Generación de mosaicos de imágenes."""
    logger.info("=== STEP 5: SAVING VISUALIZATIONS ===")
    viewer = get_default_viewer()
    output_folder = base_path / "images"
    viewer.visualize_category(train_path / "benign", limit=20, save_path=output_folder / "summary_benign.png")
    viewer.visualize_category(train_path / "malignant", limit=20, save_path=output_folder / "summary_malignant.png")

def step_hyperparameter_search():
    """Búsqueda de mejores parámetros con Optuna."""
    logger.info("=== STEP 5.5: HYPERPARAMETER SEARCH (OPTUNA) ===")
    best_params_path = Path("models/cnn/best_hyperparams.json")
    if not best_params_path.exists():
        logger.info("No se encontraron hiperparámetros. Iniciando búsqueda...")
        run_study()
    else:
        logger.info(f"✅ Parámetros encontrados en {best_params_path}. Saltando búsqueda.")

def step_cnn_training():
    """Entrenamiento final del modelo CNN."""
    logger.info("=== STEP 6: CNN TRAINING ===")
    model_path = Path("models/cnn/best_model.pth")
    if model_path.exists():
        logger.info(f"✅ El modelo '{model_path}' ya existe. Saltando entrenamiento.")
    else:
        run_training()

# --- ORQUESTADOR PRINCIPAL ---

def run_pipeline():
    base_path = Path.cwd()
    data_path = base_path / "data"
    logger.info(f"Ruta de ejecución detectada: {base_path}")

    # Lista de ejecución modular: Comenta los pasos que no necesites
    train_path = step_setup_data(base_path, data_path)
    
    step_image_enhancement(train_path)
    
    explorer, eda_service = step_pattern_analysis_and_eda(train_path, base_path)
    
    step_data_augmentation(train_path, base_path, explorer, eda_service)
    
    step_visualizations(train_path, base_path)
    
    step_hyperparameter_search()
    
    step_cnn_training()

if __name__ == "__main__":
    try:
        run_pipeline()
        logger.info("🏁 Pipeline completado con éxito.")
    except Exception as e:
        logger.error(f"❌ Error crítico en el pipeline: {e}", exc_info=True)
