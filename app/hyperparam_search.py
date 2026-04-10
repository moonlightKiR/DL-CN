import json
import optuna
import logging
import csv
from datetime import datetime
from pathlib import Path
from cnn import CNNConfig, MelanomaNet, MelanomaDataModule, MelanomaTrainer

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def objective(trial):
    """Función objetivo para Optuna."""
    # 1. Definir rangos de búsqueda inteligente
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    
    # Parámetros fijos para la búsqueda rápida
    config = CNNConfig(
        learning_rate=lr,
        weight_decay=wd,
        epochs=12,      # Épocas cortas para la búsqueda
        input_size=160
    )
    
    # 2. Preparar Datos (una vez por trial para evitar fugas)
    data_module = MelanomaDataModule(config)
    train_loader, val_loader = data_module.get_dataloaders(config.data_dir)
    
    # 3. Inicializar Modelo con parámetros del trial
    model = MelanomaNet(num_classes=2, dropout_rate=dropout)
    
    # 4. Entrenar
    trainer = MelanomaTrainer(model, config)
    trainer.fit(train_loader, val_loader)
    
    # Retornamos la mejor precisión obtenida para que Optuna la maximice
    return trainer.best_accuracy

def run_study():
    """Ejecuta el estudio de optimización de Optuna."""
    
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    db_path = log_dir / "optuna_study.db"
    
    logger.info(f"=== INICIANDO ESTUDIO CON OPTUNA (DB: {db_path}) ===")

    # Creamos el estudio (queremos maximizar la precisión)
    study = optuna.create_study(
        study_name="melanoma_optimization",
        direction="maximize",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=20)

    # REPORTE FINAL
    print("\n" + "="*50)
    print("🏆 MEJOR CONFIGURACIÓN ENCONTRADA")
    print("="*50)
    print(f"Mejor Precisión: {study.best_value:.2f}%")
    print(f"Parámetros óptimos:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # GUARDAR PARÁMETROS ÓPTIMOS AUTOMÁTICAMENTE
    best_params_path = Path("models/cnn/best_hyperparams.json")
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    # Guardar histórico de trials en CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = log_dir / f"optuna_results_{timestamp}.csv"
    study.trials_dataframe().to_csv(output_file, index=False)
    
    logger.info(f"🚀 Parámetros óptimos guardados en {best_params_path}")
    logger.info(f"✅ Estudio completado. Histórico guardado en {output_file}")

if __name__ == "__main__":
    run_study()
