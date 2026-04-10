import logging
from pathlib import Path
from cnn import CNNConfig, MelanomaNet, MelanomaDataModule, MelanomaTrainer

# Configuración de logs para entrenamiento
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training():
    """Orquestador del entrenamiento de la CNN."""
    logger.info("=== INICIANDO PIPELINE DE ENTRENAMIENTO CNN ===")
    
    # 1. Configuración
    config = CNNConfig()
    config.load_best_params() # 👈 Carga automática de los mejores parámetros de Optuna
    
    # 2. Preparar Datos
    data_module = MelanomaDataModule(config)
    train_loader, val_loader = data_module.get_dataloaders(config.data_dir)
    
    # 3. Inicializar Modelo con parámetros optimizados
    model = MelanomaNet(
        num_classes=config.num_classes, 
        dropout_rate=config.dropout_rate
    )
    
    # 4. Entrenar
    trainer = MelanomaTrainer(model, config)
    trainer.fit(train_loader, val_loader)
    
    # 5. RESUMEN FINAL
    print("\n" + "="*50)
    print("📈 RESUMEN DEL MODELO Y ENTRENAMIENTO")
    print("="*50)
    print(f"Arquitectura:    MelanomaNet")
    print(f"Parámetros:      {model.count_parameters():,}")
    print(f"Estructura:\n{model}")
    print("-" * 50)
    print(f"Mejor Precisión: {trainer.best_accuracy:.2f}%")
    print("="*50)
    
    logger.info("=== ENTRENAMIENTO COMPLETADO EXITOSAMENTE ===")

if __name__ == "__main__":
    run_training()
