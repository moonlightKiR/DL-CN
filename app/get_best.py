import optuna
import json
import logging
from pathlib import Path

# Configuración de logs mínima
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def rescue_best_model():
    """Recata el mejor trial de la base de datos de Optuna y genera el JSON."""
    
    db_path = Path("log/optuna_study.db")
    
    if not db_path.exists():
        logger.error(f"❌ Error: No se encontró la base de datos en {db_path}")
        return

    try:
        # 1. Cargar el estudio desde la base de datos SQLite
        study = optuna.load_study(
            study_name="melanoma_optimization",
            storage=f"sqlite:///{db_path}"
        )
        
        best_params = study.best_params
        best_accuracy = study.best_value
        
        # 2. Guardar el JSON para que train_cnn.py lo lea
        best_params_path = Path("models/cnn/best_hyperparams.json")
        best_params_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        # 3. Reporte Final
        print("\n" + "="*50)
        print("💉 RESCATE DE RESULTADOS (OPTUNA)")
        print("="*50)
        print(f"Mejor Precisión hasta hoy: {best_accuracy:.2f}%")
        print(f"Trials completados:       {len(study.trials)}")
        print(f"Parámetros guardados:     {best_params}")
        print("="*50)
        print(f"✅ ¡JSON generado correctamente en {best_params_path}!")
        print("Ya puedes lanzar: uv run app/train_cnn.py")

    except Exception as e:
        logger.error(f"❌ Error al cargar el estudio: {e}")

if __name__ == "__main__":
    rescue_best_model()
