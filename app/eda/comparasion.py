import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración de logging unificada similar a otros modelos del proyecto
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetComparator(ABC):
    """
    Interface Segregation / Open-Closed:
    Define una interfaz para diferentes tipos de análisis comparativos.
    """
    @abstractmethod
    def run(self, df_benign: pd.DataFrame, df_malignant: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        pass

class DescriptiveStatsComparator(DatasetComparator):
    """
    Single Responsibility: Se encarga exclusivamente de calcular 
    diferencias de medias y desviaciones típicas.
    """
    def run(self, df_benign: pd.DataFrame, df_malignant: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        results = {}
        for feature in features:
            if feature in df_benign.columns and feature in df_malignant.columns:
                mean_b = df_benign[feature].mean()
                mean_m = df_malignant[feature].mean()
                
                results[feature] = {
                    "benign_avg": float(mean_b),
                    "malignant_avg": float(mean_m),
                    "absolute_diff": float(abs(mean_m - mean_b)),
                    "percentage_change": float(((mean_m / mean_b) - 1) * 100) if mean_b != 0 else 0
                }
        return results

class EDAComparisonService:
    """
    Dependency Inversion: Depende de la abstracción DatasetComparator.
    Orquesta la comparación entre los dos DataFrames de forma segura.
    Clean Code: El usuario solo necesita interactuar con este servicio.
    """
    def __init__(self, comparator: DatasetComparator):
        self._comparator = comparator

    def execute_analysis(self, df_benign: pd.DataFrame, df_malignant: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta automáticamente las columnas numéricas comunes y ejecuta el comparador.
        """
        # Identificar características comunes (evitando metadatos)
        common_features = self._find_common_features(df_benign, df_malignant)
        
        if not common_features:
            logger.warning("No se encontraron columnas numéricas comunes para comparar.")
            return pd.DataFrame()

        logger.info(f"Comparando {len(common_features)} características comunes...")
        
        # Ejecutar comparación a través del comparador inyectado
        raw_report = self._comparator.run(df_benign, df_malignant, list(common_features))
        
        # Formatear el resultado como un DataFrame ordenado por importancia de la diferencia
        return (pd.DataFrame(raw_report).T
                .sort_values(by="absolute_diff", ascending=False))

    def _find_common_features(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Set[str]:
        # Solo comparamos columnas numéricas comunes entre ambos DataFrames
        num_df1 = set(df1.select_dtypes(include=['number']).columns)
        num_df2 = set(df2.select_dtypes(include=['number']).columns)
        
        return num_df1.intersection(num_df2)

class EDAVisualizer:
    """
    Single Responsibility: 
    Generar representaciones visuales comparativas de las métricas extraídas.
    """
    @staticmethod
    def save_comparison_plots(df_benign: pd.DataFrame, df_malignant: pd.DataFrame, metrics: List[str], save_path: Path):
        """Genera y guarda boxplots comparativos para las métricas indicadas."""
        # Filtramos solo las métricas que existen en ambos DataFrames
        valid_metrics = [m for m in metrics if m in df_benign.columns and m in df_malignant.columns]
        
        if not valid_metrics:
            logger.warning("No se han encontrado métricas válidas para graficar.")
            return

        # Combinamos los datasets para graficar con Seaborn
        df_b = df_benign.copy()
        df_b['class'] = 'benign'
        df_m = df_malignant.copy()
        df_m['class'] = 'malignant'
        df_total = pd.concat([df_b, df_m])

        # Creamos una figura con subplots horizontales
        n = len(valid_metrics)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        
        if n == 1:
            axes = [axes]

        for i, metric in enumerate(valid_metrics):
            sns.boxplot(x='class', y=metric, data=df_total, ax=axes[i], palette="Set2", hue='class', legend=False)
            axes[i].set_title(f'Distribución: {metric.replace("_", " ").title()}')
            axes[i].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        
        # Guardado físico
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"Reporte visual EDA guardado en: {save_path}")

if __name__ == "__main__":
    # Bloque de ejecución opcional si se lanza como script
    logger.info("EDA comparison module loaded.")
