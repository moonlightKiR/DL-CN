# Melanoma Deep Learning - Pipeline de Procesamiento

Este proyecto implementa un pipeline automatizado de extremo a extremo para el procesamiento, mejora y análisis exploratorio de un dataset de imágenes de melanoma (Benign vs Malignant).

## Arquitectura del Pipeline

El proceso principal se gestiona desde `app/main.py` y se divide en las siguientes etapas:

1. **STEP 1: Setup & Organization**: Descompresión de archivos ZIP y organización jerárquica en carpetas `train/test` y categorías `benigno/maligno`.
2. **STEP 1.5: Image Enhancement (FSRCNN + Laplacian)**:
   * **Super-Resolución**: Utiliza una Red Neuronal Convolucional Rápida (FSRCNN) implementada en PyTorch para escalar las imágenes a x4.
   * **Aceleración Hardware**: Optimizado para Apple Silicon (MPS), procesando la luminancia (canal Y) para máxima velocidad.
   * **Refinamiento**: Aplica un Filtro Laplaciano de OpenCV para resaltar bordes y estructuras celulares críticas en el diagnóstico.
3. **STEP 2: Pattern Analysis**: Extracción de características biométricas:
   * **Geometría**: Dimensiones y Relación de Aspecto.
   * **Color**: Distribución RGB, Brillo y Heterogeneidad (Regla ABCDE - C).
   * **Asimetría**: Puntuación de asimetría básica (Regla ABCDE - A).
4. **STEP 3: Exploratory Data Analysis (EDA)**: Generación automática de estadísticas comparativas y gráficos de caja (boxplots) para identificar patrones diferenciales entre categorías.
5. **STEP 4: Data Augmentation**: Aumento del dataset mediante transformaciones aleatorias (rotación, zoom, brillo) para mejorar el futuro entrenamiento de la CNN.
6. **STEP 5: Visualizations**: Generación de mosaicos de resumen de las imágenes procesadas.
7. **STEP 6: CNN Training**: Entrenamiento de la arquitectura MelanomaNet (optimizada para < 250k parámetros) mediante un pipeline basado en SOLID.

---

## Requisitos e Instalación

Este proyecto utiliza `uv`, el gestor de paquetes de Python más rápido del ecosistema, para garantizar la reproducibilidad y velocidad.

### 1. Instalar `uv`

Si no lo tienes instalado, puedes hacerlo en macOS con:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Configurar el Entorno

No necesitas crear entornos virtuales manualmente. `uv` se encarga de todo basándose en el archivo `pyproject.toml`:

```bash
uv sync
uv add optuna  # Requerido para la optimización inteligente
```

### 3. Modelos de Super-Resolución

Asegúrate de tener el archivo de pesos en la carpeta correcta:

* Ruta: `models/FSRCNN_x4.pth`
* Nota: El enlace de descarga para este modelo se encuentra en el archivo `models/link_model.md`.

---

## Uso del Pipeline

### 1. Procesamiento y Mejora de Datos

Para ejecutar el pipeline de procesamiento, super-resolución y EDA:

```bash
uv run app/main.py
```

### 2. Optimización de Hiperparámetros (Opcional)

Si deseas realizar una búsqueda inteligente de la mejor configuración (Learning Rate, Dropout, etc.) usando **Optuna**:

```bash
uv run app/hyperparam_search.py
```

*Este comando genera automáticamente el archivo `models/cnn/best_hyperparams.json` con los mejores resultados.*

### 3. Entrenamiento de la Red Neuronal

Para realizar el entrenamiento final de la arquitectura **MelanomaNet**:

```bash
uv run app/train_cnn.py
```

*Nota: Este script carga automáticamente los hiperparámetros del archivo JSON si existe. De lo contrario, utiliza los valores por defecto.*

### Comandos Útiles

* **Añadir una nueva librería**: `uv add nombre-libreria`
* **Ejecutar el script**: `uv run path/to/script.py`
* **Actualizar dependencias**: `uv lock --upgrade`

---

## Estructura del Proyecto

* `app/`: Código fuente del pipeline.
  * `image/`: Procesamiento y arquitectura de redes.
  * `eda/`: Análisis estadístico y aumento de datos.
* `data/`: Imágenes organizadas (Generado automáticamente).
* `images/`: Reportes visuales y gráficas generadas.
* `models/`: Pesos de modelos pre-entrenados.
