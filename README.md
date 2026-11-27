# TP2 - Sistema de Detección y Clasificación de Razas de Perros

Sistema de reconocimiento de razas con tres etapas:

**Embeddings → Búsqueda por similitud → Clasificación → Detección → Pipeline completo**

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md) (versión PDF: [Trabajo Práctico N°2.pdf](<Trabajo Práctico N°2.pdf>)).

La cátedra provee la infraestructura completa (Docker, aplicación Gradio, base de datos vectorial, API, scripts y herramientas de evaluación y visualización). El grupo hace un **fork** de este repositorio, trabaja en una **feature branch** y completa **únicamente** las funciones indicadas en cada etapa. Cualquier modificación fuera de esas funciones debe justificarse en el Pull Request.

## Objetivo del backend

API asincrónica en Python que permite:

- Búsqueda por similitud (`/search`): imagen consultada, Top-K imágenes similares y raza predicha.
- Clasificación supervisada (`/classify`): raza y score con el modelo entrenado seleccionado.
- Detección + clasificación (`/detect`): bounding boxes, raza y scores de confianza.
- Consulta del estado del procesamiento asincrónico (`/status/{job_id}`).
- Selección dinámica del modelo de embeddings (`/models` + campo `model` en `/search`).

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-dog-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Etapa 2: entrenamiento, evaluación (incl. NDCG@10 de la Etapa 1) y comparación
├── ENUNCIADO.md                 # Enunciado completo del TP
├── Trabajo Práctico N°2.pdf     # Enunciado en PDF
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra (una pestaña por etapa)
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /search, /classify, /detect, /status, /models, /files
│       ├── bootstrap.py         # Construcción de los servicios (compartida por la API y los scripts)
│       ├── config.py            # Configuración vía .env
│       ├── files.py
│       ├── schemas.py
│       ├── evaluation/
│       │   └── metrics.py       # NDCG@k, precision/recall/F1, specificity
│       ├── visualization/
│       │   └── draw.py          # Dibujo de bounding boxes
│       ├── services/
│       │   ├── similarity_service.py   # Extracción y búsqueda de embeddings
│       │   ├── classifier_service.py   # Entrenamiento y evaluación
│       │   ├── detection_service.py    # Detección YOLO y clasificación
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py      # Base vectorial en JSON
│           └── pgvector_store.py       # Base vectorial en PostgreSQL + pgvector
├── scripts/
│   ├── download_dataset.py      # Descarga el dataset de Kaggle
│   ├── build_index.py           # Indexa el dataset en la base vectorial
│   └── train_classifier.py      # Entrena/evalúa el clasificador (alternativa local a 2-train.ipynb)
├── data/
│   ├── dataset/                 # 70 Dog Breeds Image Dataset, no se versiona
│   └── embeddings.json          # Base vectorial JSON (si USE_PGVECTOR=false)
├── models/                      # Checkpoints entrenados (.pth / .onnx), no se versionan
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente

## Requisitos

- Python 3.12
- Docker (Docker Desktop incluye Docker Compose)
- Git
- Cuenta de Kaggle (para descargar el dataset)

## Dataset

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

Datos del grupo: [carpeta de distribución del TP](https://github.com/laureanodiez/computervision/tree/main/tuia-dog-recognition-app/data). La descarga del grupo debe conservar la estructura de `data/dataset/`, con el conjunto independiente, sus metadatos, las escenas de detección y el archivo `splits_finales.csv`.

```bash
python scripts/download_dataset.py
```

Si Kaggle solicita autenticación, usá las credenciales de tu cuenta. Si no las tienen, descarguen el dataset manualmente desde Kaggle y descompriman `train/`, `valid/` y `test/` dentro de `data/dataset`.

## Configurá tus modelos

Entrená tus modelos (Etapa 2) con `2-train.ipynb` (en Google Colab, con GPU) y guardá los checkpoints dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Actualizá la configuración del `.env` correspondiente (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `SIMILARITY_THRESHOLD`) para que apunte a tus modelos.

> El backend arranca aunque falten los checkpoints: `/health` informa cuáles encuentra en `models/`. Las pestañas de las Etapas 2 y 3 fallan hasta que existan.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

Es la forma en que la cátedra evalúa el trabajo.

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

El backend usa PyTorch para CPU dentro de Docker. El entrenamiento en un kernel con CUDA usa la instalación de PyTorch de ese kernel. En redes que inspeccionan HTTPS, BuildKit admite el secreto opcional `pip_ca` con las raíces públicas de confianza en formato PEM durante la instalación de dependencias.

Servicios disponibles: frontend en `http://localhost:8080`, backend en `http://localhost:8000` y PostgreSQL en `localhost:5432`.

### 2. Indexá la galería

Con la Etapa 1 implementada y el dataset en `data/dataset`, indexá el split de entrenamiento desde el contenedor del backend:

```bash
docker compose exec backend python /scripts/build_index.py --split train
```

## Opción 2 - Configurando el ambiente local

### 1. Cloná tu fork

```bash
git clone <url-de-tu-fork>
cd tuia-dog-recognition-app
git checkout -b feature/tp2-razas-perros
```

### 2. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/). También pueden usar `python3.12 -m venv .venv` y `pip`.

### 3. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 4. Activá el ambiente virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 5. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 6. Configurá las variables de entorno

Ajustá el archivo `.env.local.example` a tus necesidades y copialo a `src/.env`. El backend local y los scripts de `scripts/` leen la configuración desde ese archivo.

```bash
cp .env.local.example src/.env
```

### 7. Descargá el dataset

```bash
python scripts/download_dataset.py
```

### 8. Iniciá la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (por defecto). Con `USE_PGVECTOR=false` se usa la base vectorial JSON (`data/embeddings.json`) y no hace falta Docker para trabajar en local.

### 9. Iniciá el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 10. Iniciá el frontend (en otra terminal, con el ambiente virtual activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 11. Verificá que todo esté corriendo

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/docs` (Swagger) y `http://localhost:8000/health` (muestra el modelo seleccionado y los checkpoints encontrados)

## Scripts provistos

Se ejecutan desde la raíz del repositorio con el ambiente local activo (y PostgreSQL corriendo si `USE_PGVECTOR=true`). Leen la configuración de `src/.env` si existe:

```bash
# Descargar el dataset de Kaggle a data/dataset
python scripts/download_dataset.py

# Indexar la galería en la base vectorial (requiere la Etapa 1 implementada)
python scripts/build_index.py --split train
# ... o lo mismo desde el contenedor del backend (con docker compose up -d corriendo)
docker compose exec backend python /scripts/build_index.py --split train

# Entrenar y evaluar el clasificador localmente (alternativa a 2-train.ipynb)
python scripts/train_classifier.py --model resnet18_finetuned
```

Con `USE_PGVECTOR=false`, la indexación del split completo es lenta: el JSON se reescribe en cada inserción. Para pruebas rápidas, usen `--limit` (máximo de imágenes por raza).

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten el modelo de embeddings (`EMBEDDING_MODEL`, `EMBEDDING_DIM`), la búsqueda (`SIMILARITY_METRIC`, `SIMILARITY_THRESHOLD`, `TOP_K`), los checkpoints (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`), YOLO (`YOLO_MODEL`, `YOLO_CONF_THRESHOLD`, `YOLO_DOG_CLASS_ID`) y los paths.
3. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

| Endpoint | Descripción |
|----------|-------------|
| `POST /upload` | Sube una imagen al servidor |
| `POST /search` | Etapa 1: búsqueda por similitud (acepta `model` y `top_k`) |
| `POST /classify` | Etapa 2: clasificación con el modelo entrenado (acepta `model`) |
| `POST /detect` | Etapa 3: detección + clasificación |
| `GET /status/{job_id}` | Estado del procesamiento asincrónico |
| `GET /models` | Modelos de embeddings disponibles |
| `GET /health` | Estado del backend y checkpoints encontrados |

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/search`, `/classify`, `/detect`, `/status`, `/models`) y manejo de jobs.
- Orquestación de cada etapa: `search` e `index_image` (Etapa 1), `classify_image` (Etapa 2) y `predict` (Etapa 3: detección → recorte → clasificación → JSON).
- Selección dinámica del modelo de embeddings: `baseline`, `resnet18_finetuned` o `cnn_custom`.
- Carga de checkpoints (`.pth` / `.onnx`) desde `models/` y selección del modelo activo.
- Similitud configurable (`cosine` o `l2`) y manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Funciones auxiliares de evaluación (`lib/evaluation/metrics.py`) y de visualización (`lib/visualization/draw.py`).
- Frontend Gradio con una pestaña por etapa: Etapa 1 (imagen consultada, Top-K similares, raza predicha y selección de modelo), Etapa 2 (raza y score con el modelo entrenado) y Etapa 3 (bounding boxes, razas y scores).

## Datos y reproducibilidad

1. Desde la raíz, ejecutar `python scripts/download_dataset.py`. Los datos se organizan por raza en `data/dataset/{train,valid,test}`.
2. Colocar las imágenes independientes y su `metadata.csv` en `data/dataset/independent/`. Si el conjunto verificado está en otra carpeta del dataset, indicar su nombre con `INDEPENDENT_DIR`. El CSV registra selección, carpeta de raza, ruta local, página original, autor y licencia.
3. Ejecutar `1-pre-processing.ipynb` desde la raíz. Produce `data/dataset/splits_finales.csv` con rutas relativas, raza y split. La galería, los clasificadores y la evaluación leen el mismo archivo. La consulta del historial de candidatas es opcional.
4. Ejecutar `2-train.ipynb` con ese manifiesto. Para la evaluación del detector, colocar `data/dataset/independent_scenes.csv` con las columnas `path,kind,expected_dogs,source_url,license` y los archivos correspondientes. Incluir escenas de un perro, varios perros, escenas complejas y sin perros.
5. Los checkpoints completos van en `models/`. La configuración seleccionada por validación se escribe en `output/config_recomendada.json`. Trasladar los valores a ambos archivos de entorno, incluido `CLASSIFIER_MODEL`, y reconstruir la galería. El modelo de consulta debe coincidir con el del índice.
6. Verificar las tres pestañas en Gradio y el estado de cada operación en `/status/{job_id}`. Para correr las pruebas locales, instalar `pytest` en el entorno del proyecto y ejecutar `python -m pytest -q` desde la raíz.

## Resultados guardados

La primera notebook registra 9.346 imágenes de 70 razas antes del filtrado. De las 809 descartadas, 767 corresponden a la regla de duplicados, 5 a desenfoque y 37 a exposición, contando una sola causa por imagen. Los pares con alta similitud semántica requieren revisión visual porque distintas fotos de una misma raza pueden parecerse.

| Split | Imágenes | Razas |
|---|---:|---:|
| Train | 7.182 | 70 |
| Valid | 666 | 70 |
| Test | 689 | 70 |
| Independiente | 163 | 21 |

La ficha del dataset principal en Kaggle declara CC0 (consulta del 27/11/2026). La selección independiente contiene 184 fotos de 23 razas. Se mapearon las etiquetas Yorkshire Terrier a Yorkie y se separaron 18 fotos de razas ausentes en Kaggle, dos escenas para detección y una imagen de un par marcado por dHash. Las 163 consultas finales cubren 21 razas. Las 74 filas sin autor consignado en ese manifiesto figuran como CC0. La sección 8 del primer notebook genera el manifiesto de escenas y añade una foto de gato con licencia CC0 como control sin perros.

## Modelos

Descargá `resnet18_finetuned.pth` y `cnn_custom.pth` desde la [carpeta de modelos del grupo](https://github.com/laureanodiez/computervision/tree/main/tuia-dog-recognition-app/models) y colocá ambos archivos directamente en `models/` antes de levantar Docker.

La búsqueda inicial usa las 512 características de ResNet18 ImageNet después del pooling global. La clasificación compara ResNet18 con backbone congelado o ajuste completo y una CNN de cuatro bloques convolucionales con BatchNorm, pooling global y una capa de embedding. Los dos modelos guardan el orden de las razas en el checkpoint. El preprocesamiento compartido convierte BGR a RGB, redimensiona y normaliza con las estadísticas de ImageNet. La rotación, el flip horizontal, los cambios de brillo y contraste, el desenfoque y el ruido se aplican en entrenamiento.

La ResNet18 de ajuste completo obtuvo 87,8 % de accuracy en validación, 85,9 % en test y 47,9 % en el independiente. La CNN propia obtuvo 26,1 % en test y 8,6 % en el independiente. La búsqueda usa embeddings de ResNet18 de 512 componentes, L2 y 20 vecinos para el voto. El NDCG@10 fue 0,795 en test y 0,458 en el independiente. El umbral de similitud elegido fue cero. El clasificador trabaja con imágenes de 96 × 96 y YOLOv8n usa un umbral de 0,25. Los checkpoints finales son `models/resnet18_finetuned.pth` y `models/cnn_custom.pth`.

Los cuatro experimentos usaron cuatro épocas, lotes de 32 y semilla 42. Se optimizó la entropía cruzada con AdamW, regularización de pesos de 0,0001 y label smoothing de 0,1. En ResNet se comparó congelar el backbone con ajustarlo usando una tasa inicial de 0,0001 para el backbone y 0,001 para la cabeza. En la CNN se compararon valores de dropout de 0,3 y 0,5. Para cada corrida se conservó la época con mayor accuracy de validación.

## Problemas encontrados y soluciones

- La enumeración directa de carpetas reincorporaba imágenes descartadas. Entrenamiento, evaluación e indexación leen las rutas de `splits_finales.csv`.
- Repetir la indexación agregaba vecinos duplicados. El script reemplaza la galería y el buscador comprueba que el extractor de consulta coincida con el del índice.
- El backend en Docker intentaba conectarse a PostgreSQL en localhost. Compose carga el entorno del contenedor con el host `postgres` y verifica la disponibilidad de la API en `/health`.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- `EMBEDDING_DIM` debe coincidir con la dimensión del embedding del modelo elegido. Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- La base vectorial guarda los embeddings de un modelo a la vez: al cambiar de modelo de embeddings hay que reindexar (`scripts/build_index.py`).
- Las rutas de la galería se guardan relativas a `DATA_PATH` (por ejemplo, `dataset/train/<raza>/<imagen>.jpg`): una base indexada desde el entorno local también sirve para el backend en Docker, porque comparten PostgreSQL y la carpeta `data/`.
- Los modelos (`*.pth`, `*.pt`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen los checkpoints en `models/` y súbanlos a un link **público de solo lectura** para que los docentes puedan descargarlos.
- Los pesos preentrenados (torchvision/timm y YOLO) se descargan en el primer uso, por lo que se necesita conexión a internet.

## Entrega

- Pull Request desde su feature branch hacia `main` **de su propio repositorio**, **sin mergear**, con el link subido al campus. No abran el PR contra el repositorio de la cátedra.
- Si el repositorio es privado, los docentes deben tener acceso.
- Notebooks `1-pre-processing.ipynb` y `2-train.ipynb` ejecutadas, con sus salidas.
- Para aprobar, el sistema debe andar sin errores con:

```bash
docker compose build
docker compose up
```

**Fecha de entrega:** a definir por la cátedra. Solo se consideran los commits hasta esa fecha y hora; se recomienda trabajar de forma incremental.

El detalle del contenido del PR, la plantilla de descripción y el checklist están en [ENUNCIADO.md](ENUNCIADO.md#8-entrega-pull-request).
