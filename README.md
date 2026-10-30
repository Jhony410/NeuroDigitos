# NeuroDígitos — Reconocimiento de dígitos manuscritos (0–9) con un perceptrón multicapa

**Curso:** Redes Neuronales · **Actividad:** PA1 — Trabajo de la unidad 1
**Estudiante:** Jhon Elias Aguilar Anccori · Universidad Nacional del Altiplano de Puno

<p align="center">
  <img src="results/screenshots/04_digito_7.png" alt="Interfaz reconociendo el dígito 7" width="520">
</p>

## ¿Qué hace este proyecto?

NeuroDígitos es un programa en Python que:

1. **Entrena** un **perceptrón multicapa (MLP)** con TensorFlow/Keras sobre el dataset
   **MNIST** (70 000 imágenes de dígitos escritos a mano, 28×28 px).
2. **Evalúa** el modelo en el conjunto de prueba oficial y genera métricas, matriz de
   confusión, curvas de aprendizaje y ejemplos de aciertos/errores.
3. Ofrece una **interfaz gráfica (Tkinter)** en la que el usuario **dibuja un dígito con
   el mouse** y la red lo reconoce, mostrando:
   - el dígito predicho y su confianza,
   - las probabilidades de las 10 clases (barras),
   - la imagen de 28×28 que realmente recibe la red tras el preprocesamiento.
4. Permite **reconocer dígitos desde archivos de imagen** por línea de comandos (`predict.py`).

> **Alcance:** reconoce **un solo dígito por imagen** (0–9). No reconoce números de varias
> cifras, letras ni fotografías arbitrarias.

## Resultados

| Métrica (conjunto de prueba oficial, 10 000 imágenes) | Valor |
|---|---|
| Exactitud | **97,90 %** |
| Pérdida (entropía cruzada) | 0,0710 |
| F1 macro | 0,9788 |
| Errores | 210 de 10 000 |
| Mejor época (EarlyStopping) | 12 de 17 |
| Parámetros entrenables | 109 386 |
| Tiempo de entrenamiento (CPU) | ≈ 35 s |

Los valores pueden variar ligeramente en otra computadora (ver [Reproducibilidad](#reproducibilidad)).

## Capturas

### Interfaz gráfica

| Inicio | Dígito 3 | Dígito 8 |
|---|---|---|
| ![Inicio](results/screenshots/02_inicio.png) | ![Dígito 3](results/screenshots/04_digito_3.png) | ![Dígito 8](results/screenshots/04_digito_8.png) |

| Dígito pequeño en la esquina (se recentra) | Fuera de alcance: «12» | Modelo ausente |
|---|---|---|
| ![Esquina](results/screenshots/07_digito_pequeno_esquina.png) | ![12](results/screenshots/06_fuera_de_alcance_12.png) | ![Sin modelo](results/screenshots/01_modelo_ausente.png) |

Montaje de los 10 dígitos reconocidos en la interfaz:

![Montaje 0-9](results/figures/montaje_gui_0_9.png)

### Entrenamiento y evaluación

| Curvas de aprendizaje | Matriz de confusión |
|---|---|
| ![Curvas](results/figures/curvas_aprendizaje.png) | ![Matriz](results/figures/matriz_confusion.png) |

| Pipeline de preprocesamiento del dibujo |
|---|
| ![Preprocesamiento](results/figures/pipeline_preprocesamiento.png) |

| Ejemplos de MNIST | Ejemplos mal clasificados |
|---|---|
| ![MNIST](results/figures/ejemplos_mnist.png) | ![Errores](results/figures/ejemplos_incorrectos.png) |

## Arquitectura

### Arquitectura de la red neuronal (MLP)

```mermaid
flowchart LR
    A["Entrada<br/>imagen 28×28"] --> B["Flatten<br/>784"]
    B --> C["Dense 128<br/>ReLU · He normal"]
    C --> D["Dropout 0.2"]
    D --> E["Dense 64<br/>ReLU · He normal"]
    E --> F["Dense 10<br/>Softmax"]
    F --> G["P(0) … P(9)"]
```

| Hiperparámetro | Valor |
|---|---|
| Optimizador | Adam (lr = 0,001) |
| Pérdida | Sparse categorical cross-entropy |
| Tamaño de lote | 128 |
| Épocas máximas | 50, con EarlyStopping (paciencia 5, restaura mejores pesos) |
| Validación | 10 % del entrenamiento, estratificado (54 000 / 6 000) |
| Semilla | 42 |

Todos los hiperparámetros están centralizados en `digit_recognizer/config.py`.

### Arquitectura del software

```mermaid
flowchart TB
    subgraph Scripts["Puntos de entrada"]
        TRAIN["train.py"]
        EVAL["evaluate.py"]
        APP["app.py"]
        PRED["predict.py"]
    end

    subgraph Paquete["Paquete digit_recognizer/"]
        CFG["config.py<br/>rutas e hiperparámetros"]
        DATA["data.py<br/>carga MNIST, normaliza,<br/>split estratificado"]
        MODEL["model.py<br/>construye el MLP"]
        TRN["training.py<br/>entrenamiento + EarlyStopping"]
        EVM["evaluation.py<br/>métricas y figuras"]
        PRE["preprocessing.py<br/>dibujo → 28×28 estilo MNIST"]
        INF["inference.py<br/>carga modelo y predice"]
        GUI["gui.py<br/>interfaz Tkinter"]
    end

    subgraph Artefactos["Archivos"]
        NPZ[("data/mnist.npz")]
        KER[("models/mlp_mnist.keras")]
        RES[("results/<br/>métricas, figuras, capturas")]
    end

    TRAIN --> TRN
    TRN --> DATA & MODEL
    DATA --> NPZ
    TRN -->|guarda| KER
    TRN -->|history.json| RES

    EVAL --> EVM
    EVM --> DATA & INF
    EVM -->|figuras y métricas| RES

    APP --> GUI
    GUI --> PRE & INF
    PRED --> PRE & INF
    INF -->|carga| KER

    CFG -.-> DATA & MODEL & TRN & EVM & INF & GUI
```

### Flujo de reconocimiento en la interfaz

```mermaid
sequenceDiagram
    actor U as Usuario
    participant G as gui.py
    participant P as preprocessing.py
    participant I as inference.py
    participant M as MLP (Keras)

    U->>G: Dibuja en el lienzo 280×280 y pulsa «Reconocer»
    G->>P: preprocess_canvas(imagen)
    Note over P: Recorta el trazo → escala a caja 20×20<br/>→ pega en 28×28 → centra por centro de masa<br/>→ normaliza a [0, 1]
    P-->>G: arreglo 28×28
    G->>I: predict_probabilities(modelo, lote)
    I->>M: forward
    M-->>I: 10 probabilidades (softmax)
    I-->>G: probabilidades
    G-->>U: Dígito predicho, barras y vista 28×28
```

### Estructura de carpetas

```
NeuroDigitos/
├── app.py                    # abre la interfaz gráfica
├── train.py                  # entrena y guarda el modelo
├── evaluate.py               # evalúa en prueba y genera figuras/métricas
├── predict.py                # reconoce un dígito desde un archivo de imagen
├── requirements.txt
├── digit_recognizer/
│   ├── config.py             # rutas e hiperparámetros
│   ├── data.py               # carga de MNIST, normalización, validación estratificada
│   ├── model.py              # arquitectura del MLP
│   ├── training.py           # entrenamiento con EarlyStopping
│   ├── evaluation.py         # métricas, matriz de confusión, curvas
│   ├── preprocessing.py      # dibujo -> imagen 28x28 estilo MNIST
│   ├── inference.py          # carga del modelo y predicción
│   └── gui.py                # interfaz Tkinter
├── tests/
│   ├── test_project.py       # pruebas automáticas (pytest)
│   └── test_regressions.py   # pruebas de regresión (unittest)
├── scripts/                  # capturas automáticas (Linux) y figuras del informe
├── data/mnist.npz            # copia local de MNIST
├── models/mlp_mnist.keras    # modelo entrenado (incluido)
└── results/                  # métricas (.json/.txt), figuras y capturas
```

## Paso a paso: instalación y ejecución

### Requisitos

- **Python 3.12 o 3.13** (64 bits). En Windows, el instalador oficial de
  [python.org](https://www.python.org/downloads/) ya incluye **Tkinter**.
- **Git** (para clonar el repositorio).
- ~1,5 GB libres para TensorFlow.

### 1. Clonar el repositorio

```bash
git clone https://github.com/Jhony410/NeuroDigitos.git
cd NeuroDigitos
```

### 2. Crear el entorno virtual e instalar dependencias

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> En Linux, si falta Tkinter: `sudo apt install python3-tk`.

### 3. Abrir la interfaz gráfica (`app.py`)

El modelo entrenado ya está incluido en `models/mlp_mnist.keras`, así que **no es
necesario entrenar** para usar la aplicación:

```powershell
# Windows
.\.venv\Scripts\python.exe app.py
```

```bash
# Linux / macOS (con el entorno activado)
python app.py
```

Uso de la ventana:

| Acción | Cómo |
|---|---|
| Dibujar | Mantener el botón izquierdo del mouse sobre el lienzo negro |
| Reconocer | Botón **Reconocer** o tecla **Enter** |
| Limpiar | Botón **Limpiar** o tecla **Esc** |

A la derecha se muestran el dígito predicho, la confianza, las probabilidades de cada
clase y la imagen 28×28 que recibe la red.

### 4. (Opcional) Reentrenar el modelo

```powershell
python train.py               # entrena (≈ 1 min en CPU) y guarda models/mlp_mnist.keras
python train.py --epochs 5    # prueba rápida
python train.py --evaluate    # entrena y luego evalúa
```

> Reentrenar sobrescribe el modelo y los resultados de `results/`.

### 5. (Opcional) Evaluar y generar figuras

```powershell
python evaluate.py
```

Genera `results/test_metrics.json`, `results/classification_report.txt` y las figuras en
`results/figures/`.

### 6. (Opcional) Reconocer desde un archivo de imagen

```powershell
python predict.py imagen.png
```

Devuelve un JSON con el dígito y las diez probabilidades. Use un único dígito aislado,
con fondo uniforme y buen contraste. Los PNG transparentes se componen sobre blanco.

### 7. (Opcional) Ejecutar las pruebas

```powershell
python -m pytest -v
python -m unittest discover -s tests -p test_regressions.py -v
```

> En Windows, sustituya `python` por `.\.venv\Scripts\python.exe` si no activó el entorno.

## Sobre los datos

Si existe `data/mnist.npz` se usa esa copia local. Si se borra, Keras descarga MNIST
automáticamente la primera vez (requiere internet) y lo guarda en `~/.keras/datasets`.

## Reproducibilidad

Se fija la semilla 42 (Python, NumPy y TensorFlow) y se activa el determinismo de
operaciones. Aun así, otro sistema operativo, otro procesador/GPU o versiones distintas
de las librerías pueden producir diferencias pequeñas en los números finales. El
experimento original usó Python 3.12 / TensorFlow 2.19.1; la revisión en Windows usa
Python 3.13 / TensorFlow 2.21.0. Las métricas guardadas incluyen fecha, entorno y SHA-256
del modelo.

## Capturas automáticas (solo Linux)

`scripts/capture_screenshots.py` mueve el mouse real con `xdotool` sobre un servidor
gráfico virtual (Xvfb) para dibujar los dígitos 0–9 y guardar capturas. En Windows basta
con usar `python app.py` y dibujar a mano.
