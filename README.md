# Recomendador de Revistas Científicas  

---

## Descripción del proyecto

Este proyecto implementa un **sistema inteligente de recomendación de revistas científicas** basado en el contenido textual de artículos académicos.  
Dado el **título**, **abstract** y **palabras clave** de un artículo, el sistema es capaz de **recomendar la revista científica más adecuada** para su envío.

El sistema se ha desarrollado siguiendo dos enfoques complementarios:

- **Aproximación clásica** basada en NLP tradicional  
  (TF-IDF + SVM)
- **Aproximación conexionista** basada en modelos de lenguaje profundo  
  (BERT fine-tuned para clasificación)

Ambos enfoques se entrenan y evalúan sobre artículos reales publicados en revistas de la editorial **Elsevier** entre los años **2020–2024**.

---

## Objetivos

- Construir un recomendador automático de revistas científicas  
- Comparar modelos clásicos vs modelos basados en deep learning  
- Analizar las confusiones entre revistas con temáticas similares  
- Generar visualizaciones claras y comparables de los resultados  

---

## Estructura del proyecto

```
src/
├── train_classical.py     # Entrenamiento TF-IDF + SVM
├── train_bert.py          # Entrenamiento BERT
├── plots.py               # Funciones reutilizables de visualización
├── data_loader.py         # Carga y preprocesado de los datos
├── utils.py               # Funciones auxiliares (texto, etiquetas)
├── config.py              # Rutas y configuración global

data/
└── raw/
    ├── 1 Applied Ergonomics/
    ├── 2 Expert Systems with Applications/
    ├── 3 Neural Networks/
    └── 4 Journal of Visual Communication and Image Representation/

outputs/
└── reports/
    ├── confusion_matrix_classical.png
    ├── f1_scores_classical.png
    ├── accuracy_per_class_classical.png
    ├── confusion_matrix_bert.png
    ├── f1_scores_bert.png
    └── accuracy_per_class_bert.png

models/
├── classical/
│   └── model.joblib
└── bert/
    ├── final_model/
    └── label_encoder.joblib
```

---

## Formato de los datos

Cada revista contiene uno o varios archivos `.json` con la siguiente estructura:

```json
{
  "title": "...",
  "journal": "...",
  "year": "2021",
  "abstract": "...",
  "keywords": ["keyword1", "keyword2", "..."]
}
```

Cada carpeta representa **una clase (revista)**.

---

## Metodología

### Aproximación clásica (TF-IDF + SVM)

- Preprocesado del texto (concatenación de título, abstract y keywords)
- Representación vectorial mediante **TF-IDF**
- Clasificación con **Linear SVM**
- Evaluación mediante:
  - Matriz de confusión normalizada
  - F1-score por revista
  - Accuracy por revista

---

### Aproximación conexionista (BERT)

- Fine-tuning de **BERT base uncased**
- Clasificación multiclase
- Selección automática del mejor modelo mediante validación
- Evaluación con las **mismas métricas y visualizaciones** que el modelo clásico

---

## Visualizaciones generadas

Para cada enfoque se generan automáticamente:

- **Matriz de confusión normalizada**
- **F1-score por revista**
- **Accuracy por revista** (diagonal de la matriz de confusión)

---

## Ejecución

### 1️. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

### 2️. Entrenar modelo clásico

```bash
python src/train_classical.py
```

---

### 3️. Entrenar modelo BERT

```bash
python src/train_bert.py
```

---

## Resultados esperados

- El modelo clásico presenta un buen rendimiento en revistas con vocabulario especializado  
- Existen confusiones entre revistas con fuerte solapamiento temático  
- BERT reduce parte de estas confusiones gracias al modelado contextual del lenguaje  
