# Reconocimiento de Animales con IA

## 📂 Estructura del Proyecto

- **`main.py`**: Archivo principal para cargar el modelo y realizar predicciones.
- **`train.py`**: Define la arquitectura del modelo y el proceso de entrenamiento.
- **`utils.py`**: Funciones auxiliares para la carga de datos y preprocesamiento.
- **`images.py`**: Visualización de imágenes y batches del dataset.

---

## 🚀 `main.py`

Este archivo es el punto de entrada para realizar predicciones con el modelo entrenado.

### Funciones Clave:

- **`load_trained_model(path_pth, dataset)`**:
  Carga el modelo preentrenado desde un archivo `.pth` y lo configura en modo de evaluación.
  
- **`predict_image(image_path, model, device, classes)`**:
  Toma una imagen, la preprocesa, la pasa por el modelo y devuelve la clase predicha.

### Flujo de Trabajo:
1. Se carga el dataset para obtener las clases.
2. Se carga el modelo entrenado.
3. Se realiza la predicción sobre una imagen de prueba.

### Conceptos Clave:
- **Transformaciones de imágenes**: Asegura que las imágenes de prueba sean compatibles con el modelo.
- **Modo de Evaluación (`model.eval()`)**: Desactiva funciones como el Dropout para obtener resultados deterministas.

---

## 🧠 `train.py`

Este archivo contiene el código para definir y entrenar el modelo.

### Funciones Clave:

- **`get_model(dataset)`**:
  Carga un modelo **ResNet18** preentrenado y ajusta la última capa para adaptarse al número de clases del dataset.

- **`train_model(dataset, train_loader)`**:
  Realiza el proceso de entrenamiento usando **Cross-Entropy Loss** y el optimizador **Adam**.

### Conceptos Clave:
- **Transfer Learning**: Reutilización de un modelo preentrenado para una nueva tarea, lo que reduce el tiempo de entrenamiento.
- **Optimización (Adam)**: Algoritmo eficiente para ajustar los pesos del modelo.
- **Cross-Entropy Loss**: Métrica que mide la diferencia entre la predicción del modelo y la etiqueta real.

---

## 📦 `utils.py`

Proporciona funciones de apoyo para la carga y preprocesamiento de datos.

### Funciones Clave:

- **`downloadDataset()`**:
  Descarga el dataset desde Kaggle utilizando la librería `kagglehub`.

- **`loadDataset()`**:
  Carga el dataset descargado, aplica transformaciones de preprocesamiento y lo divide en conjuntos de entrenamiento y prueba.

### Conceptos Clave:
- **DataLoader**: Facilita la carga eficiente de datos en lotes durante el entrenamiento.
- **Transformaciones (`transforms`)**: Normalización, redimensionamiento y conversión de imágenes en tensores.

---

## 🖼️ `images.py`

Se encarga de la visualización de imágenes para verificar la calidad de los datos.

### Funciones Clave:

- **`imshow(image)`**:
  Desnormaliza y muestra una imagen usando `matplotlib`.

- **`visualize_batch(dataset, data_loader)`**:
  Visualiza un batch de imágenes con sus respectivas etiquetas para asegurarse de que la carga de datos sea correcta.

### Conceptos Clave:
- **Desnormalización**: Inversión de la normalización para que las imágenes se vean correctamente al mostrarlas.
- **Visualización**: Importante para validar que el preprocesamiento de datos sea el adecuado.

---

## 🔄 Flujo Completo del Proyecto

1. **Cargar y Preprocesar Datos (`utils.py`)**:
   - Descarga del dataset.
   - Transformación de imágenes.
   - División en conjuntos de entrenamiento y prueba.

2. **Entrenamiento del Modelo (`train.py`)**:
   - Definición del modelo ResNet18.
   - Ajuste de la última capa para el número de clases del dataset.
   - Entrenamiento con optimización y cálculo de pérdidas.

3. **Evaluación y Visualización (`images.py`)**:
   - Visualización de batches para ver la calidad de los datos.

4. **Predicción de Nuevas Imágenes (`main.py`)**:
   - Carga del modelo entrenado.
   - Predicción sobre imágenes nuevas.

---

## 🤔 Conceptos Avanzados

- **Transfer Learning**: Aprovechar modelos ya entrenados en grandes datasets para tareas específicas con menos datos.
- **Batch Processing**: Procesar múltiples imágenes al mismo tiempo para acelerar el entrenamiento.
- **GPU vs. CPU**: El uso de GPU permite cálculos más rápidos en redes neuronales.

---

## 📊 Mejora Continua

- Aumentar el número de épocas para mejorar la precisión.
- Probar diferentes optimizadores o tasas de aprendizaje.
- Implementar técnicas de data augmentation para mejorar la robustez del modelo.
