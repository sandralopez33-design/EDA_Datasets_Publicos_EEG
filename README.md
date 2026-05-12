# Análisis Exploratorio de Datos (EDA) - Señales EEG y Espectrogramas

**Autora:** Sandra Gabriela Lopez Pacheco  
**Institución:** Universidad Autónoma de Baja California (UABC) - Facultad de Ingeniería, Arquitectura y Diseño (FIAD)  
**Profesor y Director de Tesis:** Dr. Everardo Inzunza González  

## 📌 Descripción del Proyecto
Este repositorio contiene los Notebooks de Jupyter / Scripts de Python correspondientes a la **Tarea Académica para Tesistas: Búsqueda, caracterización y análisis exploratorio de 5 datasets públicos**. 

El código fuente aquí alojado procesa bioseñales y extrae características en el dominio del tiempo y la frecuencia, culminando en la generación de **espectrogramas 2D**. Estas imágenes topográficas son la base para la metodología de tesis, donde se utilizarán arquitecturas de Deep Learning (CNN, Vision Transformers y LSTM) para detectar patrones asociados a problemas de salud mental (ansiedad y depresión).

## 🛠️ Requisitos e Instalación
Para ejecutar el entorno reproducible, se requiere **Python 3.11+** y las siguientes librerías:
- `mne` (Procesamiento neurofisiológico y submuestreo a 128 Hz)
- `pandas` y `numpy` (Manipulación de matrices)
- `matplotlib`, `seaborn` y `scipy` (Generación de gráficas y cálculo de Densidad Espectral)

## 📊 Visualizaciones Generadas
El script de Análisis Exploratorio (EDA) arroja las siguientes 5 visualizaciones exigidas en la rúbrica:
1. Gráfico de distribución y balanceo de las clases.
2. Histograma de variables clave (ej. edad o distribución clínica).
3. Trazado de la señal EEG cruda en el dominio del tiempo.
4. Matriz (Heatmap) de correlación espacial de Pearson entre electrodos.
5. **Espectrograma 2D** evaluando las frecuencias de 0 a 40 Hz.

🖼️ *Las capturas de pantalla de los resultados pueden visualizarse en la carpeta `/capturas` de este mismo repositorio.*
