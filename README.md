## Contexto del Proyecto
Este es mi **primer proyecto** aplicando herramientas de Ciencia de Datos. El objetivo principal fue consolidar conocimientos en el ecosistema de **Python** y entender el ciclo de vida de un modelo de Machine Learning, enfrentando desafíos reales como la limpieza de datos y la interpretación de correlaciones bajas.

## Sobre el Proyecto
El proyecto busca predecir el índice de popularidad (0-100) de una canción en Spotify basándose en métricas del artista y de la pista. Fue un gran desafío de aprendizaje donde trabajé con datos históricos y actuales (2025).

## Tecnologías y Herramientas
* **Python**: Lenguaje principal.
* **Pandas & Numpy**: Manipulación y limpieza de datos.
* **Matplotlib & Seaborn**: Análisis exploratorio visual.
* **Scikit-Learn**: Implementación del modelo **Random Forest Regressor**.
* **Joblib**: Persistencia del modelo para uso futuro.

## Estructura del Repositorio
* `data/`: Contiene los archivos CSV originales y el dataset procesado.
* `src/`: 
    * `data_cleaning.py`: Script para la limpieza automática de datos.
    * `predict_popularity.py`: Script interactivo para usar el modelo.
    * `predictor_model.pkl`: El modelo ya entrenado.
* `notebooks/`: Cuadernos de Jupyter con todo el proceso de análisis y pruebas.

## Resultados del Modelo
A pesar de ser mi primer ejercicio de regresión, logré optimizar el modelo para obtener:
* **Error Absoluto Medio (MAE):** 8.02 puntos.
* Esto significa que el modelo predice la popularidad con un margen de error muy bajo, siendo capaz de distinguir entre un posible "Hit" y una canción con poca tracción.

## Cómo ejecutar el Predictor Interactivo
Si quieres probar el modelo con tus propios datos:

1. Cómo usar el Predictor
1.  Clona este repositorio:
    ```bash
    git clone [https://github.com/deed1502/spotify-popularity-predictor.git](https://github.com/deed1502/spotify-popularity-predictor.git)
    ```
2.  Instala las librerías necesarias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ejecuta el script interactivo:
    ```bash
    python src/predict.py
    ```