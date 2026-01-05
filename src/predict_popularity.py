import pandas as pd
import joblib
import os

def predict_popularity(datos_nueva_cancion):
    modelo_cargado = joblib.load('../src/hit_predictor_model.pkl')
    columnas = joblib.load('../src/model_columns.pkl')
    
    entrada = pd.DataFrame([datos_nueva_cancion], columns=columnas)
    resultado = modelo_cargado.predict(entrada)
    return resultado[0]


def load_model_and_predict():

    if not os.path.exists('src/hit_predictor_model.pkl'):
        print('Error: No se encuentra el modelo guardado. Primero ejecute el entrenamiento.')
        return

    modelo = joblib.load('src/hit_predictor_model.pkl')
    columnas = joblib.load('src/model_columns.pkl')

    print('\n--- 🎧 Spotify Hit Predictor 🎧 ---')
    print('Introduce los datos de la canción para calcular su popularidad estimada.\n')

    try:
        explicit = int(input('¿Es explícita? (1 para Sí, 0 para No): '))
        art_pop = float(input('Popularidad actual del artista (0-100): '))
        followers = float(input('Número de seguidores del artista: '))
        total_tracks = int(input('Total de canciones en el álbum: '))
        duration = float(input('Duración de la canción (en minutos, ej: 3.5): '))

        datos_entrada = pd.DataFrame([[explicit, art_pop, followers, total_tracks, duration]], 
                                     columns=columnas)

        prediccion = modelo.predict(datos_entrada)[0]

        print(f'El modelo estima que la canción tendrá una popularidad de {prediccion:.2f}/100')
        print('Nota: El margen de error promedio del modelo es de +/- 8.02 puntos')

    except ValueError:
        print(' Error: Por favor, introduce solo valores numéricos válidos.')

if __name__ == '__main__':
    load_model_and_predict()

