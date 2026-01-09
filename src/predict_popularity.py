import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

base_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_dir, '..', 'data', 'processed', 'spotify_total.csv')

model_path = os.path.join(base_dir, 'predictor_model.pkl')
columns_path = os.path.join(base_dir, 'model_columns.pkl')

def train_model():
    """Función para entrenar el modelo si no existe."""
    if not os.path.exists(data_path):
        print(f"Error: No se encuentra el dataset en {data_path}")
        return None, None
    
    df = pd.read_csv(data_path, index_col='track_id')
    X = df.drop(['track_name', 'artist_name', 'track_popularity'], axis=1)
    y = df['track_popularity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    error = mean_absolute_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)

    print(f'El modelo se equivoca, en promedio, por {error:.2f} puntos de popularidad.')
    print(f'Precisión del modelo (R2): {r2:.2f}')
    
    joblib.dump(modelo, model_path)
    joblib.dump(X.columns.tolist(), columns_path)

    print('Modelo guardado')
    return modelo, X.columns.tolist()

def predict_popularity(datos_nueva_cancion):
    modelo_cargado = joblib.load(model_path)
    columnas = joblib.load(columns_path)
    
    entrada = pd.DataFrame([datos_nueva_cancion], columns=columnas)
    resultado = modelo_cargado.predict(entrada)
    return resultado[0]


def load_model_and_predict():

    if os.path.exists(model_path) and os.path.exists(columns_path):
        modelo = joblib.load(model_path)
        columnas = joblib.load(columns_path)
    else:
        answer = input('\nNo se encontró el modelo. ¿Desea entrenarlo? (S/N)\n')
        if answer.upper() == 'S':
            modelo, columnas = train_model()
        else:
            return
            

    print('--- Spotify Popularity Predictor ---')
    print('Introduce los datos de la canción para calcular su popularidad estimada\n')

    try:
        explicit = int(input('¿Es explícita? (1 para Sí, 0 para No): '))
        art_pop = float(input('Popularidad actual del artista (0-100): '))
        followers = float(input('Número de seguidores del artista: '))
        total_tracks = int(input('Total de canciones en el álbum: '))
        duration = float(input('Duración de la canción (en minutos, ej: 3.5): '))

        datos_entrada = pd.DataFrame([[explicit, art_pop, followers, total_tracks, duration]], 
                                     columns=columnas)

        prediccion = modelo.predict(datos_entrada)[0]
        
        print(f'\nEl modelo estima que la canción tendrá una popularidad de {prediccion:.2f}/100')
        print('Nota: El margen de error promedio del modelo es de +/- 8.02 puntos')
        input('Presione enter para continuar')

    except ValueError:
        print('Error: Por favor, introduce solo valores numéricos válidos.')

if __name__ == '__main__':
    
    print('--- Spotify Popularity Predictor ---')
    while True:
        print('Bienvenido a Spotify Popularity Predictor')
        print('1. Entrenar modelo\n2. Hacer predicción\n0. Salir')
        option = input('Seleccione la opción que desea ejecutar: ')
        if option == '1':
            train_model()
        elif option == '2':
            load_model_and_predict()
        elif option == '0':
            print('¡Hasta luego!')
            break
        else:
            print('Por favor seleccione una opción válida')

