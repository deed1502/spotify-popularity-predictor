import pandas as pd
import os

def clean_data(classic_path, modern_path):
    """
    Lee y limpia los datasets de Spotify.
    
    :param classic_path: Path del csv con los datos de 2009 a 2023
    :param modern_path: Path del csv con los datos de 2025
    """
    df_classic = pd.read_csv(classic_path)
    df_modern = pd.read_csv(modern_path)
    
    df_classic['track_duration_min'] = df_classic['track_duration_ms']/60000

    columns_to_keep = [
        'track_id',
        'track_name',
        'artist_name', 
        'track_popularity',
        'explicit', 
        'artist_popularity',
        'artist_followers', 
        'album_total_tracks',
        'track_duration_min'
    ]
    
    df_classic_filtered = df_classic[columns_to_keep].copy()
    df_modern_filtered = df_modern[columns_to_keep].copy()
    df_final = pd.concat([df_modern_filtered, df_classic_filtered], ignore_index=True)
    df_final['explicit'] = df_final['explicit'].astype(int)
    df_final = df_final.dropna()
    df_final = df_final.set_index('track_id')
    return  df_final
    

if __name__ == '__main__':
    df_clean = clean_data('data/raw/track_data_final.csv', 'data/raw/spotify_data clean.csv')
    os.makedirs("data/processed", exist_ok=True)
    df_clean.to_csv('data/processed/spotify_total.csv')
    print(f"Datos guardados en: data/processed/spotify_total.csv")