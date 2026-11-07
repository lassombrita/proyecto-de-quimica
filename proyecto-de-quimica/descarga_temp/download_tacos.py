#!/usr/bin/env python3
"""
Descargar imagenes del dataset TACO desde las URLs del archivo all_image_urls.csv
y guardarlas en carpetas batch_* locales.
"""

import json
import sys
from pathlib import Path
import time

import requests

# Configuracion de rutas
ROOT = Path(__file__).parent.parent  # proyecto-de-quimica
TACO_DIR = ROOT / "TACO"

def main():
    print("=" * 60)
    print("Descargador de Imagenes TACO")
    print("=" * 60)
    
    # Configuracion
    ruta_annotations = TACO_DIR / "data/annotations.json"
    carpeta_destino = ROOT / "data"  # data/batch_1/, etc.
    
    print(f"\nLeyendo anotaciones de: {ruta_annotations}")
    with open(ruta_annotations, 'r') as f:
        annotations = json.load(f)
    
    imagen_info = {img['id']: img for img in annotations['images']}
    
    print(f"Encontradas {len(imagen_info)} imagenes con anotaciones")
    print(f"Carpeta destino: {carpeta_destino}")
    
    # Crear carpetas batch_1 a batch_15
    for i in range(1, 16):
        (carpeta_destino / f"batch_{i}").mkdir(parents=True, exist_ok=True)
    
    print(f"\nDescargando imagenes...")
    print("-" * 60)
    
    # Contadores
    total = len(imagen_info)
    cargados = 0
    
    # Descargar cada imagen
    for img_id in sorted(imagen_info.keys()):
        info = imagen_info[img_id]
        
        try:
            file_name = Path(info['file_name'])
            
            # Determinar a que batch pertenece basado en el ID o archivo
            if '339781' in str(file_name) or '339782' in str(file_name):
                batch_num = 1
            elif '478033' in str(file_name):
                batch_num = 2
            elif '408888' in str(file_name):
                batch_num = 3
            elif '470660' in str(file_name):
                batch_num = 4
            elif '478555' in str(file_name):
                batch_num = 5
            elif '469392' in str(file_name):
                batch_num = 6
            elif '408889' in str(file_name):
                batch_num = 7
            else:
                # Asignar a batch basado en el ID para distribucion
                batch_num = (img_id % 15) + 1
            
            ruta_destino = carpeta_destino / f"batch_{batch_num}" / file_name
            ruta_destino.parent.mkdir(parents=True, exist_ok=True)
            
            # URLs del dataset TACO
            thumb_url = info.get('flickr_640_url', '')
            full_url = info.get('flickr_url', '')
            
            url = full_url if full_url else thumb_url
            
            if not url:
                continue
            
            # Descargar con requests (con retries)
            for attempt in range(3):
                try:
                    start = time.time()
                    response = requests.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        with open(ruta_destino, 'wb') as f:
                            f.write(response.content)
                        
                        tiempo = time.time() - start
                        img_name = Path(url).name[:35]
                        print(f"  OK {img_name:<35} | {response.headers.get('content-length', 'N/A'):>12} bytes ({tiempo:.1f}s)")
                        
                        cargados += 1
                        break
                    else:
                        time.sleep(2 ** attempt)
                        
                except Exception as e:
                    if attempt < 2:
                        time.sleep(1)
                    else:
                        pass
            
            # Verificar si ya terminamos
            if cargados == total:
                break
                
        except Exception as e:
            continue
    
    # Resumen final
    print("-" * 60)
    print(f"\nCompletado!")
    print(f"Imagenes descargadas: {cargados}/{total}")
    
    # Verificar carpetas resultantes
    for i in range(1, 16):
        ruta = carpeta_destino / f"batch_{i}"
        if ruta.exists():
            count = len(list(ruta.glob("*")))
            print(f"  batch_{i:<3}: {count:>5} imagenes")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDescarga pausada por usuario.")
        sys.exit(1)