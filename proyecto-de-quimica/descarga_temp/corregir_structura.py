import shutil
from pathlib import Path
import re
import json

# Leer anotaciones para entender la estructura correcta
ruta_annotations = Path("TACO/data/annotations.json")
with open(ruta_annotations, 'r') as f:
    data = json.load(f)

imagenes_info = {img['id']: img for img in data['images']}
categorias = {str(cat['id']): cat['name'] for cat in data['categories']}

print("Corrigiendo estructura de carpetas batch...")
print("=" * 60)

# Mapeo de nombres de archivo a batches basado en ID del image
def asignar_batch(img_id, file_name):
    """Determinar a qué batch pertenece una imagen basada en su ID."""
    # Reglas aproximadas basadas en el dataset TACO original
    if 1 <= img_id <= 200: return 1
    elif 201 <= img_id <= 400: return 2
    elif 401 <= img_id <= 600: return 3
    elif 601 <= img_id <= 800: return 4
    elif 801 <= img_id <= 1000: return 5
    elif 1001 <= img_id <= 1200: return 6
    elif 1201 <= img_id <= 1400: return 7
    else: 
        # Para IDs fuera de rango, usar modulo
        batch = (img_id % 15) + 1
        return batch

# Obtener lista de archivos jpg existentes
archivos_jpg = list(Path("data").rglob("*.jpg"))
print(f"Encotrados {len(archivos_jpg)} archivos .jpg en carpetas")

# Mover cada archivo a su ubicación correcta
movidos = 0
errores = 0
for src in archivos_jpg:
    try:
        # Extraer ID de la imagen (el nombre del archivo puede tener prefijos duplicados)
        file_name = src.name
        img_id = None
        
        # Buscar ID en las anotaciones por nombre de archivo
        for rid, rinfo in imagenes_info.items():
            if rinfo.get('file_name', '') == file_name or file_name in rinfo.get('flickr_url', ''):
                img_id = rid
                break
        
        if img_id is None:
            # Intentar inferir batch basado en patrón del nombre
            match = re.search(r'(\d{5,6})\.jpg$', file_name)
            if match:
                num = int(match.group(1))
                # Asignar batch basado en rangos conocidos de TACO
                if num <= 200: batch_num = 1
                elif num <= 400: batch_num = 2
                elif num <= 600: batch_num = 3
                elif num <= 800: batch_num = 4
                elif num <= 1000: batch_num = 5
                elif num <= 1200: batch_num = 6
                elif num <= 1400: batch_num = 7
                else: batch_num = (num % 15) + 1
            else:
                continue
            
            # Determinar carpeta destino correcta (sin duplicados)
            batch_name = f"batch_{batch_num}"
            ruta_origen = src.parent
            ruta_destino = Path("data") / batch_name / file_name
            
            if str(ruta_destino) != str(src):
                ruta_destino.parent.mkdir(parents=True, exist_ok=True)
                
                # Eliminar de origen si existe duplicado
                shutil.move(str(src), str(ruta_destino))
                print(f"  OK {src.name[:35]:<35} -> data/{batch_name}/{file_name}")
                movidos += 1
                
    except Exception as e:
        errores += 1

print("=" * 60)
print(f"CORRECCION COMPLETADA:")
print(f"  Movidos: {movidos}")
print(f"  Errores: {errores}")