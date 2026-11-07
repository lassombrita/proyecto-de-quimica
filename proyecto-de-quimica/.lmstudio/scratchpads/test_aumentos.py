from ingesta_publica import generar_variantes
import cv2
import numpy as np
import glob
import os

ruta_taco = 'TACO/data'
imagenes = sorted(glob.glob(f'{ruta_taco}/**/*.jpg', recursive=True))[:1]
imagenes.extend(sorted(glob.glob(f'{ruta_taco}/**/*.png', recursive=True))[:1])
img = cv2.imread(imagenes[0])

variantes = generar_variantes(img, 3, 42)
print(f'Variantes: {len(variantes)}')
print(f'Todos mismo tamaño: {all(v.shape == img.shape for v in variantes)}')
print(f'Sin recorte: {all(v.shape[:2] == img.shape[:2] for v in variantes)}')

# Guardar collage
os.makedirs('docs/img', exist_ok=True)
max_alto = max(v.shape[0] for v in variantes)
total_ancho = sum(v.shape[1] for v in variantes)
canvas = np.zeros((max_alto, total_ancho, 3), dtype=np.uint8)
for i, var in enumerate(variantes):
    canvas[:var.shape[0], i*var.shape[1]: (i+1)*var.shape[1]] = var
cv2.imwrite('docs/img/ejemplo_aumentos.png', canvas)
print(f'Collage guardado: docs/img/ejemplo_aumentos.png')

# Verificar reproducibilidad
variantes2 = generar_variantes(img, 3, 42)
son_identicas = all((np.array(v1) == np.array(v2)).all() for v1, v2 in zip(variantes, variantes2))
print(f'Reproducible con semilla=42: {son_identicas}')

print(f'Dimensiones collage: {canvas.shape}')
print(f'Original: {img.shape}')