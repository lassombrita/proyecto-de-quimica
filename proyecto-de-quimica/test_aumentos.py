#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pruebas automáticas de Albumentations v2.x

Esta carpeta contiene pruebas unitarias para el sistema de clasificación.
No requieren datasets externos - generan imágenes sintéticas y datos JSON ficticios.

Ejecución:
    python -m pytest tests/ -q --tb=short
"""

from ingesta_publica import generar_variantes
import cv2
import numpy as np
import glob
import os

ruta_taco = 'TACO/data'
imagenes = sorted(glob.glob(f'{ruta_taco}/**/*.jpg', recursive=True))[:1]
imagenes.extend(sorted(glob.glob(f'{ruta_taco}/**/*.png', recursive=True))[:1])
img = cv2.imread(imagenes[0])

# Con n_variedades=3 deberíamos tener 3 variantes (original + 2 aumentadas)
variantes = generar_variantes(img, 3, 42)
print(f'Con n=3: {len(variantes)} variantes')
for i, var in enumerate(variantes):
    print(f'  Variante {i}: shape={var.shape}')

# Probar con n_variedades=5
variantes2 = generar_variantes(img, 5, 42)
print(f'\nCon n=5: {len(variantes2)} variantes')

# Guardar ejemplo collage
os.makedirs('docs/img', exist_ok=True)
max_alto = max(v.shape[0] for v in variantes)
total_ancho = sum(v.shape[1] for v in variantes)
canvas = np.zeros((max_alto, total_ancho, 3), dtype=np.uint8)
for i, var in enumerate(variantes):
    canvas[:var.shape[0], i*var.shape[1]: (i+1)*var.shape[1]] = var
cv2.imwrite('docs/img/ejemplo_aumentos.png', canvas)
print(f'\nCollage guardado: docs/img/ejemplo_aumentos.png')

# Verificar reproducibilidad
variantes3 = generar_variantes(img, 3, 42)
son_identicas = all((np.array(v1) == np.array(v2)).all() for v1, v2 in zip(variantes, variantes3))
print(f'Reproducible con semilla=42: {son_identicas}')

print(f'\nDimensiones collage: {canvas.shape}')
print(f'Original: {img.shape}')