#!/usr/bin/env python
"""
Genera una cuadrícula con 12 recortes de diferentes categorias de TACO
para diagnosticar que el recorte no desplaza ni rota los objetos.
Toma las primeras 12 imagenes disponibles del dataset.
"""
import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

TACO_RUTA = Path("TACO/data")
RUTA_SALIDA = Path("docs/img/diagnostico_recortes.png")

print("Diagnosticando recortes de TACO (tomando primeras 12 imagenes)...")

# Obtener lista de todos los archivos jpg existentes
all_img_files = []
for batch_dir in TACO_RUTA.iterdir():
    if not batch_dir.is_dir() or not batch_dir.name.startswith('batch_'):
        continue
    for ext in ['.jpg', '.JPG']:
        all_img_files.extend(batch_dir.glob(f"*{ext}"))

print(f"Encontradas {len(all_img_files)} imagenes en disco")

# Tomar las primeras 12 imagenes por orden alfabético directo (simple y efectivo)
recortes = []
if len(all_img_files) >= 12:
    sorted_files = sorted(all_img_files)[:12]
    print(f"Procesando {len(sorted_files)} imagenes...")
    
    for img_file in sorted_files:
        try:
            img_bgr = cv2.imread(str(img_file))
            if img_bgr is not None and img_bgr.size > 0:
                h, w, _ = img_bgr.shape
                
                # Recortar con margen 0.15 (igual que ingesta_publica.py)
                x1, y1 = max(0, int(w * 0.15)), max(0, int(h * 0.15))
                x2, y2 = min(img_bgr.shape[1], int(w + w * 0.15)), min(img_bgr.shape[0], int(h + h * 0.15))
                
                recorte = img_bgr[y1:y2, x1:x2]
                if recorte is not None and recorte.size > 0:
                    temp_path = Path("temp_recortes") / f"recorte_{len(recortes):04d}.jpg"
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(temp_path), recorte)
                    
                    recortes.append({
                        'archivo': str(img_file),
                        'bbox': (x1, y1, x2-x1, y2-y1),  # bbox del recorte
                        'recorte': str(temp_path),
                        'nombre_archivo': img_file.name
                    })
                    
                    if len(recortes) == 12:
                        break
        except Exception as e:
            print(f"  Error con {img_file.name}: {e}")

print(f"\nCargados {len(recortes)} recortes totales")
for r in recortes[:5]:
    print(f"  - {r['nombre_archivo']} (bbox: {r['bbox']})")

if len(recortes) >= 12:
    # Crear cuadricula usando matplotlib
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, rec in enumerate(recortes[:12]):
        img_path = Path(rec['recorte'])
        if img_path.exists():
            try:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None and img_bgr.size > 0:
                    h, w, _ = img_bgr.shape
                    target_size = 160
                    scale = target_size / max(h, w)
                    if scale < 1:
                        img_small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
                    else:
                        img_small = img_bgr
                    
                    axes[i].imshow(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
                    
                    x, y, w_orig, h_orig = rec['bbox']
                    px, py = w_orig * 0.15, h_orig * 0.15
                    
                    axes[i].set_title(
                        f"{rec['nombre_archivo']}\n{x},{y}\n+/-[{int(px)},{int(py)}]",
                        fontsize=7
                    )
                    axes[i].axis('off')
            except Exception as e:
                print(f"  Error procesando {rec['archivo']}: {e}")
                axes[i].text(0.5, 0.5, "Error", ha='center', va='center')
                axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, "Archivo no encontrado", ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(RUTA_SALIDA, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"\nGuardado en: {RUTA_SALIDA}")
else:
    print(f"\nSolo se encontraron {len(recortes)} recortes (necesitamos 12)")
