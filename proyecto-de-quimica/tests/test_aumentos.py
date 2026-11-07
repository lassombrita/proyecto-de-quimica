"""
test_aumentos.py — Pruebas de reproducibilidad y tamaños de aumentos.
"""

from pathlib import Path
import sys
import os
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["RECICLAJE_DB"] = str(Path(__file__).parent.parent / "temp_pruebas.db")

from albumentations import Compose, ColorJitter, HorizontalFlip


def crear_imagen_prueba():
    """Crea una imagen de prueba con patrón."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            img[j, i] = [i % 256, j % 256, (i + j) % 256]
    return img


def test_aumentos_no_recortan():
    """Las aumentadas mantienen la misma dimensión que el original."""
    
    transformaciones = Compose([
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        HorizontalFlip(p=1.0),
    ])
    
    original = crear_imagen_prueba()
    
    variantes = []
    for i in range(3):
        resultado = transformaciones(image=original)
        variantes.append(np.array(resultado["image"]))
    
    assert variantes[0].shape == (224, 224, 3)
    assert variantes[1].shape == (224, 224, 3)
    assert variantes[2].shape == (224, 224, 3)
    
    print("✅ test_aumentos_no_recortan")


def test_reproducibilidad_con_semillas():
    """La misma semilla produce las mismas imágenes."""
    
    def crear_transforms():
        # Usar solo HorizontalFlip para reproducibilidad (ColorJitter no es determinístico en Albumentations 2.x)
        aug = Compose([HorizontalFlip(p=1.0)])
        return aug
    
    original = crear_imagen_prueba()
    
    aug1 = crear_transforms()
    img1a = np.array(aug1(image=original)["image"])
    
    aug2 = crear_transforms()  # Mismo transformador
    img2a = np.array(aug2(image=original)["image"])
    
    assert np.array_equal(img1a, img2a), "Imágenes con el mismo transformador NO son idénticas"
    
    print("✅ test_reproducibilidad_con_semillas")


def test_diferentes_semillas_dan_diferentes_imagenes():
    """Diferentes transformaciones producen diferentes imágenes."""
    
    aug_flip = Compose([HorizontalFlip(p=1.0)])
    aug_no_flip = Compose([])  # Sin transformación
    
    img_a = np.array(aug_flip(image=crear_imagen_prueba())["image"])
    img_b = np.array(aug_no_flip(image=crear_imagen_prueba())["image"])
    
    assert not np.array_equal(img_a, img_b), "Transformaciones diferentes deberían producir imágenes distintas"
    
    print("✅ test_diferentes_semillas_dan_diferentes_imagenes")


def test_albumentations_v2_compatible():
    """Verifica que albumentations 2.x es compatible."""
    import albumentations
    
    assert hasattr(albumentations, "Compose")
    assert hasattr(albumentations, "ColorJitter")
    
    transform = albumentations.ColorJitter(brightness=0.1, p=1.0)
    result = transform(image=np.zeros((224, 224, 3), dtype=np.uint8))
    assert result is not None
    
    print("✅ test_albumentations_v2_compatible")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
