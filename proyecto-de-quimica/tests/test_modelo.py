"""
test_modelo.py — Pruebas de que el modelo carga y clasifica una imagen.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["RECICLAJE_DB"] = str(Path(__file__).parent.parent / "temp_pruebas.db")

def test_modelo_carga_sin_error():
    """El modelo carga con cv2.dnn sin lanzar excepciones."""
    
    modelo_onnx = Path("models/trashnet_mobilenetv2.onnx")
    if not modelo_onnx.exists():
        print("⏭️  test_modelo_carga_sin_error (OMITIDA): modelo .onnx no encontrado")
        return None
    
    import cv2
    import numpy as np
    
    try:
        net = cv2.dnn.readNetFromONNX(str(modelo_onnx))
        
        img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        img_blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (224, 224), 
                                          (123.68, 116.78, 103.94), swapRB=True)
        
        net.setInput(img_blob)
        output = net.forward()
        
        print(f"   Salida shape: {output.shape}")
        print("✅ test_modelo_carga_sin_error")
        return "passed"
    
    except Exception as e:
        print(f"❌ test_modelo_carga_sin_error FALLA: {e}")
        raise


def test_modelo_clasifica_imagen():
    """El modelo clasifica una imagen y devuelve probabilidades."""
    
    modelo_onnx = Path("models/trashnet_mobilenetv2.onnx")
    if not modelo_onnx.exists():
        print("⏭️  test_modelo_clasifica_imagen (OMITIDA): modelo .onnx no encontrado")
        return None
    
    import cv2
    import numpy as np
    
    try:
        net = cv2.dnn.readNetFromONNX(str(modelo_onnx))
        
        img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        img_blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (224, 224), 
                                          (123.68, 116.78, 103.94), swapRB=True)
        
        net.setInput(img_blob)
        output = net.forward()
        
        assert len(output) == 1, "Debe haber 1 predicción"
        assert len(output[0]) == 6, f"Debe tener 6 clases, got {len(output[0])}"
        
        prob_sum = float(np.sum(output[0]))
        tol = 5e-3
        assert abs(prob_sum - 1.0) < tol, f"Probabilidades no suman 1.0: {prob_sum}"
        
        pred_idx = np.argmax(output[0])
        print(f"   Predicción: clase {pred_idx} con confianza {output[0][pred_idx]:.2%}")
        
        print("✅ test_modelo_clasifica_imagen")
        return "passed"
    
    except Exception as e:
        print(f"❌ test_modelo_clasifica_imagen FALLA: {e}")
        raise


def test_salida_no_es_vacia():
    """La salida no es un array vacío."""
    
    modelo_onnx = Path("models/trashnet_mobilenetv2.onnx")
    if not modelo_onnx.exists():
        print("⏭️  test_salida_no_es_vacia (OMITIDA): modelo .onnx no encontrado")
        return None
    
    import cv2
    import numpy as np
    
    try:
        net = cv2.dnn.readNetFromONNX(str(modelo_onnx))
        
        img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        img_blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (224, 224),
                                          (123.68, 116.78, 103.94), swapRB=True)
        
        net.setInput(img_blob)
        output = net.forward()
        
        assert len(output) > 0, "La salida no debe estar vacía"
        assert len(output[0]) > 0, "Las probabilidades no deben estar vacías"
        
        print("✅ test_salida_no_es_vacia")
        return None
    
    except Exception as e:
        print(f"❌ test_salida_no_es_vacia FALLA: {e}")
        raise


if __name__ == "__main__":
    import pytest
    
    print("\n🔬 Pruebas del modelo\n")
    
    try:
        result1 = test_modelo_carga_sin_error()
        result2 = test_modelo_clasifica_imagen()
        result3 = test_salida_no_es_vacia()
        
        print(f"\n✅ Todas las pruebas completadas")
    except Exception as e:
        print(f"\n❌ Algunas pruebas fallaron: {e}")
