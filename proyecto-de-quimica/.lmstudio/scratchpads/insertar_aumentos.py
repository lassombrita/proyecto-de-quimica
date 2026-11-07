# Funcion para aumentar datos (no recorta)
def generar_variantes(imagen_original, n_variedades=3, semilla=42):
    import albumentations as A
    from PIL import Image
    import random
    rgb = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)
    pil_imagen = Image.fromarray(rgb)
    transformacion = A.Compose([A.Flip(p=0.5), A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0), A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6), A.GaussNoise(var_limit=(10.0, 50.0), p=0.4)], is_check_shapes=True)
    random.seed(semilla)
    np.random.seed(semilla)
    variantes = [imagen_original]
    for _ in range(n_variedades - 1):
        try:
            resultado = transformacion(image=pil_imagen)
            pil_imagen = Image.fromarray(resultado['image'])
            bgr = cv2.cvtColor(np.array(pil_imagen), cv2.COLOR_RGB2BGR)
            variantes.append(bgr)
        except:
            random.seed(semilla + 1)
            np.random.seed(semilla + 1)
    return variantes
