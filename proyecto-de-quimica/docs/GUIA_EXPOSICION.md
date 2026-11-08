# Guía para la exposición — Sistema de Reciclaje Inteligente

## 1. Checklist (el día anterior y 1 hora antes)

**Preparación en tu computador (día anterior)**
- [ ] `python -m pytest -q` → deben pasar todas las pruebas.
- [ ] La base de datos tiene datos: abre la app y revisa que el tablero muestre gráficas.
      Si está vacía: `python ingesta_publica.py --ruta <TACO/data> --aumentos 2 --distribuir-dias 30`
- [ ] Copia `data/reciclaje.db` a una USB o a Drive **con la app cerrada**
      (si existen `reciclaje.db-wal` y `reciclaje.db-shm`, cópialos también).
- [ ] Graba un video corto (1-2 min) de la app funcionando con la cámara: es tu plan B.
- [ ] Ten a mano `RESULTADOS.md` y, si reentrenaste, `models/reporte_reentrenamiento.txt`.

**En el computador de la exposición (1 hora antes)**
- [ ] `pip install -r requirements.txt` (requiere internet, ~10 min).
- [ ] Pega la base en `proyecto-de-quimica/data/reciclaje.db`.
- [ ] `streamlit run app.py` → acepta el permiso de cámara en el navegador.
- [ ] Prueba con 2-3 residuos: una botella PET, una lata y un paquete de papas.

## 2. Guion de demostración (8-10 minutos)

**1. Problema (1 min).**
> "En Colombia la Resolución 2184 de 2019 define tres contenedores: blanco (aprovechables),
> negro (no aprovechables) y verde (orgánicos). Separar mal contamina el reciclaje. Este
> sistema reconoce el residuo con una cámara y sugiere el contenedor."

**2. Clasificador en vivo (2-3 min).** Muestra una botella PET (acierta bien) y luego un
paquete de papas o un icopor. Si se equivoca, **úsalo a tu favor**: es la brecha de dominio.

**3. La brecha de dominio (2 min).** Muestra `RESULTADOS.md`:
> "El modelo se entrenó con TrashNet: fotos de objetos limpios sobre fondo blanco, de EE. UU.
> Al medirlo con 800 fotos reales de basura (dataset TACO) bajó a 32,7 %. Peor aún: 8 de cada 10
> residuos no aprovechables los mandaba al contenedor blanco, la mayoría como papel o
> cartón. El icopor es blanco y opaco y los paquetes arrugados parecen papel."

**4. La solución (2 min).**
- *Ingesta de datos públicos:* TACO trae miles de objetos ya etiquetados (formato COCO);
  un diccionario los traduce a las clases de la norma colombiana (paquete de papas →
  No aprovechable Multicapa, icopor → No aprovechable Icopor, aerosol → Metal).
- *Aumento de datos:* cada foto se convierte en variantes que imitan una webcam: destellos
  sobre empaques metalizados, cambios de luz, encuadre imperfecto y compresión de video.
- *Transfer learning:* se conserva la red que «sabe ver» y se reentrena solo la capa final.
- Si reentrenaste, muestra la tabla antes/después de `models/reporte_reentrenamiento.txt`.

**5. Tablero (1-2 min).** Filtra por origen (`camara`, `taco`, `taco_aug`) y muestra la
sección «Validación con bancos públicos» (exactitud y matriz de confusión).

**6. Cierre (30 s).**
> "El aporte no es solo clasificar: es medir honestamente dónde falla el modelo con
> residuos colombianos y tener un proceso reproducible para corregirlo."

## 3. Preguntas probables del jurado

**¿Qué es la brecha de dominio?**
> Es la diferencia entre los datos con que se entrenó el modelo y los que ve en la vida real.
> Aquí: fotos de estudio de EE. UU. frente a basura real y empaques colombianos. Por eso
> la exactitud baja de 89 % (en TrashNet) a 32,7 % (en TACO).

**¿Para qué sirve el aumento de datos?**
> Para que el modelo vea, durante el entrenamiento, las condiciones de la webcam: brillos,
> poca luz, objetos descentrados, imagen comprimida. No inventa objetos nuevos; muestra el
> mismo objeto en condiciones distintas para que el modelo no dependa de una foto perfecta.

**¿Qué es transfer learning?**
> Usar una red ya entrenada con 1,2 millones de imágenes (ImageNet) como «ojos» del sistema
> y entrenar solo la última capa, la que decide la clase. Así se logra con miles de fotos y
> en minutos, en un portátil, lo que desde cero requeriría millones de fotos y GPU.

**¿Por qué el icopor y los paquetes de papas van al contenedor negro?**
> Los paquetes de papas son multicapa (plástico + aluminio): las capas no se pueden separar
> para reciclarlas. El icopor casi siempre llega contaminado con comida y tiene muy poco
> peso por volumen, por lo que en la práctica no se recupera. (El Tetra Pak, aunque también
> es multicapa, sí se aprovecha en Colombia y va al blanco.)

**¿Cómo saben que los resultados no están inflados?**
> Separamos las fotos por lotes: el modelo se entrena con los lotes 1 a 11 de TACO, se ajusta
> con el 12 y 13, y se evalúa con el 14 y 15, que nunca vio. Además, el script solo reemplaza
> el modelo si mejora en fotos nuevas sin empeorar más de 3 puntos en TrashNet, y guarda una
> copia del modelo anterior. Hay 53 pruebas automáticas que verifican el sistema.

## 4. Datos clave (tarjeta de apoyo)

| Dato | Valor |
|---|---|
| Exactitud del modelo en TrashNet (fotos de prueba) | 89,2 % |
| Exactitud en fotos reales (TACO, 752 objetos) | 32,7 % |
| No aprovechables enviados al contenedor correcto | 17,8 % |
| Contenido del contenedor blanco que es basura | 40,5 % |
| Pruebas automáticas | 53 de 53 OK |
| Exactitud en fotos reales tras reentrenar (webcam) | 57,0 % (antes 21,0 %) |
| Icopor reconocido tras reentrenar | 87,5 % (antes 6,2 %) |

## 5. Plan B
1. Si la cámara falla: reproduce el video grabado.
2. Si la app no arranca: muestra `RESULTADOS.md` y el reporte de reentrenamiento.
3. Mantén la calma: explicar con honestidad una limitación suma puntos.
