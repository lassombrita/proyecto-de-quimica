"""
vision_engine.py — Motor Multi-Engine de Clasificación de Residuos
==================================================================
Sistema de Visión por Computador para la clasificación de residuos sólidos
según la Resolución 2184 de 2019 (Colombia).

Soporta 4 motores intercambiables:
    1. "Motor Básico (Color)":
       Heurística basada en el espacio HSV promedio. Permite reproducir y
       demostrar fallos cromáticos (ej. jugo de manzana clasificado como cartón).
    2. "Motor Geométrico (Forma/Contornos)":
       Visión clásica con OpenCV (Canny, contornos, relación de aspecto, solidez,
       análisis de cuello). Sin dependencia cromática.
    3. "Motor IA (MobileNetV2)":
       Red neuronal convolucional ligera entrenada con TrashNet y ejecutada localmente
       mediante cv2.dnn (sin necesidad de TensorFlow/PyTorch).
    4. "Motor Híbrido Estricto":
       Combina la inferencia de la red neuronal como motor principal y la geometría
       morfológica como factor de apoyo y refinamiento (PET vs PEAD), sin bloqueos
       destructivos ante fondos complejos.

Política de Incertidumbre (Fallback):
    Si la confianza del motor cae por debajo del umbral del 50%, el residuo se
    clasifica automáticamente como "No identificado" y se asigna al contenedor NEGRO
    para prevenir la contaminación del lote de reciclaje.

Autor: Johan Lasso & Equipo de Arquitectura de Software
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ===========================================================================
# 1. CATÁLOGO DE CATEGORÍAS Y CONTENEDORES (Res. 2184 de 2019)
# ===========================================================================
CATEGORIAS: dict[str, dict[str, str]] = {
    "PET": {
        "contenedor": "Blanco",
        "descripcion": "Polietileno tereftalato (botellas de bebidas). Código de resina #1.",
    },
    "PEAD": {
        "contenedor": "Blanco",
        "descripcion": "Polietileno de alta densidad (envases de detergente, galones). Código #2.",
    },
    "Vidrio": {
        "contenedor": "Blanco",
        "descripcion": "Botellas y frascos de vidrio limpios y secos.",
    },
    "Metal": {
        "contenedor": "Blanco",
        "descripcion": "Latas de aluminio y acero (bebidas y conservas).",
    },
    "Papel/Cartón": {
        "contenedor": "Blanco",
        "descripcion": "Papel y cartón limpios y secos (cajas, hojas, sobres).",
    },
    "No aprovechable": {
        "contenedor": "Negro",
        "descripcion": "Residuo no reciclable (servilletas usadas, empaques metalizados).",
    },
    "No identificado": {
        "contenedor": "Negro",
        "descripcion": "Incertidumbre alta (<50%). Por seguridad se envía al contenedor "
                       "negro para evitar contaminar los materiales aprovechables.",
    },
}

# Colores HEX oficiales para representar cada contenedor en la UI
COLORES_CONTENEDOR: dict[str, str] = {
    "Blanco": "#FFFFFF",
    "Negro": "#111827",
    "Verde": "#16A34A",
}

# Clases de salida del modelo TrashNet
CLASES_TRASHNET = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Mapeo directo de clases TrashNet a categorías del sistema
MAPEO_TRASHNET: dict[str, str] = {
    "cardboard": "Papel/Cartón",
    "paper": "Papel/Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "plastic": "PET",  # Si hay morfología, se puede especializar a PEAD
    "trash": "No aprovechable",
}

# Parámetros generales y umbrales
ANCHO_PROCESAMIENTO = 480
UMBRAL_CONFIANZA_MINIMA = 50.0  # Umbral de fallback al contenedor negro
RUTA_MODELO_DEFAULT = Path(__file__).parent / "models" / "trashnet_mobilenetv2.onnx"


# ===========================================================================
# 2. ESTRUCTURAS DE DATOS Y DIAGNÓSTICO
# ===========================================================================
@dataclass
class Morfologia:
    """Parámetros geométricos extraídos de los contornos."""
    area_relativa: float = 0.0
    relacion_aspecto: float = 0.0
    rectangularidad: float = 0.0
    solidez: float = 0.0
    vertices: int = 0
    ratio_cuello: float = 1.0
    descentrado_cuello: float = 0.0
    forma: str = "No aislada / Fondo complejo"


@dataclass
class ResultadoClasificacion:
    """Estructura de respuesta unificada consumida por Streamlit y SQLite."""
    material: str
    confianza: float
    contenedor: str
    descripcion: str
    forma: str
    metodo: str
    motivo: str
    top_predicciones: list[tuple[str, float]] = field(default_factory=list)
    diagnostico_forma: str = ""
    diagnostico_color: str = ""
    imagen_anotada: np.ndarray | None = None


# ===========================================================================
# 3. CLASIFICADOR MULTI-ENGINE
# ===========================================================================
class ClasificadorResiduos:
    """
    Motor integral de visión por computador para clasificación de residuos.
    Implementa 4 estrategias de inferencia con tolerancia a fallos.
    """

    NOMBRE_MODELO = "EcoVision Multi-Engine v2.5"

    def __init__(self, ruta_modelo: Path = RUTA_MODELO_DEFAULT) -> None:
        self.ruta_modelo = Path(ruta_modelo)
        self.red_onnx: cv2.dnn.Net | None = None
        self.error_carga_dnn: str | None = None

        # Carga tolerante a fallos del modelo ONNX
        if self.ruta_modelo.exists():
            try:
                self.red_onnx = cv2.dnn.readNetFromONNX(str(self.ruta_modelo))
            except Exception as e:
                self.red_onnx = None
                self.error_carga_dnn = f"Error al inicializar ONNX: {e}"
        else:
            self.error_carga_dnn = f"Archivo ONNX no encontrado en {self.ruta_modelo}"

    # -----------------------------------------------------------------------
    # Carga y Normalización de Entrada
    # -----------------------------------------------------------------------
    @staticmethod
    def _cargar_imagen(imagen) -> np.ndarray:
        """Convierte st.camera_input, PIL o bytes en imagen BGR normalizada."""
        if isinstance(imagen, Image.Image):
            rgb = np.array(imagen.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(imagen, np.ndarray):
            bgr = imagen.copy()
        else:
            datos = imagen.getvalue() if hasattr(imagen, "getvalue") else bytes(imagen)
            arreglo = np.frombuffer(datos, dtype=np.uint8)
            bgr = cv2.imdecode(arreglo, cv2.IMREAD_COLOR)

        if bgr is None:
            raise ValueError("No se pudo decodificar el flujo de imagen.")

        alto, ancho = bgr.shape[:2]
        if ancho > ANCHO_PROCESAMIENTO:
            factor = ANCHO_PROCESAMIENTO / ancho
            bgr = cv2.resize(bgr, (ANCHO_PROCESAMIENTO, int(alto * factor)), interpolation=cv2.INTER_AREA)

        return bgr

    # -----------------------------------------------------------------------
    # MOTOR 1: Análisis Básico por Color (HSV)
    # -----------------------------------------------------------------------
    def _clasificar_por_color(self, bgr: np.ndarray) -> ResultadoClasificacion:
        """
        Clasificador heurístico basado en el color promedio del centro.
        Demuestra la vulnerabilidad ante líquidos de color (ej. jugo en botella PET).
        """
        alto, ancho = bgr.shape[:2]
        c_y1, c_y2 = int(alto * 0.25), int(alto * 0.75)
        c_x1, c_x2 = int(ancho * 0.25), int(ancho * 0.75)
        zona_central = bgr[c_y1:c_y2, c_x1:c_x2]

        hsv = cv2.cvtColor(zona_central, cv2.COLOR_BGR2HSV)
        tono = float(np.median(hsv[:, :, 0]))
        sat = float(np.mean(hsv[:, :, 1]))
        val = float(np.mean(hsv[:, :, 2]))

        # Color promedio en Hexadecimal
        b_mean, g_mean, r_mean = [int(v) for v in zona_central.reshape(-1, 3).mean(axis=0)]
        color_hex = f"#{r_mean:02X}{g_mean:02X}{b_mean:02X}"
        diag_color = f"HSV: (H={tono:.0f}, S={sat:.0f}, V={val:.0f}) | Hex: {color_hex}"

        # Heurística cromática original (con sus sesgos conocidos)
        if sat < 40 and val > 160:
            material = "PET"
            conf = 78.0
            motivo = "Tono claro/transparente de baja saturación detectado en el centro."
        elif sat < 45 and 80 < val <= 160:
            material = "Metal"
            conf = 74.0
            motivo = "Gris neutro con baja saturación asociado a latas metálicas."
        elif val < 60:
            material = "No aprovechable"
            conf = 80.0
            motivo = "Brillo muy bajo (material oscuro/sucio o residuo no reciclable)."
        elif 35 <= tono <= 85 and sat >= 40:
            material = "Vidrio"
            conf = 76.0
            motivo = "Color verdoso predominante característico de botellas de vidrio."
        elif 8 <= tono < 28 and sat >= 35:
            # SESGO CRÍTICO: Una botella transparente llena de jugo de manzana se ve marrón/ámbar
            material = "Papel/Cartón"
            conf = 82.0
            motivo = "Color marrón/ocre predominante asignado erróneamente a Papel/Cartón (falso positivo)."
        elif 85 < tono <= 135 and sat >= 35:
            material = "PEAD"
            conf = 75.0
            motivo = "Tonalidad azul predominante (típica de envases plásticos de detergente)."
        else:
            material = "No aprovechable"
            conf = 52.0
            motivo = "Mezcla de colores indefinida sin patrón cromático claro."

        # Imagen anotada: rectángulo indicador de zona de muestreo central
        anotada = bgr.copy()
        cv2.rectangle(anotada, (c_x1, c_y1), (c_x2, c_y2), (0, 165, 255), 2)
        cv2.putText(anotada, f"Muestreo Color: {color_hex}", (c_x1, max(20, c_y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        rgb_anotada = cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB)

        cat_info = CATEGORIAS[material]
        return ResultadoClasificacion(
            material=material,
            confianza=conf,
            contenedor=cat_info["contenedor"],
            descripcion=cat_info["descripcion"],
            forma="Análisis cromático (sin detección de contorno)",
            metodo="Motor Básico (Color)",
            motivo=motivo,
            top_predicciones=[],
            diagnostico_forma="N/A (Motor cromático puro)",
            diagnostico_color=diag_color,
            imagen_anotada=rgb_anotada,
        )

    # -----------------------------------------------------------------------
    # Segmentación y Morfología (OpenCV)
    # -----------------------------------------------------------------------
    @staticmethod
    def _segmentar_y_extraer_morfologia(bgr: np.ndarray) -> tuple[Morfologia, np.ndarray | None, np.ndarray]:
        """
        Segmenta el objeto principal mediante gradientes y binarización adaptativa.
        Calcula descriptores geométricos independientes de la iluminación y el color.
        """
        gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        suave = cv2.GaussianBlur(gris, (5, 5), 0)

        # Canny edge detection adaptativo
        mediana = float(np.median(gris))
        lim_inf = int(max(20, 0.66 * mediana))
        lim_sup = int(min(240, 1.33 * mediana))
        bordes = cv2.Canny(suave, lim_inf, lim_sup)

        # Cierre morfológico para conectar contornos desconectados
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mascara = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel, iterations=2)
        mascara = cv2.dilate(mascara, kernel, iterations=1)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        alto_img, ancho_img = gris.shape
        area_total = alto_img * ancho_img
        mejor_contorno = None
        max_area = 0.0

        for c in contornos:
            area = cv2.contourArea(c)
            # Filtro de escala: debe ocupar entre 2.5% y 95% del encuadre
            if 0.025 * area_total <= area <= 0.95 * area_total:
                if area > max_area:
                    max_area = area
                    mejor_contorno = c

        if mejor_contorno is None:
            return Morfologia(), None, gris

        # Descriptores de forma
        area = max_area
        perimetro = cv2.arcLength(mejor_contorno, True)
        casco = cv2.convexHull(mejor_contorno)
        area_casco = max(1.0, cv2.contourArea(casco))
        solidez = area / area_casco

        # Rectángulo rotado mínimo
        (_, _), (rw, rh), _ = cv2.minAreaRect(mejor_contorno)
        largo = max(rw, rh)
        corto = max(1.0, min(rw, rh))
        relacion_aspecto = largo / corto
        rectangularidad = area / max(1.0, rw * rh)

        # Polígono aproximado
        aprox = cv2.approxPolyDP(mejor_contorno, 0.025 * perimetro, True)
        vertices = len(aprox)

        # Análisis de cuello (perfil de ancho):
        x, y, w, h = cv2.boundingRect(mejor_contorno)
        recorte_mascara = np.zeros_like(gris)
        cv2.drawContours(recorte_mascara, [mejor_contorno], -1, 255, thickness=cv2.FILLED)
        recorte_obj = recorte_mascara[y:y + h, x:x + w]

        # Ancho por filas
        perfil = (recorte_obj > 0).sum(axis=1).astype(float)
        n_filas = len(perfil)
        if n_filas >= 10:
            cuerpo = perfil[int(0.3 * n_filas):int(0.7 * n_filas)]
            ancho_cuerpo = float(np.median(cuerpo)) if len(cuerpo) else float(w)
            extremo1 = float(np.median(perfil[:max(2, n_filas // 6)]))
            extremo2 = float(np.median(perfil[-max(2, n_filas // 6):]))
            min_extremo = min(extremo1, extremo2)
            ratio_cuello = min_extremo / max(1.0, ancho_cuerpo)

            # Posición del centroide del extremo más angosto para ver si está desplazado
            es_arriba = extremo1 < extremo2
            filas_cuello = slice(0, max(2, n_filas // 8)) if es_arriba else slice(-max(2, n_filas // 8), n_filas)
            sub_recorte = recorte_obj[filas_cuello, :]
            coords_x = np.where(sub_recorte > 0)[1]
            if len(coords_x):
                cx_cuello = float(coords_x.mean())
                descentrado = abs(cx_cuello - (w / 2.0)) / max(1.0, w)
            else:
                descentrado = 0.0
        else:
            ratio_cuello = 1.0
            descentrado = 0.0

        # Etiqueta geométrica descriptiva
        if ratio_cuello < 0.65:
            forma = "Envase con pico lateral" if descentrado > 0.15 else "Botella con cuello"
        elif 1.2 <= relacion_aspecto <= 2.8 and solidez > 0.78 and rectangularidad < 0.92:
            forma = "Cilindro (Lata)"
        elif rectangularidad >= 0.82 and solidez > 0.85:
            forma = "Rectangular (Caja u Hoja)"
        else:
            forma = "Forma irregular o compuesta"

        morf = Morfologia(
            area_relativa=area / area_total,
            relacion_aspecto=relacion_aspecto,
            rectangularidad=rectangularidad,
            solidez=solidez,
            vertices=vertices,
            ratio_cuello=ratio_cuello,
            descentrado_cuello=descentrado,
            forma=forma,
        )
        return morf, mejor_contorno, gris

    # -----------------------------------------------------------------------
    # MOTOR 2: Motor Geométrico Puro (OpenCV)
    # -----------------------------------------------------------------------
    def _clasificar_por_geometria(self, bgr: np.ndarray) -> ResultadoClasificacion:
        """Clasifica basándose únicamente en los contornos y relaciones de aspecto."""
        morf, contorno, _ = self._segmentar_y_extraer_morfologia(bgr)

        anotada = bgr.copy()
        if contorno is None:
            # Fallback inmediato: no se puede determinar contorno
            rgb_anotada = cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB)
            info = CATEGORIAS["No identificado"]
            return ResultadoClasificacion(
                material="No identificado",
                confianza=0.0,
                contenedor=info["contenedor"],
                descripcion=info["descripcion"],
                forma="Sin contorno claro",
                metodo="Motor Geométrico (Forma/Contornos)",
                motivo="No se encontró un contorno de objeto aislable frente al fondo.",
                top_predicciones=[],
                diagnostico_forma="Sin contorno detectado. Fondo con textura o iluminación débil.",
                imagen_anotada=rgb_anotada,
            )

        # Dibujar silueta y rectángulo rotado
        cv2.drawContours(anotada, [contorno], -1, (0, 230, 115), 3)
        caja = cv2.boxPoints(cv2.minAreaRect(contorno)).astype(int)
        cv2.drawContours(anotada, [caja], -1, (255, 140, 0), 2)
        rgb_anotada = cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB)

        diag_forma = (
            f"Aspecto: {morf.relacion_aspecto:.2f} | Rectangularidad: {morf.rectangularidad:.2f} | "
            f"Solidez: {morf.solidez:.2f} | Ratio Cuello: {morf.ratio_cuello:.2f} | "
            f"Forma: {morf.forma}"
        )

        # Reglas morfológicas deterministas
        if morf.ratio_cuello < 0.65:
            if morf.descentrado_cuello > 0.15:
                material = "PEAD"
                conf = 75.0
                motivo = f"Pico desplazado ({morf.descentrado_cuello:.0%}): silueta de galón o detergente (PEAD)."
            else:
                material = "PET"
                conf = 78.0
                motivo = f"Cuello angosto (ratio {morf.ratio_cuello:.2f}) y cuerpo simétrico: silueta de botella PET."
        elif 1.2 <= morf.relacion_aspecto <= 2.8 and morf.solidez > 0.78 and morf.rectangularidad < 0.92:
            material = "Metal"
            conf = 72.0
            motivo = f"Relación de aspecto {morf.relacion_aspecto:.2f} con paredes rectas: perfil de lata metálica."
        elif morf.rectangularidad >= 0.82 and morf.solidez > 0.85:
            material = "Papel/Cartón"
            conf = 74.0
            motivo = f"Alta rectangularidad ({morf.rectangularidad:.0%}) y solidez ({morf.solidez:.0%}): silueta de caja u hoja."
        else:
            material = "No identificado"
            conf = 35.0
            motivo = "Forma geométrica irregular que no encaja en plantillas estándar de reciclaje."

        # Control de umbral de fallback
        if conf < UMBRAL_CONFIANZA_MINIMA:
            material = "No identificado"
            motivo = f"Incertidumbre geométrica: confianza ({conf:.1f}%) por debajo del {UMBRAL_CONFIANZA_MINIMA:.0f}%."

        info = CATEGORIAS[material]
        return ResultadoClasificacion(
            material=material,
            confianza=conf,
            contenedor=info["contenedor"],
            descripcion=info["descripcion"],
            forma=morf.forma,
            metodo="Motor Geométrico (Forma/Contornos)",
            motivo=motivo,
            top_predicciones=[],
            diagnostico_forma=diag_forma,
            imagen_anotada=rgb_anotada,
        )

    # -----------------------------------------------------------------------
    # Inferencia Neuronal (MobileNetV2 + ONNX)
    # -----------------------------------------------------------------------
    def _inferir_red(self, bgr_crop: np.ndarray) -> tuple[dict[str, float], list[tuple[str, float]]]:
        """
        Ejecuta la inferencia sobre el recorte con cv2.dnn garantizando:
        - Conversión BGR -> RGB.
        - Redimensionamiento exacto a (224, 224).
        - Normalización estándar de ImageNet (media y desviación estándar).
        """
        if self.red_onnx is None:
            raise RuntimeError(self.error_carga_dnn or "Modelo ONNX no disponible.")

        # Conversión obligatoria BGR a RGB
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

        # Normalización de ImageNet
        media = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        desv = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_norm = (rgb_resized - media) / desv

        # Formato NCHW esperado por MobileNetV2
        blob = np.transpose(rgb_norm, (2, 0, 1))[np.newaxis, :]
        self.red_onnx.setInput(blob)
        logits = self.red_onnx.forward().flatten()

        # Softmax estable
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        prob_dict = {clase: float(probs[i]) for i, clase in enumerate(CLASES_TRASHNET)}
        top_sorted = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        top3_porcentajes = [(k, round(v * 100.0, 1)) for k, v in top_sorted[:3]]

        return prob_dict, top3_porcentajes

    # -----------------------------------------------------------------------
    # MOTOR 3: Motor IA Puro (MobileNetV2)
    # -----------------------------------------------------------------------
    def _clasificar_por_ia(self, bgr: np.ndarray) -> ResultadoClasificacion:
        """Clasificación basada exclusivamente en la red neuronal convolucional."""
        if self.red_onnx is None:
            # Fallback seguro si no hay archivo ONNX
            info = CATEGORIAS["No identificado"]
            return ResultadoClasificacion(
                material="No identificado",
                confianza=0.0,
                contenedor=info["contenedor"],
                descripcion=info["descripcion"],
                forma="N/A",
                metodo="Motor IA (MobileNetV2)",
                motivo=f"No se pudo cargar la red neuronal: {self.error_carga_dnn}",
                diagnostico_forma="DNN no disponible",
                imagen_anotada=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            )

        morf, contorno, _ = self._segmentar_y_extraer_morfologia(bgr)

        # Si hay contorno claro, recortamos el objeto con margen del 10%
        if contorno is not None:
            x, y, w, h = cv2.boundingRect(contorno)
            pad_x, pad_y = int(w * 0.1), int(h * 0.1)
            recorte = bgr[max(0, y - pad_y):min(bgr.shape[0], y + h + pad_y),
                          max(0, x - pad_x):min(bgr.shape[1], x + w + pad_x)]
            diag_forma = f"Objeto delimitado ({w}x{h} px) | Aspecto: {morf.relacion_aspecto:.2f}"
        else:
            # Si el fondo es ruidoso o no hay contorno claro, usamos el encuadre central al 80%
            alto, ancho = bgr.shape[:2]
            recorte = bgr[int(alto * 0.1):int(alto * 0.9), int(ancho * 0.1):int(ancho * 0.9)]
            diag_forma = "Encuadre completo (fondo complejo o sin contorno nítido)"

        prob_dict, top3 = self._inferir_red(recorte)
        top_clase, top_prob_pct = top3[0]

        # Anotación en imagen
        anotada = bgr.copy()
        if contorno is not None:
            cv2.drawContours(anotada, [contorno], -1, (255, 170, 0), 2)
        cv2.putText(anotada, f"IA: {top_clase} ({top_prob_pct:.1f}%)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        rgb_anotada = cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB)

        # Regla de incertidumbre: si top_prob_pct < UMBRAL_CONFIANZA_MINIMA -> Negro
        if top_prob_pct < UMBRAL_CONFIANZA_MINIMA:
            material = "No identificado"
            motivo = f"Certeza IA insuficiente ({top_prob_pct:.1f}% < {UMBRAL_CONFIANZA_MINIMA:.0f}%)."
            conf = top_prob_pct
        else:
            material = MAPEO_TRASHNET.get(top_clase, "No aprovechable")
            conf = top_prob_pct
            motivo = f"Red neuronal MobileNetV2 identifica '{top_clase}' con {top_prob_pct:.1f}% de confianza."

        info = CATEGORIAS[material]
        return ResultadoClasificacion(
            material=material,
            confianza=conf,
            contenedor=info["contenedor"],
            descripcion=info["descripcion"],
            forma=morf.forma if contorno is not None else "Vista general",
            metodo="Motor IA (MobileNetV2)",
            motivo=motivo,
            top_predicciones=top3,
            diagnostico_forma=diag_forma,
            imagen_anotada=rgb_anotada,
        )

    # -----------------------------------------------------------------------
    # MOTOR 4: Motor Híbrido Estricto (Red Neuronal + Soporte de Forma)
    # -----------------------------------------------------------------------
    def _clasificar_hibrido(self, bgr: np.ndarray) -> ResultadoClasificacion:
        """
        Lógica corregida de fusión:
        - La red neuronal tiene el peso prioritario para clasificar el material.
        - La geometría apoya la decisión (por ejemplo, distinguiendo PEAD vs PET por pico/cuello)
          y refuerza la confianza (+8%) si hay coincidencia estructural.
        - NUNCA descarta ni bloquea la predicción por contradicción con el contorno,
          evitando los falsos rechazos ante fondos de habitación complejos.
        """
        if self.red_onnx is None:
            # Fallback suave hacia el motor geométrico si no está el ONNX
            return self._clasificar_por_geometria(bgr)

        morf, contorno, _ = self._segmentar_y_extraer_morfologia(bgr)

        # Recorte guiado por contorno o encuadre central
        if contorno is not None:
            x, y, w, h = cv2.boundingRect(contorno)
            pad_x, pad_y = int(w * 0.1), int(h * 0.1)
            recorte = bgr[max(0, y - pad_y):min(bgr.shape[0], y + h + pad_y),
                          max(0, x - pad_x):min(bgr.shape[1], x + w + pad_x)]
        else:
            alto, ancho = bgr.shape[:2]
            recorte = bgr[int(alto * 0.1):int(alto * 0.9), int(ancho * 0.1):int(ancho * 0.9)]

        prob_dict, top3 = self._inferir_red(recorte)
        top_clase, top_prob_pct = top3[0]

        # Anotación visual
        anotada = bgr.copy()
        if contorno is not None:
            cv2.drawContours(anotada, [contorno], -1, (0, 220, 100), 2)
            caja = cv2.boxPoints(cv2.minAreaRect(contorno)).astype(int)
            cv2.drawContours(anotada, [caja], -1, (255, 160, 20), 1)

        diag_forma = (
            f"Forma: {morf.forma} | Aspecto: {morf.relacion_aspecto:.2f} | "
            f"Cuello: {morf.ratio_cuello:.2f} | Solidez: {morf.solidez:.2f}"
        )

        # Fusión no destructiva:
        # 1. Base del material viene del mapeo de la red neuronal
        material_base = MAPEO_TRASHNET.get(top_clase, "No aprovechable")
        conf_final = top_prob_pct
        soporte_motivo = ""

        # 2. Refinamiento geométrico según la clase
        if top_clase == "plastic":
            if morf.ratio_cuello < 0.65:
                if morf.descentrado_cuello > 0.15:
                    material_base = "PEAD"
                    soporte_motivo = "Forma con pico lateral apoya envase de polietileno (PEAD)."
                    conf_final = min(99.0, conf_final + 8.0)
                else:
                    material_base = "PET"
                    soporte_motivo = "Forma con cuello simétrico confirma botella PET."
                    conf_final = min(99.0, conf_final + 8.0)
            else:
                material_base = "PET"
                soporte_motivo = "Clasificado como plástico por la red neuronal."
        elif top_clase == "metal" and ("Cilindro" in morf.forma or morf.relacion_aspecto >= 1.2):
            soporte_motivo = "Morfología cilíndrica confirma lata metálica."
            conf_final = min(99.0, conf_final + 8.0)
        elif top_clase in ("cardboard", "paper") and ("Rectangular" in morf.forma or morf.rectangularidad >= 0.80):
            soporte_motivo = "Morfología rectangular corrobora caja o empaque de papel/cartón."
            conf_final = min(99.0, conf_final + 8.0)
        else:
            soporte_motivo = f"Inferencia liderada por red neuronal ('{top_clase}')."

        # 3. Lógica de Fallback ante incertidumbre (<50%)
        if conf_final < UMBRAL_CONFIANZA_MINIMA:
            material = "No identificado"
            motivo = (f"Certeza baja ({conf_final:.1f}% < {UMBRAL_CONFIANZA_MINIMA:.0f}%). "
                      f"Se remite a contenedor Negro por precaución.")
            conf_final = round(conf_final, 1)
        else:
            material = material_base
            motivo = f"IA: {top_clase} ({top_prob_pct:.1f}%). {soporte_motivo}"
            conf_final = round(conf_final, 1)

        cv2.putText(anotada, f"{material} ({conf_final:.1f}%)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        rgb_anotada = cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB)

        info = CATEGORIAS[material]
        return ResultadoClasificacion(
            material=material,
            confianza=conf_final,
            contenedor=info["contenedor"],
            descripcion=info["descripcion"],
            forma=morf.forma,
            metodo="Motor Híbrido Estricto",
            motivo=motivo,
            top_predicciones=top3,
            diagnostico_forma=diag_forma,
            imagen_anotada=rgb_anotada,
        )

    # -----------------------------------------------------------------------
    # API Pública Principal
    # -----------------------------------------------------------------------
    def predecir(self, imagen, motor: str = "Motor Híbrido Estricto") -> ResultadoClasificacion:
        """
        Ejecuta la clasificación según el motor solicitado.

        Args:
            imagen: Foto capturada por st.camera_input, PIL o arreglo BGR.
            motor: Uno de ["Motor Híbrido Estricto", "Motor IA (MobileNetV2)",
                           "Motor Geométrico (Forma/Contornos)", "Motor Básico (Color)"]
        """
        bgr = self._cargar_imagen(imagen)

        if motor == "Motor Básico (Color)":
            return self._clasificar_por_color(bgr)
        elif motor == "Motor Geométrico (Forma/Contornos)":
            return self._clasificar_por_geometria(bgr)
        elif motor == "Motor IA (MobileNetV2)":
            return self._clasificar_por_ia(bgr)
        else:  # "Motor Híbrido Estricto" por defecto
            return self._clasificar_hibrido(bgr)
