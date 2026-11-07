# Auditoría técnica: video en tiempo real, base de datos e ingesta de datasets públicos

**Proyecto:** Sistema de Reciclaje Inteligente  
**Archivos afectados:** `app.py`, `database.py`, `ingesta_publica.py` (nuevo), `requirements.txt`, `assets/styles.css`, `entrenamiento/evaluar_sistema.py`  
**Fecha:** noviembre de 2026

---

## 1. Diagnóstico del lag de cámara

### Causa raíz: el modelo de ejecución de Streamlit
Streamlit vuelve a ejecutar **todo el script** de arriba abajo en cada interacción. Con `st.camera_input` (o con un bucle de `cv2.VideoCapture` + `st.image`) cada fotograma implica:

1. El navegador codifica una imagen JPEG y la sube por WebSocket.
2. Se vuelve a ejecutar `app.py` completo: CSS, barra lateral, consultas a SQLite, `hashlib.md5` sobre los bytes.
3. La inferencia (segmentación OpenCV + MobileNetV2) corre **en el mismo hilo que dibuja la interfaz**, así que la página se congela mientras dura.
4. Se serializan la imagen anotada y los widgets y se reenvían al navegador.

Resultado: de 1 a 3 "fotogramas" por segundo, con la interfaz congelada entre uno y otro. Si se usa un bucle `while True` con `st.image`, los frames además **se acumulan en cola** y el retraso crece sin límite.

### Hallazgos secundarios
| # | Hallazgo | Impacto |
|---|----------|---------|
| H1 | `cv2.dnn.Net` compartido por `@st.cache_resource` sin candado | Con dos pestañas o dos hilos, la salida se corrompe o el proceso se bloquea |
| H2 | `database.py` dependía de `streamlit` (`@st.cache_resource`) | No se podía usar desde otro hilo ni desde un script independiente |
| H3 | Una conexión SQLite global + `Lock` para lecturas y escrituras | El tablero bloqueaba a la cámara y viceversa |
| H4 | MobileNetV2 es un clasificador **de conjunto cerrado** | Con la mesa vacía responde "Vidrio 78 %": registraría el fondo y nunca sabría que el objeto se retiró |
| H5 | Paleta del tablero: PET (#3B82F6) y PEAD (#8B5CF6) | Indistinguibles para daltonismo (ΔE 1,3) |
| H6 | `evaluar_sistema.py` usaba `modelo.motor` (no existe) | `AttributeError` al ejecutarlo |
| H7 | `.gitignore` excluye `data/` pero `data/reciclaje.db` está versionado | Datos de prueba viajan con el repositorio |

## 2. Solución: pipeline WebRTC desacoplado

```
Navegador ──WebRTC──► recv() [hilo aiortc, ~2 ms]  ──► frame + HUD ──► Navegador
                         │ guarda solo el último frame
                         ▼
                  buzón de 1 frame (los viejos se descartan)
                         │
                         ▼
          _bucle_inferencia() [hilo propio, ≤ 8 inf/s]
            1. DetectorPresencia (MOG2) en la zona de interés ── vacío → no infiere
            2. MobileNetV2 / OpenCV (con candado compartido)
            3. EstabilizadorDetecciones (6 de 8 votos, confianza > 60 %)
            4. SQLite (un registro por objeto)
                         │
                         ▼
              estado compartido ◄── st.fragment(run_every=1 s): solo se redibuja el panel
```

Decisiones clave:

* **`recv()` nunca infiere.** Solo copia el frame y dibuja el último resultado. Mediana medida: **1,7 ms** (p95 2,6 ms).
* **Buzón de un frame.** Si la IA tarda más que la cámara, los frames intermedios se descartan en lugar de encolarse. La latencia queda acotada.
* **Tasa de inferencia limitada** (`INTERVALO_INFERENCIA_S = 0.12`), para dejar CPU libre al códec VP8 de aiortc, que codifica por software.
* **Resolución 640×480** en `media_stream_constraints`: el motor reduce de todos modos a 480 px, y pedir 720p cuadruplica el costo de (de)codificación.
* **`st.fragment(run_every=1.0)`** actualiza solo el panel de resultados. Se verificó en el código de Streamlit 1.67 que los reruns de fragmento no incrementan `script_run_count`, así que `streamlit-webrtc` no reinicia el stream.
* **Estabilizador + detector de presencia.** Un objeto que permanece frente a la cámara se registra **una sola vez**. El sistema se rearma cuando la zona queda vacía. Las escenas vacías no se registran nunca.
* **Modo de respaldo "Foto única"** (`st.camera_input`) para redes donde WebRTC no conecta.

### Mediciones (CPU, sin GPU)
| Métrica | Antes | Después |
|---|---|---|
| Fotogramas mostrados | 1–3 fps, con congelamiento | 18–28 fps fluidos |
| Costo por frame en el hilo de video | inferencia completa (~15–80 ms) | 1,7 ms |
| Inferencias de IA | 1 por rerun | ~8/s (configurable) |
| Reruns completos de la página | 1 por frame | 0 mientras hay video |

Prueba de punta a punta: Chrome headless con cámara falsa → WebRTC → IA → SQLite. Secuencia vacío/vidrio/vacío/lata/vacío/cartón: **1 registro por objeto y 0 registros de fondo**.

## 3. Base de datos (`database.py`)

* Tabla `clasificaciones(id, fecha_hora, material, confianza, contenedor)` más columnas de trazabilidad opcionales: `origen`, `motor`, `etiqueta_real` y `archivo`.
* Restricciones `CHECK` (confianza entre 0 y 100; contenedor ∈ {Blanco, Negro, Verde}) e índices por fecha, origen y material.
* **WAL + `busy_timeout` + una conexión corta por operación.** Lectores y escritores no se bloquean, y funciona desde cualquier hilo o proceso. Prueba: 4 hilos × 50 inserciones simultáneas sin errores.
* Índice único parcial `(origen, archivo, motor)`: relanzar la ingesta no duplica registros.
* **Migración automática** desde `eventos_reciclaje`. La tabla antigua se conserva como `eventos_reciclaje_migrada_v1`.
* Las agregaciones del tablero se hacen en SQL (`GROUP BY`). Con **50 000 registros**, todas las consultas del tablero suman 144 ms sin caché. El tablero además cachea por la "firma" `(COUNT, MAX(id))` de la tabla.
* Se mantienen `insert_evento`, `get_estadisticas` y `reset_db` por compatibilidad.

## 4. Ingesta de bancos públicos (`ingesta_publica.py`)

* Detecta automáticamente **TrashNet** (carpetas por clase), **TACO** (`annotations.json` en formato COCO) o una carpeta genérica.
* TACO: recorta cada *bounding box* con un margen del 15 % y aplica la **orientación EXIF** (muchas fotos de celular vienen rotadas). Mapea las **60 categorías** oficiales a los grupos del sistema.
* Simula la cámara: reduce a 640 px y aplica compresión JPEG con calidad 75.
* Inserta por lotes en una sola transacción. Opciones `--dry-run`, `--reporte-csv`, `--distribuir-dias` y `--borrar-origen`.
* **Fuga de datos:** el modelo se entrenó con TrashNet. `--solo-prueba` reproduce exactamente el 15 % reservado (semilla 42). Resultado medido con simulación de webcam: **85,3 % de exactitud** (89,2 % reportado sin degradación). TACO es la validación verdaderamente independiente.

## 5. Despliegue

* `localhost` funciona sin configuración adicional. Para acceder desde otro equipo de la red, el navegador exige **HTTPS** para usar la cámara: `streamlit run app.py --server.sslCertFile cert.pem --server.sslKeyFile key.pem`.
* Detrás de NAT estricto (redes universitarias, 4G) hace falta un servidor **TURN**, además de STUN. `streamlit-webrtc` incluye `get_twilio_ice_servers` y `get_cloudflare_ice_servers`.
