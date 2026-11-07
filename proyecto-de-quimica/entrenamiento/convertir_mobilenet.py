"""
convertir_mobilenet.py — Convierte MobileNetV2 (Caffe, ImageNet) a ONNX.

Origen de los pesos: https://github.com/shicai/MobileNet-Caffe (licencia BSD-3).
OpenCV 5 ya no lee modelos Caffe, por eso se reconstruye la red en ONNX:
    * BatchNorm + Scale se "pliegan" dentro de cada convolución (misma salida,
      menos operaciones).
    * La entrada del ONNX es RGB normalizado con media/desviación de ImageNet
      (lo que produce vision_mock.ModeloTrashNet); el cambio a BGR y la escala
      de Caffe (media [103.94,116.78,123.68], factor 0.017) se hacen dentro del grafo.

Uso:
    python convertir_mobilenet.py <deploy.prototxt> <modelo.caffemodel> <salida.onnx> [--con-fc]
"""
import sys

import numpy as np
import onnx
from google.protobuf import text_format
from onnx import TensorProto, helper, numpy_helper

import caffe_pb2

MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], np.float32)   # RGB
DESV_IMAGENET = np.array([0.229, 0.224, 0.225], np.float32)    # RGB
MEDIA_CAFFE_BGR = np.array([103.94, 116.78, 123.68], np.float32)
ESCALA_CAFFE = 0.017


def convertir(ruta_proto, ruta_pesos, ruta_salida, con_fc=False):
    estructura = caffe_pb2.NetParameter()
    text_format.Merge(open(ruta_proto).read(), estructura)
    pesos_net = caffe_pb2.NetParameter()
    pesos_net.ParseFromString(open(ruta_pesos, "rb").read())
    pesos = {l.name: [np.array(b.data, np.float32).reshape(b.shape.dim) for b in l.blobs]
             for l in pesos_net.layer}

    nodos, inicializadores = [], []
    alias = {}                                   # nombre caffe -> nombre del tensor ONNX

    def const(nombre, arr):
        inicializadores.append(numpy_helper.from_array(arr.astype(np.float32), nombre))
        return nombre

    # --- Preprocesamiento dentro del grafo: RGB-ImageNet -> escala Caffe ---
    # pixel_c = 255*(s_c*x + m_c);  caffe_c = (pixel_c - M_c) * 0.017
    media_caffe_rgb = MEDIA_CAFFE_BGR[::-1]
    a = 255 * DESV_IMAGENET * ESCALA_CAFFE
    b = (255 * MEDIA_IMAGENET - media_caffe_rgb) * ESCALA_CAFFE
    nodos.append(helper.make_node("Mul", ["input", const("pre_a", a.reshape(1, 3, 1, 1))], ["pre_mul"]))
    nodos.append(helper.make_node("Add", ["pre_mul", const("pre_b", b.reshape(1, 3, 1, 1))], ["pre_out"]))
    alias["data"] = "pre_out"

    capas = list(estructura.layer)
    i = 0
    primera_conv = True
    salida = None
    while i < len(capas):
        capa = capas[i]
        t = capa.type
        if t == "Convolution":
            if capa.name == "fc7" and not con_fc:
                break
            W = pesos[capa.name][0].copy()
            bias = pesos[capa.name][1].copy() if len(pesos[capa.name]) > 1 else np.zeros(W.shape[0], np.float32)
            if primera_conv:                     # La red espera BGR: se reordenan los canales
                W = W[:, ::-1, :, :].copy()
                primera_conv = False
            # Plegado de BatchNorm + Scale siguientes
            j = i + 1
            if j < len(capas) and capas[j].type == "BatchNorm":
                media, var, factor = pesos[capas[j].name]
                factor = factor[0] if factor[0] != 0 else 1.0
                media, var = media / factor, var / factor
                eps = capas[j].batch_norm_param.eps or 1e-5
                gamma, beta = pesos[capas[j + 1].name]
                k = gamma / np.sqrt(var + eps)
                W = W * k[:, None, None, None]
                bias = (bias - media) * k + beta
                ultimo = capas[j + 1]
                j += 2
            else:
                ultimo = capa
            p = capa.convolution_param
            kernel = p.kernel_size[0] if p.kernel_size else 1
            stride = p.stride[0] if p.stride else 1
            pad = p.pad[0] if p.pad else 0
            nombre = capa.name.replace("/", "_")
            nodos.append(helper.make_node(
                "Conv", [alias[capa.bottom[0]], const(nombre + "_W", W), const(nombre + "_b", bias)],
                [nombre], kernel_shape=[kernel, kernel], strides=[stride, stride],
                pads=[pad] * 4, group=p.group or 1))
            alias[ultimo.top[0]] = nombre
            alias[capa.top[0]] = nombre
            salida = nombre
            i = j
            continue
        if t == "ReLU":
            nombre = capa.name.replace("/", "_")
            nodos.append(helper.make_node("Relu", [alias[capa.bottom[0]]], [nombre]))
            alias[capa.top[0]] = nombre
            salida = nombre
        elif t == "Split":
            for top in capa.top:
                alias[top] = alias[capa.bottom[0]]
        elif t == "Eltwise":
            nombre = capa.name.replace("/", "_")
            nodos.append(helper.make_node("Add", [alias[x] for x in capa.bottom], [nombre]))
            alias[capa.top[0]] = nombre
            salida = nombre
        elif t == "Pooling":
            nombre = capa.name.replace("/", "_")
            nodos.append(helper.make_node("GlobalAveragePool", [alias[capa.bottom[0]]], [nombre]))
            alias[capa.top[0]] = nombre
            salida = nombre
        elif t in ("Softmax", "Input"):
            pass
        else:
            raise ValueError(f"Capa no soportada: {t} ({capa.name})")
        i += 1

    nodos.append(helper.make_node("Flatten", [salida], ["salida"]))
    dim = 1000 if con_fc else 1280
    grafo = helper.make_graph(
        nodos, "mobilenet_v2",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("salida", TensorProto.FLOAT, [1, dim])],
        inicializadores)
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 13)])
    modelo.ir_version = 8
    onnx.checker.check_model(modelo)
    onnx.save(modelo, ruta_salida)
    print(f"Guardado {ruta_salida}: {len(nodos)} nodos, salida de {dim} dimensiones")


if __name__ == "__main__":
    convertir(sys.argv[1], sys.argv[2], sys.argv[3], con_fc="--con-fc" in sys.argv)
