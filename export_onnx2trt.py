# -*- coding: utf-8 -*-
# !@time: 2020/11/22 上午7:27
# !@author: superMC @email: 18758266469@163.com
# !@fileName: export_onnx2trt.py

import onnx
import onnx_tensorrt.backend as backend
import numpy as np

model = onnx.load("weights/v3.1/yolov5m.onnx")
engine = backend.prepare(model, device='CUDA:0')
input_data = np.random.random(size=(1, 3, 640, 640)).astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
print(output_data.shape)
