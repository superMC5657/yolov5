# -*- coding: utf-8 -*-
# !@time: 2020/11/22 下午1:54
# !@author: superMC @email: 18758266469@163.com
# !@fileName: export_torch2trt.py
from torch2trt import torch2trt, tensorrt_converter, get_arg, trt, nn, F, torch, add_missing_trt_tensors, \
    add_module_test
from torch2trt.converters.sigmoid import convert_sigmoid
import models
from models.experimental import attempt_load


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


@tensorrt_converter('torch.nn.functional.hardtanh')
def convert_hardtanh(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    min_val = get_arg(ctx, 'min_val', pos=1, default=-1.0)
    max_val = get_arg(ctx, 'max_val', pos=2, default=1.0)
    output = ctx.method_return

    layer = ctx.network.add_activation(input._trt, trt.ActivationType.CLIP)
    layer.alpha = min_val
    layer.beta = max_val

    output._trt = layer.get_output(0)


model = attempt_load("weights/v3.1/yolov5m.pt", map_location=torch.device('cpu')).eval().cuda()
for k, m in model.named_modules():
    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
        m.act = Hardswish()  # assign activation
    # if isinstance(m, models.yolo.Detect):
    #     m.forward = m.forward_export  # assign forward (optional)
x = torch.ones((1, 3, 640, 640)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))
