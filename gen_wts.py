import torch
import struct
from utils.torch_utils import select_device
from models.yolo import Model

# Initialize
device = select_device('cpu')
# Load model
model = torch.load('weights/v3.1/yolov5s.pt', map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

# torch.save(model.state_dict(),'weights/v3.1/yolov5l_resave.pt')
# model = Model('models/yolov5l.yaml')
# static_dict = torch.load('weights/v3.1/yolov5l_resave.pt')
# model.load_state_dict(static_dict)  # 没差别

f = open('weights/v3.1/yolov5s.wts', 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f', float(vv)).hex())
    f.write('\n')
