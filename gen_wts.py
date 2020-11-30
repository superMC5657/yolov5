import os

import torch
import struct

from tqdm import tqdm

from utils.general import strip_optimizer
from utils.torch_utils import select_device

# Initialize
device = select_device('cpu')
# Load model
model_name = 'runs/train/scratch_v5_t/weights/epoch_16_total_loss_0.085034.pt'
model = torch.load(model_name, map_location=device)['model'].float().half()  # load to FP16
model.to(device).eval()
for p in model.parameters():
    p.requires_grad = False
# torch.save(model.state_dict(),'weights/v3.1/yolov5l_resave.pt')
# model = Model('models/yolov5l.yaml')
# static_dict = torch.load('weights/v3.1/yolov5l_resave.pt')
# model.load_state_dict(static_dict)  # 没差别

f = open('wts/' + 'yolov5t' + '.wts', 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in tqdm(model.state_dict().items()):
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f', float(vv)).hex())
    f.write('\n')
