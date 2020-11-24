# -*- coding: utf-8 -*-
# !@time: 2020/6/20 上午8:37
# !@author: superMC @email: 18758266469@163.com
# !@fileName: yolov5.py

from utils.datasets import *
from utils.general import non_max_suppression, scale_coords


def yolov5_model(weight='weights/yolov5l.pt', use_cuda=True, half=True):
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model = torch.load(weight, map_location=device)['model'].float()

    # 因为某些问题 load 强烈依赖于models内容 为了可以多地复用,先 init 模型再加载模型参数. 此处为保存模型state_dict()"只包括'model'的"
    # torch.save(model.state_dict(), 'weights/yolov5l_resave.pt')

    # model = yolo.Model(model_cfg=model_cfg).to(device)
    # model.load_state_dict(torch.load(weight))
    model.eval()
    if half:
        model.half()
    return model


def image_normal(image, image_size):
    img = letterbox(image, new_shape=image_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img


def crop_box(image, box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    return image[y1:y2, x1:x2]


def detect_person(model, image, use_cuda=True, half=True):
    img = image_normal(image, image_size=640)
    # image 为numpy 格式的原始图片
    img = torch.from_numpy(img)
    if use_cuda:
        img = img.cuda()
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # single image
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    # Apply NMS
    # predict [bbox,conf,classes]
    pred = non_max_suppression(pred, 0.4, 0.5)[0]

    transform_det = scale_coords(img.shape[2:], pred[:, :4], image.shape).round()
    transform_det = transform_det.cpu().detach().numpy()
    transform_det = transform_det.astype(int).tolist()
    person_images = []
    for xyxy in transform_det:
        person_images.append(crop_box(image, xyxy))

    return person_images, transform_det


if __name__ == '__main__':
    model = yolov5_model()
    image = cv2.imread('inference/images/zidane.jpg')
    person_images, boxes = detect_person(model, image)
    for person_image in person_images:
        cv2.imshow('demo', person_image)
        cv2.waitKey(0)
