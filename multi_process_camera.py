# -*- coding: utf-8 -*-
# !@time: 2020/11/30 下午7:26
# !@author: superMC @email: 18758266469@163.com
# !@fileName: multi_process_camera.py
import ctypes
import multiprocessing as mp
import random
import time

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision

INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.4
IOU_THRESHOLD = 0.5
time_sleep_pre = 0.001
time_sleep_inference = 0.001
time_sleep_post = 0.001
time_sleep_process = 1


def preprocess_image(image_raw):
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = INPUT_W / w
    r_h = INPUT_H / h
    if r_h > r_w:
        tw = INPUT_W
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((INPUT_H - th) / 2)
        ty2 = INPUT_H - th - ty1
    else:
        tw = int(r_h * w)
        th = INPUT_H
        tx1 = int((INPUT_W - tw) / 2)
        tx2 = INPUT_W - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image, h, w


def post_process(output, origin_h, origin_w):
    # Get the num of boxes detected
    num = int(output[0])
    # Reshape to a two dimentional ndarray
    pred = np.reshape(output[1:], (-1, 6))[:num, :]
    # to a torch Tensor
    pred = torch.Tensor(pred).cuda()
    # Get the boxes
    boxes = pred[:, :4]
    # Get the scores
    scores = pred[:, 4]
    # Get the classid
    classid = pred[:, 5]
    # Choose those boxes that score > CONF_THRESH
    si = scores > CONF_THRESH
    boxes = boxes[si, :]
    scores = scores[si]
    classid = classid[si]
    # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    boxes = xywh2xyxy(origin_h, origin_w, boxes)
    # Do nms
    indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
    result_boxes = boxes[indices, :].cpu()
    result_scores = scores[indices].cpu()
    result_classid = classid[indices].cpu()
    return result_boxes, result_scores, result_classid


def xywh2xyxy(origin_h, origin_w, x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    r_w = INPUT_W / origin_w
    r_h = INPUT_H / origin_h
    if r_h > r_w:
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
        y /= r_w
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        y /= r_h

    return y


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA)


def inference(q_put, q_get, engine_file_path):
    cuda.init()
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    ctypes.CDLL(PLUGIN_LIBRARY)
    # Create a Context on this device,
    cfx = cuda.Device(0).make_context()
    stream = cuda.Stream()
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(engine_file_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    bindings = []

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(cuda_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)

    while True:
        if q_put.qsize() == 0:
            time.sleep(time_sleep_pre)
            continue
        inf_start = time.time()
        image_raw, input_image, h, w = q_put.get()
        if type(w) == bool:
            q_get.put((False, False, False, False))
            time.sleep(time_sleep_process)  # 等待
            break
        cfx.push()
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        result_boxes, result_scores, result_classid = post_process(
            output, h, w
        )
        q_get.put((image_raw, result_boxes, result_scores, result_classid))
        print("inference:", time.time() - inf_start)


class ImgStream:
    def __init__(self, input_video_url, output_video_url):
        self.input_video_url = input_video_url
        self.output_video_url = output_video_url
        input_cap = cv2.VideoCapture(input_video_url)

        self.src_video_fps = input_cap.get(cv2.CAP_PROP_FPS)
        self.video_size = (
            int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

        self.categories = [
            'vehicle',
            'bicycle',
            'person',
            'sign',
            'light',
        ]
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.categories]
        input_cap.release()

    def queue_img_put(self, q_put):
        input_cap = cv2.VideoCapture(self.input_video_url)
        while True:
            if q_put.qsize() == 10:
                time.sleep(time_sleep_inference)
                continue
            pre_start = time.time()
            is_opened, frame = input_cap.read()
            if is_opened:
                image, h, w = preprocess_image(frame)
                q_put.put((frame, image, h, w))
            else:
                q_put.put((False, False, False, False))
                time.sleep(time_sleep_process)
                break
            print("preprocess:", time.time() - pre_start)
        input_cap.release()

    def queue_img_get(self, q_get):
        output_cap = cv2.VideoWriter(
            self.output_video_url,
            cv2.VideoWriter_fourcc(*'MJPG'),  # 编码器
            self.src_video_fps,
            self.video_size
        )
        while True:
            if q_get.qsize() == 0:
                time.sleep(time_sleep_post)
                continue
            post_start = time.time()
            image_raw, result_boxes, result_scores, result_classid = q_get.get()
            if type(result_classid) == bool:
                break
            for i in range(len(result_boxes)):
                box = result_boxes[i]
                plot_one_box(
                    box,
                    image_raw,
                    color=self.colors[int(result_classid[i])],
                    label="{}:{:.2f}".format(
                        self.categories[int(result_classid[i])], result_scores[i]
                    ),
                )

            # 　Save image
            cv2.imshow('fxxk', image_raw)
            k = cv2.waitKey(1)
            if k & 0xff == ord('q'):
                break
            output_cap.write(image_raw)
            print("postprocess:", time.time() - post_start)
        output_cap.release()
        cv2.destroyAllWindows()


def run():
    origin_img_q = mp.Queue(maxsize=10)
    result_img_q = mp.Queue(maxsize=10)
    img_stream = ImgStream('inference/video/demo.mp4', 'inference/output_video/output.avi')

    processes = [
        mp.Process(target=img_stream.queue_img_put, args=(origin_img_q,)),
        mp.Process(target=inference, args=(origin_img_q, result_img_q, 'build/yolov5t.engine')),
        mp.Process(target=img_stream.queue_img_get, args=(result_img_q,)),
    ]
    processes_num = len(processes)

    for i in range(processes_num):
        processes[i].start()
    for i in range(processes_num):
        processes[i].join()


if __name__ == '__main__':
    run()
