import argparse

import os
import time
from math import sqrt

from SelfUtils.get_img import out_file_name

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory E:\strongsort\strong6.0
# print("ROOT:",ROOT)
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args,
                                  check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


@torch.no_grad()
def run(
        source='',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,

        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference

):
    # source 填入需要跑的文件
    # 1. 去判断填入的变量的类型：文件、url、摄像头等
    global per_v_thres, car_v_thres, person_tru, car_tru, motor_v_thres, motor_tru, bus_tru, truck_tru, regional_alarm_pts
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)  # 匹配是不是视频类型
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:  # 如果符合url和视频文件， 先检查是不是有文件在项目里，没有的话开始通过url进行下载
        source = check_file(source)  # download

    # Directories 目录
    if not isinstance(yolo_weights, list):  # single yolo model  单个lovo模型
        exp_name = str(yolo_weights).rsplit('/', 1)[-1].split('.')[0]  # 模型名称，例如 yolov5m
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights  如果传入的是列表，列表里长度为1
        exp_name = yolo_weights[0].split(".")[0]  # 取文件名，例如 yolov5m
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'  # 读不到
    # exp_name 这个就是reid 的模型
    exp_name = name if name is not None else exp_name + "\\" + str(strong_sort_weights).split('\\')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run 对保存的路径进行校验，有就直接保存，没有就创建文件夹保存
    # print("save_dir:",save_dir)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model  加载模型
    device = select_device(device)  # 选择设备进行运行
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)  # 匹配设备的状态和相关模块是否可以用，不行就下载
    stride, names, pt = model.stride, model.names, model.pt  # 步态、 名字、 模型文件
    imgsz = check_img_size(imgsz, s=stride)  # check image size  检查图片的尺寸

    # Dataloader  # 数据加载器，如果source是txt文件，就先加载；如果不是就看看有没有dataset的文件
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT  初始化sort，读取yaml文件
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)
    # Create as many strong sort instances as there are video sources
    # 如果是视频文件的话，创建sort实例
    strongsort_list = []
    # nr_sources 来源与dataset/ =1， 所以就是循环给strong_list加元组
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                'E:/strongsort/strong6.0/weights/osnet_x0_25_msmt17.pt',
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )

    outputs = [None] * nr_sources

    # Run tracking  运行追踪
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources  # 总帧数和当前帧数

    carid_list = []
    carid_mes = {}
    personid_list = []
    personid_mes = {}
    motorid_list = []
    motorid_mes = {}
    busid_list = []
    busid_mes = {}
    truckid_list = []
    truckid_mes = {}

    count = 0

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()  # 获取时间
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()  # 获取时间
        dt[0] += t2 - t1  # 计算花费的时间
        # print("frame_idx: ", frame_idx)
        # Inference 推断
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2  # 推断所需要的时间

        # Apply NMS  应用
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3  # 推断所需要的时间

        # Process detections  物体检测
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        # bbox_left = output[0]
                        # bbox_top = output[1]
                        # bbox_w = output[2] - output[0]
                        # bbox_h = output[3] - output[1]

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]

                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id

                            # 运算部分
                            # 车
                            if c == 2:
                                if id not in carid_list:  # 不存在，就存储id和存储信息
                                    carid_list.append(id)
                                    # 存储x的中心点、y的中心点、w宽度、出现在第几帧
                                    x_center_new = output[0]
                                    y_center_new = output[1]
                                    count += 1
                                    v_pre = 20
                                    carid_mes[str(id)] = [x_center_new, y_center_new, output[2] - output[0], frame_idx,
                                                          v_pre, count]
                                    label = f'{id} {names[c]} {v_pre} km/h '
                                    annotator.box_label(bboxes, label, color=colors(0, True))
                                else:  # 存在，就判断位置信息
                                    if carid_mes.get(str(id))[5] < 20:
                                        x_center_old, y_center_old, car_piex, seen, v_pre = carid_mes.get(str(id))[0: 5]
                                        count += 1
                                        carid_mes[str(id)] = [x_center_new, y_center_new, car_piex, seen, v_pre, count]
                                        if not v_pre:
                                            label = f' {names[c]} {v_pre}km/h '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0 <= v_pre < 6:
                                            label = f' {names[c]} idling '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 6 <= v_pre < 0.9 * car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0.9 * car_v_thres <= v_pre < car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(1, True))
                                        elif car_v_thres <= v_pre:  # 将图片保存到本地
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(2, True))
                                            out_file_name(int(time.time()), c, id, v_pre, im0)
                                    elif carid_mes.get(str(id))[5] >= 8:
                                        count = 0
                                        x_center_old, y_center_old = carid_mes.get(str(id))[0], carid_mes.get(str(id))[
                                            1]
                                        x_center_new, y_center_new = output[0], output[1]
                                        pixel_rem_car = sqrt(
                                            (x_center_new - x_center_old) ** 2 + (y_center_new - y_center_old) ** 2)
                                        tru_rem = round(car_tru / (output[2] - output[0]) * pixel_rem_car, 2)
                                        # 计算时间
                                        seen_old = carid_mes.get(str(id))[3]
                                        t = round((int(frame_idx) - int(seen_old)) * (1 / 25) * 3, 2)
                                        # 计算速度
                                        v_pre = round(tru_rem / t * 3.6, 2)
                                        carid_mes[str(id)] = [x_center_new, y_center_new, output[2] - output[0],
                                                              frame_idx, v_pre, count]
                                        if not v_pre:
                                            label = f' {names[c]}  '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0 <= v_pre < 6:
                                            label = f' {names[c]} idling '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 6 <= v_pre < 0.9 * car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0.9 * car_v_thres <= v_pre < car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(1, True))
                                        elif car_v_thres <= v_pre:  # 将图片保存到本地
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(2, True))
                                            out_file_name(int(time.time()), c, id, v_pre, im0)
                            # 人
                            elif c == 0:
                                if id not in personid_list:  # 不存在，就存储id和存储信息
                                    personid_list.append(id)
                                    # 存储框框的底部中心点、w宽度、出现在第几帧
                                    x_center_new = output[0]
                                    y_center_new = output[1]
                                    count += 1
                                    v_pre = 0.6
                                    personid_mes[str(id)] = [x_center_new, y_center_new, output[2] - output[0],
                                                             frame_idx, v_pre, count]
                                    label = f' {names[c]} {v_pre} km/h '
                                    annotator.box_label(bboxes, label, color=colors(0, True))
                                else:  # 存在，就开始循环
                                    if personid_mes.get(str(id))[5] < 8:
                                        x_center_new, y_center_new, per_piex, seen, v_pre = personid_mes.get(str(id))[
                                                                                            0: 5]
                                        count += 1
                                        personid_mes[str(id)] = [x_center_new, y_center_new, per_piex, seen, v_pre,
                                                                 count]
                                        # print(personid_mes)
                                        if not v_pre:

                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0 <= v_pre < 1:
                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 1 <= v_pre < 0.9 * per_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0.9 * per_v_thres <= v_pre < per_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(1, True))
                                        elif per_v_thres <= v_pre:  # 将图片保存到本地
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(2, True))
                                            out_file_name(int(time.time()), c, id, v_pre, im0)
                                    elif personid_mes.get(str(id))[5] >= 8:
                                        count = 0
                                        # 计算距离
                                        x_center_old, y_center_old = personid_mes.get(str(id))[0], \
                                                                     personid_mes.get(str(id))[1]
                                        x_center_new, y_center_new = output[0], output[1]
                                        pixel_rem_person = sqrt(
                                            (x_center_new - x_center_old) ** 2 + (y_center_new - y_center_old) ** 2)
                                        tru_rem = round(person_tru / (output[2] - output[0]) * pixel_rem_person, 2)
                                        # 计算时间
                                        seen_old = personid_mes.get(str(id))[3]
                                        # print(f"当前处于第{frame_idx}帧，上一次物体出现在第{seen_old}帧")
                                        t = round((int(frame_idx) - int(seen_old)) * (1 / 25) * 3.8, 2)
                                        # 计算速度
                                        v_pre = round(tru_rem / t * 3.6, 2)
                                        print(f"{id}当前的速度：{v_pre} Km/h!，花费了{t}秒，走了{tru_rem}的距离")
                                        # 计算之后更新字典里的id信息
                                        personid_mes[str(id)] = [x_center_new, y_center_new, output[2] - output[0],
                                                                 frame_idx, v_pre, count]
                                        # print(personid_mes)
                                        # 将速度写在标签中
                                        if not v_pre:
                                            label = f' {names[c]} '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0 <= v_pre < 1:
                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 1 <= v_pre < 0.9 * per_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0.9 * per_v_thres <= v_pre < per_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(1, True))
                                        elif per_v_thres <= v_pre:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(2, True))
                                            out_file_name(int(time.time()), c, id, v_pre, im0)

                            # 摩托车
                            elif c == 3:
                                if id not in motorid_list:  # 不存在，就存储id和存储信息
                                    motorid_list.append(id)
                                    # 存储框框的底部中心点、w宽度、出现在第几帧
                                    x_center_new = output[0]
                                    y_center_new = output[1]
                                    count += 1
                                    v_pre = 6
                                    motorid_mes[str(id)] = [x_center_new, y_center_new, output[2] - output[0],
                                                            frame_idx, v_pre, count]
                                    label = f' {names[c]} {v_pre} km/h '
                                    annotator.box_label(bboxes, label, color=colors(0, True))
                                else:  # 存在，就开始循环
                                    if motorid_mes.get(str(id))[5] < 8:
                                        x_center_new, y_center_new, per_piex, seen, v_pre = motorid_mes.get(str(id))[
                                                                                            0: 5]
                                        count += 1
                                        motorid_mes[str(id)] = [x_center_new, y_center_new, per_piex, seen, v_pre,
                                                                count]
                                        if not v_pre:
                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0 <= v_pre < 1:
                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 1 <= v_pre < 0.9 * motor_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0.9 * motor_v_thres <= v_pre < motor_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(1, True))
                                        elif motor_v_thres <= v_pre:  # 将图片保存到本地
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(2, True))
                                            out_file_name(int(time.time()), c, id, v_pre, im0)
                                    elif motorid_mes.get(str(id))[5] >= 8:
                                        count = 0
                                        # 计算距离
                                        x_center_old, y_center_old = motorid_mes.get(str(id))[0], \
                                                                     motorid_mes.get(str(id))[1]
                                        x_center_new, y_center_new = output[0], output[1]
                                        pixel_rem_motor = sqrt(
                                            (x_center_new - x_center_old) ** 2 + (y_center_new - y_center_old) ** 2)
                                        tru_rem = round(motor_tru / (output[2] - output[0]) * pixel_rem_motor, 2)
                                        # 计算时间
                                        seen_old = motorid_mes.get(str(id))[3]
                                        # print(f"当前处于第{frame_idx}帧，上一次物体出现在第{seen_old}帧")
                                        t = round((int(frame_idx) - int(seen_old)) * (1 / 25) * 3.8, 2)
                                        # 计算速度
                                        v_pre = round(tru_rem / t * 3.6, 2)
                                        print(f"{id}当前的速度：{v_pre} Km/h!，花费了{t}秒，走了{tru_rem}的距离")
                                        # 计算之后更新字典里的id信息
                                        motorid_mes[str(id)] = [x_center_new, y_center_new, output[2] - output[0],
                                                                frame_idx, v_pre, count]

                                        # 将速度写在标签中
                                        if not v_pre:
                                            label = f' {names[c]} '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0 <= v_pre < 1:
                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 1 <= v_pre < 0.9 * motor_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0.9 * motor_v_thres <= v_pre < motor_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(1, True))
                                        elif motor_v_thres <= v_pre:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(2, True))
                                            out_file_name(int(time.time()), c, id, v_pre, im0)

                            # 公共汽车
                            elif c == 5:
                                if id not in busid_list:  # 不存在，就存储id和存储信息
                                    busid_list.append(id)
                                    # 存储框框的底部中心点、w宽度、出现在第几帧
                                    x_center_new = output[0]
                                    y_center_new = output[1]
                                    count += 1
                                    v_pre = 0.6
                                    busid_mes[str(id)] = [x_center_new, y_center_new, output[2] - output[0],
                                                          frame_idx, v_pre, count]
                                    label = f' {names[c]} {v_pre} km/h '
                                    annotator.box_label(bboxes, label, color=colors(0, True))
                                else:  # 存在，就开始循环
                                    if busid_mes.get(str(id))[5] < 8:
                                        x_center_new, y_center_new, per_piex, seen, v_pre = busid_mes.get(str(id))[
                                                                                            0: 5]
                                        count += 1
                                        busid_mes[str(id)] = [x_center_new, y_center_new, per_piex, seen, v_pre,
                                                              count]
                                        # print(personid_mes)
                                        if not v_pre:
                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0 <= v_pre < 1:
                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 1 <= v_pre < 0.9 * car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0.9 * car_v_thres <= v_pre < car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(1, True))
                                        elif car_v_thres <= v_pre:  # 将图片保存到本地
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(2, True))
                                            out_file_name(int(time.time()), c, id, v_pre, im0)
                                    elif busid_mes.get(str(id))[5] >= 8:
                                        count = 0
                                        # 计算距离
                                        x_center_old, y_center_old = busid_mes.get(str(id))[0], \
                                                                     busid_mes.get(str(id))[1]
                                        x_center_new, y_center_new = output[0], output[1]
                                        pixel_rem_bus = sqrt(
                                            (x_center_new - x_center_old) ** 2 + (y_center_new - y_center_old) ** 2)
                                        tru_rem = round(bus_tru / (output[2] - output[0]) * pixel_rem_bus, 2)
                                        # 计算时间
                                        seen_old = busid_mes.get(str(id))[3]
                                        # print(f"当前处于第{frame_idx}帧，上一次物体出现在第{seen_old}帧")
                                        t = round((int(frame_idx) - int(seen_old)) * (1 / 25) * 3.8, 2)
                                        # 计算速度
                                        v_pre = round(tru_rem / t * 3.6, 2)
                                        print(f"{id}当前的速度：{v_pre} Km/h!，花费了{t}秒，走了{tru_rem}的距离")
                                        # 计算之后更新字典里的id信息
                                        busid_mes[str(id)] = [x_center_new, y_center_new, output[2] - output[0],
                                                              frame_idx, v_pre, count]

                                        # 将速度写在标签中
                                        if not v_pre:
                                            label = f' {names[c]} '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0 <= v_pre < 1:
                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 1 <= v_pre < 0.9 * car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0.9 * car_v_thres <= v_pre < car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(1, True))
                                        elif car_v_thres <= v_pre:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(2, True))
                                            out_file_name(int(time.time()), c, id, v_pre, im0)

                            # 卡车
                            elif c == 7:
                                if id not in truckid_list:  # 不存在，就存储id和存储信息
                                    truckid_list.append(id)
                                    # 存储框框的底部中心点、w宽度、出现在第几帧
                                    x_center_new = output[0]
                                    y_center_new = output[1]
                                    count += 1
                                    v_pre = 0.6
                                    truckid_mes[str(id)] = [x_center_new, y_center_new, output[2] - output[0],
                                                            frame_idx, v_pre, count]
                                    label = f' {names[c]} {v_pre} km/h '
                                    annotator.box_label(bboxes, label, color=colors(0, True))
                                else:  # 存在，就开始循环
                                    if truckid_mes.get(str(id))[5] < 8:
                                        x_center_new, y_center_new, per_piex, seen, v_pre = truckid_mes.get(
                                            str(id))[
                                                                                            0: 5]
                                        count += 1
                                        truckid_mes[str(id)] = [x_center_new, y_center_new, per_piex, seen, v_pre,
                                                                count]
                                        # print(personid_mes)
                                        if not v_pre:
                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0 <= v_pre < 1:
                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 1 <= v_pre < 0.9 * car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0.9 * car_v_thres <= v_pre < car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(1, True))
                                        elif car_v_thres <= v_pre:  # 将图片保存到本地
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(2, True))
                                            out_file_name(int(time.time()), c, id, v_pre, im0)
                                    elif truckid_mes.get(str(id))[5] >= 8:
                                        count = 0
                                        # 计算距离
                                        x_center_old, y_center_old = truckid_mes.get(str(id))[0], \
                                                                     truckid_mes.get(str(id))[1]
                                        x_center_new, y_center_new = output[0], output[1]
                                        pixel_rem_truck = sqrt(
                                            (x_center_new - x_center_old) ** 2 + (
                                                    y_center_new - y_center_old) ** 2)
                                        tru_rem = round(truck_tru / (output[2] - output[0]) * pixel_rem_truck, 2)
                                        # 计算时间
                                        seen_old = truckid_mes.get(str(id))[3]
                                        # print(f"当前处于第{frame_idx}帧，上一次物体出现在第{seen_old}帧")
                                        t = round((int(frame_idx) - int(seen_old)) * (1 / 25) * 3.8, 2)
                                        # 计算速度
                                        v_pre = round(tru_rem / t * 3.6, 2)
                                        print(f"{id}当前的速度：{v_pre} Km/h!，花费了{t}秒，走了{tru_rem}的距离")
                                        # 计算之后更新字典里的id信息
                                        truckid_mes[str(id)] = [x_center_new, y_center_new, output[2] - output[0],
                                                                frame_idx, v_pre, count]

                                        # 将速度写在标签中
                                        if not v_pre:
                                            label = f' {names[c]} '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0 <= v_pre < 1:
                                            label = f' {names[c]} standstill '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 1 <= v_pre < 0.9 * car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(0, True))
                                        elif 0.9 * car_v_thres <= v_pre < car_v_thres:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(1, True))
                                        elif car_v_thres <= v_pre:
                                            label = f' {names[c]} {v_pre} km/h '
                                            annotator.box_label(bboxes, label, color=colors(2, True))

                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                    c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                # LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')
                # s: video 1/1 (x/sumframe) 文件 + 文件的w x h + 识别到的内容 + 模型需要的时间

            else:
                strongsort_list[i].increment_ages()
                # LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 25, im0.shape[1], im0.shape[0]
                        # fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str,
                        default='datas/8.mp4',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))  # 检查环境是否满足条件
    run(**vars(opt))


if __name__ == "__main__":
    # 定义类的信息
    person_tru = 1
    per_v_thres = 6.8  # 人
    car_tru = 2.0  # 车
    car_v_thres = 40
    motor_tru = 1.2
    motor_v_thres = 40
    bus_tru = 3
    truck_tru = 3.2
    # start = time.time()
    opt = parse_opt()  # 超参数内容
    print("opt ：", opt)
    main(opt)  # 将超参数内容传递给main函数
    end = time.time()
    # print(f"Cost-time: {end - start}:.2f")
