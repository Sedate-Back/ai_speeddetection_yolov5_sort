# Yolov5 + StrongSORT with OSNet





<div align="center">
<p>
<img src="MOT16_eval/track_pedestrians.gif" width="400"/> <img src="MOT16_eval/track_all.gif" width="400"/> 
</p>
<br>
<div>
<a href="https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/actions"><img src="https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>
<br>  
<a href="https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

</div>



## Before you run the tracker

1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

`pip install -r requirements.txt`


## Tracking sources

Tracking can be run on most video formats

```bash
$ python track.py --source 0  # webcam
                           img.jpg  # image
                           vid.mp4  # video
                           path/  # directory
                           path/*.jpg  # glob
                           'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                           'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```


## Select object detection and ReID model

### Yolov5

There is a clear trade-off between model inference speed and accuracy. In order to make it possible to fulfill your inference speed/accuracy needs
you can select a Yolov5 family model for automatic download

```bash


$ python track.py --source 0 --yolo_model yolov5n.pt --img 640
                                          yolov5s.pt
                                          yolov5m.pt
                                          yolov5l.pt 
                                          yolov5x.pt --img 1280
                                          ...
```

### StrongSORT

The above applies to StrongSORT models as well. Choose a ReID model based on your needs from this ReID [model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)

```bash


$ python track.py --source 0 --strong-sort-weights osnet_x0_25_market1501.pt
                                                   osnet_x0_5_market1501.pt
                                                   osnet_x0_75_msmt17.pt
                                                   osnet_x1_0_msmt17.pt
                                                   ...
```

## Filter tracked classes

By default the tracker tracks all MS COCO classes.

If you only want to track persons I recommend you to get [these weights](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) for increased performance

```bash
python track.py --source 0 --yolo_model yolov5/weights/crowdhuman_yolov5m.pt --classes 0  # tracks persons, only
```

If you want to track a subset of the MS COCO classes, add their corresponding index after the classes flag

```bash
python track.py --source 0 --yolo_model yolov5s.pt --classes 16 17  # tracks cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov5 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero.





----------

上述内容为mikel-brostrom书写的内容，包括Yolov5&StrongSort的环境配置和快速开始



基于这个项目，笔者对识别后的信息进行处理，**实现了对人、车的速度检测算法**。

特此声明：速度结果只可作为依靠，不可作为判断的证据或标准。该算法目前还有很多优化空间，会在时间条件允许的情况下，进行优化和更新。



## 实现原理

通过设定物体的现实世界宽度，和画面中物体的像素宽度的比值，去等比物体在画面中移动的像素距离，预测出物体在现实世界的移动距离s

再用数据存储工具，对物体出现的时间进行计算，取出现的第一帧为初始位置，再通过8帧，进行时间的计算，得出时间t

通过v=s/t得v，再通过记录v的变化幅度和趋势，得v均值



此算法可以设置阈值，当计算出物体的瞬时速度超过设置的阈值的时候，可以对图像进行导出，留存证据。



## 项目结构

```python
--strong6.0
----.github  
----datas     -> 数据文件夹
----MOT16_eval 
----output_self -> 证据保存文件夹
----runs    ->  运行结果文件夹
----SelfUtils  -> 自用插件（图像生成插件和获取点位信息插件）
----strong_sort  -> strongsort追踪算法源码
----weights -> 权重文件
----yolov5 -> yolov5目标检测源码
----.gitgnore 
----.gitmodules
----LICENSE
----README.md  
----requirements.txt   -> 项目所需包
----track.py -> 追踪程序
```



## 项目说明

```python
if __name__ == "__main__":
    # 定义类的信息
    person_tru = 1 # 人的真实宽度
    per_v_thres = 6.8  # 人移动速度阈值
    car_tru = 2.0  # 车的真实宽度
    car_v_thres = 40 # 车移动速度阈值
    motor_tru = 1.2 # 摩托的真实宽度
    motor_v_thres = 40   # 摩托移动速度阈值
    bus_tru = 3    # 公交车的真实宽度
    truck_tru = 3.2  # 卡车的真实宽度

    # start = time.time()
    opt = parse_opt()  # 超参数内容
    print("opt ：", opt)
    main(opt)  # 将超参数内容传递给main函数
    end = time.time()
    print(f"Cost-time: {end - start}:.2f") # 获取程序运行时间
```



## 项目说明

详情请看项目注释



