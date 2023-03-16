import cv2
import numpy as np


# pts=np.array([[86, 422], [387, 224], [623, 416], [559, 571], [144, 573]], np.int32)
# cv2.polylines(img, [pts], True, (0, 0, 255), 2)     # True表示该图形为封闭图形
# pts = []


# 用鼠标获取点坐标，并加入数组
def onmouse_draw_rect(event, x, y, flags, draw_rects):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        pts.append([x, y])

    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)


# 换成视频的话
# 首先是需要读取pts是否为空，非空可以直接根据pts绘制多边形,直接开启摄像头进行检测
# 为空，读取摄像头第一帧的图像进行标记，标记成功后，按q退出，并在视频中一直出现矩形框
# 如果视频中有多个区域需要检测，可以再开启一次绘制，参数我设置为 pts_obj

# 判断pts是否为空

# pts=[[86, 422], [387, 224], [623, 416], [559, 571], [144, 573]]
# pts = []
# cap = cv2.VideoCapture("rtsp://admin:yyit86536280@192.168.1.109/cam/realmonitor?channel=1&subtype=1")
# ret, frame = cap.read()
# while True:
#
#     if not pts:
#         # 读取视频流的第一帧，并等待
#         ret, frame = cap.read()
#         cv2.namedWindow("frame", 0)
#         cv2.setMouseCallback("frame", onmouse_draw_rect)
#         cv2.imshow("frame", frame)
#         # 开始利用鼠标进行回调
#         c = cv2.waitKey(0)
#         # 获取完点坐标后保存到数组
#         if c == ord("q"):
#             cv2.destroyAllWindows()
#     else:  # 存在就开始绘制，绘制需要无限循环
#         pts_array = np.array(pts, np.int32)
#
#         ret, frame = cap.read()
#         cv2.polylines(frame, [pts_array], True, (0, 0, 255), 2)
#         cv2.imshow("frame", frame)
#         c = cv2.waitKey(1)
#         if c == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()


# 上面绘制功能和展示功能已经实现，这时候需要看看能不能将track那边的画面拿过来这边看；
# 首先需要找到img0进行绘制
# 在vid-show里面看看


def show_regonal_alarm(frame, pts):
    """区域告警"""

    def onmouse_draw_rect(event, x, y, flags, draw_rects):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            pts.append([x, y])

        if event == cv2.EVENT_LBUTTONUP:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)

    cv2.namedWindow("frame", 0)
    cv2.setMouseCallback("frame", onmouse_draw_rect)

    cv2.imshow("frame", frame)
    c = cv2.waitKey(0)
    if c == ord("q"):
        cv2.destroyAllWindows()
        return pts


# 获取区域之后，封闭图形的多个点坐标我们就可以拿到了
#
