import time
import cv2

# 定义一个转换函数，入参为当前时间time.time()
def out_file_name(ts, c, id, v_pre, img):
    dt = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(int(float(ts))))
    out_file = str(dt) + "_" + str(c) + "_" + str(id)+ "_" + str(v_pre) + '.png'
    out_path = f"output_self/{str(dt)[0:10]}/" + out_file
    out_path = str(out_path)
    cv2.imwrite(out_path, img)
    # print("Save Successful!")
