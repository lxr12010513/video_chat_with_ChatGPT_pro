from paddleocr import PaddleOCR, draw_ocr
import os
import numpy as np
import cv2
from difflib import SequenceMatcher

def OCR():
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True)  # need to run only once to download and load model into memory

    # 指定图片所在目录
    directory = 'keyframe_extract\extract_result'

    # 获取目录下所有文件
    files = os.listdir(directory)

    # 过滤出图片文件
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 自定义排序函数，根据文件名中的数字排序
    def sort_key(filename):
        # 提取出数字部分
        parts = filename.split('_')
        number_part = parts[1].split('.')[0]  
        return int(number_part)

    # 按数字顺序排序文件
    image_files = sorted(image_files, key=sort_key)
    
    OCR_index = np.arange(len(image_files)-1)
    print(OCR_index)
    OCR_results = []
    for image in image_files:
        result = ocr.ocr(directory+'/'+image, cls=True)
        if result and result[0]:
            OCR_result = []
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    OCR_result.append(line[1][0])
            OCR_results.append(" | ".join(OCR_result))
        else:
            OCR_results.append("")
    OCR_results = ' '.join([f"Second {i+1} : {j}\n" for i,j in zip(OCR_index,OCR_results)])
    return OCR_results


def OCR(video_path):
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True)  # need to run only once to download and load model into memory

    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))  # 获取视频的帧率
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的帧数
    duration = frame_count / fps if fps > 0 else 0  # 计算视频时长
    count = 0
    OCR_results = []
    last_result = ""

    if duration<120:
        k = 1
    elif duration<360:
        k = 2
    elif duration<960:
        k = 8 
    else:
        k = 16

    while True:
        success, image = video.read()
        if not success:
            break
        if count % (k*fps) == 0:  # 每秒提取一帧
            # 直接对帧进行OCR处理
            result = ocr.ocr(image, cls=True)
            if result and result[0]:
                OCR_result = []
                for idx in range(len(result)):
                    res = result[idx]
                    for line in res:
                        OCR_result.append(line[1][0])
                OCR_result = " | ".join(OCR_result)
                if SequenceMatcher(None, OCR_result, last_result).ratio()<0.8:
                    OCR_results.append(OCR_result)
                last_result = OCR_result
        count += 1
    OCR_results = '\n'.join(OCR_results)
    video.release()
    return OCR_results


if __name__ == '__main__':
    print('---------------------')
    print(OCR('./images/yoga.mp4'))