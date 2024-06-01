import whisper
import time

path = "video/peanut.mp4"
model = whisper.load_model("medium")
T1 = time.time()
result = model.transcribe(path)
print(result["text"])
T2 = time.time()
print('程序运行时间:%s秒' % ((T2 - T1)))