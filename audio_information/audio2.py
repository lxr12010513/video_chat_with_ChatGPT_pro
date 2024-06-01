from whisper_jax import FlaxWhisperPipline
import time
# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-base")

print("begin")
# JIT compile the forward call - slow, but we only do once
text = pipeline("audio/peanut.mp3")
print(text)
T1 = time.time()
# used cached function thereafter - super fast!!
text = pipeline("audio/peanut.mp3")
print(text)
T2 = time.time()
print('程序运行时间:%s秒' % ((T2 - T1)))