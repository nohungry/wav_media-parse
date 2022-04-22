import wave
import matplotlib.pyplot as plt
import numpy as np

# 解析WAV音檔的頻率
# ---------------------------------------------------------

f = wave.open(r"C:\Users\norman_cheng\Desktop\voice\testusing_folder\copycat_test\double.wav", "rb")
fft_size = 4000
origin_params = f.getparams()
nchannels, sampwidth, framerate, nframes = origin_params[:4]

origin_str_data = f.readframes(nframes)
f.close()

# 可以由nchannels判斷單聲道/ 雙聲道
if nchannels == 1:
    wave_data = np.frombuffer(origin_str_data, dtype=np.short)
    wave_data = wave_data.T
    time = np.arange(0, nframes) * (1.0 / framerate)
    fft_y1 = np.fft.rfft(wave_data[0][0:0 + fft_size - 1]) / fft_size  # 左聲部
    fft_y1 = np.abs(fft_y1)
elif nchannels == 2:
    wave_data = np.frombuffer(origin_str_data, dtype=np.short)
    wave_data.shape = -1, 2
    wave_data = wave_data.T
    time = np.arange(0, nframes) * (1.0 / framerate)
    fft_y1 = np.fft.rfft(wave_data[0][0:0 + fft_size - 1]) / fft_size  # 左聲部
    fft_y2 = np.fft.rfft(wave_data[1][0:0 + fft_size - 1]) / fft_size  # 右聲部
    fft_y1 = np.abs(fft_y1)
    fft_y2 = np.abs(fft_y2)

# 音訊資訊
print(nchannels, sampwidth, framerate, nframes, time[-1])

# 畫波形
plt.subplot(211)
plt.plot(time, wave_data[0])
plt.subplot(212)
plt.plot(time, wave_data[1], c="g")
plt.xlabel("time (seconds)")
plt.show()

# 製作WAV音檔
# 可以利用上面解析的頻率域再反轉製作一樣的音檔
# ---------------------------------------------------------

time = 1
framerate = 44100

def sinSweepFrq(f0, f1, framerate, time):
    nframes = time * framerate
    k = (f1 - f0) / time
    t = np.linspace(0, time, num=nframes)
    # f = f0 + kt
    # w = 2 * pi * f = 2 * pi * f0 + 2 * pi * kt
    # w 對 t 積分得 phase
    # phase = 2 * pi * f0 * t + pi * kt^2
    phase = 2 * np.pi * f0 * t + np.pi * k * t * t

    return np.sin(phase)


nframes = time * framerate

n = np.linspace(0, time, num=nframes)

f_start = 20  # Hz
f_stop = 1000  # Hz

sinus_f_start = np.sin(2 * np.pi * f_start * n)
sinus_f_stop = np.sin(2 * np.pi * f_stop * n)
# 產生 10秒 44.1kHz 的 20Hz - 1kHz 的 sine
sinus_f_sweep = sinSweepFrq(f_start, f_stop, framerate, time)

# 先放大以免轉換 np.short 為 0，並且太小聲
wave_data = sinus_f_sweep * 10000
wave_data = wave_data.astype(np.short)

# 寫入檔案
f = wave.open(r"C:\Users\norman_cheng\Desktop\voice\testusing_folder\copycat_test\sweep_single.wav", "wb")

# 設定聲道數、sample 大小(byte) 和取樣頻率
f.setnchannels(1)
f.setnchannels(2)
f.setsampwidth(sampwidth)
f.setframerate(framerate)
# 寫入波形數據
f.writeframes(wave_data.tostring())
f.close()

# 畫圖確認
fig = plt.figure()

fig.add_subplot(411)
plt.plot(n, sinus_f_start)
fig.add_subplot(412)
plt.plot(n, sinus_f_stop)
fig.add_subplot(413)
plt.plot(n, sinus_f_sweep)
fig.add_subplot(414)
plt.plot(n, wave_data)
plt.show()