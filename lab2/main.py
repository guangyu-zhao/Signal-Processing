# 收集四段不同的⻓度为30s的语音
# 预处理音频信号: 统一原始信号的采样率至8000Hz
# 编码处理: 将4段音频的频谱合并到一段采样率为48000Hz的音频中
# 解码处理: 将叠加在一起的多路语音分开，分离得到多路语音

import librosa
import numpy as np
import soundfile

sr = 8000  # 采样率，始终是8000Hz
output_sr = 48000  # 合并频率，始终是48000Hz


def encode(waves: list):
    """编码：将保存4段音频时域采样的 list 中的每段音频分别做 FFT 后，将频谱合并放入一个 ndarray 中，
    对这一 ndarray 整体做 IFFT 得到一个同样长的 ndarray"""
    freq = [np.fft.fft(wave) for wave in waves]
    half_len = len(freq[0]) // 2
    merged = np.concatenate((freq[0][0:half_len], freq[1][0:half_len], freq[2][0:half_len], freq[3][0:half_len],
                             np.zeros(len(freq[0]) * 2), freq[3][half_len:], freq[2][half_len:], freq[1][half_len:],
                             freq[0][half_len:]))
    encoded = np.real(np.fft.ifft(merged))
    return encoded


def decode(wave_merge):
    """解码：将编码后产生的 ndarray 先做 FFT 得到合并的频域数据，分为四段后分别做 IFFT，放入一个 list"""
    freq_merge = np.fft.fft(wave_merge)
    unit_half = len(freq_merge) // 12
    cutted = []
    for i in range(4):
        cutted.append(
            np.concatenate((freq_merge[i * unit_half: (i + 1) * unit_half],
                            freq_merge[(11 - i) * unit_half: (12 - i) * unit_half])))
    decoded: list = [np.real(np.fft.ifft(freq)) for freq in cutted]
    return decoded


if __name__ == '__main__':

    for N in (30, 1, 2, 5, 10):  # 将音频切为N秒一小段

        # 加载音频时域采样信息
        input_waves = []
        for i in range(4):
            y, _ = librosa.load(f"./input_{i}.wav", sr=sr)
            input_waves.append(y)

        # 将编码后的 ndarray 写入文件，模拟传输过程
        encoded = []
        total_len = len(input_waves[0])
        parts = total_len // (N * sr)
        for i in range(parts):
            encoded.append(encode([wave[(total_len // parts) * i: (total_len // parts) * (i + 1)] for wave in input_waves]))
        encoded_merged = np.concatenate(encoded)
        soundfile.write(f"encoded_N={N}.wav", encoded_merged, output_sr)

        # 读入之前写出的文件，模拟信号接收过程
        receive, _ = librosa.load(f"encoded_N={N}.wav", sr=None)

        output_waves = [[], [], [], []]
        for i in range(parts):
            waves_decode = decode(receive[output_sr * N * i: output_sr * N * (i + 1)])
            for j in range(4):
                output_waves[j].append(waves_decode[j])

        wave_after = [np.concatenate(waves) for waves in output_waves]

        # 写出经过编码-解码处理后的音频文件
        for i, wave in enumerate(wave_after):
            soundfile.write(f"output_{i}_N={N}.wav", wave, sr)

        print(f"N = {N} finished")
