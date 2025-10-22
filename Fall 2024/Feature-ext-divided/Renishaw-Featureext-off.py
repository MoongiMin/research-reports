import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# 파일 경로
file_path = r"C:\Users\mmgzz\Desktop\ME Lab Machine Learning\Renishaw Sound\Renishaw Sound\Renishaw2\20241108_180130.163479Z_off_state_sound0-5min.wav"

# 데이터 로드
sound, rate = librosa.load(file_path, sr=None)
print(f"Sample Rate: {rate} Hz")
print(f"Sound Shape: {sound.shape}")

# 오디오 데이터 시각화
plt.figure(figsize=(16, 5))
plt.title("Raw Sound Wave")
plt.plot(sound)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# 스펙트로그램 계산
nperseg = int(rate * 1)  # 1초 길이의 세그먼트
noverlap = int(rate * 0.8)  # 80% 오버랩

f, t, sound_stft = signal.spectrogram(sound, fs=rate, nperseg=nperseg, noverlap=noverlap)
print(f"STFT Shape: {sound_stft.shape}")

# 스펙트로그램 시각화
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f, 10 * np.log10(sound_stft), cmap="jet")
plt.title("Spectrogram (STFT)", fontsize=15)
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Frequency [Hz]", fontsize=12)
plt.colorbar(label="Amplitude [dB]")
plt.show()

# 데이터 세그먼트화
segment_length = int(rate * 1)  # 1초 세그먼트
overlap = int(rate * 0.8)  # 80% 오버랩
segments = []

for start in range(0, len(sound) - segment_length, segment_length - overlap):
    segment = sound[start : start + segment_length]
    segments.append(segment)

segments = np.array(segments)
print(f"Number of Segments: {segments.shape[0]}")
print(f"Segment Shape: {segments.shape}")

# 세그먼트 저장
save_path = r"C:\Users\mmgzz\Desktop\ME Lab Machine Learning\Renishaw Segments"
os.makedirs(save_path, exist_ok=True)

np.save(os.path.join(save_path, "segments.npy"), segments)
print(f"Segments saved at: {save_path}")

# 확장된 특징 추출 함수
def extract_features(segment, sr):
    rms = np.sqrt(np.mean(segment**2))  # RMS: 신호의 에너지 크기를 나타냄. Running 상태에서는 에너지가 더 큼.
    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()  # Spectral Centroid: 주파수 분포의 중심(소리의 밝기). Running 상태에서 더 높을 가능성.
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr).mean()  # Spectral Bandwidth: 주파수 분포의 폭(소리의 복잡성). Running 상태에서 대역폭이 더 넓음.
    zero_crossing_rate = librosa.feature.zero_crossing_rate(segment).mean()  # Zero Crossing Rate: 신호의 0을 지나는 횟수(소리의 거칠기). Running 상태에서 더 높을 가능성.
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).mean(axis=1)  # MFCC: 소리의 주파수 특징을 요약. Running 상태의 복잡한 음향 패턴을 포착.
    spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr).mean(axis=1)  # Spectral Contrast: 주파수 대역 간의 에너지 대비. Running 상태에서 대비가 더 클 가능성.

    
    # Combine all features into one array
    return [rms, spectral_centroid, spectral_bandwidth, zero_crossing_rate] + list(mfcc) + list(spectral_contrast)

# 각 세그먼트에서 특징 추출
features = []

for segment in segments:
    features.append(extract_features(segment, rate))

features = np.array(features)
print(f"Feature Shape: {features.shape}")

# 특징 저장
feature_columns = ["RMS", "Spectral Centroid", "Spectral Bandwidth", "Zero Crossing Rate"] + \
                  [f"MFCC_{i}" for i in range(1, 14)] + \
                  [f"Spectral Contrast_{i}" for i in range(1, 8)]

feature_save_path = os.path.join(save_path, "features_off.csv")
pd.DataFrame(features, columns=feature_columns).to_csv(
    feature_save_path, index=False
)
print(f"Features saved at: {feature_save_path}")
