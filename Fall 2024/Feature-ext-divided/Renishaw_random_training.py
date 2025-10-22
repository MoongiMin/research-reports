import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 파일 경로 설정
new_wav_file_path = r"C:\Users\mmgzz\Desktop\ME Lab Machine Learning\Testing\20241108_180130.163479Z_off_testing2.wav"
combined_features_path = r"C:\Users\mmgzz\Desktop\ME Lab Machine Learning\Renishaw Segments\combined_features.csv"

# 새 오디오 파일 전처리
def preprocess_wav(file_path, sr=22050, segment_length=1, overlap=0.8):
    print(f"Processing file: {file_path}")
    # 오디오 로드
    sound, rate = librosa.load(file_path, sr=sr)
    print(f"Sample Rate: {rate} Hz")
    print(f"Sound Shape: {sound.shape}")

    # 데이터 세그먼트화
    segment_len_samples = int(rate * segment_length)  # 1초 세그먼트
    overlap_samples = int(rate * overlap)  # 오버랩 길이
    segments = []

    for start in range(0, len(sound) - segment_len_samples, segment_len_samples - overlap_samples):
        segment = sound[start : start + segment_len_samples]
        segments.append(segment)

    segments = np.array(segments)
    print(f"Number of Segments: {segments.shape[0]}")

    # 특징 추출 함수
    def extract_features(segment):
        rms = np.sqrt(np.mean(segment**2))  # RMS
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=rate).mean()  # Spectral Centroid
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=rate).mean()  # Spectral Bandwidth
        zero_crossing_rate = librosa.feature.zero_crossing_rate(segment).mean()  # Zero Crossing Rate
        mfcc = librosa.feature.mfcc(y=segment, sr=rate, n_mfcc=13).mean(axis=1)  # MFCC
        spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=rate).mean(axis=1)  # Spectral Contrast
        return [rms, spectral_centroid, spectral_bandwidth, zero_crossing_rate] + list(mfcc) + list(spectral_contrast)

    # 모든 세그먼트에서 특징 추출
    features = [extract_features(segment) for segment in segments]
    feature_columns = ["RMS", "Spectral Centroid", "Spectral Bandwidth", "Zero Crossing Rate"] + \
                      [f"MFCC_{i}" for i in range(1, 14)] + \
                      [f"Spectral Contrast_{i}" for i in range(1, 8)]
    return pd.DataFrame(features, columns=feature_columns), len(segments)

# 새 데이터 전처리
new_data_features, num_segments = preprocess_wav(new_wav_file_path)

# 병합된 데이터 불러오기
combined_features = pd.read_csv(combined_features_path)
X = combined_features.drop(columns=["Label"]).values
y = combined_features["Label"].values

# 모델 학습 준비
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_scaled, y)

# 새로운 데이터에 대해 예측 수행
new_data_scaled = scaler.transform(new_data_features.values)
new_predictions = model.predict(new_data_scaled)

# 새 데이터 라벨 설정 (사용자가 지정)
true_label = 0  # 이 파일이 실제 Printing(1) 상태라고 가정
true_labels = [true_label] * num_segments  # 모든 세그먼트에 동일한 라벨 적용

# 결과 비교 및 출력
print("\n--- Predictions for New WAV File ---")
for i, pred in enumerate(new_predictions):
    state = "Printing" if pred == 1 else "Off"
    print(f"Segment {i + 1}: Predicted State = {state}")

# 상태별 세그먼트 개수 출력
printing_count = sum(new_predictions)
off_count = len(new_predictions) - printing_count
print("\n--- Summary ---")
print(f"Total Segments: {len(new_predictions)}")
print(f"Printing Segments (1): {printing_count}")
print(f"Off Segments (0): {off_count}")

# 정확도 계산
print("\n--- Evaluation ---")
print(f"True Label: {'Printing' if true_label == 1 else 'Off'}")
print(f"Accuracy: {accuracy_score(true_labels, new_predictions):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, new_predictions))
print("\nClassification Report:")
print(classification_report(true_labels, new_predictions))
