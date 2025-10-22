import pandas as pd

# 각 상태에 해당하는 파일 경로
features_printing_path = r"C:\Users\mmgzz\Desktop\ME Lab Machine Learning\Renishaw Segments\features_printing.csv"
features_off_path = r"C:\Users\mmgzz\Desktop\ME Lab Machine Learning\Renishaw Segments\features_off.csv"

# Printing State 데이터 로드 및 레이블 추가
features_printing = pd.read_csv(features_printing_path)
features_printing["Label"] = 1  # Printing State

# Off State 데이터 로드 및 레이블 추가
features_off = pd.read_csv(features_off_path)
features_off["Label"] = 0  # Off State

# 두 데이터 병합
features_combined = pd.concat([features_printing, features_off], ignore_index=True)

# 병합 결과 확인
print(features_combined.head())
print(features_combined.tail())

# 병합된 데이터를 저장
combined_features_path = r"C:\Users\mmgzz\Desktop\ME Lab Machine Learning\Renishaw Segments\combined_features.csv"
features_combined.to_csv(combined_features_path, index=False)
print(f"Combined Features saved at: {combined_features_path}")
