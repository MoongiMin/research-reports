import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: 병합된 데이터 로드
combined_features_path = r"C:\Users\mmgzz\Desktop\ME Lab Machine Learning\Renishaw Segments\combined_features.csv"
features_combined = pd.read_csv(combined_features_path)

# Step 2: 데이터와 레이블 분리
X = features_combined.drop(columns=["Label"]).values  # 특징 데이터
y = features_combined["Label"].values                # 레이블

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4: 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: 모델 학습 (Random Forest)
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Step 6: 모델 평가
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Step 7: 새로운 데이터 예측 (테스트 데이터로 예측)
print("\nSample Predictions:")
for i in range(len(X_test)):
    prediction = model.predict([X_test[i]])
    print(f"Sample {i+1}: Predicted = {prediction[0]}, Actual = {y_test[i]}")
