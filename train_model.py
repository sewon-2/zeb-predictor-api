# 📌 라이브러리 로드
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import AdamW
from keras.losses import LogCosh
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 📌 GPU 사용 안함 (CPU로 실행)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 📌 데이터 불러오기 (openpyxl 엔진 추가)
file_path = "C:/python_my_projects/homepage2/학습.xlsx"
df = pd.read_excel(file_path, sheet_name="차트_정리", engine="openpyxl")

# 📌 입력(X) / 출력(Y) 데이터 분리
X = df[['연면적', '창면적비', '열관류율_지붕', '열관류율_벽체', '열관류율_바닥', '에너지 자립률']]
Y = df[['PV 설치 규모']]

# 📌 데이터 분할 (훈련 80%, 테스트 20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 📌 데이터 정규화 (표준화)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

# 📌 랜덤 시드 고정 (결과 재현 가능)
np.random.seed(42)

# 📌 최적화된 TensorFlow 모델 생성
model = Sequential([
    BatchNormalization(input_shape=(X_train.shape[1],)),  # BatchNorm을 맨 앞에 배치
    Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001)), 
    BatchNormalization(),
    Dropout(0.05),  

    Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Dropout(0.05),

    Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Dropout(0.05),

    Dense(1)  # 🔹 출력층 (PV 설치 규모 예측)
])

# 📌 AdamW Optimizer 적용 (Lookahead 제거)
optimizer = AdamW(learning_rate=0.00002, weight_decay=1e-4)

# 📌 모델 컴파일 및 학습 준비
model.compile(optimizer=optimizer, loss=LogCosh())

# 📌 Early Stopping 적용
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# 📌 모델 학습 (Epoch=5000 설정)
history = model.fit(
    X_train, Y_train,
    epochs=5000,
    batch_size=512,  # 배치 크기 증가
    validation_data=(X_test, Y_test),
    callbacks=[early_stopping]
)

# 📌 모델 평가
loss = model.evaluate(X_test, Y_test)
print(f"\n✅ 모델 손실(MSE): {loss}")

# 📌 새로운 데이터 예측
new_data = np.array([[2000, 0.3, 0.2, 0.7, 0.4, 0.75]])  # 예제 데이터
new_data_scaled = scaler_X.transform(new_data)  # 정규화 적용
predicted_pv = model.predict(new_data_scaled)

# 📌 예측값 복원 (정규화된 값을 원래 스케일로 변환)
predicted_pv = scaler_Y.inverse_transform(predicted_pv)

# 📌 ZEB 인증 등급 계산 (에너지 자립률 기준)
energy_self_sufficiency = new_data[0][-1]  # 마지막 값이 에너지 자립률
if energy_self_sufficiency >= 1.0:
    zeb_grade = "ZEB 1등급"
elif 0.8 <= energy_self_sufficiency < 1.0:
    zeb_grade = "ZEB 2등급"
elif 0.6 <= energy_self_sufficiency < 0.8:
    zeb_grade = "ZEB 3등급"
elif 0.4 <= energy_self_sufficiency < 0.6:
    zeb_grade = "ZEB 4등급"
elif 0.2 <= energy_self_sufficiency < 0.4:
    zeb_grade = "ZEB 5등급"
else:
    zeb_grade = "ZEB 인증 불가"

# 📌 최종 예측 결과 출력
print(f"\n🔹 예측된 PV 설치 규모: {predicted_pv[0][0]:.2f} kW")
print(f"🔹 ZEB 인증 등급: {zeb_grade}")

# 📌 학습된 모델 저장 (h5 형식)
model.save("my_final_best_model.h5")
print("\n✅ 학습된 모델이 my_final_best_model.h5로 저장되었습니다.")
