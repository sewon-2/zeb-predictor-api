from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ✅ 모델 & 데이터 로드 (경로 수정)
file_path = "./학습.xlsx"
df = pd.read_excel(file_path, sheet_name="차트_정리")

# ✅ 입력 데이터 스케일링 설정
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X = df[['연면적', '창면적비', '열관류율_지붕', '열관류율_벽체', '열관류율_바닥', '에너지 자립률']]
Y = df[['PV 설치 규모']]

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# ✅ 학습된 모델 로드
model = tf.keras.models.load_model("my_final_best_model.h5")

def validate_input(data):
    """
    입력 데이터 검증 함수
    """
    try:
        # 연면적 (정수, 1 이상)
        area = int(data['area'])
        if area < 1:
            return "연면적은 1㎡ 이상이어야 합니다."

        # 창면적비 (소수점 첫째 자리까지 허용, 1~100%)
        window_ratio = round(float(data['windowRatio']), 1)
        if window_ratio < 1 or window_ratio > 100:
            return "창면적비는 1~100% 사이여야 합니다."

        # 열관류율 (소수점 셋째 자리까지 허용, 0 이상)
        roof_U = round(float(data['roofU']), 3)
        wall_U = round(float(data['wallU']), 3)
        floor_U = round(float(data['floorU']), 3)

        if not (roof_U >= 0 and wall_U >= 0 and floor_U >= 0):
            return "열관류율(지붕, 벽체, 바닥)은 0 이상의 값이어야 합니다."

        # 목표 자립률 (정수, 1 이상 입력 가능)
        target = int(data['target'])
        if target < 1:
            return "목표 자립률은 1% 이상이어야 합니다."

        return None  # 모든 값이 유효하면 None 반환
    except (ValueError, TypeError):
        return "입력 값이 올바르지 않습니다. 숫자 값을 입력하세요."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ 입력 데이터 검증 추가
        data = request.json
        validation_error = validate_input(data)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        # ✅ 사용자가 입력한 값을 학습 모델에 맞게 변환
        input_data = np.array([
            int(data['area']),  # 연면적 (정수, 그대로 사용)
            round((float(data['windowRatio']) / 100) - 0.01, 3),  # 창면적비 변환 (입력값 / 100 - 0.01)
            round(float(data['roofU']), 3),  # 열관류율_지붕 (그대로 사용)
            round(float(data['wallU']), 3),  # 열관류율_벽체 (그대로 사용)
            round(float(data['floorU']), 3),  # 열관류율_바닥 (그대로 사용)
            round(int(data['target']) / 100, 3)  # 목표 자립률 변환 (입력값 / 100)
        ]).reshape(1, -1)

        # ✅ 데이터 정규화
        input_scaled = scaler_X.transform(input_data)

        # ✅ 모델 예측
        predicted_pv_scaled = model.predict(input_scaled)

        # ✅ 예측값 복원
        predicted_pv = scaler_Y.inverse_transform(predicted_pv_scaled)

        # ✅ JSON 직렬화 오류 방지를 위한 float 변환
        predicted_pv_value = float(predicted_pv[0][0])

        # ✅ ZEB 인증 등급 판별 개선
        energy_self_sufficiency = int(data['target'])  # 입력값 기준으로 판별 (변환 X)
        if energy_self_sufficiency >= 100:
            zeb_grade = "ZEB 인증 1등급"
        elif 80 <= energy_self_sufficiency < 100:
            zeb_grade = "ZEB 인증 2등급"
        elif 60 <= energy_self_sufficiency < 80:
            zeb_grade = "ZEB 인증 3등급"
        elif 40 <= energy_self_sufficiency < 60:
            zeb_grade = "ZEB 인증 4등급"
        elif 20 <= energy_self_sufficiency < 40:
            zeb_grade = "ZEB 인증 5등급"
        else:
            zeb_grade = "ZEB 인증 불가"

        return jsonify({
            "predicted_pv": int(round(predicted_pv_value)),
            "zeb_grade": zeb_grade
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
