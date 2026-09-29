import streamlit as st
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
# 윈도우 환경 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 앞서 만든 모델 파일들에서 함수를 불러옵니다.
from ae_model import predict_defect_size
from surrogate_model import calculate_pressure_drop

# UI 화면 구성 시작
st.set_page_config(page_title="Aero-Twin 품질 진단", layout="wide")
st.title("Aero-Twin: Pre-fab 배관 조인트 지능형 품질 검사 🛠️")
st.markdown("현장 작업자 태블릿 시뮬레이션 화면입니다.")

st.sidebar.header("검사 제어 패널")
# 현장에서 스캔 버튼을 누르는 상황을 가정
scan_button = st.sidebar.button("음향 센서 스캔 시작", type="primary")

if scan_button:
    with st.spinner("배관 체결부 음향 데이터 수집 및 분석 중..."):
        time.sleep(1.5) # 실제 스캔하는 것처럼 시간 지연 연출
        
        # 1. 음향 데이터 가상 생성 및 결함 판독
        # (실제 환경에서는 센서에서 들어오는 데이터지만, 데모를 위해 임의 생성)
        dummy_data = torch.randn(1, 1, 100) 
        defect_size = abs(predict_defect_size(dummy_data)) * 5.0 # 보기 좋게 스케일링
        
        # 2. 유체역학 대체 모델로 압력 손실 계산
        pressure_drop = calculate_pressure_drop(defect_size) * 10.0 + 5.0

    st.success("진단이 완료되었습니다!")
    
    # 3. 결과를 화면에 깔끔하게 3개의 열로 나누어 표시
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="탐지된 미세 Gap (결함 크기)", value=f"{defect_size:.2f} mm")
    with col2:
        st.metric(label="예상 압력 강하율", value=f"{pressure_drop:.1f} %", delta="-효율 저하", delta_color="inverse")
    with col3:
        # 판정 기준 (예: 2mm 이상이면 재시공)
        if defect_size > 2.0:
            st.error("🚨 품질 기준 미달: 체결 토크 재조정 필요")
        else:
            st.success("✅ 품질 기준 통과")

    # 4. 시각화 (디지털 트윈 모사)
    st.subheader("유체 저항 시뮬레이션 (Surrogate Model Mapping)")
    st.info("※ 고사양 연산 없이 단 1초 만에 유체 손실 영향을 렌더링했습니다.")
    
    # 시각적 효과를 위한 가상 데이터 그래프
    x = np.linspace(0, 10, 100)
    # 결함 크기에 따라 난류(흔들림)가 심해지는 것을 그래프로 표현
    y_normal = np.sin(x) 
    y_defect = np.sin(x) + np.random.normal(0, defect_size*0.2, 100)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, y_normal, label="정상 상태 유동", color="blue", alpha=0.5)
    ax.plot(x, y_defect, label=f"현재 상태 유동 (Gap: {defect_size:.2f}mm)", color="red")
    ax.set_title("배관 내부 유체 난류 및 압력 변동 예측")
    ax.legend()
    
    st.pyplot(fig)
else:
    st.info("왼쪽 패널에서 '음향 센서 스캔 시작' 버튼을 눌러주세요.")