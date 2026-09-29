# Aero-Twin : Pre-fab 배관 조인트 지능형 품질 검사 플랫폼

Aero-Twin은 하이테크 공장 및 데이터센터 건설 현장에 적용되는 '무용접 모듈화 배관(Pre-fab)'의 미세 체결 불량을 실시간으로 판독하고 시각화하는 디지털 트윈 솔루션입니다.

# 핵심 기능
1. Edge-AI 음향 데이터 판독: 조인트 체결 시 발생하는 Acoustic Emission(고주파 음향) 파형을 1D-CNN 딥러닝이 현장 태블릿에서 즉각 분석하여 미세 Gap 및 토크 미달 수치화.
2. PINN 기반 유체 시뮬레이션: 무거운 Navier-Stokes 방정식을 엣지 디바이스에서 직접 풀지 않고, 사전 학습된 Surrogate Model을 통해 결함에 따른 유체 저항(난류, 압력 강하)을 1초 만에 3D 렌더링.

# 저장소 구조 (Repository Structure)
- `app.py`: 현장 작업자 태블릿용 실시간 검사 UI (Streamlit 데모)
- `ae_model.py`: 음향 파형 분석 1D-CNN 모델 아키텍처
- `surrogate_model.py`: 유체 역학 대체 모델 아키텍처
- `train.py`: 센서 데이터 기반 AI 모델 학습 파이프라인
- `requirements.txt`: 구동에 필요한 라이브러리 목록

# 로컬 데모 실행 방법 (How to Run)
본 프로젝트는 심사위원의 이해를 돕기 위한 MVP(Minimum Viable Product) UI가 포함되어 있습니다.

```bash
# 1. 필요 패키지 설치
pip install -r requirements.txt

# 2. 모델 학습 시뮬레이션 (선택)
python train.py

# 3. 현장 태블릿 UI 데모 실행
streamlit run app.py