import torch
import torch.nn as nn

# 1D-CNN 기반의 음향 데이터(Acoustic Emission) 분석 모델 클래스
class AcousticEmissionCNN(nn.Module):
    def __init__(self):
        super(AcousticEmissionCNN, self).__init__()
        # 현장 소음(시계열 데이터)에서 특징을 추출하는 합성곱(Convolution) 계층
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 추출된 특징을 바탕으로 최종 결함 크기(Gap, 단위: mm)를 예측하는 선형 계층
        self.fc1 = nn.Linear(16 * 50, 32)
        self.fc2 = nn.Linear(32, 1) # 최종 출력값 1개 (결함 크기)

    def forward(self, x):
        # x: 센서에서 들어오는 음향 파형 데이터
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # 1차원으로 평탄화
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 모델 테스트용 함수 (실제 데이터 대신 가짜 데이터를 넣어 작동 확인)
def predict_defect_size(dummy_audio_data):
    model = AcousticEmissionCNN()
    model.eval() # 평가 모드로 전환
    with torch.no_grad():
        # 가짜 데이터를 모델에 통과시켜 예상 결함 크기 도출
        prediction = model(dummy_audio_data)
    return prediction.item()