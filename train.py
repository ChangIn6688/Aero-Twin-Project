import torch
import torch.nn as nn
import torch.optim as optim
from ae_model import AcousticEmissionCNN

# 현장에서 수집될 데이터라고 가정 (센서 데이터 생성)
# 실제 프로젝트에서는 현장에서 수집한 .csv나 .wav 파일을 불러오는 부분이 됩니다.
def get_dummy_training_data(num_samples=1000):
    # 1000개의 샘플 데이터 (가짜 데이터)
    inputs = torch.randn(num_samples, 1, 100)
    # 정답 라벨 (실제 결함 크기, 0~5mm 사이)
    targets = torch.rand(num_samples, 1) * 5.0
    return inputs, targets

def train_model():
    print("🚀 Aero-Twin AI 모델 학습을 시작합니다...")
    
    # 1. 모델과 최적화 도구(Optimizer) 설정
    model = AcousticEmissionCNN()
    criterion = nn.MSELoss() # 오차를 계산하는 함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    inputs, targets = get_dummy_training_data()
    
    # 2. 에포크(Epoch, 반복 학습 횟수) 설정
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        
        # 모델 예측
        outputs = model(inputs)
        
        # 오차(Loss) 계산
        loss = criterion(outputs, targets)
        
        # 가중치 업데이트 (학습)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss(오차): {loss.item():.4f}")
        
    # 3. 학습이 완료된 모델 가중치 저장 (.pth 파일)
    # 이 과정을 거쳐야 모바일 기기(app.py)에서 가벼운 파일만 가져가서 쓸 수 있습니다.
    torch.save(model.state_dict(), "aero_twin_model.pth")
    print("✅ 학습 완료! 모델이 'aero_twin_model.pth'로 저장되었습니다.")

if __name__ == "__main__":
    train_model()