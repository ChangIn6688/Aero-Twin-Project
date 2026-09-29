import torch
import torch.nn as nn

# PINN 학습 결과를 압축한 대체 모델(Surrogate Model)
class FluidSurrogateModel(nn.Module):
    def __init__(self):
        super(FluidSurrogateModel, self).__init__()
        # 결함 크기(Gap)라는 1개의 입력값을 받아 압력 강하율(Pressure Drop)을 계산
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # 최종 출력값 1개 (압력 손실율 %)
        )

    def forward(self, defect_size):
        return self.network(defect_size)

def calculate_pressure_drop(defect_size_mm):
    model = FluidSurrogateModel()
    model.eval()
    
    # 입력값을 파이토치 텐서(데이터 형태)로 변환
    input_tensor = torch.tensor([[defect_size_mm]], dtype=torch.float32)
    
    with torch.no_grad():
        pressure_drop = model(input_tensor)
        
    return pressure_drop.item()