import torch
import torch.nn as nn

class simpleNet(nn.Module) :
    def __init__ (self) : #신경망 구조 생성
        super (simpleNet, self).__init__()
        self.fn1 = nn.Linear(17, 10) #17개의 입력측에서 10개의 2번째 층으로 연결
        self.fn2 = nn.Linear(10, 5) #10개의 2번째 층에서 5개의 3번째 층으로 연결
        self.fn3 = nn.Linear(5, 1) #5개의 3번째 층에서 1개의 출력층로 연결

    def forward(self, x) : #순전파 함수 생성
        x = torch.relu(self.fn1(x)) #첫번째 순전파에서 활성화 함수 적용
        x = self.fn2(x) #두번째 순전파
        x = self.fn3(x) #세번째 순전파
        return x

model = simpleNet() #model이라는 신경망 정의
loss_fn = nn.MSELoss() #loss함수 정의
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) #grad를 구해서 parameter에 lr*grad를 빼는 최적화 함수 정의

inputs = torch.randn(1, 17) #17개의 입력층에 들어갈 입력 데이터 생성
target = torch.randn(1, 1) #출력층에서 출력되길 목표하는 출력 데이터 생성
print(target)

for epoch in range(100) :
    outputs = model(inputs) #실제 출력 데이터
    loss = loss_fn(outputs, target)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    #각 연결층의 w, b값들을 훈련시키기

    if (epoch + 1) % 10 == 0 :
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}') #loss는 0에 근접해 감
        print(outputs) #outputs은 target에 근접해감
