#=======================
#tensor		：ただの多次元配列（NumPyの強化版）
#nn.Module	：モデルの設計図
#optimizer	：パラメータを更新するやつ
#loss.backward()：誤差を逆に伝えて学習
#=======================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ======================
# データ準備
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

# ======================
# モデル定義
# ======================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 13 * 13, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.leaky_relu(self.conv1(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = self.fc1(x)
        return x

net = Net()

# ======================
# 学習設定
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# ======================
# 学習ループ
# ======================
for epoch in range(30):
    running_loss = 0.0

    for images, labels in trainloader:
        optimizer.zero_grad()

        outputs = net(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}")

print("Finished Training")