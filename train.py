import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import Net

# ======================
# データ準備
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
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
# モデル
# ======================
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# ======================
# 学習
# ======================
loss_list = []

for epoch in range(30):
    running_loss = 0.0

    for images, labels in trainloader:
        optimizer.zero_grad()

        outputs = net(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    loss_list.append(running_loss)

    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}")

print("Finished Training")

# ======================
# 保存
# ======================
torch.save(net.state_dict(), "model.pth")
print("Model saved!")

# ======================
# 精度確認
# ======================
correct = 0
total = 0

with torch.no_grad():
    for images, labels in trainloader:
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

# ======================
# グラフ
# ======================
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()