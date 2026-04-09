#=======================
#tensor		：ただの多次元配列（NumPyの強化版）
#nn.Module	：モデルの設計図
#optimizer	：パラメータを更新するやつ
#loss.backward()：誤差を逆に伝えて学習
#=======================

import torch #PyTorch
import torch.nn as nn #活性化関数、損失関数などを定義利用。ニューラルネットワーク構築用モジュール。
import torch.optim as optim #最適化アルゴリズム。損失関数の値の最小化のためのパラメータの重みやバイアスを更新するもの。
import torchvision #画像や動画のデータセットの準備。回転などの変形をする。
import torchvision.transforms as transforms #画像のリサイズ、回転、反転、正規化など。
import matplotlib.pyplot as plt #グラフの表示

# ======================
# データ準備
# ======================
transform = transforms.Compose([ #前処理を纏めてする関数。
    transforms.ToTensor(), #画像をTensorに変換。画素値を0~255->0~1に正規化。
    transforms.Normalize((0.5,), (0.5)) #-1~1の範囲にスケーリング。
])

trainset = torchvision.datasets.MNIST(
    root='./data', #./dataファイルに保存場所指定。
    train=True, #学習データを使う
    download=True, #データがなければ自動ダウンロード。初回のみ。2回目以降はローカルを使用。
    transform=transform #前処理を適応。
)

trainloader = torch.utils.data.DataLoader( #でーた読み込み器の作成。ミニバッチにデータをまとめる。
    trainset, #さっき作ったMNISTデータのtrainsetを使う。
    batch_size=64, #batch_sizeを64にして、ひとまとまり64枚である定義。
    shuffle=True #データを毎回ランダムに並び替える。偏り防止の学修性能向上用。
)

# ======================
# モデル定義
# ======================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        self.conv1 = nn.Conv2d(1, 16, 3) #1チャンネルの白黒画像。を16チャンネルの特徴マップに。エッジや輪郭。簡単な特徴
        self.conv2 = nn.Conv2d(16, 32, 3) #16->32。より高度な特徴を抽出。曲線やパーツ
        self.pool = nn.MaxPool2d(2, 2) #情報の圧縮。サイズを半分に。
        self.fc1 = nn.Linear(32 * 5 * 5, 10) #特徴を全部繋げた入力。出力を10種類に。
        
    def forward(self, x):
        x = self.pool(torch.nn.functional.leaky_relu(self.conv1(x))) #元画像から特徴を抽出して、活性化関数で非線形に。そして圧縮。
        x = self.pool(torch.nn.functional.leaky_relu(self.conv2(x))) #特徴マップからさらに高度な特徴を抽出して、活性化関数で非線形に。そして圧縮。
        x = x.view(-1, 32 * 5 * 5) #flattenで一次元化。Linearに渡す準備。
        x = self.fc1(x) #10種類に分類
        return x
    
net = Net()

# ======================
# 学習設定
# ======================
criterion = nn.CrossEntropyLoss() #どれだけまちっがているのか測る
optimizer = optim.Adam(net.parameters(), lr=0.001) #どう直すか決める

# ======================
# 学習ループ
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
    acc_list = []
    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}")
    
print("Finished Training")

correct = 0
total = 0

# ======================
# 精度評価
# ======================
with torch.no_grad():
    for images, labels in trainloader:
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()