import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import Net
from pytorch_grad_cam import GradCAM

# ======================
# モデル読み込み
# ======================
net = Net()
net.load_state_dict(torch.load("model.pth"))
net.eval()

# ======================
# データ
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
])

dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# ======================
# Grad-CAM
# ======================
target_layer = net.conv2

cam = GradCAM(model=net, target_layers=[target_layer])

dataiter = iter(loader)
images, labels = next(dataiter)

input_tensor = images[0].unsqueeze(0)

grayscale_cam = cam(input_tensor=input_tensor)

# ======================
# 可視化
# ======================
img = images[0].squeeze().numpy()
cam_image = grayscale_cam[0]

plt.imshow(img, cmap='gray')
plt.imshow(cam_image, cmap='jet', alpha=0.5)
plt.title(f"Label: {labels[0].item()}")
plt.savefig("gradcam_result.png")
plt.show()