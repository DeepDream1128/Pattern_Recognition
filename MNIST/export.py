import torch
import torch.onnx
from model import Net
# 假设model是你的PyTorch模型
# dummy_input是一个符合模型输入大小的张量，比如对于MNIST为1x1x28x28
# 加载你的预训练模型
model = Net()
model.load_state_dict(torch.load('mnist_cnn.pt'))
model.eval()

# 创建一个模拟的输入，对于MNIST来说，输入是28x28的灰度图像
dummy_input = torch.randn(1, 1, 28, 28)

# 导出模型
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=['input'], output_names=['output'])