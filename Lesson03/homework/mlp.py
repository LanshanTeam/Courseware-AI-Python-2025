import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.init import normal_

# ------------------------------
# 导入原始数据集
# ------------------------------
# 读取xor_dataset.csv文件（确保文件在当前目录）
data = np.loadtxt('xor_dataset.csv', delimiter=',')
print('数据集大小：', len(data))
print(data[:5])

# 划分训练集与测试集
ratio = 0.8
split = int(ratio * len(data))
np.random.seed(0)
data = np.random.permutation(data)
x_train, y_train = data[:split, :2], data[:split, -1].reshape(-1, 1)
x_test, y_test = data[split:, :2], data[split:, -1].reshape(-1, 1)

# ------------------------------
# 手动实现MLP
# ------------------------------
class Layer:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    def update(self, learning_rate):
        pass

class Linear(Layer):    
    def __init__(self, num_in, num_out, use_bias=True):
        self.num_in = num_in
        self.num_out = num_out
        self.use_bias = use_bias
        self.W = np.random.normal(loc=0, scale=1.0, size=(num_in, num_out))
        if use_bias:
            self.b = np.zeros((1, num_out))
        
    def forward(self, x):
        self.x = x
        self.y = x @ self.W
        if self.use_bias:
            self.y += self.b
        return self.y
    
    def backward(self, grad):
        self.grad_W = self.x.T @ grad / grad.shape[0]
        if self.use_bias:
            self.grad_b = np.mean(grad, axis=0, keepdims=True)
        grad = grad @ self.W.T
        return grad
    
    def update(self, learning_rate):
        self.W -= learning_rate * self.grad_W
        if self.use_bias:
            self.b -= learning_rate * self.grad_b

class Identity(Layer):
    def forward(self, x):
        return x
    def backward(self, grad):
        return grad

class Sigmoid(Layer):  
    def forward(self, x):
        self.x = x
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    def backward(self, grad):
        return grad * self.y * (1 - self.y)

class Tanh(Layer):
    def forward(self, x):
        self.x = x
        self.y = np.tanh(x)
        return self.y
    def backward(self, grad):
        return grad * (1 - self.y **2)

class ReLU(Layer):
    def forward(self, x):
        self.x = x
        self.y = np.maximum(x, 0)
        return self.y
    def backward(self, grad):
        return grad * (self.x >= 0)

activation_dict = { 
    'identity': Identity,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU
}

class MLP:
    def __init__(self, layer_sizes, use_bias=True, activation='relu', out_activation='identity'):
        self.layers = []
        num_in = layer_sizes[0]
        for num_out in layer_sizes[1: -1]:
            self.layers.append(Linear(num_in, num_out, use_bias))
            self.layers.append(activation_dict[activation]())
            num_in = num_out
        self.layers.append(Linear(num_in, layer_sizes[-1], use_bias))
        self.layers.append(activation_dict[out_activation]())
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

# 手动实现MLP的训练与测试
print("\n===== 手动实现MLP结果 =====")
num_epochs = 1000
learning_rate = 0.1
batch_size = 128
eps = 1e-7

mlp_manual = MLP(layer_sizes=[2, 4, 1], use_bias=True, out_activation='sigmoid')

losses_manual = []
test_losses_manual = []
test_accs_manual = []
for epoch in range(num_epochs):
    st = 0
    loss = 0.0
    while True:
        ed = min(st + batch_size, len(x_train))
        if st >= ed:
            break
        x = x_train[st: ed]
        y = y_train[st: ed]
        y_pred = mlp_manual.forward(x)
        grad = (y_pred - y) / (y_pred * (1 - y_pred) + eps)
        mlp_manual.backward(grad)
        mlp_manual.update(learning_rate)
        train_loss = np.sum(-y * np.log(y_pred + eps) - (1 - y) * np.log(1 - y_pred + eps))
        loss += train_loss
        st += batch_size

    losses_manual.append(loss / len(x_train))
    y_pred = mlp_manual.forward(x_test)
    test_loss = np.sum(-y_test * np.log(y_pred + eps) - (1 - y_test) * np.log(1 - y_pred + eps)) / len(x_test)
    test_acc = np.sum(np.round(y_pred) == y_test) / len(x_test)
    test_losses_manual.append(test_loss)
    test_accs_manual.append(test_acc)
    
print('测试精度：', test_accs_manual[-1])

# 可视化手动实现结果
plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.plot(losses_manual, color='blue', label='train loss')
plt.plot(test_losses_manual, color='red', ls='--', label='test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Cross-Entropy Loss')
plt.legend()

plt.subplot(122)
plt.plot(test_accs_manual, color='red')
plt.ylim(top=1.0)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.show()

# ------------------------------
# PyTorch实现MLP
# ------------------------------
torch_activation_dict = {
    'identity': lambda x: x,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'relu': torch.relu
}

class MLP_torch(nn.Module):
    def __init__(self, layer_sizes, use_bias=True, activation='relu', out_activation='identity'):
        super().__init__()
        self.activation = torch_activation_dict[activation]
        self.out_activation = torch_activation_dict[out_activation]
        self.layers = nn.ModuleList()
        num_in = layer_sizes[0]
        for num_out in layer_sizes[1:]:
            self.layers.append(nn.Linear(num_in, num_out, bias=use_bias))
            normal_(self.layers[-1].weight, std=1.0)
            self.layers[-1].bias.data.fill_(0.0)
            num_in = num_out

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.out_activation(x)
        return x

# PyTorch实现MLP的训练与测试
print("\n===== PyTorch实现MLP结果 =====")
num_epochs = 1000
learning_rate = 0.1
batch_size = 128
eps = 1e-7
torch.manual_seed(0)

mlp_torch = MLP_torch(layer_sizes=[2, 4, 1], use_bias=True, out_activation='sigmoid')
opt = torch.optim.SGD(mlp_torch.parameters(), lr=learning_rate)

losses_torch = []
test_losses_torch = []
test_accs_torch = []
for epoch in range(num_epochs):
    st = 0
    loss = []
    while True:
        ed = min(st + batch_size, len(x_train))
        if st >= ed:
            break
        x = torch.tensor(x_train[st: ed], dtype=torch.float32)
        y = torch.tensor(y_train[st: ed], dtype=torch.float32).reshape(-1, 1)
        y_pred = mlp_torch(x)
        train_loss = torch.mean(-y * torch.log(y_pred + eps) - (1 - y) * torch.log(1 - y_pred + eps))
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        loss.append(train_loss.detach().numpy())
        st += batch_size

    losses_torch.append(np.mean(loss))
    with torch.inference_mode():
        x = torch.tensor(x_test, dtype=torch.float32)
        y = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        y_pred = mlp_torch(x)
        test_loss = torch.sum(-y * torch.log(y_pred + eps) - (1 - y) * torch.log(1 - y_pred + eps)) / len(x_test)
        test_acc = torch.sum(torch.round(y_pred) == y) / len(x_test)
        test_losses_torch.append(test_loss.detach().numpy())
        test_accs_torch.append(test_acc.detach().numpy())

print('测试精度：', test_accs_torch[-1])

# 可视化PyTorch实现结果
plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.plot(losses_torch, color='blue', label='train loss')
plt.plot(test_losses_torch, color='red', ls='--', label='test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('PyTorch_Cross-Entropy Loss')
plt.legend()

plt.subplot(122)
plt.plot(test_accs_torch, color='red')
plt.ylim(top=1.0)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('PyTorch_Test Accuracy')
plt.show()