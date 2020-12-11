import torch
import torch

# model = Net(10, 2)
def u(x, y): return 3+y+0.5*(y**2-x**2)
model_path = './weights/problem_laplace_1.bin'
model = torch.load(model_path)

# radius = math.sqrt(6)
# phi = 2*math.pi*random.random()
# r = radius * random.random()
# data = [r*math.cos(phi), r*math.sin(phi)]
# print(data)
data = [1,1]
data_tensor = torch.Tensor(data).resize(2, 1)
print(model(data_tensor).item())
print(u(data[0], data[1]))

