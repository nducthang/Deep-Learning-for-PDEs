from operator import imatmul
import torch
import numpy as np
import math

model_path = './Kết quả thực nghiệm/Phương trình 4/pt4_4tang_128/model.bin'
v = 0.025
lamda = 1/(2*v) - math.sqrt(1/(4*(v**2))+4*math.pi**2)

def split_grid():
    x1 = np.linspace(-0.5, 1.0, 500)
    x2 = np.linspace(-0.5, 1.5, 500)
    X, Y = np.meshgrid(x1, x2)
    zs = np.array([[x, y] for x,y in zip(np.ravel(X), np.ravel(Y))])
    return zs

def extract_solution(input):
    x1 = input.take(torch.tensor([0]))
    x2 = input.take(torch.tensor([1]))
    pi = math.pi
    u1 = 1-math.exp(lamda*x1)*math.cos(2*pi*x2)
    u2 = lamda/(2*pi)*math.exp(lamda*x1)*math.sin(2*pi*x2)
    p = 0.5*(1-math.exp(2*lamda*x1))
    return [u1, u2, p]

if __name__ == '__main__':
    model = torch.load(model_path)
    samples = split_grid()
    L2_error = 0
    for point in samples:
        test_point_input = torch.Tensor(point).resize(2, 1)
        u_h = model(test_point_input).detach().numpy().T[0]
        u = extract_solution(test_point_input)
        L2_error += (u_h-u)**2
    L2_error /= len(samples)
    print(math.sqrt(np.sum(L2_error)/3))
