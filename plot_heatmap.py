import numpy as np
import seaborn as sb
import math
import matplotlib.pyplot as plt
import torch

def plot_heatmap(uh, u, x1_min, x1_max, x2_min, x2_max):
    x1 = np.linspace(x1_min, x1_max, 100)
    x2 = np.linspace(x2_min, x2_max, 100)
    X, Y = np.meshgrid(x1,x2)

    data = np.zeros((100, 100))
    for i in range(len(X)):
        for j in range(len(X[0])):
            data[i][j] = math.fabs(uh(X[i][j], Y[i][j])-u(X[i][j], Y[i][j]))
    heat_map = sb.heatmap(data)
    plt.title("|u2_h-u2|")
    plt.show()

def plot_error_u(uh1, uh2, u1, u2, x1_min, x1_max, x2_min, x2_max):
    x1 = np.linspace(x1_min, x1_max, 100)
    x2 = np.linspace(x2_min, x2_max, 100)
    X, Y = np.meshgrid(x1,x2)

    data = np.zeros((100, 100))
    for i in range(len(X)):
        for j in range(len(X[0])):
            data[i][j] = math.fabs(uh1(X[i][j], Y[i][j])-u1(X[i][j], Y[i][j])) + math.fabs(uh2(X[i][j], Y[i][j])-u2(X[i][j], Y[i][j])) 
    heat_map = sb.heatmap(data)
    plt.title("|uh_1-u_1| + |uh_2-u2|")
    plt.savefig('./images/N-S_u_error-2-layer.png')


if __name__ == '__main__':
    v = 0.025
    lamda = 1/(2*v) - math.sqrt(1/(4*(v**2))+4*math.pi**2)
    model_path = './Kết quả thực nghiệm/Phương trình 4/pt4_3tang_128/model.bin'
    model = torch.load(model_path)

    def uh1(x, y):
        # return 0
        data = [x, y]
        data_tensor = torch.Tensor(data).resize(2, 1)
        return model(data_tensor)[0].item()

    def uh2(x, y):
        # return 0
        data = [x, y]
        data_tensor = torch.Tensor(data).resize(2, 1)
        return model(data_tensor)[1].item()

    def u1(x1, x2):
        return 1-math.exp(lamda*x1)*math.cos(2*math.pi*x2)

    def u2(x1, x2):
        return lamda/(2*math.pi)*math.exp(lamda*x1)*math.sin(2*math.pi*x2)

    # plot_heatmap(uh, u2, -0.5, 1.0, -0.5, 1.5)
    plot_error_u(uh1, uh2, u1, u2, -0.5, 1.0, -0.5, 1.5)
