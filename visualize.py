from operator import matmul
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import random

model_path = './weights/problem_laplace_1.bin'
model = torch.load(model_path)


def u(x, y): return 3+y+0.5*(y**2-x**2)


def u_predict(x, y):
    data = [x, y]
    data_tensor = torch.Tensor(data).resize(2, 1)
    return model(data_tensor).item()

def visualize(num_element=1000):
    a1, a2, b1, b2 = -math.sqrt(6), math.sqrt(6), -math.sqrt(6), math.sqrt(6)
    x1 = np.linspace(a1, a2, num=num_element)
    x2 = np.linspace(b1, b2, num=num_element)
    X, Y = np.meshgrid(x1, x2)

    Z = u(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('u')
    
    Z_predict = np.zeros((num_element, num_element))

    for i in range(num_element):
        for j in range(num_element):
            Z_predict[i][j] = u_predict(X[i][j], Y[i][j])

    ax.plot_surface(X, Y, Z, color='red') # truth
    ax.plot_surface(X, Y, Z_predict, color='blue') # predict
    plt.show(block=False)
    plt.draw()
    plt.pause(10)
    # ax.plot_wireframe(X, Y, Z, color='green')
    # ax.plot_wireframe(X, Y, Z_predict, color='blue')
    # plt.show()

if __name__ == '__main__':
    visualize()