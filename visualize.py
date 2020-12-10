from operator import matmul
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import random

model_path = './weights/model1.thang'
model = torch.load(model_path)


def u(x, y): return 3+y+0.5*(y**2-x**2)


def u_predict(x, y):
    data = [x, y]
    data_tensor = torch.Tensor(data).resize(2, 1)
    return model(data_tensor).item()

def visualize(num_element=100):
    x, y = [], []

    radius = math.sqrt(6)
    for i in range(20000):
        phi = 2*math.pi*random.random()
        r = radius * random.random()
        x.append(r*math.cos(phi))
        y.append(r*math.sin(phi))

    X, Y = np.meshgrid(x, y)
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

    ax.plot_wireframe(X, Y, Z, color='green')
    ax.plot_wireframe(X, Y, Z_predict, color='blue')
    plt.show()