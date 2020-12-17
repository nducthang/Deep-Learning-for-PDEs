import random
from numpy.core.records import _OrderedCounter
import torch
from torch.autograd.variable import Variable
from equation import Equation
import utils
import math
import numpy as np
from torch.autograd import grad
from torch import optim
import tqdm

class Problem_4(Equation):
    def __init__(self, model):
        super().__init__(self)
        self.name = "Phương trình Navier-Stokes ổn định"
        self.model = model
        self.ndim = 2
        self.v = 0.025
        self.lamda = 1/(2*self.v) - math.sqrt(1/(4*(self.v**2))+4*math.pi**2)

    def generate_data(self):
        omega_points = []
        boundary_points = []

        for i in range(self.num_point):
            x1 = random.uniform(-0.5, 1.0)
            x2 = random.uniform(-0.5, 1.5)
            omega_points.append([x1, x2])

        for i in range(self.num_point//4):
            x2 = -0.5
            x1 = random.uniform(-0.5, 1.0)
            boundary_points.append([x1, x2])
        for i in range(self.num_point//4):
            x1 = 1.0
            x2 = random.uniform(-0.5, 1.5)
            boundary_points.append([x1, x2])
        for i in range(self.num_point//4):
            x2 = 1.5
            x1 = random.uniform(-0.5, 1.0)
            boundary_points.append([x1, x2])
        for i in range(self.num_point - len(boundary_points)):
            x1 = -0.5
            x2 = random.uniform(-0.5, 1.5)
            boundary_points.append([x1, x2])

        return omega_points, boundary_points

    def calculate_loss(self, samples):
        L1 = L2 = L3 = 0
        omega_points, boundary_points = samples
        for p in omega_points:
            point = Variable(torch.Tensor(p).resize(
                self.ndim, 1), requires_grad=True)
            out = self.model(point)
            L_u1 = -self.v * utils.laplacian_u1(out[0], point) + (out[0]*utils.u1x1(out[0], point) + out[1]*utils.u1x2(out[0], point)) + utils.px1(out[2], point)
            L_u2 = -self.v * utils.laplacian_u2(out[1], point) + (out[1]*utils.u2x1(out[1], point) + out[0]*utils.u2x2(out[1], point)) + utils.px2(out[2], point)
            L1 += L_u1**2 + L_u2**2
            L2 += (utils.u1x1(out[0], point)+utils.u2x2(out[1], point))**2

        # boundary 1
        n = len(boundary_points)//4
        for p in boundary_points[:n]:
            point = Variable(torch.Tensor(p).resize(
                self.ndim, 1), requires_grad=True)
            out = self.model(point)
            L3 += (out[0] - 1 + math.exp(self.lamda * p[0] * math.cos(-math.pi)))**2
            L3 += (out[1] - self.lamda/(2*math.pi) * math.exp(self.lamda * p[0]) * math.cos(-math.pi))**2
        # boundary 2
        for p in boundary_points[n:2*n]:
            point = Variable(torch.Tensor(p).resize(
                self.ndim, 1), requires_grad=True)
            out = self.model(point)
            L3 += (out[0]-1+math.exp(self.lamda)*math.cos(2*math.pi*p[1]))**2
            L3 += (out[1]-self.lamda/(2*math.pi) * math.exp(self.lamda)*math.sin(2*math.pi*p[1]))**2
        # boundary 3
        for p in boundary_points[2*n:3*n]:
            point = Variable(torch.Tensor(p).resize(
                self.ndim, 1), requires_grad=True)
            out = self.model(point)
            L3 += (out[0] - 1 + math.exp(self.lamda*p[0])*math.cos(2*math.pi))**2
            L3 += (out[1]-self.lamda/(2*math.pi)*math.exp(self.lamda*p[0])*math.sin(3*math.pi))**2
        # boundary 4
        for p in boundary_points[3*n:]:
            point = Variable(torch.Tensor(p).resize(
                self.ndim, 1), requires_grad=True)
            out = self.model(point)
            L3 += (out[0]-1+math.exp(self.lamda*(-0.5))*math.cos(2*math.pi*p[1]))**2
            L3 += (out[1]-self.lamda/(2*math.pi)*math.exp(self.lamda)*math.sin(2*math.pi*p[1]))**2

        L1 = L1/len(omega_points)
        L2 = L2/len(omega_points)
        L3 = L3/len(boundary_points)

        loss = L1+L2+L3
        return loss

    def extract_solution(self, input):
        x1 = input.take(torch.tensor([0]))
        x2 = input.take(torch.tensor([1]))
        pi = math.pi
        u1 = 1-math.exp(self.lamda*x1)*math.cos(2*pi*x2)
        u2 = self.lamda/(2*pi)*math.exp(self.lamda*x1)*math.sin(2*pi*x2)
        p = 0.5*(1-math.exp(2*self.lamda*x1))
        return [u1, u2, p]

    def split_grid(self):
        x1 = np.linspace(-0.5, 1.0, 100)
        x2 = np.linspace(-0.5, 1.5, 150)
        X, Y = np.meshgrid(x1, x2)
        zs = np.array([[x, y] for x,y in zip(np.ravel(X), np.ravel(Y))])
        return zs

    def calculate_l2_error(self, samples):
        L2_error = 0
        for point in samples:
            test_point_input = torch.Tensor(point).resize(self.ndim, 1)
            u_h = self.model(test_point_input).detach().numpy().T[0]
            u = self.extract_solution(test_point_input)
            L2_error += (u_h-u)**2
        L2_error /= len(samples)
        return math.sqrt(np.sum(L2_error)/3)
