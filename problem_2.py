from random import random
import torch
from torch import optim
from torch.types import Number
from equation import Equation
from utils import laplacian
import math
import torch
import random
from torch.autograd.variable import Variable


class Problem_2(Equation):
    def __init__(self, model):
        super().__init__(self)
        self.name = "Phương trình laplace 2"
        self.model = model
        self.ndim = 2
        
    def generate_data(self):
        omega_points = []
        boundary_points = []

        for i in range(self.num_point):
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            omega_points.append([x, y])

        for i in range(self.num_point//4):
            y = 0
            x = random.uniform(0, 1)
            boundary_points.append([x, y])
        for i in range(self.num_point//4):
            x = 1
            y = random.uniform(0, 1)
            boundary_points.append([x, y])
        for i in range(self.num_point//4):
            y = 1
            x = random.uniform(0, 1)
            boundary_points.append([x, y])
        for i in range(self.num_point - len(boundary_points)):
            x = 0
            y = random.uniform(0, 1)
            boundary_points.append([x, y])

        return omega_points, boundary_points

    def f_function(self, input):
        x = input.take(torch.tensor([0]))
        y = input.take(torch.tensor([1]))
        pi = math.pi
        return 2*(pi**2)*math.sin(pi*x)*math.sin(pi*y)

    def boundary_condition(self, input):
        return 0

    def extract_solution(self, input):
        x = input.take(torch.tensor([0]))
        y = input.take(torch.tensor([1]))
        return math.sin(math.pi*x)*math.sin(math.pi*y)

    def calculate_loss(self, samples):
        L1 = L2 = 0
        omega_points, boundary_points = samples

        for omega in omega_points:
            point = Variable(torch.Tensor(omega).resize(
                self.ndim, 1), requires_grad=True)
            out = self.model(point)
            L1 += (-laplacian(out, point) - self.f_function(point))**2

        for boundary in boundary_points:
            b_point = Variable(torch.Tensor(boundary).resize(
                self.ndim, 1), requires_grad=True)
            b_out = self.model(b_point)
            L2 += (b_out - self.boundary_condition(b_point))**2

        L1 = L1/len(omega_points)
        L2 = L2/len(boundary_points)

        loss = L1 + L2
        return loss

    def calculate_l2_error(self, samples):
        test_omega, test_boundary = samples
        samples = test_omega + test_boundary
        L2_error = 0
        for point in samples:
            test_point_input = torch.Tensor(point).resize(self.ndim, 1)
            L2_error += (self.model(test_point_input) -
                         self.extract_solution(test_point_input))**2
        L2_error /= len(samples)
        return math.sqrt(L2_error.item())
