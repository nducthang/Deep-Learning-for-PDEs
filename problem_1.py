from equation import Equation
from utils import laplacian
import math
import random
import torch
from torch.autograd.variable import Variable
import numpy as np


class Problem_1(Equation):
    def __init__(self, model):
        super().__init__(self)
        self.name = "Phương trình laplace 1"
        self.model = model
        self.ndim = 2

    def generate_data(self):
        omega_points = []
        boundary_points = []

        radius = math.sqrt(6)
        for i in range(self.num_point):
            phi = 2*math.pi*random.random()
            r = radius * random.random()
            omega_points.append([r*math.cos(phi), r*math.sin(phi)])

        for i in range(self.num_point):
            phi = 2 * math.pi * random.random()
            boundary_points.append(
                [radius * math.cos(phi), radius*math.sin(phi)])

        return omega_points, boundary_points

    def boundary_condition(self, input):
        return input.take(torch.tensor([1])) + input.take(torch.tensor([1]))**2

    def extract_solution(self, input):
        return 3 + input.take(torch.tensor([1])) + 0.5*(input.take(torch.tensor([1]))**2 - input.take(torch.tensor([0]))**2)

    def calculate_loss(self, samples):
        L1 = L2 = 0
        omega_points, boundary_points = samples

        for omega in omega_points:
            point = Variable(torch.Tensor(
                omega).resize(self.ndim, 1), requires_grad=True)
            out = self.model(point)
            L1 += (-laplacian(out, point))**2

        for boundary in boundary_points:
            b_point = Variable(torch.Tensor(
                boundary).resize(self.ndim, 1), requires_grad=True)
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
