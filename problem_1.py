from equation import Equation
from utils import laplacian
import math
import random
import torch
from torch.autograd.variable import Variable
import numpy as np


class Problem_1(Equation):
    def __init__(self, model, name=None, num_point=1000):
        super().__init__(model, name, num_point=num_point)
        self.name = "Phương trình Laplace"

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
            point_input = Variable(torch.Tensor(
                omega).resize(2, 1), requires_grad=True)
            point_output = self.model(point_input)
            L1 += (-laplacian(point_output, point_input))**2

        for boudary in boundary_points:
            b_point_input = Variable(torch.Tensor(
                boudary).resize(2, 1), requires_grad=True)
            b_point_output = self.model(b_point_input)
            L2 += (b_point_output - self.boundary_condition(b_point_input))**2

        L1 = L1/len(omega_points)
        L2 = L2/len(boundary_points)

        loss = L1 + L2
        return loss
