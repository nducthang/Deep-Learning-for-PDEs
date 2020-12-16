import math
from utils import laplacian
import torch
from torch import tensor
from torch.autograd import grad
from torch.autograd.variable import Variable
from equation import Equation
import random


class Problem_3(Equation):
    def __init__(self, model):
        super().__init__(self)
        self.name = "Phương trình truyền nhiệt phụ thuộc thời gian"
        self.model = model
        self.ndim = 3

    def generate_data(self):
        points = []
        omega_points = []
        boundary_points = []

        # (x,y,t) random
        for i in range(self.num_point):
            t = random.uniform(0, 1)
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            points.append([x, y, t])

        # (x, y) in Omega, t = 0
        for i in range(self.num_point):
            t = 0
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            omega_points.append([x, y, t])

        # (x,y) in boundary, t random
        for i in range(self.num_point//4):
            t = random.uniform(0, 1)
            y = 0
            x = random.uniform(0, 1)
            boundary_points.append([x, y, t])
        for i in range(self.num_point//4):
            t = random.uniform(0, 1)
            x = 1
            y = random.uniform(0, 1)
            boundary_points.append([x, y, t])
        for i in range(self.num_point//4):
            t = random.uniform(0, 1)
            y = 1
            x = random.uniform(0, 1)
            boundary_points.append([x, y, t])
        for i in range(self.num_point - len(boundary_points)):
            t = random.uniform(0, 1)
            x = 0
            y = random.uniform(0, 1)
            boundary_points.append([x, y, t])

        return points, omega_points, boundary_points

    def ut(self, output_vector, input_vector):
        return grad(output_vector, input_vector, create_graph=True)[0].take(torch.tensor([2]))

    def f_function(self, input):
        x = input.take(torch.tensor([0]))
        y = input.take(torch.tensor([1]))
        t = input.take(torch.tensor([2]))
        return (1+2*(math.pi)**2)*math.exp(t)*math.sin(math.pi*x)*math.sin(math.pi*y)

    def boundary_condition(self, input):
        return 0

    def initial_condition(self, input):
        x = input.take(torch.tensor([0]))
        y = input.take(torch.tensor([1]))
        return math.sin(math.pi*x)*math.sin(math.pi*y)

    def extract_solution(self, input):
        x = input.take(torch.tensor([0]))
        y = input.take(torch.tensor([1]))
        t = input.take(torch.tensor([2]))
        return math.exp(t)*math.sin(math.pi*x)*math.sin(math.pi*y)

    def calculate_loss(self, samples):
        L1 = L2 = L3 = 0
        points, omega_points, boundary_points = samples
        for p in points:
            point = Variable(torch.tensor(p).resize(
                self.ndim, 1), requires_grad=True)
            out = self.model(point)
            L1 += (self.ut(out, point) - laplacian(out,
                                                   point) - self.f_function(point))**2

        for p in omega_points:
            point = Variable(torch.tensor(p).resize(
                self.ndim, 1), requires_grad=True)
            out = self.model(point)
            L2 += (out - self.initial_condition(point))**2

        for p in boundary_points:
            point = Variable(torch.tensor(p).resize(
                self.ndim, 1), requires_grad=True)
            out = self.model(point)
            L3 += (out)**2

        L1 = L1/len(points)
        L2 = L2/len(omega_points)
        L3 = L3/len(boundary_points)

        loss = L1 + L2 + L3
        return loss

    def calculate_l2_error(self, samples):
        points, omega_points, boundary_points = samples
        samples = points + omega_points + boundary_points
        L2_error = 0
        for p in samples:
            point = torch.Tensor(p).resize(self.ndim, 1)
            L2_error += (self.model(point)-self.extract_solution(point))**2

        L2_error /= len(samples)
        return math.sqrt(L2_error.item())
