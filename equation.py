from matplotlib import pyplot as plt
import torch
from torch import optim
import tqdm
import math


class Equation:
    def __init__(self, model, name=None, num_point=1000):
        self.num_point = num_point
        self.model = model
        self.losses = []
        self.errors = []
        self.num_iterator = 0
        self.name = name

    def generate_data(self):
        pass

    def calculate_loss(self, samples):
        pass

    def extract_solution(self, input):
        pass

    def calculate_l2_error(self, samples):
        test_omega, test_boundary = samples
        samples = test_omega + test_boundary
        L2_error = 0
        for point in samples:
            test_point_input = torch.Tensor(point).resize(2, 1)
            L2_error += (self.model(test_point_input) -
                         self.extract_solution(test_point_input))**2
        L2_error /= len(samples)
        return math.sqrt(L2_error.item())

    def train(self, num_iterator=50):
        self.num_iterator = self.num_iterator + num_iterator
        optimizer = optim.AdamW(self.model.parameters())
        for i in tqdm.tqdm(range(num_iterator)):
            samples = self.generate_data()
            optimizer.zero_grad()

            loss = self.calculate_loss(samples)
            loss.backward()
            self.losses.append(loss.item())

            optimizer.step()
            test_points = self.generate_data()
            L2_error = self.calculate_l2_error(test_points)
            self.errors.append(L2_error)

    def save_model(self, path):
        torch.save(self.model, path)

    def save_loss(self, path):
        with open(path, "w") as f:
            for idx, l in enumerate(self.losses, start=1):
                f.write(str(idx)+","+str(l)+"\n")

    def save_l2_error(self, path):
        with open(path, "w") as f:
            for idx, e in enumerate(self.errors, start=1):
                f.write(str(idx)+","+str(e)+"\n")

    def plot_loss(self, path):
        epochs = [i+1 for i in range(self.num_iterator)]

        plt.figure()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss của ' + self.name)
        plt.plot(epochs, self.losses)
        plt.savefig(path)

    def plot_l2_error(self, path):
        epochs = [i+1 for i in range(self.num_iterator)]

        plt.figure()
        plt.xlabel('Iterations')
        plt.xlabel('L2 error')
        plt.title('Sai số L2 của ' + self.name)
        plt.plot(epochs, self.errors)
        plt.savefig(path)
