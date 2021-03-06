from matplotlib import pyplot as plt
import torch
from torch import optim
import tqdm
import math
import multiprocessing as mp

class Equation:
    def __init__(self, model, name = None, ndim = None, num_point=100):
        self.num_point = num_point
        self.model = model
        self.losses = []
        self.errors = []
        self.num_iterator = 0
        self.name = name
        self.ndim = ndim
        self.step_save = 200

    def generate_data(self):
        pass

    def calculate_loss(self, samples):
        pass

    def extract_solution(self, input):
        pass

    def calculate_l2_error(self, samples):
        pass
    
    def split_grid(self):
        pass

    def train(self, num_iterator=50, path_model = None):
        self.num_iterator = self.num_iterator + num_iterator
        optimizer = optim.AdamW(self.model.parameters())
        test_points = self.split_grid()
        for i in tqdm.tqdm(range(num_iterator)):
            samples = self.generate_data()
            optimizer.zero_grad()

            # pool = mp.Pool()
            # loss = pool.map(self.calculate_loss, samples)
            loss = self.calculate_loss(samples)
            loss.backward()
            self.losses.append(loss.item())

            optimizer.step()
            # L2_error = self.calculate_l2_error(test_points)
            # self.errors.append(L2_error)
        # if (i+1) % self.step_save == 0:
        # self.save_model(path_model + 'model_' + str(i+1) + '.bin')
        self.save_model(path_model + 'model' + '.bin')

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
