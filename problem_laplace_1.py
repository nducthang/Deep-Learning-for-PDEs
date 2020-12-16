import torch
from torch.autograd import grad
from torch.autograd.variable import Variable
import random
import numpy as np
import matplotlib.pyplot as plt
from Net import Net
import math
import torch.optim as optim
import tqdm


class Problem_Laplace_1:
    def __init__(self, name, num_point=1000):
        self.name = name
        self.num_point = num_point

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

    def laplacian(self, output_vector, input_vector):
        gradient = grad(output_vector, input_vector, create_graph=True)[0]
        ux = gradient.take(torch.tensor([0]))
        uxx = grad(ux, input_vector, create_graph=True)[
            0].take(torch.tensor([0]))
        uy = gradient.take(torch.tensor([1]))
        uyy = grad(uy, input_vector, create_graph=True)[
            0].take(torch.tensor([1]))
        return uxx+uyy

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
            point_output = model(point_input)
            L1 += (-self.laplacian(point_output, point_input))**2

        for boudary in boundary_points:
            b_point_input = Variable(torch.Tensor(
                boudary).resize(2, 1), requires_grad=True)
            b_point_output = model(b_point_input)
            L2 += (b_point_output - self.boundary_condition(b_point_input))**2

        L1 = L1/len(omega_points)
        L2 = L2/len(boundary_points)

        loss = L1 + L2
        return loss

    def calculate_error(self, samples):
        L2_error = 0
        for point in samples:
            test_point_input = torch.Tensor(point).resize(2, 1)
            L2_error += (model(test_point_input) -
                         self.extract_solution(test_point_input))**2
        L2_error /= len(samples)
        return L2_error


if __name__ == '__main__':
    # Init neural network with layer size: 10, dim input: 2
    model = Net(20, 2)

    num_iteration = 50
    model.init_weights()

    equation = Problem_Laplace_1(name='problem_laplace_1')

    losses = []
    errors = []

    optimizer = optim.AdamW(model.parameters())

    for i in tqdm.tqdm(range(num_iteration)):
        samples = equation.generate_data()
        optimizer.zero_grad()

        loss = equation.calculate_loss(samples)
        loss.backward()  # calculate gradient of loss w.r.t the parameters
        losses.append(loss.item())

        optimizer.step()

        # Calculate square error
        test_omega, test_boundary = equation.generate_data()
        test_points = test_omega + test_boundary
        L2_error = equation.calculate_error(test_points)
        errors.append(L2_error.item())

    print("Saving model...")
    torch.save(model, "./weights/" + equation.name + ".bin")

    print("Plotting...")
    epochs = [i+1 for i in range(num_iteration)]

    plt.figure(0)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title("Loss of " + str(equation.name))
    plt.plot(epochs, losses)
    plt.savefig('./plots/loss_' + equation.name + '.png')

    plt.figure(1)
    plt.xlabel('Iterations')
    plt.ylabel('L2 error')
    plt.title("MSE of " + str(equation.name))
    plt.plot(epochs, np.log(errors))
    plt.savefig('./plots/error_' + equation.name + '.png')

    print("Saving loss and error to .txt files...")
    with open("./loss/" + equation.name + ".txt", "w") as f:
        for idx, l in enumerate(losses, start=1):
            f.write(str(idx)+","+str(l)+"\n")

    with open("./error/" + equation.name + ".txt", "w") as f:
        for idx, e in enumerate(errors, start=1):
            f.write(str(idx)+","+str(e)+"\n")

    print("Done!")
