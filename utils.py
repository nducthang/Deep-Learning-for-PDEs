from torch.autograd import grad
import torch


def laplacian(output_vector, input_vector):
    gradient = grad(output_vector, input_vector, create_graph=True)[0]
    ux = gradient.take(torch.tensor([0]))
    uxx = grad(ux, input_vector, create_graph=True)[0].take(torch.tensor([0]))
    uy = gradient.take(torch.tensor([1]))
    uyy = grad(uy, input_vector, create_graph=True)[0].take(torch.tensor([1]))
    return uxx+uyy

def laplacian_u1(output_vector, input_vector):
    gradient = grad(output_vector, input_vector, create_graph=True)[0]
    ux1 = gradient.take(torch.tensor([0]))
    ux1x1 = grad(ux1, input_vector, create_graph=True)[0].take(torch.tensor([0]))
    ux2 = gradient.take(torch.tensor([1]))
    ux2x2 = grad(ux2, input_vector, create_graph=True)[0].take(torch.tensor([1]))
    return ux1x1 + ux2x2

def u1x1(output_vector, input_vector):
    gradient = grad(output_vector, input_vector, create_graph=True)[0]
    u1x1 = gradient.take(torch.tensor([0]))
    return u1x1
def u1x2(output_vector, input_vector):
    gradient = grad(output_vector, input_vector, create_graph=True)[0]
    u1x2 = gradient.take(torch.tensor([1]))
    return u1x2

def px1(output_vector, input_vector):
    gradient = grad(output_vector, input_vector, create_graph=True)[0]
    px1 = gradient.take(torch.tensor([0]))
    return px1

def px2(output_vector, input_vector):
    gradient = grad(output_vector, input_vector, create_graph=True)[0]
    px2 = gradient.take(torch.tensor([1]))
    return px2

def laplacian_u2(output_vector, input_vector):
    gradient = grad(output_vector, input_vector, create_graph=True)[0]
    ux1 = gradient.take(torch.tensor([0]))
    ux1x1 = grad(ux1, input_vector, create_graph=True)[0].take(torch.tensor([0]))
    ux2 = gradient.take(torch.tensor([1]))
    ux2x2 = grad(ux2, input_vector, create_graph=True)[0].take(torch.tensor([1]))
    return ux1x1 + ux2x2

def u2x1(output_vector, input_vector):
    gradient = grad(output_vector, input_vector, create_graph=True)[0]
    u2x1 = gradient.take(torch.tensor([0]))
    return u2x1

def u2x2(output_vector, input_vector):
    gradient = grad(output_vector, input_vector, create_graph=True)[0]
    u2x2 = gradient.take(torch.tensor([1]))
    return u2x2
