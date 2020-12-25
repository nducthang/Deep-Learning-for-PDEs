from torch.nn.modules import module
from problem_1 import Problem_1
from problem_2 import Problem_2
from problem_3 import Problem_3
from problem_4 import Problem_4

from Model.HL import PDENetLight
from Model.NeuralNetwork import NeuralNetwork
import torch
# import multiprocessing as mp
# torch.multiprocessing.set_start_method('spawn')
# try:
#     mp.set_start_method('spawn')
# except RuntimeError:
#     pass

PATH_MODEL = './results/pt3_3tang/'
PATH_LOSS_IMAGE = './results/pt3_3tang/loss.png'
PATH_L2_ERROR_IMAGE = './results/pt3_3tang/l2_error.png'
PATH_LOSS_TXT = './results/pt3_3tang/loss.txt'
PATH_L2_ERROR_TXT = './results/pt3_3tang/l2_error.txt'

NUM_ITERATOR = 15000

if __name__ == '__main__':
    # model = PDENetLight(3, 10)
    model = NeuralNetwork(3, 1, 16, 3)
    model.init_weights()
    # device = torch.device("cuda:0")
    # model.to(device)
    # model.cuda()
    # model.to('cuda')
    # model = torch.nn.DataParallel(model)
    # model.cuda()
    # dim_input, dim_output, hidden_size, num_hidden_layer
    # model = NeuralNetwork(2, 1, 16, 2)


    problem = Problem_3(model)
    problem.train(num_iterator=NUM_ITERATOR, path_model=PATH_MODEL)
    problem.plot_loss(PATH_LOSS_IMAGE)
    # problem.plot_l2_error(PATH_L2_ERROR_IMAGE)
    problem.save_loss(PATH_LOSS_TXT)
    # problem.save_l2_error(PATH_L2_ERROR_TXT)
