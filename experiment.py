from torch.nn.modules import module
from problem_1 import Problem_1
from problem_2 import Problem_2
from problem_3 import Problem_3
from problem_4 import Problem_4

from Model.HL import PDENetLight
from Model.NeuralNetwork import NeuralNetwork


PATH_MODEL = './results/test8/'
PATH_LOSS_IMAGE = './results/test8/loss.png'
PATH_L2_ERROR_IMAGE = './results/test8/l2_error.png'
PATH_LOSS_TXT = './results/test8/loss.txt'
PATH_L2_ERROR_TXT = './results/test8/l2_error.txt'

NUM_ITERATOR = 1000

if __name__ == '__main__':
    # model = PDENetLight(3, 10)
    # dim_input, dim_output, hidden_size, num_hidden_layer
    model = NeuralNetwork(3, 1, 16, 1)
    model.init_weights()

    problem = Problem_3(model)
    problem.train(num_iterator=NUM_ITERATOR, path_model=PATH_MODEL)
    problem.plot_loss(PATH_LOSS_IMAGE)
    problem.plot_l2_error(PATH_L2_ERROR_IMAGE)
    problem.save_loss(PATH_LOSS_TXT)
    problem.save_l2_error(PATH_L2_ERROR_TXT)
