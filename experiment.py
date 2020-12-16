from torch.nn.modules import module
from problem_1 import Problem_1
from problem_2 import Problem_2
from problem_3 import Problem_3

from Model.Net import Net
from Model.Net2 import Net2
from Model.HL import PDENetLight

if __name__ == '__main__':
    # model = Net2(20, 3)
    model = PDENetLight(3, 10)
    model.init_weights()

    problem = Problem_3(model)
    problem.train(num_iterator=200)
    problem.save_model('./model.bin')
    problem.plot_loss('./loss.png')
    problem.plot_l2_error('./l2_error.png')
    problem.save_loss('./loss.txt')
    problem.save_l2_error('./l2_errors.txt')
