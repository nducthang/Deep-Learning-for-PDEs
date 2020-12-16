from problem_1 import Problem_1
from problem_2 import Problem_2
from Model.Net import Net
from Model.Net2 import Net2

if __name__ == '__main__':
    model = Net(20, 2)
    # model.init_weights()

    problem = Problem_1(model)
    problem.train(num_iterator=200)
    problem.save_model('./model.bin')
    problem.plot_loss('./loss.png')
    problem.plot_l2_error('./l2_error.png')
    problem.save_loss('./loss.txt')
    problem.save_l2_error('./l2_errors.txt')
