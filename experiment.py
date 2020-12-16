from problem_1 import Problem_1
from Net import Net
if __name__ == '__main__':
    model = Net(20, 2)
    model.init_weights()

    problem = Problem_1(model)
    problem.train()
    problem.save_model('./model.bin')
    problem.plot_loss('./loss.png')
    problem.plot_l2_error('./l2_error.png')
    problem.save_loss('./loss.txt')
    problem.save_l2_error('./l2_erros.txt')
