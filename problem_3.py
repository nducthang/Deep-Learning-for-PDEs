from equation import Equation


class Problem_3(Equation):
    def __init__(self, model, name = None, ndim, num_point=1000):
        super().__init__(model, name=name, ndim=ndim, num_point=num_point)