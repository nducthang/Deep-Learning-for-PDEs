from equation import Equation
from utils import laplacian


class Problem_2(Equation):
    def __init__(self, model, name = None, num_point=1000):
        super().__init__(model, name=name, num_point=num_point)
        self.name = "Phương trình Laplace 2"

    def generate_data(self):
        pass

    def boundary_condition(self, input):
        pass

    def extract_solution(self, input):
        pass

    def calculate_loss(self, samples):
        pass