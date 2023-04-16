import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error as mae



class gradient_descent():
    """
    the class will be constructed based on a functon and its inputs.
    """

    def __init__(self, func: callable, input_data: np.array, parameters: np.array):
        """
        the contructor of the gradient descent class.
        :param func: function which takes some parameters and input data. This function should have 2 inputs: input data,
        and "parameters" which is described below. It should get the input data first (func(input_data, parameters)).
        :param input_data: training data on which the parameters are going to be optimized.
        :param parameters: the parameters which are going to be optimized. For example in neural networks, weights
        and biases are adjusted.
        """

        self.func = func
        self.input_data = input_data
        self.parameters = parameters


    def loss_function(self, first_input, second_input, loss='mae'):
        """
        This is the loss function; the function we want to minimize for optimization.
        :param second_input: the second input for the loss function
        :param first_input: the first input for the loss function
        :param loss: str:['mae', 'mse']. mean absolute error or mean squared error
        :return: func output
        """
        assert loss in ['mae', 'mse']; "loss should be 'mea' or 'mse'"
        if loss == 'mae':
            return mae(first_input - second_input)
        if loss == 'mse':
            return mse(first_input - second_input)


    def derivative(self, idx=None):
        """
        a function to get a point's derivative.
        :param idx: in case of partial derivative, this parameter tells us which dimension of the parameters should be
        used to get the derivative.
        :return: gradient of the function at the point of the parameters.
        """
        epsilon = (random.random()**2)

        if idx:
            greater_parameter = self.parameters[idx] + epsilon
            greater_output = self.func(self.input_data, greater_parameter)
            lesser_parameter = self.parameters[idx] - epsilon
            lesser_output = self.func(self.input_data, lesser_parameter)
            der = (greater_output - lesser_output)/(greater_parameter - lesser_parameter)
        else:
            greater_parameter = self.parameters + epsilon
            greater_output = self.func(self.input_data, greater_parameter)
            lesser_parameter = self.parameters - epsilon
            lesser_output = self.func(self.input_data, lesser_parameter)
            der = (greater_output - lesser_output) / (greater_parameter - lesser_parameter)

        return der

    def gradient(self):
        """
        a function to calculate the gradient of the function at a point (inputs)
        :return: gradient of the
        """

    def gradient_step(self):
        """
        one step of parameters adjustment
        :return: adjusted parameters
        """
