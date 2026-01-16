import torch


class WeightsTransformFuncsRegistry:

    weights_transform_funcs = {}

    @classmethod
    def register(cls, func):
        cls.weights_transform_funcs[func.__name__] = func

    @classmethod
    def get(cls, name):
        return cls.weights_transform_funcs[name]

@WeightsTransformFuncsRegistry.register
def identity(weights):
    return weights

@WeightsTransformFuncsRegistry.register
def standardize(weights):
    return (weights - weights.mean()) / weights.std()

@WeightsTransformFuncsRegistry.register
def substract_min(weights):
    return weights - weights.min()

@WeightsTransformFuncsRegistry.register
def softmax(weights):
    assert weights.ndim == 1, "Softmax only supports 1D weights"
    return torch.softmax(weights, dim=-1)

@WeightsTransformFuncsRegistry.register
def quadratic(weights):
    return weights ** 2
