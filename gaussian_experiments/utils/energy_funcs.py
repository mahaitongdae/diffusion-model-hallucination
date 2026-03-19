import numpy as np
import torch

def norm(x):
    if isinstance(x, torch.Tensor):
        return torch.norm(x, dim=1)
    else:
        return np.linalg.norm(x, axis=1)
    
def abs(x):
    if isinstance(x, torch.Tensor):
        return torch.abs(x)
    else:
        return np.abs(x)

def clip(x, min=0., max=1.):
    if isinstance(x, torch.Tensor):
        return torch.clip(x, min=min, max=max)
    else:
        return np.clip(x, a_min=min, a_max=max)

class EnergyFunctionsRegistry:

    # Use a class-level registry so decorators work without needing an instance
    energy_functions = {}

    @classmethod
    def register(cls, energy_fn):
        cls.energy_functions[energy_fn.__name__] = energy_fn
        return energy_fn

    @classmethod
    def get(cls, name):
        if name not in cls.energy_functions:
            raise ValueError(f"Energy function {name} not found")
        return cls.energy_functions[name]

@EnergyFunctionsRegistry.register
def energy_func_linear(x):
    if x.ndim == 1:
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(1)
        else:
            x = x[:, np.newaxis]
    return -norm(x) + 1.

@EnergyFunctionsRegistry.register
def energy_func_two_gaussian(x, l=1.2, a1=0.3, a2=0.7):
    if x.ndim == 1:
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(1)
        else:
            x = x[:, np.newaxis]
    if isinstance(x, torch.Tensor):
        energy = a1 * torch.exp(-norm(x)**2 /
                                 (l**2)) + a2 * torch.exp(-norm(x - 3.)**2 /
                                                           (l**2))
    else:
        energy = a1 * np.exp(-norm(x)**2 /
                              (l**2)) + a2 * np.exp(-norm(x - 3.)**2 / (l**2))
    return energy

@EnergyFunctionsRegistry.register
def energy_func_two_cusps(x, l=0.8, a1=0.3, a2=0.7):
    if x.ndim == 1:
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(1)
        else:
            x = x[:, np.newaxis]
    if isinstance(x, torch.Tensor):
        energy = a1 * torch.exp(-abs(x) /
                                 (l**2)) + a2 * torch.exp(-abs(x - 3.) /
                                                           (l**2))
    else:
        energy = a1 * np.exp(-abs(x) /
                              (l**2)) + a2 * np.exp(-abs(x - 3.) / (l**2))
    return energy

@EnergyFunctionsRegistry.register
def energy_func_log_two_gaussian(x, l=1):
    if x.ndim == 1:
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(1)
        else:
            x = x[:, np.newaxis]
    if isinstance(x, torch.Tensor):
        energy = 0.2 * torch.exp(-norm(x)**2 /
                                 (l**2)) + 0.8 * torch.exp(-norm(x - 3.)**2 /
                                                           (l**2))
        energy = torch.log(energy)
    else:
        energy = 0.2 * np.exp(-norm(x)**2 /
                              (l**2)) + 0.8 * np.exp(-norm(x - 3.)**2 / (l**2))
        energy = np.log(energy)
    return energy

@EnergyFunctionsRegistry.register
def energy_func_linear_clip(x):
    if x.ndim == 1:
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(1)
        else:
            x = x[:, np.newaxis]
    energy = -norm(x) + 1.
    return clip(energy, min=0., max=1.)

@EnergyFunctionsRegistry.register
def energy_func_quadratic(x):
    if x.ndim == 1:
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(1)
        else:
            x = x[:, np.newaxis]
    return norm(x) ** 2

@EnergyFunctionsRegistry.register
def energy_func_exponential(x):
    if x.ndim == 1:
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(1)
        else:
            x = x[:, np.newaxis]
    return np.exp(norm(x))

@EnergyFunctionsRegistry.register
def energy_func_negative_quadratic(x):
    if x.ndim == 1:
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(1)
        else:
            x = x[:, np.newaxis]
    energy = -norm(x) ** 2
    return energy - energy.min()

@EnergyFunctionsRegistry.register
def energy_function_exponential_negative_quadratic(x):
    if x.ndim == 1:
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(1)
        else:
            x = x[:, np.newaxis]
    return np.exp(-norm(x) ** 2)


def plot_energy_functions_3d():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    X, Y = np.meshgrid(x, y)
    Z = energy_func_linear(np.stack([X, Y], axis=0))
    plt.figure()
    fig, axs = plt.subplots(1, 5, subplot_kw={'projection': '3d'})
    axs[0].plot_surface(X, Y, Z)
    axs[1].plot_surface(X, Y, energy_func_quadratic(np.stack([X, Y], axis=0)))
    axs[2].plot_surface(X, Y, energy_func_exponential(np.stack([X, Y],
                                                               axis=0)))
    axs[3].plot_surface(
        X, Y, energy_func_negative_quadratic(np.stack([X, Y], axis=0)))
    axs[4].plot_surface(
        X, Y,
        energy_function_exponential_negative_quadratic(np.stack([X, Y],
                                                                axis=0)))
    plt.show()


def plot_energy_functions_2d(bounds=[-6, 6]):
    import matplotlib.pyplot as plt
    x = np.linspace(bounds[0], bounds[1], 1000)
    fig, axs = plt.subplots(1, 7, figsize=(20, 5))
    axs[0].plot(x, energy_func_linear(x))
    axs[1].plot(x, energy_func_quadratic(x))
    axs[2].plot(x, energy_func_exponential(x))
    axs[3].plot(x, energy_func_negative_quadratic(x))
    axs[4].plot(x, energy_function_exponential_negative_quadratic(x))
    axs[5].plot(x, energy_func_two_gaussian(x))
    axs[6].plot(x, energy_func_two_cusps(x))
    # plt.show()
    plt.savefig("energy_functions_2d.pdf", dpi=300)

def get_best_possible_val(energy_fn):
    x = np.linspace(-10, 10, 1000)
    return energy_fn(x).max()


if __name__ == "__main__":
    plot_energy_functions_2d()
    # print(EnergyFunctionsRegistry.energy_functions)
    # print(EnergyFunctionsRegistry.get("energy_func_linear_clip")(np.array([1, 2, 3])))
    # print(get_best_possible_val(EnergyFunctionsRegistry.get("energy_func_two_gaussian"))