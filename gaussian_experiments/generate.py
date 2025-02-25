import numpy as np
from ddpm_torch.toy import *
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.stats import entropy

if __name__ == '__main__':
    model_path = '/home/haitong/PycharmProjects/diffusion-model-hallucination/gaussian_experiments/chkpts/edm_UnbalancedGaussian2D_200000_rssm_uniform_0/ddpm_UnbalancedGaussian2D_gen_0.pt'
    # replace with model_path
    true_data = np.load(Path(model_path).parent / 'real_dataset.npy')
    trainloader = DataStreamer("UnbalancedGaussian2D", batch_size=10000, num_batches=2, modes=2)
    evaluator = Evaluator(
        true_data=np.concatenate([
            next(iter(trainloader)) for _ in range(2)
        ]), eval_batch_size=1000, max_eval_count=2000, value_range=[0, 10])

    # diffusion = GaussianDiffusion(
    #     betas=betas, model_mean_type="eps", model_var_type="fixed-large", loss_type="mse")
    diffusion = EDMDiffusion(sigma_min=0.001, sigma_max=30, sample_steps=20, device=torch.device("cuda"))
    # These are sigma_min, sigma_max, sample_step in train_toy_edm.py, feel free to change if you change the training hyperparameters.
    model = Decoder(2, 128, 3).to(torch.device("cuda"))
    # mid_features, num_temporal layers in hyperparameters.
    model.load_state_dict(torch.load(model_path)['model'])


    def denoise_fn(x_t, t):
        return model(x_t, t)


    def sample_fn(n):
        shape = (n,) + (2,)
        sample = diffusion.p_sample(
            denoise_fn=denoise_fn, shape=shape, device="cuda", noise=None)
        return sample.detach().cpu().numpy()

        # if evaluator is not None:


    eval_results = evaluator.eval(sample_fn)
    gen_data = eval_results['x_gen']
    plt.figure(figsize=(3, 3))
    # Generate example 2D data
    x = gen_data[:, 0]
    y = gen_data[:, 1]

    # Create the figure and gridspec layout
    fig = plt.figure(figsize=(3, 3))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

    # Scatter plot
    scatter_ax = fig.add_subplot(grid[1:, :-1])
    scatter_ax.scatter(x, y, alpha=0.5, s=0.5)
    scatter_ax.set_xlim([-6, 6])
    scatter_ax.set_ylim([-6, 6])
    scatter_ax.set_xticks(np.array([-6, -3, 0, 3, 6]))
    scatter_ax.set_yticks(np.array([-6, -3, 0, 3, 6]))
    scatter_ax.grid(True)

    # Histogram for X-axis
    x_hist_ax = fig.add_subplot(grid[0, :-1], sharex=scatter_ax)
    x_hist_ax.hist(x, bins=2, color='blue', alpha=0.7, weights=np.ones_like(x) / len(x))
    # x_hist_ax.axis('off')  # Hide x-ticks and labels
    x_hist_ax.grid(True)
    x_hist_ax.set_yticks([0.2, 0.8, ])
    x_hist_ax.tick_params(axis='x', which='both', labelbottom=False)

    # Histogram for Y-axis
    y_hist_ax = fig.add_subplot(grid[1:, -1], sharey=scatter_ax)
    y_hist_ax.hist(y, bins=2, orientation='horizontal', color='green', alpha=0.7, weights=np.ones_like(y) / len(y))
    y_hist_ax.grid(True)
    y_hist_ax.tick_params(axis='y', which='both', labelleft=False)
    y_hist_ax.set_xticks([0.2, 0.8])



    bins = 50
    hist1, bin_edges = np.histogram(gen_data, bins=bins, density=True)
    hist2, _ = np.histogram(true_data, bins=bin_edges, density=True)

    # Avoid division by zero
    hist1 += 1e-10
    hist2 += 1e-10

    # Normalize histograms to get probabilities
    prob1 = hist1 / np.sum(hist1)
    prob2 = hist2 / np.sum(hist2)

    # Compute KL divergence
    kl_divergence = entropy(prob1, prob2)
    print("KL Divergence:", kl_divergence)

    plt.suptitle('KL Divergence: {:.3f}'.format(kl_divergence))

    # Adjust the layout to avoid overlaps
    plt.tight_layout()

    # Show the plot
    plt.show()
