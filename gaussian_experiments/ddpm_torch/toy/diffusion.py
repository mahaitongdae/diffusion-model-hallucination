import math
import torch
from .. import diffusion
from ..functions import normal_kl, continuous_gaussian_loglik, flat_mean
import numpy as np


def get_schedule(num_steps, sigma_min, sigma_max, device = None, schedule_type = 'polynomial', schedule_rho = 7,
                 net = None):
    """
    Get the time schedule for sampling.
    Get from the AMED_solvers, http://arxiv.org/abs/2312.00094

    Args:
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings.
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        device: A torch device.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_type: A `float`. Time step exponent.
        net: A pre-trained diffusion model. Required when schedule_type == 'discrete'.
    Returns:
        a PyTorch tensor with shape [num_steps].
    """
    if schedule_type == 'polynomial':
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (sigma_max ** (1 / schedule_rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / schedule_rho) - sigma_max ** (1 / schedule_rho))) ** schedule_rho
    elif schedule_type == 'logsnr':
        logsnr_max = -1 * torch.log(torch.tensor(sigma_min))
        logsnr_min = -1 * torch.log(torch.tensor(sigma_max))
        t_steps = torch.linspace(logsnr_min.item(), logsnr_max.item(), steps=num_steps, device=device)
        t_steps = (-t_steps).exp()
    elif schedule_type == 'time_uniform':
        epsilon_s = 1e-3
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (
                sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
        step_indices = torch.arange(num_steps, device=device)
        vp_beta_d = 2 * (torch.log(torch.tensor(sigma_min).cpu() ** 2 + 1) / epsilon_s - torch.log(
            torch.tensor(sigma_max).cpu() ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = torch.log(torch.tensor(sigma_max).cpu() ** 2 + 1) - 0.5 * vp_beta_d
        t_steps_temp = (1 + step_indices / (num_steps - 1) * (epsilon_s ** (1 / schedule_rho) - 1)) ** schedule_rho
        t_steps = vp_sigma(vp_beta_d.clone().detach().cpu(), vp_beta_min.clone().detach().cpu())(
            t_steps_temp.clone().detach().cpu())
    elif schedule_type == 'discrete':
        assert net is not None
        t_steps_min = net.sigma_inv(torch.tensor(sigma_min, device=device))
        t_steps_max = net.sigma_inv(torch.tensor(sigma_max, device=device))
        step_indices = torch.arange(num_steps, device=device)
        t_steps_temp = (t_steps_max + step_indices / (num_steps - 1) * (
                t_steps_min ** (1 / schedule_rho) - t_steps_max)) ** schedule_rho
        t_steps = net.sigma(t_steps_temp)
    else:
        raise ValueError("Got wrong schedule type {}".format(schedule_type))

    return t_steps.to(device)


class GaussianDiffusion(diffusion.GaussianDiffusion):

    def get_true_score_unbalanced_gmm(self, x_t, t):
        x_t = x_t.detach_().requires_grad_(True)
        energy = torch.log(0.8 * torch.exp(- torch.linalg.norm(x_t - 1.5 * torch.ones_like(x_t), axis=1) ** 2 / 2)
                           + 0.2 * torch.exp(- torch.linalg.norm(x_t + 1.5 * torch.ones_like(x_t), axis=1) ** 2 / 2))
        score = torch.autograd.grad(energy.sum(), x_t)[0]
        scale = self._extract(self.sqrt_one_minus_alphas_bar, t, x_t)
        return - scale * score

    def q_sample(self, x_0, t, noise = None):
        if noise is None:
            noise = torch.randn_like(x_0)
        coef1 = self._extract(self.sqrt_alphas_bar, t, x_0)
        coef2 = self._extract(self.sqrt_one_minus_alphas_bar, t, x_0)
        return coef1 * x_0 + coef2 * noise

    def p_mean_var(self, denoise_fn, x_t, t, clip_denoised, return_pred):
        B, D = x_t.shape
        out = denoise_fn(x_t, t)

        if self.model_var_type == "learned":
            assert all(out.shape == (B, 2 * D))
            out, model_logvar = out.chunk(2, dim=1)
            model_var = torch.exp(model_logvar)
        elif self.model_var_type in ["fixed-small", "fixed-large"]:
            model_var, model_logvar = self._extract(self.fixed_model_var, t, x_t), \
                self._extract(self.fixed_model_logvar, t, x_t)
        else:
            raise NotImplementedError(self.model_var_type)

        # calculate the mean estimate
        _clip = lambda x: x  # (lambda x: x.clamp(-3., 3.)) if clip_denoised else (lambda x: x)
        if self.model_mean_type == "mean":  # noqa
            pred_x_0 = _clip(self._pred_x_0_from_mean(x_t=x_t, mean=out, t=t))
            model_mean = out
        elif self.model_mean_type == "x_0":
            pred_x_0 = _clip(out)
            model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0, x_t=x_t, t=t)
        elif self.model_mean_type == "eps":
            pred_x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=out, t=t))
            model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0, x_t=x_t, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        if return_pred:
            return model_mean, model_var, model_logvar, pred_x_0
        else:
            return model_mean, model_var, model_logvar

    # === log likelihood ===
    # bpd: bits per dimension

    def _loss_term_bpd(self, denoise_fn, x_0, x_t, t, clip_denoised, return_pred):
        # calculate L_t
        # t = 0: negative log likelihood of decoder, -\log p(x_0 | x_1)
        # t > 0: variational lower bound loss term, KL term
        true_mean, _, true_logvar = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)  # noqa
        model_mean, _, model_logvar, pred_x_0 = self.p_mean_var(
            denoise_fn, x_t=x_t, t=t, clip_denoised=clip_denoised, return_pred=True)
        kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
        kl = flat_mean(kl) / math.log(2.)  # natural base to base 2
        decoder_nll = continuous_gaussian_loglik(x_0, model_mean, logvar=model_logvar).neg()
        decoder_nll = flat_mean(decoder_nll) / math.log(2.)
        output = torch.where(t.to(kl.device) > 0, kl, decoder_nll)
        return (output, pred_x_0) if return_pred else output


class EDMDiffusion:

    def __init__(self,
                 sigma_min, sigma_max, sample_steps, sampler='heun', device=torch.device('cpu'),):
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.t_steps = get_schedule(sample_steps, self.sigma_min, self.sigma_max).to(device)
        self.timesteps = len(self.t_steps)

    def p_sample(self,
                 denoise_fn,
                 sampler='heun',
                 shape = None, device = torch.device("cpu"), noise = None, seed = None):
        rng = None
        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        if noise is None:
            latents = torch.empty(shape, device=device).normal_(generator=rng)
        else:
            latents = noise.to(device)
        # for ti in range(self.timesteps - 1, -1, -1):
        #     t.fill_(ti)
        #     x_t = self.p_sample_step(denoise_fn, x_t, t, generator=rng)
        if sampler == "heun":
            x_t = self.heun_sample(denoise_fn, latents, return_inters=False)
        elif sampler == 'euler':
            x_t = self.euler_sample(denoise_fn, latents, return_inters=False)
        else:
            raise NotImplementedError("Sampler not implemented")
        return x_t

    # @torch.inference_mode()
    # def pf_ode_sample_step(self, denoise_fn, x_t, t, tm1, clip_denoised = True, return_pred = False, generator = None):
    #     # self.model_mean_type
    #     assert self.model_mean_type == 'x_0'
    #     x_0 = denoise_fn(x_t, t)
    #     # Euler step
    #     score = (x_t - x_0)

    @torch.inference_mode()
    def heun_sample(self, denoise_fn, latents, return_inters=False, S_churn=0):
        x_next = latents * self.t_steps[0]
        inters = [x_next.unsqueeze(0)]
        for i, (t_cur, t_next) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            gamma = min(S_churn / self.timesteps, np.sqrt(2) - 1)
            t_hat = torch.as_tensor(t_cur + gamma * t_cur) # t_hat = t_cur if no S_churn
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * torch.randn_like(x_cur) #  x_hat = x_cur if no S_churn

            # Euler step.
            denoised = denoise_fn(x_hat, t_hat)
            d_cur = (x_cur - denoised) / t_hat # here d is the \epsilon, if you try to get logp then
            # logp = - d_cur / t_cur
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            denoised = denoise_fn(x_next, t_next)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            if return_inters:
                inters.append(x_next.unsqueeze(0))
        if return_inters:
            return torch.cat(inters, dim=0).to(latents.device)
        return x_next

    @torch.inference_mode()
    def euler_sample(self, denoise_fn, latents, return_inters=False, S_churn=0):
        x_next = latents * self.t_steps[0]
        inters = [x_next.unsqueeze(0)]
        for i, (t_cur, t_next) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            gamma = min(S_churn / self.timesteps, np.sqrt(2) - 1)
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)  # t_hat = t_cur if no S_churn
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * torch.randn_like(x_cur)  # x_hat = x_cur if no S_churn

            # Euler step.
            denoised = denoise_fn(x_hat, t_hat)
            d_cur = (x_cur - denoised) / t_hat  # here d is the \epsilon, if you try to get logp then
            # logp = - d_cur / t_cur
            x_next = x_hat + (t_next - t_hat) * d_cur

            if return_inters:
                inters.append(x_next.unsqueeze(0))
        if return_inters:
            return torch.cat(inters, dim=0).to(latents.device)
        return x_next

    @torch.inference_mode()
    def heun_sample_plus_noise(self, denoise_fn, latents, return_inters = False):
        x_next = latents * self.t_steps[0]
        inters = [x_next.unsqueeze(0)]
        for i, (t_cur, t_next) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Euler step.
            denoised = denoise_fn(x_cur, t_cur)
            d_cur = (x_cur - denoised) / t_cur  # here d is the \epsilon, if you try to get logp then
            # logp = - d_cur / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            denoised = denoise_fn(x_next, t_next)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
            if return_inters:
                inters.append(x_next.unsqueeze(0))
        if return_inters:
            return torch.cat(inters, dim=0).to(latents.device)
        return x_next

    def train_losses(self, denoise_fn, x_0, t, noise = None):
        """
        VE loss,  https://github.com/NVlabs/edm/blob/main/training/loss.py

        """
        if noise is None:
            noise = torch.randn_like(x_0)
        rnd_uniform = torch.rand(x_0.shape[0], device=x_0.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        # weight = 1 / sigma ** 2
        weight = torch.ones(x_0.shape[0], device=x_0.device)
        model_out = denoise_fn(x_0 + noise * sigma.unsqueeze(1), sigma)
        losses = torch.sum(weight.unsqueeze(1) * (model_out - x_0) ** 2, dim=-1)

        return losses, torch.linalg.norm(model_out, axis=1).mean().item()
