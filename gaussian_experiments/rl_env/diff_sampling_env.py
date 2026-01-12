# from solver_utils import *
# from solvers_amed import get_denoised, init_hook, get_amed_prediction
# from sample import StackedRandomGenerator
# from rsl_rl.env import VecEnv
import torch
import os
from gaussian_experiments.ddpm_torch.toy import *
import torch.nn.functional as F
import numpy as np
from gaussian_experiments.utils import plot_with_kl_2d


class DiffSamplingEnv(object):

    def __init__(self,
                 # net,
                 # batch_seeds,
                 # dataset_name,
                 batch_size,
                 # num_steps,
                 # schedule_type,
                 # schedule_rho,
                 # outdir = None,
                 device = 'cuda',
                 reward_scale = 100,
                 reward_func = "hallucination_var",
                 **kwargs):
        # Pick latents and labels.
        self.reward_func = reward_func
        self.device = torch.device(device)
        self.batch_size = self.num_envs = batch_size
        self.num_actions = 2
        self.device = torch.device(device)

        model_path = '/home/haitong/PycharmProjects/diffusion-model-hallucination/gaussian_experiments/chkpts/edm_UnbalancedGaussian2D_200000_rssm_uniform_0/ddpm_UnbalancedGaussian2D_gen_0.pt'
        self.net = Decoder(2, 128, 3).to(torch.device("cuda"))
        # mid_features, num_temporal layers in hyperparameters.
        self.net.load_state_dict(torch.load(model_path)['model'])
        self.diffusion = EDMDiffusion(sigma_min=0.001, sigma_max=30, sample_steps=20, device=torch.device("cuda"))
        self.solver_kwargs = kwargs
        self.reward_scale = reward_scale
        self.t_steps = self.diffusion.t_steps
        self.reset()

    def denoise_fn(self, xt, t):
        return self.net(xt, t)

    def reset(self, batch_seeds = None):
        rng = None
        if batch_seeds is not None:
            rng = torch.Generator(self.device).manual_seed(batch_seeds)
        latents = torch.empty((self.batch_size, 2), device=self.device).normal_(generator=rng).requires_grad_(False)
        self.current_step = 0
        self.x = latents * self.t_steps[0]
        # class_labels = c = uc = None
        # self.current_step = 0
        # self.class_labels, self.c, self.uc = self.__reset_class_labels(batch_seeds)
        # self.traj_mean = None
        # self.sum_of_square = None
        #
        # # Get UNet enc out as observations.
        # unet_enc_out, hook = init_hook(self.net, self.class_labels)
        # _ = get_denoised(self.net, self.x, self.t_steps[0],
        #                  class_labels=self.class_labels,
        #                  condition=self.c,
        #                  unconditional_condition=self.uc)
        # hook.remove()
        # self.observation = torch.mean(unet_enc_out[-1], dim=1)
        self.traj_mean = None
        # self.cumulative_rew = torch.zeros([batch_size], device=self.device)
        return self.x, {}

    def get_observations(self):

        flatten_obs = self.x
        t_cur = self.t_steps[self.current_step] * torch.ones([self.batch_size, 1]).to(self.device)
        t_next = self.t_steps[self.current_step + 1] * torch.ones([self.batch_size, 1]).to(self.device)
        whole_obs = torch.cat([flatten_obs, t_cur, t_next], dim=1)
        return (whole_obs, {'observations': {'critic': whole_obs}})

    def preprocess_actions(self, raw_actions, scale=0.2):
        """
        change raw actions to

        Args:
            raw_actions: actions from [-1, 1]

        Returns:
            actions: r in [0, 1]
            scale_dir: [1 - scale, 1 + scale]

        """
        raw_r, raw_scale_dir = raw_actions[:, 0], raw_actions[:, 1]
        r = (raw_r + 1) / 2
        scale_dir = raw_scale_dir * scale + 1.
        return r, scale_dir

    @torch.inference_mode()
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action.astype(np.float32)).to(self.device)
        r, scale_dir = self.preprocess_actions(action)
        r = r.reshape(-1, 1)
        scale_dir = scale_dir.reshape(-1, 1)
        t_cur = self.t_steps[self.current_step]
        t_next = self.t_steps[self.current_step + 1]
        x_cur = self.x

        denoised_cur = self.denoise_fn(x_cur, t_cur)
        d_cur = (x_cur - denoised_cur) / t_cur  # here d is the \epsilon, if you try to get logp then

        t_mid = (t_next ** r) * (t_cur ** (1 - r))  # r=1 -> t_next, r=0 -> t_cur
        x_mid = x_cur + (t_mid - t_cur) * d_cur

        # Euler step.
        denoised_mid = self.denoise_fn(x_mid, t_mid)
        d_mid = (x_mid - denoised_mid) / t_mid  # here d is the \epsilon, if you try to get logp then
        x_next = x_cur + scale_dir * (t_next - t_cur) * d_mid
        self.x = x_next

        # Apply 2nd order correction.
        # denoised = denoise_fn(x_next, t_next)
        # d_prime = (x_next - denoised) / t_next
        # x_next = x_cur + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        # if return_inters:
        #     inters.append(x_next.unsqueeze(0))

        if self.reward_func == 'hallucination_var':
            rewards = self.get_hallucination_var_rewards(denoised_cur)
        elif self.reward_func == 'change_of_direction':
            rewards = self.get_change_of_direction_rewards(d_mid)
        elif self.reward_func == 'grad_norm':
            rewards = torch.norm(d_mid, dim=1)
        dones = self.get_dones()
        obs = self.get_observations()[0]
        self.current_step += 1

        return obs, rewards, dones, {"observations": {}, "pred_cur": denoised_cur}

    # def step_euler(self):
    #     t_cur = self.t_steps[self.current_step]
    #     t_next = self.t_steps[self.current_step + 1]
    #     x_cur = self.x
    #
    #     # By default we set use_afs = False.
    #
    #     denoised = self.denoise_fn(x_cur, t_cur)
    #
    #     d_cur = (x_cur - denoised) / t_cur
    #     hook.remove()
    #     t_cur = t_cur.reshape(-1, 1, 1, 1)
    #     t_next = t_next.reshape(-1, 1, 1, 1)
    #
    #     # t_mid = (t_next ** r) * (t_cur ** (1 - r))
    #     x_next = x_cur + (t_next - t_cur) * d_cur
    #     self.x = x_next
    #
    #     rewards = self.get_rewards(denoised)
    #     dones = self.get_dones()
    #     self.current_step += 1
    #
    #     return (unet_enc_out, t_cur, t_next), rewards, dones, {}

    def get_hallucination_var_rewards(self, x):
        with torch.no_grad():
            # if self.reward_func == "hallucination_var":
            n = self.current_step + 1
            if self.traj_mean is None:
                self.traj_mean = x.clone()
                self.sum_of_square = x.clone() ** 2
                self.mean_flatten_var = torch.zeros([self.batch_size, ]).to(self.device)
            else:
                self.traj_mean = self.traj_mean + 1 / n * (x - self.traj_mean)
                self.sum_of_square = self.sum_of_square + x ** 2
            traj_var = self.sum_of_square / n - self.traj_mean ** 2
            new_mean_flatten_var = torch.mean(torch.reshape(traj_var, [self.batch_size, -1]), dim=1)
            reward = new_mean_flatten_var - self.mean_flatten_var
            self.mean_flatten_var = new_mean_flatten_var
            return self.reward_scale * reward
            # elif self.reward_func == 'change of direction':
            #     if self.current_step == 0:
            #         return 0

    def get_change_of_direction_rewards(self, d):
        with torch.no_grad():
            if self.current_step == 0:
                reward = torch.zeros([self.batch_size, ]).to(self.device)
                self.last_d = d
            else:
                d_norm = F.normalize(d, p=2, dim=1)
                last_d_norm = F.normalize(self.last_d, p=2, dim=1)
                reward = (d_norm * last_d_norm).sum(dim=-1)
            return self.reward_scale * reward


    def get_current_mean_flatten_var(self):
        return self.mean_flatten_var

    def get_dones(self):
        if self.current_step != len(self.t_steps) - 2:
            return torch.zeros([self.batch_size], device=self.device)
        else:
            return torch.ones([self.batch_size], device=self.device)

    def save_images(self, img = None, grid = True, outdir = None):
        from torchvision.utils import make_grid, save_image
        images = self.x if img is None else img
        if outdir is None:
            if grid:
                outdir = os.path.join(f"./samples/grids/{self.dataset_name}", f"env_rollout_nfe{len(self.t_steps) - 1}")
            else:
                outdir = os.path.join(f"./samples/{self.dataset_name}", f"env_rollout_nfe{len(self.t_steps) - 1}")
        if grid:
            images = torch.clamp(images / 2 + 0.5, 0, 1)
            os.makedirs(outdir, exist_ok=True)
            nrows = int(images.shape[0] ** 0.5)
            image_grid = make_grid(images, nrows, padding=0)
            save_image(image_grid, os.path.join(outdir, "grid.png"))

    # def sample_via_sample_fn(self, ):
    #     from solvers_amed import euler_sampler
    #     img = euler_sampler(self.net, self.latents, self.class_labels, num_steps=1000, schedule_type='time_uniform',
    #                         schedule_rho=1.0)
    #     return img


if __name__ == '__main__':

    import torch
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    import numpy as np


    batch_size = 2048
    num_steps = 1000
    env = DiffSamplingEnv(
        batch_size=batch_size
    )
    env.reset() # batch_seeds=torch.randint(low=0, high=1000, size=(batch_size,))
    # img = env.sample_via_sample_fn()
    done = False
    rs = []
    obses = []
    while not done:
        obs, r, done, _ = env.step(np.vstack([-1 * np.ones(batch_size), np.zeros(batch_size)]).T)
        rs.append(r.cpu().numpy())
        obses.append(obs.cpu().numpy())
        done = torch.any(done)
    print(env.current_step, done)
    sum = np.sum(rs, axis=0)
    plt.plot(rs)
    var = np.array(rs).sum(axis=0)
    plt.show()
    obs = obs.cpu().numpy()
    obses = np.array(obses)[:, :, :2]
    var = np.var(obses, axis=0).mean(axis=-1)
    print(var)
    scatter = plt.scatter(obs[:, 0], obs[:, 1], c=var, )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Value')
    plt.show()
    # plot_with_kl_2d(obs)
    # env.save_images()



