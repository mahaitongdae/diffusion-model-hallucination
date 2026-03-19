import math
import numpy as np
import os
import csv
import torch
import torch.nn as nn
from contextlib import nullcontext
from tqdm import tqdm

gaussian_experiments_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(gaussian_experiments_dir)

from ..functions import discrete_klv2d, hist2d
from ddpm_torch.utils import save_scatterplot
from ddpm_torch.utils.train import DummyScheduler, RunningStatistics, EMA
from utils.energy_funcs import EnergyFunctionsRegistry
import wandb
from matplotlib import pyplot as plt
import datetime
import imageio


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        diffusion,
        epochs,
        trainloader,
        scheduler=None,
        shape=None,
        grad_norm=0,
        use_ema=False,
        ema_decay=0.9999,
        device=torch.device("cpu"),
        eval_intv=1,
        chkpt_intv=10,
        gen=0,
        args=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.epochs = epochs
        self.start_epoch = 0
        self.trainloader = trainloader
        if shape is None:
            shape = next(iter(trainloader)).shape[1:]
            if not shape:
                shape = (1, )
        self.shape = tuple(shape)
        print("Shape: ", self.shape)
        self.scheduler = DummyScheduler() if scheduler is None else scheduler
        self.grad_norm = grad_norm
        self.device = device
        self.eval_intv = eval_intv
        self.chkpt_intv = chkpt_intv
        self.args = args
        self.gen = gen

        self.use_ema = use_ema
        self.ema = EMA(self.model,
                       decay=ema_decay) if use_ema else nullcontext()

        self.stats = RunningStatistics(loss=None)
        self.epoch_count = 0

    def _run_eval(self, evaluator, sample_fn, epoch, eval_csv_path=None,
                  image_dir=None, image_paths=None, **plot_kwargs):
        """Run evaluation and save results to CSV and image."""
        self.model.eval()
        eval_results = evaluator.eval(sample_fn)

        # Log energy stats to CSV
        if eval_csv_path and "energy" in eval_results:
            energy_vals = np.array(eval_results["energy"])
            energy_mean = float(np.mean(energy_vals))
            energy_std = float(np.std(energy_vals))
            energy_stderr = float(energy_std / np.sqrt(len(energy_vals)))
            csv_exists = os.path.exists(eval_csv_path)
            with open(eval_csv_path, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not csv_exists:
                    writer.writerow(["epoch", "energy_mean", "energy_std", "energy_stderr"])
                writer.writerow([epoch, energy_mean, energy_std, energy_stderr])

        # Save image
        x_gen = eval_results.pop("x_gen", None)
        if x_gen is not None and image_dir:
            if "1d" in self.args.dataset:
                fig, ax1 = plt.subplots(1, 1, figsize=(4, 3))
                ax1.hist(x_gen, bins=40, alpha=0.7, density=True, label='Initial Policy')
                ax1.legend(loc='upper left')
                ax1.set_ylabel("PolicyDensity")
                h1, l1 = ax1.get_legend_handles_labels()
                ax1.legend().remove()
                h2, l2 = [], []
                if "reweighting" in self.args.loss_type and epoch >= self.diffusion.reweighting_rl_warmup_epochs:
                    x = np.linspace(-6, 6, 1000)
                    # energy = self.diffusion.energy_fn(x)
                    energy = evaluator.eval_energy_fn(x, **evaluator.energy_fn_kwargs)
                    ax2 = ax1.twinx()
                    ax2.plot(x, energy, label='Rewards', color='tab:orange')
                    ax2.set_ylabel("Rewards")
                    ax2.legend(loc='upper right')
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax2.legend().remove()
                ax1.legend(h1 + h2, l1 + l2, loc='upper left')
                ax1.set_xlabel("Epoch: {}".format(epoch - 300))
                plt.tight_layout()
                img_path = os.path.join(image_dir, f"{epoch}.jpg")
                plt.savefig(img_path)
                plt.close()
                if image_paths is not None:
                    image_paths.append(img_path)
            else:
                img_path = os.path.join(image_dir, f"{epoch}.jpg")
                save_scatterplot(img_path, x_gen, **plot_kwargs)
                if image_paths is not None:
                    image_paths.append(img_path)

        return eval_results, x_gen

    def loss(self, x):
        B = x.shape[0]
        T = self.diffusion.timesteps
        t = torch.randint(T, size=(B, ), dtype=torch.int64, device=self.device)
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        loss, info = self.diffusion.train_losses(
            self.model, x_0=x, t=t, epoch=self.epoch_count)
        assert loss.shape == (B, )
        return loss, info

    def step(self, x):
        # X has shape (1000, 2)
        B = x.shape[0]
        loss, info = self.loss(x)
        loss = loss.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # gradient clipping by global norm
        if self.grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     max_norm=self.grad_norm)
        self.optimizer.step()
        if self.use_ema and hasattr(self.ema, "update"):
            self.ema.update()
        self.stats.update(B, loss=loss.item() * B)

    def train(self,
              evaluator=None,
              chkpt_path=None,
              image_dir=None,
              eval_csv_path=None,
              warmup_chkpt_path=None,
              warmup_epochs=None,
              eval_at_start=False,
              **plot_kwargs):

        image_paths = []

        def sample_fn(n):
            shape = (n, ) + self.shape
            with self.ema if self.use_ema else nullcontext():
                sample = self.diffusion.p_sample(denoise_fn=self.model,
                                                 shape=shape,
                                                 device=self.device,
                                                 noise=None)
            return sample.cpu().numpy()

        # Evaluate at start if requested (e.g., after loading warmup checkpoint)
        if eval_at_start and evaluator is not None and self.start_epoch > 0:
            self._run_eval(evaluator, sample_fn, self.start_epoch,
                           eval_csv_path, image_dir, image_paths, **plot_kwargs)

        for e in range(self.start_epoch, self.epochs):
            self.epoch_count = e
            self.stats.reset()
            self.model.train()
            print(f"{e + 1}/{self.epochs} epochs")
            # with tqdm(self.trainloader, desc=f"{e + 1}/{self.epochs} epochs") as t:
            for i, x in enumerate(self.trainloader):
                self.step(x.to(self.device))
                # t.set_postfix(self.current_stats)
                if i == len(self.trainloader) - 1:
                    eval_results = dict()
                    x_gen = None
                    if (e + 1) % self.eval_intv == 0 and e + 1 >= self.diffusion.reweighting_rl_warmup_epochs:
                        if evaluator is not None:
                            eval_results, x_gen = self._run_eval(
                                evaluator, sample_fn, e + 1,
                                eval_csv_path, image_dir, image_paths, **plot_kwargs)
                        if self.args.log_results:
                            wandb.log(self.current_stats)
                    results = dict()
                    results.update(self.current_stats)
                    results.update(eval_results)
            # adjust learning rate every epoch before checkpoint
            self.scheduler.step()
            if warmup_chkpt_path and warmup_epochs and (e + 1) == warmup_epochs:
                self.save_checkpoint(warmup_chkpt_path,
                                     epoch=e + 1,
                                     warmup_checkpoint=True)
            if not (e + 1) % self.chkpt_intv and chkpt_path:
                self.save_checkpoint(chkpt_path, epoch=e + 1, **results)

        # Final evaluation
        x_gen = None
        if evaluator is not None:
            _, x_gen = self._run_eval(evaluator, sample_fn, self.epochs,
                                      eval_csv_path, image_dir, image_paths, **plot_kwargs)

        if image_dir and image_paths:
            print(f"Creating GIF in {image_dir}...")
            gif_path = os.path.join(image_dir, "training_progress.gif")
            with imageio.get_writer(gif_path, mode='I', duration=500, loop=0) as writer:
                for filename in image_paths:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            print(f"GIF saved to {gif_path}")

        return x_gen

    @property
    def current_stats(self):
        return self.stats.extract()

    @property
    def trainees(self):
        roster = ["model", "optimizer"]
        if self.use_ema:
            roster.append("ema")
        if self.scheduler is not None:
            roster.append("scheduler")
        return roster

    def load_checkpoint(self, chkpt_path):
        chkpt = torch.load(chkpt_path, map_location=self.device)
        if "model" in chkpt:
            self.model.load_state_dict(chkpt["model"])
        if "optimizer" in chkpt:
            self.optimizer.load_state_dict(chkpt["optimizer"])
        if self.scheduler is not None and "scheduler" in chkpt:
            self.scheduler.load_state_dict(chkpt["scheduler"])
        if self.use_ema and "ema" in chkpt:
            self.ema.load_state_dict(chkpt["ema"])
        self.start_epoch = chkpt.get("epoch", 0)

    def save_checkpoint(self, chkpt_path, **extra_info):
        chkpt = []
        for k, v in self.named_state_dicts():
            chkpt.append((k, v))
        for k, v in extra_info.items():
            chkpt.append((k, v))
        torch.save(dict(chkpt), chkpt_path)

    def named_state_dicts(self):
        for k in self.trainees:
            yield k, getattr(self, k).state_dict()


class Evaluator:
    def __init__(
            self,
            true_data,
            eval_batch_size=500,
            max_eval_count=30000,
            value_range=(-3, 3),
            eps=1e-9
    ):
        self.eval_batch_size = eval_batch_size
        self.max_eval_count = max_eval_count
        self.bins = math.floor(math.sqrt(self.max_eval_count // 10))
        self.value_range = value_range
        self.eps = eps
        self.true_hist = self.get_histogram(true_data)
        self.true_hist.setflags(write=False)  # noqa; make true_hist read-only

    def get_histogram(self, data):
        hist = 0
        for i in range(0, len(data), self.eval_batch_size):
            hist += hist2d(
                data[i:(i + self.eval_batch_size)], bins=self.bins, value_range=self.value_range)
        hist /= np.sum(hist) + self.eps  # avoid zero-division
        return hist

    def eval(self, sample_fn):
        x_gen = []
        gen_hist = 0
        for _ in range(0, self.max_eval_count, self.eval_batch_size):
            x_gen.append(sample_fn(self.eval_batch_size))
            gen_hist += hist2d(
                x_gen[-1], bins=self.bins, value_range=self.value_range)
        gen_hist /= np.sum(gen_hist) + self.eps
        return {
            "kld": discrete_klv2d(gen_hist, self.true_hist),
            "x_gen": np.concatenate(x_gen, axis=0)
        }


class Evaluator1D:

    def __init__(self,
                 true_data,
                 eval_batch_size=500,
                 max_eval_count=30000,
                 value_range=(-3, 3),
                 eps=1e-9,
                 eval_energy_fn="energy_func_two_gaussian",
                 energy_fn_kwargs={}):
        self.eval_batch_size = eval_batch_size
        self.max_eval_count = max_eval_count
        self.eps = eps
        self.eval_energy_fn = EnergyFunctionsRegistry.get(eval_energy_fn)
        self.energy_fn_kwargs = energy_fn_kwargs

    def eval(self, sample_fn):
        x_gen = []
        for j in range(0, self.max_eval_count, self.eval_batch_size):
            samples = sample_fn(self.eval_batch_size)
            x_gen.extend(samples)
        return {"x_gen": np.array(x_gen), "energy": self.eval_energy_fn(np.array(x_gen), **self.energy_fn_kwargs)}
