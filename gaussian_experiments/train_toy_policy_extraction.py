import numpy as np
import os
import torch
from ddpm_torch.toy import *
from ddpm_torch.utils import seed_all
from torch.optim import Adam, lr_scheduler
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import wandb
import datetime
import sys

f_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(f_dir)


def parse_arguments():
    parser = ArgumentParser()

    # Dataset
    parser.add_argument("--dataset",
                        choices=["gaussian1d", "Uniform1d", "gaussian1dv2"],
                        default="gaussian1dv2")
    parser.add_argument("--size", default=200000, type=int)
    parser.add_argument('--num_modes', type=int, default=3,
                        help='Number of Gaussian modes')
    parser.add_argument('--modes', type=int, nargs='+', default=[1, 2, 3],
                        help='Means of the Gaussians')
    parser.add_argument("--initial_mean", default=1.0, type=float,
                        help="Mean of initial gaussian for gaussian1dv2")

    # Training
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--batch-size", default=2048, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--lr-warmup", default=0, type=int)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", default=0.95, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    # Model
    parser.add_argument("--mid-features", default=128, type=int)
    parser.add_argument("--num-temporal-layers", default=3, type=int)

    # Diffusion
    parser.add_argument("--timesteps", default=50, type=int)
    parser.add_argument("--beta-schedule",
                        choices=["quad", "linear", "cosine"],
                        default="linear")
    parser.add_argument("--beta-start", default=0.001, type=float)
    parser.add_argument("--beta-end", default=0.3, type=float)
    parser.add_argument("--model-mean-type",
                        choices=["mean", "x_0", "eps"],
                        default="eps")
    parser.add_argument("--model-var-type",
                        choices=["learned", "fixed-small", "fixed-large"],
                        default="fixed-small")
    parser.add_argument("--loss-type",
                        choices=["mse", "reweighting_rl"],
                        default="reweighting_rl")

    # Reweighting RL
    parser.add_argument("--energy_fn", default='energy_func_linear', type=str)
    parser.add_argument("--weights_transform_fn", default='standardize-softmax', type=str)
    parser.add_argument("--reweighting_rl_warmup_epochs", default=300, type=int)
    parser.add_argument("--sample-x0-noise-std", default=0.0, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--energy_fn_kwargs_l", default=1.2, type=float)
    parser.add_argument("--weights_transform_kwargs_min_value", default=-0.1, type=float)

    # Checkpointing
    parser.add_argument("--chkpt_dir", default=os.path.join(root_dir, "chkpts"), type=str)
    parser.add_argument("--chkpt-intv", default=100, type=int)
    parser.add_argument("--eval-intv", default=10, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exp_str", default="0", type=str)
    parser.add_argument("--group_name", default="", type=str)

    # Logging
    parser.add_argument("--wandb_project_name", default="ddpm_hallucination", type=str)
    parser.add_argument("--log_results", action="store_true")
    

    return parser.parse_args()


def main():
    args = parse_arguments()
    assert args.num_modes == len(args.modes)

    # Store name for checkpoints
    args.store_name = "_".join([
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        args.loss_type, "seed", str(args.seed),
        "add_noise", str(args.sample_x0_noise_std),
        "temp", str(args.temperature),
        args.exp_str, args.weights_transform_fn
    ])

    seed_all(args.seed)
    print(args)

    if args.log_results:
        wandb.init(project=args.wandb_project_name,
                   name=args.store_name,
                   group=args.group_name)
        wandb.config.update(args)
        wandb.run.log_code(".")

    # Setup directories
    batch_size = args.batch_size
    num_batches = args.size // batch_size
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    chkpt_dir = os.path.join(args.chkpt_dir, f"{current_date}_{args.group_name}", args.store_name)
    os.makedirs(chkpt_dir, exist_ok=True)

    warmup_dir = os.path.join(args.chkpt_dir, "warmup_models",
                              f"{args.dataset}_im{args.initial_mean}")
    os.makedirs(warmup_dir, exist_ok=True)
    warmup_chkpt_path = os.path.join(warmup_dir, "warmup.pt")

    # Save command and args
    with open(os.path.join(chkpt_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    with open(os.path.join(chkpt_dir, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # Data
    trainloader = DataStreamer(args.dataset,
                               batch_size=batch_size,
                               num_batches=num_batches,
                               modes=args.modes,
                               gaussian1dv2_initial_mean=args.initial_mean)
    print(f"Dataset range: [{np.min(trainloader.dataset.data)}, {np.max(trainloader.dataset.data)}]")
    np.save(os.path.join(chkpt_dir, "real_dataset.npy"), trainloader.dataset.data)

    # Diffusion
    device = torch.device(args.device)
    betas = get_beta_schedule(args.beta_schedule,
                              beta_start=args.beta_start,
                              beta_end=args.beta_end,
                              timesteps=args.timesteps)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=args.model_mean_type,
        model_var_type=args.model_var_type,
        loss_type=args.loss_type,
        energy_fn=args.energy_fn,
        sample_x0_noise_std=args.sample_x0_noise_std,
        weights_transform_fn=args.weights_transform_fn,
        reweighting_rl_warmup_epochs=args.reweighting_rl_warmup_epochs,
        temperature=args.temperature,
        energy_fn_kwargs={"l": args.energy_fn_kwargs_l},
        weights_kwargs={"min_value": args.weights_transform_kwargs_min_value})

    # Model
    model = Decoder(1, args.mid_features, args.num_temporal_layers)
    model.to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = (lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: min((t + 1) / args.lr_warmup, 1.0))
                 if args.lr_warmup > 0 else None)

    # Paths
    chkpt_path = os.path.join(chkpt_dir, f"ddpm_{args.dataset}_gen_0.pt")
    energy_csv_path = os.path.join(chkpt_dir, "energy_eval.csv")
    image_dir = os.path.join(chkpt_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # Trainer
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      diffusion=diffusion,
                      epochs=args.epochs,
                      trainloader=trainloader,
                      scheduler=scheduler,
                      grad_norm=0,
                      use_ema=args.use_ema,
                      ema_decay=args.ema_decay,
                      device=device,
                      eval_intv=args.eval_intv,
                      chkpt_intv=args.chkpt_intv,
                      gen=0,
                      args=args)

    # Plot true data distribution
    plt.figure(figsize=(3, 3))
    plt.yscale("log")
    plt.hist(trainloader.dataset.data, bins=100, alpha=0.7, edgecolor='black')
    plt.title('True data dist')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "true_data.jpg"))
    plt.close()

    # Evaluator
    max_eval_count = 10240
    eval_batch_size = 2048
    true_data = iter(trainloader)
    evaluator = Evaluator1D(
        true_data=np.concatenate([
            next(true_data) for _ in range(
                min(max_eval_count // eval_batch_size, args.size // args.batch_size))
        ]),
        eval_batch_size=eval_batch_size,
        max_eval_count=max_eval_count,
        eval_energy_fn=args.energy_fn,
        energy_fn_kwargs={"l": args.energy_fn_kwargs_l})

    # Load checkpoint
    loaded_warmup = False
    if os.path.exists(warmup_chkpt_path):
        print(f"Loading warmup checkpoint from {warmup_chkpt_path}")
        trainer.load_checkpoint(warmup_chkpt_path)
        trainer.start_epoch = max(trainer.start_epoch, args.reweighting_rl_warmup_epochs - 1)
        loaded_warmup = True
    elif args.resume:
        try:
            trainer.load_checkpoint(chkpt_path)
        except FileNotFoundError:
            print("Checkpoint not found, starting from scratch...")

    # Train
    gen_dataset = trainer.train(evaluator,
                                chkpt_path=chkpt_path,
                                image_dir=image_dir,
                                eval_csv_path=energy_csv_path,
                                warmup_chkpt_path=warmup_chkpt_path,
                                warmup_epochs=args.reweighting_rl_warmup_epochs,
                                eval_at_start=loaded_warmup)

    # Save final generated data
    np.save(os.path.join(chkpt_dir, "gen_dataset.npy"), gen_dataset)
    print(f"Generated dataset shape: {gen_dataset.shape}")

    plt.figure(figsize=(3, 3))
    plt.yscale("log")
    plt.hist(gen_dataset, bins=100, alpha=0.7, edgecolor='black')
    plt.title('Generated dist')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(chkpt_dir, "generated.pdf"))
    plt.close()


if __name__ == "__main__":
    main()
