import numpy as np
import os
import torch
from ddpm_torch.toy import *
from ddpm_torch.utils import seed_all, infer_range
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

    parser.add_argument("--dataset",
                        choices=[
                            "gaussian1d", "gaussian8", "gaussian25",
                            "swissroll", "gaussian25_rotated",
                            "UnbalancedGaussian2D", "TwoMoonsToyData",
                            "Uniform1d"
                        ],
                        default="Uniform1d")
    parser.add_argument("--size", default=200000, type=int)
    parser.add_argument("--root",
                        default="~/datasets",
                        type=str,
                        help="root directory of datasets")
    parser.add_argument("--epochs",
                        default=300,
                        type=int,
                        help="total number of training epochs")
    parser.add_argument("--lr",
                        default=0.001,
                        type=float,
                        help="learning rate")
    parser.add_argument("--beta1",
                        default=0.9,
                        type=float,
                        help="beta_1 in Adam")
    parser.add_argument("--beta2",
                        default=0.999,
                        type=float,
                        help="beta_2 in Adam")
    parser.add_argument("--use-ema",
                        action="store_true",
                        help="apply EMA to model parameters during training")
    parser.add_argument("--ema-decay",
                        default=0.9999,
                        type=float,
                        help="EMA decay rate")
    parser.add_argument("--lr-warmup",
                        default=0,
                        type=int,
                        help="number of warming-up epochs")
    parser.add_argument("--batch-size", default=2048, type=int)
    parser.add_argument("--timesteps",
                        default=50,
                        type=int,
                        help="number of diffusion steps")

    parser.add_argument(
        "--beta-schedule",
        choices=["quad", "linear", "cosine", "warmup10", "warmup50", "jsd"],
        default="linear")
    parser.add_argument("--beta-start", default=0.001, type=float)
    parser.add_argument("--beta-end", default=0.3, type=float)
    parser.add_argument("--model-mean-type",
                        choices=["mean", "x_0", "eps"],
                        default="eps",
                        type=str)
    parser.add_argument("--model-var-type",
                        choices=["learned", "fixed-small", "fixed-large"],
                        default="fixed-small",
                        type=str)  # noqa
    parser.add_argument("--loss-type",
                        choices=[
                            "kl", "mse", "rssm", "idem", 'reweighting_data',
                            'reweighting_rl'
                        ],
                        default="reweighting_data",
                        type=str)
    parser.add_argument("--sampling_dist",
                        choices=["uniform", "pt", "Gaussian"],
                        default="uniform",
                        type=str,
                        help="Sampling distribution, only used in rssm loss type")
    parser.add_argument(
        "--sample-x0-noise-std",
        default=0.0,
        type=float,
        help="Std of Gaussian noise added to sampled x0 in RL reweighting")
    parser.add_argument("--image-dir", default="./images/train", type=str)
    parser.add_argument("--exp_str", default="0", type=str)
    parser.add_argument("--chkpt_dir",
                        default=os.path.join(root_dir, "chkpts"),
                        type=str)
    parser.add_argument("--chkpt-intv",
                        default=100,
                        type=int,
                        help="frequency of saving a checkpoint")
    parser.add_argument("--eval-intv", default=10, type=int)
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--resume",
                        action="store_true",
                        help="to resume training from a checkpoint")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--mid-features", default=128, type=int)
    parser.add_argument("--num-temporal-layers", default=3, type=int)

    parser.add_argument('--num_modes',
                        type=int,
                        help='Number of Modes (for 1D only)',
                        default=3)
    parser.add_argument('--modes',
                        type=int,
                        nargs='+',
                        help='Means of the Gaussians (for 1D only)',
                        default=[1, 2, 3])

    parser.add_argument("--generations", default=1, type=int)
    parser.add_argument("--num_sample_images", default=10_000, type=int)

    parser.add_argument("--wandb_project_name",
                        default="ddpm_hallucination",
                        type=str)
    parser.add_argument("--wandb_entity", default="haitongma", type=str)
    parser.add_argument("--log_results",
                        default=False,
                        action="store_true",
                        help="log results to wandb")

    parser.add_argument("--energy_fn", default='energy_func_linear', type=str)
    parser.add_argument(
        "--weights_transform_fn",
        default='standardize-softmax',
        type=str,
        help="Weights transformation function. dash separated string.")
    parser.add_argument("--reweighting_rl_warmup_epochs", default=20, type=int,
                        help="Number of warmup epochs for reweighting RL")
    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    assert args.num_modes == len(
        args.modes)  # Number of modes should be equal to the number of means.

    if "1d" in args.dataset:
        dataset_name = args.dataset + f"_{args.num_modes}_" + "".join(
            [str(mode) for mode in args.modes])
    else:
        dataset_name = args.dataset
    args.store_name = "_".join([
        f"{args.loss_type}", "seed", str(args.seed), "add_noise", str(args.sample_x0_noise_std),
        args.exp_str, args.energy_fn, args.weights_transform_fn,
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ])
    # set seed
    seed_all(args.seed)
    print(args)

    if args.log_results:
        wandb.init(project=args.wandb_project_name,
                   name=args.store_name)  # entity=args.wandb_entity
        wandb.config.update(args)
        wandb.run.log_code(".")

    # prepare toy data
    in_features = 1 if "1d" in args.dataset else 2
    dataset = args.dataset
    data_size = args.size
    root = os.path.expanduser(args.root)
    batch_size = args.batch_size
    num_batches = data_size // batch_size
    chkpt_dir = args.chkpt_dir + f"/{args.store_name}"
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    with open(os.path.join(chkpt_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    with open(os.path.join(chkpt_dir, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    trainloader = DataStreamer(dataset,
                               batch_size=batch_size,
                               num_batches=num_batches,
                               modes=args.modes)
    print("Max and Min of dataset: ", np.max(trainloader.dataset.data),
          np.min(trainloader.dataset.data))
    np.save(f"{chkpt_dir}/real_dataset.npy", trainloader.dataset.data)

    # training parameters
    device = torch.device(args.device)
    epochs = args.epochs

    # diffusion parameters
    beta_schedule = args.beta_schedule
    beta_start, beta_end = args.beta_start, args.beta_end
    timesteps = args.timesteps
    betas = get_beta_schedule(beta_schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              timesteps=timesteps)
    model_mean_type = args.model_mean_type
    model_var_type = args.model_var_type
    loss_type = args.loss_type
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        sampling_dist=args.sampling_dist,
        energy_fn=args.energy_fn,
        sample_x0_noise_std=args.sample_x0_noise_std,
        weights_transform_fn=args.weights_transform_fn,
        reweighting_rl_warmup_epochs=args.reweighting_rl_warmup_epochs)

    # model parameters
    out_features = 2 * in_features if model_var_type == "learned" else in_features
    mid_features = args.mid_features
    model = Decoder(in_features, mid_features, args.num_temporal_layers)
    model.to(device)

    # training parameters
    lr = args.lr
    beta1, beta2 = args.beta1, args.beta2
    optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

    # checkpoint path
    chkpt_dir = args.chkpt_dir + f"/{args.store_name}"
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    chkpt_path = os.path.join(chkpt_dir, f"ddpm_{dataset}_gen_0.pt")
    energy_csv_path = os.path.join(chkpt_dir, "energy_eval.csv")

    # set up image directory
    image_dir = os.path.join(chkpt_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # scheduler
    warmup = args.lr_warmup
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min(
            (t + 1) / warmup, 1.0)) if warmup > 0 else None

    # load trainer
    grad_norm = 0  # gradient global clipping is disabled
    eval_intv = args.eval_intv
    chkpt_intv = args.chkpt_intv
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      diffusion=diffusion,
                      epochs=epochs,
                      trainloader=trainloader,
                      scheduler=scheduler,
                      grad_norm=grad_norm,
                      use_ema=args.use_ema,
                      ema_decay=args.ema_decay,
                      device=device,
                      eval_intv=eval_intv,
                      chkpt_intv=chkpt_intv,
                      gen=0,
                      args=args)

    print("Len of trainloader: ", len(trainloader))
    print("Data size: ", data_size)
    plt.figure(figsize=(3, 3))
    dataloader_dataset = trainloader.dataset
    if in_features == 1:
        # Visualize histogram in case of 1D input.
        # Set log scale
        plt.yscale("log")
        plt.hist(dataloader_dataset.data,
                 bins=100,
                 alpha=0.7,
                 edgecolor='black')
    else:
        plt.scatter(*np.hsplit(next(iter(trainloader))[:2000], 2),
                    s=0.5,
                    alpha=0.7)

    plt.title('True data dist')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{image_dir}/gen.jpg")
    plt.close()

    # max_eval_count = min(data_size, 30000)
    max_eval_count = 10240  # max(args.num_sample_images, data_size)#min(data_size, data_size)
    print("Max eval count: ", max_eval_count)
    # eval_batch_size = min(max_eval_count, 30000)
    eval_batch_size = 2048
    print("Eval batch size: ", eval_batch_size)
    xlim, ylim = infer_range(trainloader.dataset)
    value_range = (xlim, ylim)
    true_data = iter(trainloader)
    if in_features == 1:
        evaluator = Evaluator1D(true_data=np.concatenate([
            next(true_data) for _ in range(
                min(max_eval_count // eval_batch_size, args.size //
                    args.batch_size))
        ]),
                                eval_batch_size=eval_batch_size,
                                max_eval_count=max_eval_count,
                                value_range=value_range)
    else:
        evaluator = Evaluator(true_data=np.concatenate([
            next(true_data) for _ in range(max_eval_count // eval_batch_size)
        ]),
                              eval_batch_size=eval_batch_size,
                              max_eval_count=max_eval_count,
                              value_range=value_range)
    if args.resume:
        try:
            trainer.load_checkpoint(chkpt_path)
        except FileNotFoundError:
            print("Checkpoint file does not exist!")
            print("Starting from scratch...")

    gen_dataset = trainer.train(evaluator,
                                chkpt_path=chkpt_path,
                                image_dir=image_dir,
                                eval_csv_path=energy_csv_path,
                                xlim=xlim,
                                ylim=ylim)
    np.save(f"{chkpt_dir}/gen_dataset.npy", gen_dataset)
    print(gen_dataset.shape)
    plt.figure(figsize=(3, 3))
    if "1d" in args.dataset:
        # Set log scale
        plt.yscale("log")
        plt.hist(gen_dataset, bins=100, alpha=0.7, edgecolor='black')
    else:
        plt.scatter(*np.hsplit(gen_dataset, 2), s=0.5, alpha=0.7)
    plt.title(f'RSSM w/ {args.sampling_dist} sample')
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.grid(True)
    plt.xticks(np.array([-6, -3, 0, 3, 6]))
    plt.yticks(np.array([-6, -3, 0, 3, 6]))
    plt.savefig(f"{chkpt_dir}/generated.pdf")
    plt.close()

    if args.log_results:
        wandb.log({
            f"Gen":
            wandb.Image(f"{chkpt_dir}/generated.pdf", caption=f"Gen 0")
        })


if __name__ == "__main__":
    main()
