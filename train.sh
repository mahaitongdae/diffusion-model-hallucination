#!/bin/bash

# add ema
# python gaussian_experiments/train_toy_policy_extraction.py --exp_str reverse_test \
# --loss-type reweighting_rl --energy_fn energy_func_linear_clip --size 20480 \
#  --log_results --use-ema

# add sample_x0_noise_std
# python gaussian_experiments/train_toy_policy_extraction.py --exp_str reverse_test \
# --loss-type reweighting_rl --energy_fn energy_func_two_gaussian --size 20480 \
#  --log_results --use-ema --sample-x0-noise-std 0.2  --weights_transform_fn identity

# CUDA_VISIBLE_DEVICES=1 python gaussian_experiments/train_toy_policy_extraction.py \
#         --exp_str test_temperature \
#         --loss-type reweighting_rl --energy_fn energy_func_two_gaussian --size 20480 --seed 42 \
#         --log_results --use-ema --sample-x0-noise-std 0.1 \
#         --weights_transform_fn identity --temperature 0.1

# CUDA_VISIBLE_DEVICES=1 python gaussian_experiments/train_toy_policy_extraction.py \
#         --exp_str test_relu_normalize \
#         --loss-type reweighting_rl --energy_fn energy_func_two_gaussian --size 20480 --seed 42 \
#         --log_results --use-ema --sample-x0-noise-std 0.05 \
#         --weights_transform_fn relu_normalize --temperature 1.0

EXP_STR="sweep_energy_fn_kwargs_l_initial_mean_2.0_fixed_softmax"
GROUP_NAME="sweep_energy_fn_kwargs_l_initial_mean_2.0_fixed_softmax"

# for sample_x0_noise_std in 0.2; do
#     # for seed in 42 43 44 45 46; do
#     for weight_pre_transform_fn in "substract_min" "standarize-relu" ""; do
#     # for temperature in 0.01 0.1 1.0 10.0; do
#         for weights_transform_fn in "identity" "quadratic" "softmax"; do
#             CUDA_VISIBLE_DEVICES=1 python gaussian_experiments/train_toy_policy_extraction.py \
#             --exp_str ${EXP_STR}_noise_${sample_x0_noise_std}_temperature_1.0 --group_name ${GROUP_NAME} \
#             --loss-type reweighting_rl --energy_fn energy_func_two_gaussian --size 20480 --seed 42 \
#             --log_results --use-ema --sample-x0-noise-std ${sample_x0_noise_std} \
#             --weights_transform_fn ${weight_pre_transform_fn}-${weights_transform_fn}
#         done
#     done
#     # done
#     # done
# done
for initial_mean in 1.0 0.0 2.0; do
    EXP_STR="sweep_energy_fn_kwargs_l_initial_mean_${initial_mean}_fixed_softmax_v2"
    GROUP_NAME="sweep_energy_fn_kwargs_l_initial_mean_${initial_mean}_fixed_softmax_v2"
    for temperature in 0.1; do
        # for energy_fn_kwargs_l in 1.0 1.2 1.4; do
        for seed in 42 43 44; do
            for weights_transform_fn in "quad_relu_normalize" "relu_normalize" "softmax"; do
                CUDA_VISIBLE_DEVICES=1 python gaussian_experiments/train_toy_policy_extraction.py \
                    --exp_str test_relu_normalize_energy_fn_kwargs_l_${energy_fn_kwargs_l} \
                    --group_name ${GROUP_NAME} \
                    --loss-type reweighting_rl \
                    --energy_fn energy_func_two_gaussian \
                    --size 20480 \
                    --seed ${seed} \
                    --use-ema \
                    --sample-x0-noise-std 0.05 \
                    --weights_transform_fn ${weights_transform_fn} \
                    --temperature ${temperature} \
                    --dataset gaussian1dv2 \
                    --epochs 400 \
                    --log_results \
                    --initial_mean ${initial_mean} \
                    --eval-intv 2
            done
        done
    done
done
