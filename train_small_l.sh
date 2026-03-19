#!/bin/bash

EXP_STR="sweep_energy_fn_kwargs_l_initial_mean_2.0_fixed_softmax"
GROUP_NAME="sweep_energy_fn_kwargs_l_initial_mean_2.0_fixed_softmax"

for initial_mean in 1.0; do
    EXP_STR="two_gaussian_small_l_fixed_eval"
    GROUP_NAME="two_gaussian_small_l_fixed_eval"
    for temperature in 0.1; do
        # for energy_fn_kwargs_l in 1.0 1.2 1.4; do
        for seed in 42; do
            for l in 1.2; do
                for weights_transform_fn in "quad_relu_normalize" "relu_normalize" "softmax"; do
                    CUDA_VISIBLE_DEVICES=1 python gaussian_experiments/train_toy_policy_extraction.py \
                        --exp_str s${seed}_w${weights_transform_fn}_im${initial_mean}_t${temperature}_l${l} \
                        --group_name ${GROUP_NAME} \
                        --initial_mean ${initial_mean} \
                        --temperature ${temperature} \
                        --loss-type reweighting_rl \
                        --energy_fn energy_func_two_gaussian \
                        --size 20480 \
                        --seed ${seed} \
                        --use-ema --sample-x0-noise-std 0.05 \
                        --weights_transform_fn ${weights_transform_fn} \
                        --energy_fn_kwargs_l ${l} \
                        --eval-intv 2 \
                        --dataset gaussian1dv2 \
                        --epochs 400 \
                        --log_results 
                done
            done
        done
    done
done
