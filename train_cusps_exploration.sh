#!/bin/bash

for initial_mean in 0.0; do
    EXP_STR="shape_cusps_v2_exploration"
    GROUP_NAME="shape_cusps_v2_exploration"
    for temperature in 0.1; do
        # for energy_fn_kwargs_l in 1.0 1.2 1.4; do
        for seed in 42 43 44; do
            # for l in 0.1 0.2 0.3; do
            for weights_transform_fn in "group_relative_relu_linear" "group_relative_relu_quadratic" "softmax"; do
                CUDA_VISIBLE_DEVICES=1 python gaussian_experiments/train_toy_policy_extraction.py \
                    --exp_str s${seed}_w${weights_transform_fn}_im${initial_mean}_t${temperature}_l0.6 \
                    --group_name ${GROUP_NAME} \
                    --initial_mean ${initial_mean} \
                    --temperature ${temperature} \
                    --loss-type reweighting_rl \
                    --energy_fn energy_func_two_cusps \
                    --size 20480 \
                    --seed ${seed} \
                    --use-ema --sample-x0-noise-std 0.05 \
                    --weights_transform_fn ${weights_transform_fn} \
                    --energy_fn_kwargs_l 0.6 \
                    --eval-intv 2 \
                    --dataset gaussian1dv2 \
                    --epochs 500 \
                    --log_results
            done
            # done
        done
    done
done
