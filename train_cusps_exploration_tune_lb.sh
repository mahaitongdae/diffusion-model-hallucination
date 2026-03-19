#!/bin/bash

for initial_mean in 0.0; do
    EXP_STR="shape_cusps_v2_exploration"
    GROUP_NAME="shape_cusps_v2_exploration"
    for temperature in 0.1; do
        # for energy_fn_kwargs_l in 1.0 1.2 1.4; do
        for weights_transform_kwargs_min_value in 0.0; do
            for energy_fn_kwargs_l in 0.65 0.7 0.75; do
                for seed in 42; do
                    # for l in 0.1 0.2 0.3; do
                    for weights_transform_fn in "group_relative_negative_linear"; do
                        CUDA_VISIBLE_DEVICES=1 python gaussian_experiments/train_toy_policy_extraction.py \
                            --exp_str s${seed}_w${weights_transform_fn}_im${initial_mean}_t${temperature}_l${energy_fn_kwargs_l}_lb${weights_transform_kwargs_min_value} \
                            --group_name ${GROUP_NAME} \
                            --initial_mean ${initial_mean} \
                            --temperature ${temperature} \
                            --loss-type reweighting_rl \
                            --energy_fn energy_func_two_cusps \
                            --size 20480 \
                            --seed ${seed} \
                            --use-ema --sample-x0-noise-std 0.05 \
                            --weights_transform_fn ${weights_transform_fn} \
                            --energy_fn_kwargs_l ${energy_fn_kwargs_l} \
                            --eval-intv 2 \
                            --dataset gaussian1dv2 \
                            --epochs 400 \
                            --log_results \
                            --weights_transform_kwargs_min_value ${weights_transform_kwargs_min_value}
                    done
                done
            done
        done
    done
done
