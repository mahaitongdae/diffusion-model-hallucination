#!/bin/bash

# add ema
# python gaussian_experiments/train_toy_policy_extraction.py --exp_str reverse_test \
# --loss-type reweighting_rl --energy_fn energy_func_linear_clip --size 20480 \
#  --log_results --use-ema

# # add sample_x0_noise_std
# python gaussian_experiments/train_toy_policy_extraction.py --exp_str reverse_test \
# --loss-type reweighting_rl --energy_fn energy_func_linear_clip --size 20480 \
#  --log_results --use-ema --sample-x0-noise-std 0.2

EXP_STR="compare_weights_transform_fns"

for seed in 42 43 44 45 46; do  
    for weights_transform_fn in "identity" "quadratic" "softmax"; do
        python gaussian_experiments/train_toy_policy_extraction.py --exp_str $EXP_STR \
        --loss-type reweighting_rl --energy_fn energy_func_two_gaussian --size 20480 --seed $seed \
        --log_results --use-ema --sample-x0-noise-std 0.2 --weights_transform_fn substract_min-${weights_transform_fn}
    done
done
