#!/bin/bash
sbatch --array=0-2 run_all/hopper_continuous.slurm
sbatch --array=0-2 run_all/hopper_feed_forward.slurm
sbatch --array=0-2 run_all/hopper_feed_forward_fov.slurm
sbatch --array=0-2 run_all/hopper_rnn.slurm
sbatch --array=0-2 run_all/hopper_transformer.slurm
