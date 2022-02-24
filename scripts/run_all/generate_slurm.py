
if __name__ == '__main__':
    run_dir = "run_all"
    filename_base_str = "{}_{}.slurm"
    base_str = \
"""#!/bin/bash
#SBATCH --job-name=run_exp_seeded
#SBATCH --open-mode=append
#SBATCH --output=./output/%j_%x_%a.out
#SBATCH --error=./output/%j_%x_%a.err
#SBATCH --export=ALL
#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --gres=gpu:1

singularity exec --nv --overlay /scratch/bne215/overlay-50G-10M.ext3:ro \
    --overlay /scratch/work/public/singularity/mujoco200-dep-cuda11.1-cudnn8-ubunutu18.04.sqf:ro \
    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c \"
source /ext3/env.sh
conda activate
echo 'task id $SLURM_ARRAY_TASK_ID'
python run_exp.py -c da_configs/greene/{}/{}/{}_$SLURM_ARRAY_TASK_ID.yaml
\"
    """
    env_names = ["humanoid", "pushbox", "slidepuck", "swimmer", "hopper"]
    algo_dirs = ["continuous", "feed_forward", "feed_forward_fov", "rnn", "transformer"]
    algo_suffix = ["continuous", "ff", "ff_fov", "rnn", "transformer"]

    for env in env_names:
        for d,s in zip(algo_dirs, algo_suffix):
            filename = "{}_{}".format(env, s)
            out_str = base_str.format(env, d, filename)
            with open(filename_base_str.format(env, d), 'w') as f:
                f.write(out_str)

    run_strs = []
    for env in env_names:
        for algo_dir in algo_dirs:
            filename = filename_base_str.format(env, algo_dir)
            run_strs.append("sbatch --array=0-2 {}/{}".format(run_dir, filename))

    with open("run_all.sh", "w") as f:
        f.write('#!/bin/bash\n')
        for r in run_strs:
            line = "{}\n".format(r)
            f.write(line)
