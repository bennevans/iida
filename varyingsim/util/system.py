
import socket
import os

def is_cassio():
    host_split = socket.gethostname().split('.')
    return len(host_split) > 1 and host_split[1] == 'cs'

def is_greene():
    host_split = socket.gethostname().split('.')
    return len(host_split) > 1 and host_split[-1] == 'cluster'

def get_true_cores():
    n_cpu = os.cpu_count()
    lines = os.popen('lscpu').readlines()
    threads_per_core = int([l for l in lines if l.startswith('Thread(s)')][0].strip()[-1])
    max_cores = n_cpu // threads_per_core

    slurm_cores = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cores:
        return min(int(slurm_cores), max_cores)
    else:
        return max_cores

def hash_config(config):
    # dict config. any changes to keys or values will change hash
    indiv_hashes = []
    for k, v in config.items():
        indiv_hashes.append(hash(k) + hash(str(v)))
    
    final_hash = 0
    for h in indiv_hashes:
        final_hash = hash(h + final_hash)
    return final_hash