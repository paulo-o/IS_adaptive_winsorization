import subprocess
import numpy as np

NUMBER_SIMS = 1000
NUMBER_OBS = 100000

seeds = np.random.permutation(np.arange(30))
for seed in seeds:
    cmd_no_traps = (f"python src/self_avoiding_walk/generate.py "
                    f"-s {seed} -ns {NUMBER_SIMS} -no {NUMBER_OBS} -t no_traps")
    p = subprocess.call(cmd_no_traps, shell=True)

    cmd_all_traps = (f"python src/self_avoiding_walk/generate.py "
                     f"-s {seed} -ns {NUMBER_SIMS} -no {NUMBER_OBS} -t all_traps")
    p = subprocess.call(cmd_all_traps, shell=True)

    cmd_int_traps = (f"python src/self_avoiding_walk/generate.py "
                     f"-s {seed} -ns {NUMBER_SIMS} -no {NUMBER_OBS} -t interior_traps")
    p = subprocess.call(cmd_int_traps, shell=True)
