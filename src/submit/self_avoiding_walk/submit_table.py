import pandas as pd

from src.utils.general_utils import get_folder

errors_folder = get_folder("eval/self_avoiding_walk/errors")

params = {
    "seed": 30,
    "threshold_set": [1e21, 5e23, 1e25, 5e26, 1e28],
    "params_steps": 10,
    "n": 11,
    "number_sims": 1000,
    "number_obs": 10000,
}

errors_file = (
    f"{params['seed']}_{params['number_sims']}_{params['number_obs']}_"
    f"{'-'.join(map(str, params['threshold_set']))}_{params['n']}.csv"
)

errors_df = pd.read_csv(f"{errors_folder}/{errors_file}", index_col=[0], header=[0, 1])
errors_df = errors_df.drop(["FITHIAN-WAGER"], axis=1, level=1)

submit_folder = get_folder("submit")
pd.set_option('display.float_format', '{:.3g}'.format)
errors_df.to_latex(f"{submit_folder}/self_avoiding_walk.tex")
