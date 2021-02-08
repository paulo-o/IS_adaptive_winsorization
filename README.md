# Code for "Robust Importance Sampling with Adaptive Winsorization"

This folder includes the code to generate all the figures, tables and plots in the paper "Robust Importance Sampling with Adaptive Winsorization". The base folder for this repository is `code/`; everything should be run from this directory. To create a conda environment with the necessary packages, run

- `conda env create -f src/utils/environment.yml`

This will create the environment `is` (for "Importance Sampling").

## Synthetic Examples

To generate data and figures containing the mean squared error (MSE) and mean absolute deviation (MAD) of different importance sampling (IS) estimators, run each of the following files:
- `python src/synthetic/beta.py`: generate data for beta example

- `python src/synthetic/chi_squared.py`: generate data for chi squared example

- `python src/synthetic/exponential.py`: generate data for exponential example

- `python src/synthetic/normal.py`: generate data for normal example

  Note the scripts above receive flags, so it is possible to run `python src/synthetic/beta.py --sim_size 1000 --n 100 --threshold_set 10 20 30 40 50 60 70 80 90 100 --seed 2020 --params_min 0.1 --params_max 1.0 --params_steps 5  `. The default flags indicate the values used to generate the figures in the paper.

For each example, the MSE and MAD of the following five estimators are created: (i) usual IS; (ii) balanced IS, as proposed in the paper; (iii) CV IS, a winsorized estimator that uses cross-validation to pick the threshold value; (iv) Fixed, which winsorizes at $\sqrt{n}$, where $n$ is the number of samples, following Pinelis; and (v) Fithian-Wager, which discards the largest ratio observed. In each example, the proposal or target distributions are changed so it is possible to observe how the estimators behave as the underlying IS problem becomes harder (generally increasing until the IS estimator has infinite variance).

The result of running each example is a Pandas DataFrame storing MSE and MAD data (in `eval/errors/synthetic/`), as well as plots exhibiting how each estimator fares (in `eval/plots/synthetic`). Note the filenames store all information needed to reproduce the figures.

To obtain the final figures used in the paper, run

- `python src/submit/synthetic/submit_plot.py`: create final plots from generated data.

The figures will be available in `submit/`, for instance, `submit/beta_MAD.png` and `submit/beta_MSE.png`.

## Self-avoiding Walks

To plot the self-avoiding walks pictured in the paper, run the file

- `python src/self_avoiding_walk/plot_saw.py`: create self-avoiding walk pictures; output is in `data/self_avoiding_walk/plots` (red denotes unsuccesful walk, blue successful).

  The following flags are available: 

  - `--n`: number of nodes in grid; edges = nodes - 1; default is 11;
  - `--type`: type of self avoiding walk (saw): all_traps generates saws that can fail by self-intersecting anywhere; interior_traps generates saws that can only by self-intersecting in the interior of the grid; and no_traps generates saws that never self-intersect
  - `--seeds`: list of seeds to use when generating saw plots,
  - `--verbose`: whether to print diagnostic information.

To generate self-avoiding walks and save their weights for posterior importance sampling estimation, run each of the following files:

- `python src/self_avoiding_walk/generate.py`: save weights for self-avoiding walk estimation, output is in `data/self_avoiding_walk/weights`

  The script above receive flags, so it is possible to run `python src/self_avoiding_walk/generate.py --type all_traps --number_sims 100 --number_obs 1000 --seed 0 --n 11 --verbose True`.  

- `python src/self_avoiding_walk/eval.py`: load weights from given example and use importance sampling to estimate the number of self-avoiding walks for that grid size. As before, five estimators are created: (i) usual IS; (ii) balanced IS, as proposed in the paper; (iii) CV IS, a winsorized estimator that uses cross-validation to pick the threshold value; (iv) Fixed, which winsorizes at $\sqrt{n}$, where $n$ is the number of samples, following Pinelis; and (v) Fithian-Wager, which discards the largest ratio observed.

  The script above receive flags, so it is possible to run `python src/self_avoiding_walk/eval.py --number_sims 100 --number_obs 1000 --seed 0 --n 11 --threshold_set 1e21 5e23 1e25 5e26 1e28 --verbose True`.  

To create the figures and tables in the paper, run:

- `python src/submit/self_avoiding_walk/submit_table.py`: creates table with MSE and MAD for the five estimators and each of the three types of proposal distributions (no traps, interior traps, all traps); the output is `submit/self_avoiding_walk.tex`.
- `python src/submit/self_avoiding_walk/submit_plots.py`: generated violin plots for the five estimators and the three proposal distributions; the output is in `submit/`, e.g., `submit/self_avoiding_walk-predictions-interior_traps-violin.png`.