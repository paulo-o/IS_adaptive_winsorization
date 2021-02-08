import numpy as np
import numba as nb
import os
import math

from src.utils.general_utils import get_folder


# # # EXPONENTIAL

# g(x) = x
# p(x) = (1/param)*exp(-x/param) ~ theta * Expo
# q(x) = exp(-x) ~ Expo

# finite variance if param < 2
# mean = param


def exponential_proposal_sample(n):
    """Get n samples from Exponential proposal.
        (This is not Numba jitted because seed is needed.)
        Previously: r_q(size=N, **args_r_q_dict)."""
    return np.random.exponential(scale=1.0, size=n)


@nb.njit
def exponential_IS_ratio(x, param):
    """Calculates the ratio g*p/q for Exponential example, where g(x)=x,
        p is param*Expo and q is Expo.
        Previously: [g(x[j]) * d_p(x[j], **args_p_dict[example_n])
                     / d_q(x[j], **args_d_q_dict)
                     for j in range(len(x))]"""
    return [
        x[j] * (1 / param) * np.exp(-x[j] / param) / np.exp(-x[j])
        for j in range(len(x))
    ]


# # # NORMAL

# g(x) = x^2
# p(x) = (1/sqrt(2*pi)) * exp(-x^2/2) ~ N(0,1)
# q(x) = (1/sqrt(2 * pi * (1/1+param))) * exp(-x^2/2(1/(1+param))) ~ N(0, 1/(1+param))

# finite variance if param < 1
# mean = 1

def normal_proposal_sample(n, param):
    """Get n samples from Normal proposal.
        (This is not Numba jitted because seed is needed.)
        Previously: r_q(size=N, **args_r_q_dict)."""
    return np.random.normal(loc=0.0, scale=np.sqrt((1 / (1 + param))), size=n)


@nb.njit
def normal_IS_ratio(x, param):
    """Calculates the ratio g*p/q for Normal example, where g(x)=x^2
        p is N(0,1) and q is N(0, np.sqrt(1/1(1+param))).
        Previously: [g(x[j]) * d_p(x[j], **args_p_dict[example_n])
                     / d_q(x[j], **args_d_q_dict)
                     for j in range(len(x))]"""
    return [
        ((x[j] ** 2) / np.sqrt(1 + param)) * np.exp((x[j] ** 2) * (param / 2))
        for j in range(len(x))
    ]


# # # Beta

# g(x) = (x*(1-x))^(-1/2)
# p(x) = 1 ~ Unif
# q(x) = (Gamma(2*param)/(2*Gamma(param))) * (x*(1-x))^param ~ Beta(param, param)

# finite variance if param < 1
# mean = pi

def beta_proposal_sample(n, param):
    """Get n samples from Beta proposal.
    (This is not Numba jitted because seed is needed.)"""
    return np.random.beta(a=param, b=param, size=n)


@nb.njit
def beta_function(param):
    return (math.gamma(param) ** 2) / math.gamma(2 * param)


@nb.njit
def beta_IS_ratio(x, param):
    """Calculates the ratio g*p/q for Beta example, where g(x)=6*x*(1-x)
        p is Unif and q is Beta(param, param)."""
    return [
        beta_function(param) * ((x[j] * (1 - x[j])) ** (1 / 2 - param))
        for j in range(len(x))
    ]


# # # Pareto

# g(x) = 1
# p(x) = 1/(x^2) ~ Pareto(1, 1)
# q(x) = param / (x^(param+1)) ~ Pareto(param, 1)

# finite variance if param < 2
# mean = 1


def pareto_proposal_sample(n, param):
    """Get n samples from Pareto(param) proposal.
    (This is not Numba jitted because seed is needed.)"""
    return np.random.pareto(a=param, size=n) + 1


@nb.njit
def pareto_IS_ratio(x, param):
    """Calculates the ratio g*p/q for Pareto example, where g(x)=1
        p is Pareto(1) and q is Pareto(param)."""
    return [
        # (10/(param * 10**(param))) * (x[j] ** (param-1))
        (1 / param) * (x[j] ** (param - 1))
        for j in range(len(x))
    ]


# # # Chi-squared

# g(x) = 1
# p(x) = (1/(2^(25) * Gamma(25))) * x^(24) * e^(-x/2) ~ Chi-squared(50)
# q(x) = (1/(2^(param/2) * Gamma(param/2))) * x^(param/2-1) * e^(-x/2)
#   ~ Chi-squared(param)

# finite variance if param < 100
# mean = 1


def chi_squared_proposal_sample(n, param):
    """Get n samples from Chi_squared(param) proposal.
    (This is not Numba jitted because seed is needed.)"""
    return np.random.chisquare(param, size=n)


@nb.njit
def chi_squared_IS_ratio(x, param):
    """Calculates the ratio g*p/q for Chi_squared example, where g(x)=1
        p is Chi_squared(25) and q is Chi_squared(param)."""
    return [
        (2 ** (param / 2 - 25))
        * (math.gamma(param / 2) / math.gamma(25))
        * (x[j] ** (25 - param / 2))
        for j in range(len(x))
    ]


def output_file_exists(example, args):
    """Check if DataFrame with MSE and MAD errors have already been generated."""

    output_filename = (
        f"{example}_{args.seed}_{'-'.join(map(str, args.threshold_set))}_"
        f"{args.params_min}_{args.params_max}_{args.params_steps}"
        f"_{args.n}_{args.sim_size}"
    )
    errors_folder = get_folder("eval/synthetic/errors")
    return os.path.exists(f"{errors_folder}/{output_filename}")
