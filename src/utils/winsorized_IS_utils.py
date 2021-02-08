import numpy as np
import numba as nb


@nb.njit()
def is_sorted(y):
    """Returns whether np.array y is already sorted in descending order."""
    return np.all(np.diff(y) >= 0)


@nb.njit()
def small_bias_condition(y, M1, M2, c, t, n):
    """Checks if condition in Balancing IS theorem holds when winsorizing at M1, M2."""
    y_M1 = winsorize(y, M1)
    y_M2 = winsorize(y, M2)
    return np.abs(np.mean(y_M1) - np.mean(y_M2)) <= c * t * (
        (np.std(y_M1) + np.std(y_M2)) / 2
    ) / np.sqrt(n)


@nb.njit()
def bis_threshold(y, winsorization_vals, c=(1 + np.sqrt(5)), t=1, verbose=False):
    assert is_sorted(winsorization_vals)

    k = len(winsorization_vals)
    for i in range(k - 2, 1 - 2, -1):
        small_bias_conditions_true = np.array(
            [
                small_bias_condition(
                    y, winsorization_vals[i], winsorization_vals[j], c, t, len(y)
                )
                for j in range(i + 1, k)
            ]
        )

        if ~np.all(small_bias_conditions_true):
            return winsorization_vals[i + 1]
    return winsorization_vals[0]


# @nb.njit
def mse(param_estimate, param_true):
    """Calculate mean squared error between estimate and true parameter value."""
    param_estimate = np.asarray(param_estimate, dtype=np.float64)
    param_true = np.asarray(param_true, dtype=np.float64)
    return np.mean((param_estimate - param_true) ** 2)


# @nb.njit
def mad(param_estimate, param_true):
    """Calculate mean absolute deviation between estimate and true parameter value."""
    param_estimate = np.asarray(param_estimate, dtype=np.float64)
    param_true = np.asarray(param_true, dtype=np.float64)
    return np.mean(np.abs(param_estimate - param_true))


@nb.njit
def winsorize(y, wins_value_sup, wins_value_inf=None):
    """Winsorize array y using winsorization upper and lower thresholds. If lower
        value is not provided, use lower=-upper.

    Args:
        y (np.array): Array to be winsorized (or capped).
        wins_value_sup (float): Upper winsorization value.
        wins_value_inf (float): Lower winsorization value.

    Returns:
        np.array: Winsorized array.
    """
    # y = np.asarray(y)

    if wins_value_inf is None:
        # If no inferior value is provided, winsorize around zero.
        wins_value_inf = -wins_value_sup

    assert (
        wins_value_sup >= wins_value_inf
    ), "Upper winsorization bound < lower winsorization bound (did you flip them?)"

    y_wins = y.copy()
    y_wins[y > wins_value_sup] = wins_value_sup
    y_wins[y < wins_value_inf] = wins_value_inf

    return y_wins


def balanced_threshold(y, trunc_vals, c=(1 + np.sqrt(5)), t=1, verbose=False):
    y = np.asarray(y, dtype=np.float64)
    trunc_vals = np.asarray(trunc_vals, dtype=np.float64)

    trunc_vals = -np.sort(-trunc_vals)  # order truncation values in decreasing order
    n_trunc_vals = len(trunc_vals)  # number of truncated values

    n = len(y)  # size of sample

    i = 0  # iterate through truncation values; index of 'old' truncated value
    trunc_val_old = trunc_vals[i]  # start with first entry in trunc_vals
    trunc_val_new = trunc_vals[i + 1]  # consider updating to lower truncation value

    y_trunc_old = winsorize(y, trunc_val_old)  # truncate sample at each value
    y_trunc_new = winsorize(y, trunc_val_new)

    mean_trunc_old = np.mean(y_trunc_old)  # finds sample mean of each truncated sample
    mean_trunc_new = np.mean(y_trunc_new)

    sd_trunc_old = np.std(y_trunc_old)  # finds sample std dev of each truncated sample
    sd_trunc_new = np.std(y_trunc_new)

    mean_vector = np.array(
        [mean_trunc_old], dtype=np.float64
    )  # adds 'old' sample mean to mean_vector
    sd_vector = np.array(
        [sd_trunc_old], dtype=np.float64
    )  # adds 'old' sample std dev to sd_vector

    trunc_further = True  # keep on truncating further

    if verbose:
        print(f"First truncation at {trunc_val_old}")

    while trunc_further:

        # Condition must hold for all previous elements in mean_vector and sd_vector
        is_condition_valid = np.all(
            [
                np.abs(var1 - mean_trunc_new)
                < c * t * (1 / np.sqrt(n)) * (var2 + sd_trunc_new) / 2
                for var1, var2 in zip(list(mean_vector), list(sd_vector))
            ]
        )

        # If condition to truncate further holds, truncate it, update all objects
        if is_condition_valid:
            if verbose:
                print(f"Further truncating to {trunc_val_new}")

            # Add sample mean and std dev truncated at new value
            # to mean_vector and sd_vector
            mean_vector = np.append(mean_vector, mean_trunc_new)
            sd_vector = np.append(sd_vector, sd_trunc_new)

            # If truncating further, make 'old' truncation values the currently new ones
            # and update 'old' truncated sample, as well as 'old' truncated sample mean
            # and 'old' std dev
            trunc_val_old, y_trunc_old = trunc_val_new, y_trunc_new
            mean_trunc_old, sd_trunc_old = mean_trunc_new, sd_trunc_new

            # If truncating further, move down along trunc_vals to obtain the 'new'
            # truncation values and update 'new' truncated sample, as well as 'new'
            # sample mean and 'new' std dev
            i = i + 1  # update index of 'old' truncated value

            # If there are no more values to truncate at, though, stop
            if i == (n_trunc_vals - 1):
                if verbose:
                    print(f"Reached max truncation value: {trunc_val_new}")
                trunc_further = False  # stop truncating futher
                break

            trunc_val_new = trunc_vals[
                i + 1
            ]  # 'new' truncated value is one after 'old'
            y_trunc_new = winsorize(y, trunc_val_new)
            mean_trunc_new = np.mean(y_trunc_new)
            sd_trunc_new = np.std(y_trunc_new)

        # If condition to truncate further fails, stop procedure
        else:
            trunc_further = False  # stop truncating futher
            if verbose:
                print(f"Stopped truncation value at: {trunc_val_old}")

    # Return list with three objects
    return mean_vector, sd_vector, trunc_val_old


# @nb.njit
def is_sorted_in_descending_order(y):
    """Returns whether np.array y is already sorted in descending order."""
    return np.all(np.diff(y) <= 0)


def cv_threshold(y, winsorization_vals, number_of_folds=5, verbose=False):
    """Picks best winsorization level chosen via k-fold CV, with k=number_of_blocks.

    Args:
        y (np.array): vector of sample points to be (possibly) truncated
        wins_vals (np.array): pre-chosen winsorization thresholds set
        number_of_blocks (int): How many blocks to use in k-fold CV.
        verbose (bool): Whether to print extra information.

    Returns:
        float: Threshold chosen via CV.
    """
    y_folded = np.array(np.array_split(y, number_of_folds), dtype=object)
    cv_errors = np.zeros((number_of_folds, len(winsorization_vals)))

    for fold in range(number_of_folds):
        y_test = y_folded[fold]
        mask = np.ones(len(y_folded), dtype=bool)
        mask[fold] = False
        y_train = np.asarray(np.concatenate(y_folded[mask]), dtype=np.float64)
        mean_out_of_sample = np.mean(y_test)
        cv_errors[fold, :] = [
            mse(winsorize(y_train, winsorization_val), mean_out_of_sample)
            for winsorization_val in winsorization_vals
        ]

    if verbose:
        print(cv_errors.mean(axis=0))
    best_winsorization_val = winsorization_vals[np.argmin(cv_errors.mean(axis=0))]
    return best_winsorization_val
