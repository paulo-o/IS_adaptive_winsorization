import os
import numba as nb
import numpy as np

from src.utils.general_utils import get_folder


@nb.njit
def is_fib_permutation(fib_permutation):
    """ Checks whether given permutation is a Fibonacci permutation.

    Input:
    fib_permutation [np.array]: a Fibonacci permutation

    Output:
    [Boolean]: whether the given permutation is a Fibonacci permutation"""

    n = len(fib_permutation)
    # ensure elements are unique
    assert n == len(np.unique(fib_permutation)), "Repeated elements in permutation"
    # ensure no -1 in fib_permutation
    assert np.all(
        np.where(fib_permutation >= 0, True, False)
    ), "Non-negative elements in permutation"

    # check if fib_permutation[i] is in {i-1, i, i+1}
    if not ((fib_permutation[0] == 0) or (fib_permutation[0] == 1)):
        # print("Position 0 is {}, not in {{0, 1}}".format(fib_permutation[0]))
        return False

    for i in range(1, n - 1):
        if not (
            (fib_permutation[i] == i - 1)
            or (fib_permutation[i] == i)
            or (fib_permutation[i] == i + 1)
        ):
            # print(
            #     "Position {} is {}, not in {{{}, {}, {}}}".format(
            #         i, fib_permutation[i], i - 1, i, i + 1
            #     )
            # )
            return False

    if not ((fib_permutation[n - 1] == n - 2) or (fib_permutation[n - 1] == n - 1)):
        # print(
        #     "Position {} is {}, not in {{{}, {}}}".format(
        #         n - 1, fib_permutation[n - 1], n - 2, n - 1
        #     )
        # )
        return False

    return True


@nb.njit
def fib(n):
    """ Function to compute the n-th number in the Fibonacci sequence in O(n),
    using dynamic programming.

    From: https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/ (method 3)

    Input:
    n [int]: term in the Fibonacci sequence we are interested in

    Output:
    b [int]: the n-th term in the Fibonacci sequence (e.g., fib(6) = 8)"""

    a = 1
    b = 1
    if n < 0:
        print("Incorrect input")
    elif n == 0:
        return a
    elif n == 1:
        return b
    else:
        for i in range(2, n):
            c = a + b
            a, b = b, c
        return b


@nb.njit
def fast_fib(n):
    """ Function to compute the n-th number in the Fibonacci sequence in O(log(n)).
    This uses the fact that if n is even and k = n/2, F(n) = [2*F(k-1) + F(k)]*F(k);
    if n is odd and k = (n + 1)/2, F(n) = F(k)*F(k) + F(k-1)*F(k-1).

    From: https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/ (method 6).

    Input:
    n [int]: term in the Fibonacci sequence we are interested in

    Output:
    f[n] [int]: the n-th term in the Fibonacci sequence (e.g., f[6] = 8)"""

    f = [0] * 1000

    # Base cases
    if n == 0:
        return 0
    if n == 1 or n == 2:
        f[n] = 1
        return f[n]

    # If fib(n) has already been computed
    if f[n]:
        return f[n]

    if n & 1:  # checks if n is odd by using bitwise 'and'
        k = (n + 1) // 2
    else:
        k = n // 2

    # Applying formula; note value (n & 1) is 1 if n is odd, else 0.
    if n & 1:
        f[n] = fast_fib(k) * fast_fib(k) + fast_fib(k - 1) * fast_fib(k - 1)
    else:
        f[n] = (2 * fast_fib(k - 1) + fast_fib(k)) * fast_fib(k)

    return f[n]


@nb.njit
def fib_permutation(n, initial_permutation):
    """ Generates a Fibonacci permutation of length n.

    Input:
    n [int]: length of Fibonacci permutation

    Output:
    fib_permutation [list]: a Fibonacci permutation of 1, ..., n
    cum_prod_neighbors [int]: cumulative product of all neighbors available
    when placing each element
    """
    # if initial_permutation is None:
    #     initial_permutation = np.array(range(n))

    cum_prod_neighbors = 1
    # initial_permutation = np.asarray(range(n))
    # initial_permutation[2::3] + initial_permutation[1::3] + initial_permutation[::3]
    fib_permutation = np.full(
        initial_permutation.shape, -1
    )  # initialize to (-1, ..., -1)

    while len(initial_permutation) > 0:
        i = initial_permutation[0]  # current first element in permutation

        # find available neighbors for i in fibonacci_permutation:
        if i == 0:
            available_neighbors = [x for x in [i, i + 1] if fib_permutation[x] == -1]
        elif i == (n - 1):
            available_neighbors = [x for x in [i - 1, i] if fib_permutation[x] == -1]
        else:
            available_neighbors = [
                x for x in [i - 1, i, i + 1] if fib_permutation[x] == -1
            ]
        cum_prod_neighbors *= len(available_neighbors)

        chosen_neighbor = np.random.choice(np.asarray(available_neighbors))
        fib_permutation[chosen_neighbor] = i

        # first element has been used; delete it
        initial_permutation = np.delete(initial_permutation, 0)
        # if i is not in position i, add chosen_neighbor to position i
        if chosen_neighbor != i:
            fib_permutation[i] = chosen_neighbor
            # chosen_neighbor has been used; delete it
            initial_permutation = np.delete(
                initial_permutation,
                np.where(initial_permutation == chosen_neighbor)[0][0],
            )

    return fib_permutation, cum_prod_neighbors


@nb.njit
def fib_permutations(n, reps=1000, seed=1, initial_permutation_type="random"):
    """ Generate Fibonacci permutations.

    Input:
    n [int]: length of Fibonacci permutation
    reps [int]: number of Fibonacci permutations to create
    seed [int]: seed for controlling randomness in the procedure
    """
    np.random.seed(seed)
    list_of_permutations = []
    list_of_cum_prod_neighbors = []

    for i in range(reps):
        if initial_permutation_type == "random":
            initial_permutation = np.random.permutation(n)
        elif initial_permutation_type == "identity":
            initial_permutation = np.arange(n)
        else:
            NotImplementedError()
        perm, cum_prod = fib_permutation(n, initial_permutation)
        # perm, cum_prod = fib_permutation(n)

        assert is_fib_permutation(perm)
        list_of_permutations.append(perm)
        list_of_cum_prod_neighbors.append(cum_prod)

    return list_of_permutations, list_of_cum_prod_neighbors


@nb.njit
def distinct_fib_permutations(list_of_permutations):
    """ List the distinct Fibonacci permutations in a list.

    Input:
    list_of_permutations [list]: list with many Fibonacci permutations

    Output:
    distinct_permutations [list]: list with only the distinct permutation
    """
    distinct_permutations = []

    for i in range(len(list_of_permutations)):
        if not any(
            np.array_equal(list_of_permutations[i], x) for x in distinct_permutations
        ):
            distinct_permutations.append(list_of_permutations[i])

    return distinct_permutations


def output_files_exist(args, file):
    """Check if script output files have already been generated."""

    if file == "generate":
        weights_folder = get_folder("data/fibonacci_permutations/weights")
        weights_filename = (
            f"weights_{args.type}_"
            f"{args.number_sims}_{args.number_obs}_{args.seed}_{args.n}.npy"
        )
        weights_file_exists = os.path.exists(f"{weights_folder}/{weights_filename}")
        return weights_file_exists
        # lengths_folder = get_folder("data/self_avoiding_walk/lengths")
        # lengths_filename = (
        #     f"lengths_{args.type}_"
        #     f"{args.number_sims}_{args.number_obs}_{args.seed}_{args.n}.npy"
        # )
        # lengths_file_exists = os.path.exists(f"{lengths_folder}/{lengths_filename}")
        # return weights_file_exists and lengths_file_exists

    elif file == "eval":
        errors_folder = get_folder("eval/fibonacci_permutations/errors")
        errors_filename = (
            f"{args.type}_{args.seed}_{args.number_sims}_{args.number_obs}_"
            f"{'-'.join(map(str, args.threshold_set))}_{args.n}.csv"
        )

        return os.path.exists(f"{errors_folder}/{errors_filename}.csv")

    else:
        raise NotImplementedError("Only checks for 'generate.py' and 'eval.py' "
                                  "have been implemented.")
