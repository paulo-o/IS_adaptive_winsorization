from fractions import Fraction
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

from src.utils.general_utils import get_folder


def gen_grid(n):
    """ Generates an (n+2)x(n+2) grid, with the edges filled with 5 (a boundary),
    the interior filled with 0 (spaces not explored by the SAW)

    Args:
    n: number of nodes in the (visible) grid; number of edges would be n-1

    Returns:
    (n+2)x(n+2): grid
    """
    grid = np.zeros((n + 2, n + 2))  # Create grid of zeros
    grid[0, :] = grid[:, n + 1] = grid[n + 1, :] = grid[
        :, 0
    ] = 5  # Create boundary of 5s
    return grid


def print_diagnostics(n, moves, log_probabilities, is_successful):
    """ Prints diagnostics for a given SAW.
    """
    print("SAW succeeded." if is_successful else "SAW failed.")
    print(f"Length is {len(log_probabilities)}, with nodes n={n}.")
    print(
        f"Probability of SAW is "
        f"{np.exp(np.sum(log_probabilities)) if is_successful else 0}."
    )
    print("")
    print("point |    probability ahead")
    for i in range(len(log_probabilities)):
        print(
            f"{moves[i]} |    "
            f"{str(Fraction(np.exp(log_probabilities[i])).limit_denominator())}"
        )


def neighbors(x):
    """ Returns the grid value of the neighbors of a given point in the grid x.
    """
    assert len(x) == 2
    return [(x[0] + 1, x[1]), (x[0], x[1] + 1), (x[0] - 1, x[1]), (x[0], x[1] - 1)]


def get_available_neighbors(x, grid):
    """ Returns the grid value of the neighbors of a given point in the grid x
    that have not been previously occupied or are beyond the boundary. It
    might return points that will trap our walk; this possibility is dealt with.
    """
    neighbors = [(x[0] + 1, x[1]), (x[0], x[1] + 1), (x[0] - 1, x[1]), (x[0], x[1] - 1)]
    return [neighbor for neighbor in neighbors(x) if grid[neighbor] == 0]


def plot_grid(
    moves,
    n,
    seed,
    type,
    is_successful=True,
    save_file=False,
    show_grid=True,
    show_ticks=True,
    style="light",
):
    """ Plots the grid with the self-avoiding random walk; the walk is in blue
    if it succeeds in reaching the upper right corner; it is red if it runs out
    of possible moves.

    Args:
    moves: set of nodes visited by the SAW
    n: number of nodes in the (visible) grid
    is_successful: whether the SAW reaches upper right corner of the grid
    save_file: boolean; if True, stores the plot as plots/plot.png
    show_grid: boolean; if True, displays grid, otherwise hides it
    show_ticks: boolean; if True, displays axes numbers, otherwise hides it
    style: if "light", then use seaborn "whitegrid" style, if "dark", uses "darkgrid"

    Returns:
    -
    """
    if style == "light":
        sns.set_style("whitegrid")
    elif style == "dark":
        sns.set_style("darkgrid")
    plot_color = "mediumblue" if is_successful else "crimson"
    moves_array = np.array(moves)
    plt.figure(figsize=(8, 8.5))
    plt.plot(
        moves_array[:, 1],
        moves_array[:, 0],
        "o",
        color=plot_color,
        linestyle="-",
        linewidth=2.0,
        markersize=10,
        clip_on=False,
        zorder=5,
    )
    plt.axis([0, n - 1, 0, n - 1])
    plt.yticks(range(0, n))
    plt.xticks(range(0, n))
    if not show_grid:
        plt.grid(False)
    if not show_ticks:
        plt.tick_params(
            axis="both",  # changes apply to both axes
            which="both",  # both major and minor ticks are affected
            bottom="off",  # ticks along the bottom edge are off
            top="off",  # ticks along the top edge are off
            left="off",
            right="off",
            labelleft="off",
            labelbottom="off",
        )
    if save_file:
        folder = get_folder("data/self_avoiding_walk/plots")
        filename = type + f"_plot_{n}_{seed}.png"
        plt.savefig(f"{folder}/{filename}", dpi=200)
        print(f" -saved: {folder}/{filename}")
    else:
        plt.show()
    plt.close()


def find_allowed_moves(n, x, grid):
    """ Returns the grid value of the allowed moves in the grid from a given
    point x; by "allowed moves" we mean points that have not been visited before
    and that will not let the walk be trapped in the boundary by going back to
    the origin (we exclude trappings onto itself, in the interior of the grid).
    """
    neighbors = [(x[0] + 1, x[1]), (x[0], x[1] + 1), (x[0] - 1, x[1]), (x[0], x[1] - 1)]
    unoccupied_neighbors = [neighbor for neighbor in neighbors if grid[neighbor] == 0]
    if x[0] == 1 or x[0] == n:  # upper or lower boundary point
        allowed_moves = unoccupied_neighbors
        if (x[0], x[1] - 1) in allowed_moves:
            allowed_moves.remove((x[0], x[1] - 1))
    elif x[1] == 1 or x[1] == n:  # left or right boundary point
        allowed_moves = unoccupied_neighbors
        if (x[0] + 1, x[1]) in allowed_moves:
            allowed_moves.remove((x[0] + 1, x[1]))
    else:  # interior point
        allowed_moves = unoccupied_neighbors
    return allowed_moves


def add_winding_number_triple(v):
    """ Finds the winding number of a vector with three elements. Here, we define
    winding number = (#left_turns-#right_turns)/2. The winding number is generally
    defined to be this quantity times pi, but we refrain from this extra
    multiplication for ease of reading the results. Hence, if the three points in
    v denote a left turn, we return 1/2; if they denote a right turn, we return
    -1/2; if neither, we return a zero.

    For example, if v=[(1, 1), (1, 2), (2, 2)], then this denotes a right turn,
    so the function returns -1/2.

    Args:
    v: vector with three points

    Returns:
    winding_number (-1/2 if right turn, 1/2 if left turn, 0 otherwise)
    """
    assert len(v) == 3
    x = v[-1]
    right_turns = [
        [(x[0] + 1, x[1] - 1), (x[0], x[1] - 1), x],
        [(x[0] - 1, x[1] - 1), (x[0] - 1, x[1]), x],
        [(x[0] + 1, x[1] + 1), (x[0] + 1, x[1]), x],
        [(x[0] - 1, x[1] + 1), (x[0], x[1] + 1), x],
    ]
    left_turns = [
        [(x[0] - 1, x[1] + 1), (x[0] - 1, x[1]), x],
        [(x[0] + 1, x[1] + 1), (x[0], x[1] + 1), x],
        [(x[0] - 1, x[1] - 1), (x[0], x[1] - 1), x],
        [(x[0] + 1, x[1] - 1), (x[0] + 1, x[1]), x],
    ]

    if [v[0], v[1], v[2]] in right_turns:
        return -1.0 / 2
    elif [v[0], v[1], v[2]] in left_turns:
        return 1.0 / 2
    else:
        return 0.0


def add_winding_number_path(path):
    """ Adds the winding number for the whole path. By 'path' we mean a closed
    curve. We append the first two elements of the path to the end as well, then
    run from element 2 till the end of the adjoined path, and for each triple
    we see whether it constitutes a curve by calling add_winding_number_triple(v).

    Args:
    path: vector whose points denote a closed path

    Returns:
    winding_number: (left_turns-right_turns)/2
    """
    path.append(path[0])
    path.append(path[1])
    winding_number = 0.0
    for i in range(2, len(path)):
        winding_number += add_winding_number_triple(path[(i - 2):(i + 1)])
    return winding_number


def north(x):
    """ Returns point to the north of x"""
    return (x[0] - 1, x[1])


def south(x):
    """ Returns point to the south of x"""
    return (x[0] + 1, x[1])


def east(x):
    """ Returns point to the east of x"""
    return (x[0], x[1] + 1)


def west(x):
    """ Returns point to the west of x"""
    return (x[0], x[1] - 1)


def rel_east(x, last_direction):
    """ Returns point to the relative east of x. That is, considering we are
    getting to x with our head looking in last_direction, returns the points
    that would be on our right side.

    Args:
    x: current position in grid
    last_direction: (absolute) direction we followed to get to x

    Returns:
    point to the relative east of x
    """
    if last_direction == "west":
        return north(x)
    elif last_direction == "east":
        return south(x)
    elif last_direction == "north":
        return east(x)
    elif last_direction == "south":
        return west(x)


def rel_west(x, last_direction):
    """ Returns point to the relative west of x. That is, considering we are
    getting to x with our head looking in last_direction, returns the points
    that would be on our left side.

    Args:
    x: current position in grid
    last_direction: (absolute) direction we followed to get to x

    Returns:
    point to the relative west of x
    """
    if last_direction == "west":
        return south(x)
    elif last_direction == "east":
        return north(x)
    elif last_direction == "north":
        return west(x)
    elif last_direction == "south":
        return east(x)


def rel_north(x, last_direction):
    """ Returns point to the relative north of x. That is, considering we are
    getting to x with our head looking in last_direction, returns the points
    that would be in front of us.

    Args:
    x: current position in grid
    last_direction: (absolute) direction we followed to get to x

    Returns:
    point in the relative north of x
    """
    if last_direction == "west":
        return west(x)
    elif last_direction == "east":
        return east(x)
    elif last_direction == "north":
        return north(x)
    elif last_direction == "south":
        return south(x)


def avoid_interior_SAW_traps(moves, grid, last_direction, verbose):
    """ Following Bousquet-Melou, ``On the importance sampling of self-avoiding
    walks'', this function returns the set of directions from the last move made
    that will not result in the walk being trapped in its interior (issues with
    the boundary are dealt with in find_allowed_moves()). Refer to figure 9 in
    the paper for each of 3 possibles cases we take care of in this function.

    Args:
    moves: vector with each point visited
    grid: grid with 0 if we haven't visited a point, 1 otherwise (and a border
    of 5s to denote we are outside of the n x n grid)
    last_direction: direction we took to get to the last move in moves

    Returns:
    unoccupied_neighbors: set of neighbors the walk can visit without getting
    trapped in the interior
    """
    x = moves[-1]  # current position is the result of last move
    neighbors = [(x[0] + 1, x[1]), (x[0], x[1] + 1), (x[0] - 1, x[1]), (x[0], x[1] - 1)]
    allowed_moves = [neighbor for neighbor in neighbors if grid[neighbor] == 0]

    # Check for case 1 in Bousquet
    # y = add_last_direction(x, last_direction)
    y = rel_north(x, last_direction)

    if grid[y] == 1:
        closed_path = moves[(moves.index(y)):(moves.index(x) + 1)]
        winding_number = add_winding_number_path(closed_path)

        if winding_number == 2.0 and rel_west(x, last_direction) in allowed_moves:
            allowed_moves.remove(rel_west(x, last_direction))
            # if verbose:
            #     print(f"Removed possibility 'relative west' at point {x}")
        elif winding_number == -2.0 and rel_east(x, last_direction) in allowed_moves:
            allowed_moves.remove(rel_east(x, last_direction))
            # if verbose:
            #     print(f"Removed possibility 'relative east' at point {x}")

    # Check for case 2, 3 in Bousquet
    # y1 is the result of turning to the relative east of x, then following the
    # direction last_direction (see Figure 9(ii)). These three points constitute
    # a left turn; we consider the path from y1 to x, and add y1_missing, the
    # point to the relative east of x, to close the loop. Depending on the
    # winding loop (ie. if we keep turning left, or right, as we go from y1 to x),
    # we exclude different points from the set of available neighbors (Figure 9(ii)
    # and Figure 9(iii)).

    # y2 is analogous, but the result of turning to the relative west of x, then
    # following the direction last_direction.

    # y1, y1_missing, y2, y2_missing = add_last_direction_diagonal(x, last_direction)
    y1_missing = rel_east(x, last_direction)  # result of turning relative east from x
    y1 = rel_north(
        y1_missing, last_direction
    )  # result of turning relative east from x, then relative west
    y2_missing = rel_west(x, last_direction)  # result of turning relative west from x
    y2 = rel_north(
        y2_missing, last_direction
    )  # result of turning relative west from x, then relative east

    if grid[y1] == 1:
        closed_path = moves[(moves.index(y1)):(moves.index(x) + 1)]
        closed_path.append(y1_missing)  # append missing point to close path
        winding_number = add_winding_number_path(
            closed_path
        )  # finds winding_number of a closed path

        if winding_number == 2.0:
            if rel_west(x, last_direction) in allowed_moves:
                allowed_moves.remove(rel_west(x, last_direction))
                # if verbose:
                #     print(f"Removed possibility 'relative west' at point {x}")
            if rel_north(x, last_direction) in allowed_moves:
                allowed_moves.remove(rel_north(x, last_direction))
                # if verbose:
                #     print(f"Removed possibility 'relative north' at point {x}")
        elif winding_number == -2.0 and rel_east(x, last_direction) in allowed_moves:
            allowed_moves.remove(rel_east(x, last_direction))
            # if verbose:
            #     print(f"Removed possibility 'relative east' at point {x}")

    if grid[y2] == 1:
        closed_path = moves[(moves.index(y2)):(moves.index(x) + 1)]
        closed_path.append(y2_missing)
        winding_number = add_winding_number_path(closed_path)
        if winding_number == 2.0 and rel_west(x, last_direction) in allowed_moves:
            allowed_moves.remove(rel_west(x, last_direction))
            # if verbose:
            #     print(f"Removed possibility 'relative west' at point {x}")
        elif winding_number == -2.0:
            if rel_east(x, last_direction) in allowed_moves:
                allowed_moves.remove(rel_east(x, last_direction))
                # if verbose:
                #     print(f"Removed possibility 'relative east' at point {x}")
            if rel_north(x, last_direction) in allowed_moves:
                allowed_moves.remove(rel_north(x, last_direction))
                # if verbose:
                #     print(f"Removed possibility 'relative north' at point {x}")

    return allowed_moves


def find_allowed_moves_no_traps(n, moves, grid, verbose):
    """ Returns the grid value of the allowed moves in the grid from a given
    point x; by "allowed moves" we mean points that have not been visited before
    and that will not let the walk be trapped in the boundary by going back to
    the origin (we exclude trappings onto itself, in the interior of the grid).
    """
    x = moves[-1]  # current position is the result of last move

    x_old = moves[-2]  # previous move to x
    if x_old[0] == x[0] and x_old[1] - 1 == x[1]:
        last_direction = "west"  # coming from the west
    elif x_old[0] == x[0] and x_old[1] + 1 == x[1]:
        last_direction = "east"  # coming from the east
    elif x_old[0] - 1 == x[0] and x_old[1] == x[1]:
        last_direction = "north"  # coming from the north
    elif x_old[0] + 1 == x[0] and x_old[1] == x[1]:
        last_direction = "south"  # coming from the south
    else:
        raise Exception("Interior point not coming from east, west, north or south!")

    # Avoid moves that will trap the SAW in its interior
    allowed_moves = avoid_interior_SAW_traps(moves, grid, last_direction, verbose)

    # If we are in the upper or lower boundary, avoid going left
    if x[0] == 1 or x[0] == n:
        if (x[0], x[1] - 1) in allowed_moves:
            allowed_moves.remove((x[0], x[1] - 1))

    # If we are in the left or right boundary, avoid going down
    elif x[1] == 1 or x[1] == n:
        if (x[0] + 1, x[1]) in allowed_moves:
            allowed_moves.remove((x[0] + 1, x[1]))

    return allowed_moves


def run_SAW(n, type="all_traps", verbose=True):
    """ Runs a self-avoiding random walk in a grid with n x n nodes.

    Args:
    n: number of nodes in the (visible) grid

    Returns:
    moves: set of nodes visited by the SAW
    log_probabilities: log of the probability of each edge in the SAW
    is_successful: whether the SAW reaches upper right corner of the grid
    """
    log_probabilities = []
    moves = []
    grid = gen_grid(n)  # create grid
    x = (n, 1)  # SAW starts at the origin
    grid[x] = 1  # mark origin as visited
    moves.append(x)  # append origin to the moves vector

    if type == "no_traps":
        # First step in the walk; this is so the 'for' loop below can assume
        # a direction for each step (ie. coming from north, south, east, west)
        allowed_moves = [(n - 1, 1), (n, 2)]
        number_of_allowed_moves = len(allowed_moves)
        log_probabilities.append(np.log(1.0 / number_of_allowed_moves))
        x = random.choice(allowed_moves)  # randomly choose one of the edges to visit
        moves.append(x)  # append the move
        grid[x] = 1  # mark position as visited

        while True:
            allowed_moves = find_allowed_moves_no_traps(n, moves, grid, verbose)
            number_of_allowed_moves = len(allowed_moves)
            if (
                number_of_allowed_moves == 0
            ):  # all neighbors have been visited; SAW fails
                # print("Failed run!")
                is_successful = False
                break
            # if there is at least one edge to visit, append to log_probabilities
            # the current value of log(1/number_of_allowed_moves)
            log_probabilities.append(np.log(1.0 / number_of_allowed_moves))
            x = random.choice(
                allowed_moves
            )  # randomly choose one of the edges to visit
            moves.append(x)  # append the move
            grid[x] = 1  # mark position as visited

            if x == (1, n):  # reaches upper right corner; SAW succeeds
                # print("Successful run!")
                is_successful = True
                break

        return moves, log_probabilities, is_successful

    else:
        while True:
            if type == "all_traps":
                allowed_moves = [
                    neighbor for neighbor in neighbors(x) if grid[neighbor] == 0
                ]
            elif type == "interior_traps":
                allowed_moves = find_allowed_moves(n, x, grid)
            number_of_allowed_moves = len(allowed_moves)
            if (
                number_of_allowed_moves == 0
            ):  # all neighbors have been visited; SAW fails
                # print("Failed run!")
                is_successful = False
                break

            # if there is at least one edge to visit, append to log_probabilities
            # the current value of log(1/number_of_allowed_moves)
            log_probabilities.append(np.log(1.0 / number_of_allowed_moves))
            x = random.choice(
                allowed_moves
            )  # randomly choose one of the edges to visit
            moves.append(x)  # append the move
            grid[x] = 1  # mark position as visited

            if x == (1, n):  # reaches upper right corner; SAW succeeds
                # print("Successful run!")
                is_successful = True
                break

        return moves, log_probabilities, is_successful


def get_number_of_saws(n):
    """Return actual number of saws for gridsize."""
    if n == 11:
        return 1568758030464750013214100
    else:
        raise NotImplementedError("Only n=11 has hardcoded answer.")


def output_files_exist(args, file):
    """Check if script output files have already been generated."""

    if file == "generate":
        weights_folder = get_folder("data/self_avoiding_walk/weights")
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
        errors_folder = get_folder("eval/self_avoiding_walk/errors")
        errors_filename = (
            f"{args.seed}_{args.number_sims}_{args.number_obs}_"
            f"{'-'.join(map(str, args.threshold_set))}_{args.n}.csv"
        )

        return os.path.exists(f"{errors_folder}/{errors_filename}.csv")

    else:
        raise NotImplementedError("Only checks for 'generate.py' and 'eval.py' "
                                  "have been implemented.")
