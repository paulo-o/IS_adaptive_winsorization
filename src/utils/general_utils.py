import argparse
import os
import time
import warnings


def get_folder(folder_path, verbose=True):
    """Creates folder, if it doesn't exist, and returns folder path.
    Args:
        folder_path (str): Folder path, either existing or to be created.
    Returns:
        str: folder path.
    """
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        if verbose:
            print(f"-created directory {folder_path}")
    return folder_path


def is_notebook() -> bool:
    """Returns True if code is being executed interactively as a Jupyter notebook
    and False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class _TicToc(object):
    """
    Author: Hector Sanchez
    Date: 2018-07-26
    Description: Class that allows you to do 'tic toc' to your code.

    This class was based on https://github.com/hector-sab/ttictoc, which is
    distributed under the MIT license. It prints time information between
    successive tic() and toc() calls.

    Example:

        from src.utils.general_utils import tic,toc

        tic()
        tic()
        toc()
        toc()
    """

    def __init__(self, name="", method="time", nested=False, print_toc=True):
        """
        Args:
            name (str): Just informative, not needed
            method (int|str|ftn|clss): Still trying to understand the default
                options. 'time' uses the 'real wold' clock, while the other
                two use the cpu clock. To use your own method,
                do it through this argument

                Valid int values:
                    0: time.time | 1: time.perf_counter | 2: time.proces_time
                    3: time.time_ns | 4: time.perf_counter_ns
                    5: time.proces_time_ns

                Valid str values:
                  'time': time.time | 'perf_counter': time.perf_counter
                  'process_time': time.proces_time | 'time_ns': time.time_ns
                  'perf_counter_ns': time.perf_counter_ns
                  'proces_time_ns': time.proces_time_ns

                Others:
                  Whatever you want to use as time.time
            nested (bool): Allows to do tic toc with nested with a
                single object. If True, you can put several tics using the
                same object, and each toc will correspond to the respective tic.
                If False, it will only register one single tic, and
                return the respective elapsed time of the future tocs.
            print_toc (bool): Indicates if the toc method will print
                the elapsed time or not.
        """
        self.name = name
        self.nested = nested
        self.tstart = None
        if self.nested:
            self.set_nested(True)

        self._print_toc = print_toc

        self._int2strl = [
            "time",
            "perf_counter",
            "process_time",
            "time_ns",
            "perf_counter_ns",
            "process_time_ns",
        ]
        self._str2fn = {
            "time": [time.time, "s"],
            "perf_counter": [time.perf_counter, "s"],
            "process_time": [time.process_time, "s"],
            "time_ns": [time.time_ns, "ns"],
            "perf_counter_ns": [time.perf_counter_ns, "ns"],
            "process_time_ns": [time.process_time_ns, "ns"],
        }

        if type(method) is not int and type(method) is not str:
            self._get_time = method

        if type(method) is int and method < len(self._int2strl):
            method = self._int2strl[method]
        elif type(method) is int and method > len(self._int2strl):
            self._warning_value(method)
            method = "time"

        if type(method) is str and method in self._str2fn:
            self._get_time = self._str2fn[method][0]
            self._measure = self._str2fn[method][1]
        elif type(method) is str and method not in self._str2fn:
            self._warning_value(method)
            self._get_time = self._str2fn["time"][0]
            self._measure = self._str2fn["time"][1]

    def __warning_value(self, item):
        msg = f"Value '{item}' is not a valid option. Using 'time' instead."
        warnings.warn(msg, Warning)

    def __enter__(self):
        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()

    def __exit__(self, type, value, traceback):
        self.tend = self._get_time()
        if self.nested:
            self.elapsed = self.tend - self.tstart.pop()
        else:
            self.elapsed = self.tend - self.tstart

        self._print_elapsed()

    def _print_elapsed(self):
        """
        Prints the elapsed time
        """
        if self.name != "":
            name = "[{}] ".format(self.name)
        else:
            name = self.name
        print(
            "-{0}elapsed time: {1:.3g} ({2})".format(name, self.elapsed, self._measure)
        )

    def tic(self):
        """
        Defines the start of the timing.
        """
        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()

    def toc(self, print_elapsed=None):
        """
        Defines the end of the timing.
        """
        self.tend = self._get_time()
        if self.nested:
            if len(self.tstart) > 0:
                self.elapsed = self.tend - self.tstart.pop()
            else:
                self.elapsed = None
        else:
            if self.tstart:
                self.elapsed = self.tend - self.tstart
            else:
                self.elapsed = None

        if print_elapsed is None:
            if self._print_toc:
                self._print_elapsed()
        else:
            if print_elapsed:
                self._print_elapsed()

        # return(self.elapsed)

    def set_print_toc(self, set_print):
        """
        Indicate if you want the timed time printed out or not.

        Args:
          set_print (bool): If True, a message with the elapsed time
            will be printed.
        """
        if type(set_print) is bool:
            self._print_toc = set_print
        else:
            warnings.warn(
                "Parameter 'set_print' not boolean. " "Ignoring the command.", Warning,
            )

    def set_nested(self, nested):
        """
        Sets the nested functionality.
        """
        # Assert that the input is a boolean
        if type(nested) is bool:
            # Check if the request is actually changing the
            # behaviour of the nested tictoc
            if nested != self.nested:
                self.nested = nested

                if self.nested:
                    self.tstart = []
                else:
                    self.tstart = None
        else:
            warnings.warn(
                "Parameter 'nested' not boolean. " "Ignoring the command.", Warning,
            )


class TicToc(_TicToc):
    def tic(self, nested=True):
        """
        Defines the start of the timing.
        """
        if nested:
            self.set_nested(True)

        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()


__TICTOC_8320947502983745 = TicToc()
tic = __TICTOC_8320947502983745.tic
toc = __TICTOC_8320947502983745.toc
