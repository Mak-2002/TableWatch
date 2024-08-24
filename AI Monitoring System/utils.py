import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def are_tables_similar(coords1, coords2, threshold=50):
    """Check if two sets of coordinates represent the same table."""
    x1_diff = abs(coords1[0] - coords2[0])
    y1_diff = abs(coords1[1] - coords2[1])
    x2_diff = abs(coords1[2] - coords2[2])
    y2_diff = abs(coords1[3] - coords2[3])

    return x1_diff < threshold and y1_diff < threshold and x2_diff < threshold and y2_diff < threshold
