import numpy as np
from numba import cuda
import random

@cuda.jit
def random_move_kernel(board, move):
    idx = cuda.threadIdx.x
    if idx == 0:
        empty_cells = []
        for i in range(9):
            if board[i] == 0:
                empty_cells.append(i)
        if len(empty_cells) > 0:
            move[0] = empty_cells[random.randint(0, len(empty_cells) - 1)]
