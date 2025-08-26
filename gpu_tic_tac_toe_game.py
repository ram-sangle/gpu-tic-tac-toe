import numpy as np
from numba import cuda
import json
from gpu_random_player import random_move_kernel

# Minimax strategy for GPU 2 (CPU-based for simplicity)
def minimax(board, player):
    winner = check_winner(board)
    if winner == 1:
        return -1
    elif winner == 2:
        return 1
    elif 0 not in board:
        return 0

    moves = []
    for i in range(9):
        if board[i] == 0:
            board[i] = player
            score = minimax(board, 3 - player)
            moves.append((score, i))
            board[i] = 0

    if player == 2:
        return max(moves)[1]
    else:
        return min(moves)[1]

def check_winner(board):
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if board[a] == board[b] == board[c] and board[a] != 0:
            return board[a]
    return 0

def print_board(board):
    symbols = [' ', 'X', 'O']
    for i in range(3):
        print('|'.join(symbols[board[3*i+j]] for j in range(3)))
        if i < 2:
            print('-'*5)

def main():
    board = np.zeros(9, dtype=np.int32)
    history = []
    move_gpu = np.zeros(1, dtype=np.int32)
    d_board = cuda.to_device(board)
    d_move = cuda.to_device(move_gpu)

    for turn in range(9):
        player = 1 if turn % 2 == 0 else 2
        if player == 1:
            random_move_kernel[1, 1](d_board, d_move)
            move_gpu = d_move.copy_to_host()
            move = move_gpu[0]
        else:
            move = minimax(board.copy(), player)

        if board[move] == 0:
            board[move] = player
            history.append(board.copy().tolist())
            print("Player {0} moves to {1}".format(player, move))
            print_board(board)
            winner = check_winner(board)
            if winner != 0:
                print("Player {0} wins!".format(winner))
                break
        else:
            print("Invalid move by Player {0} at {1}".format(player, move))
            break
    else:
        print("Game ends in a draw.")

    with open("tic_tac_toe_history.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()
