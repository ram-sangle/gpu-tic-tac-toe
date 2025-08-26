# gpu-tic-tac-toe
GPU Tic-Tac-Toe : Test 

This project implements a classic Tic-Tac-Toe game where two GPUs act as competing players. Each GPU is assigned a strategy to decide its moves. The game board is represented as a 3x3 matrix, and the GPUs take turns placing their marks (X or O) until one wins or the game ends in a draw.

GPU 1 uses a random move selection strategy.
GPU 2 uses a minimax-based strategy for optimal play.
The goal is to simulate a competition between two GPU kernels, each making decisions independently and updating the shared game state. The game can be replayed using a sequence of board states saved after each move.

Code Description
The code is structured into the following components:

Game Engine (CPU-side):

Initializes the board.
Manages turn-taking between GPUs.
Checks for win/draw conditions.
Stores the board state after each move for replay.
GPU Kernels:

GPU 1 Kernel: Implements a simple random move generator.
GPU 2 Kernel: Implements a minimax algorithm with alpha-beta pruning for optimal move selection.
Data Transfer:

The board state is transferred to the GPU before each move.
The selected move is returned to the CPU for validation and update.
Replay Generator:

Stores each board state in a list.
Outputs the sequence as a JSON or CSV file for visualization.

How to Run
Ensure you have:

A CUDA-compatible GPU
NVIDIA drivers installed
Python with numba, numpy, and json libraries
Place both files in the same directory.

Run the game:
python gpu_tic_tac_toe_game.py

This will simulate a game between two GPU players and save the move history in tic_tac_toe_history.json.
