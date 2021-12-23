import chess

from project.Custom_Agent.Board_utility import BoardUtility
from project.Custom_Agent.Chess_agent import ChessAgent

epochs = 10000
batches = 32

utility = BoardUtility()

agent = ChessAgent(utility, 1)
epsilon = 0.8

training_dataset = []

for i in range(epochs):
    board = chess.Board()
    epsilon *= 0.9
    while True:
        if board.is_checkmate() or board.is_stalemate():
            break
        move, utility_val, board = agent.train_move(board)
        board.push(move)
        training_dataset.append((move, utility_val, board))



