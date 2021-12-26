import chess
import numpy as np

from project.chess_utilities.utility import Utility


class BoardUtility(Utility):
    def __init__(self) -> None:
        pass

    @staticmethod
    def one_hot_board(board: chess.Board):
        board_array = np.zeros(shape=(8, 8, 13))
        one_hot_pieces = {
            # White pieces
            "R": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "N": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "B": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Q": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "K": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "P": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

            # Empty tiles
            "E": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

            # Black pieces
            "r": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "n": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "b": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "q": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "k": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "p": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
        current_tile = 0
        for i in range(8):
            for j in range(8):
                current_piece = board.piece_at(current_tile)
                if current_piece == None:
                    board_array[i][j] = one_hot_pieces["E"]
                else:
                    board_array[i][j] = one_hot_pieces[current_piece.symbol()]
                current_tile += 1
        return board_array

    # hier wordt board info door het neural network gehaald om de utility te berekenen
    def board_value(self, board: chess.Board):
        pass
