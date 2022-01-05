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

    # (onvermeidelijke) schaakmat vinden binnen x zetten
    @staticmethod
    def mate_in_x(board: chess.Board, move_count, depth = 0):
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            # schaakmat is nog niet gevonden
            # alle mogelijke zetten van de tegenstander overlopen
            # als voor elke zet een schaakmat mogelijk is, dan werkt het algoritme wel
            elif move_count > depth + 1: # mogen nog dieper gaan
                mate_possible = True
                for opponents_move in list(board.legal_moves):
                    board.push(opponents_move)
                    # dieper naar chekmate zoeken
                    new_move = BoardUtility.mate_in_x(board, move_count, depth+1)
                    board.pop()
                    # als geen chekmate gevonden is voor deze zet, dan hebben we geen garantie op schaakmat voor deze diepte
                    if new_move is None:
                        # board.pop()
                        mate_possible = False
                        break
                # als de volledige movelist volledig overlopen wordt betekent dit dat alle zetten kunnen resulteren in schaakmat, dus deze zet is ok
                if mate_possible:
                    board.pop()
                    return move
            board.pop()
        return None


    # hier wordt board info door het neural network gehaald om de utility te berekenen
    def board_value(self, board: chess.Board):
        pass
