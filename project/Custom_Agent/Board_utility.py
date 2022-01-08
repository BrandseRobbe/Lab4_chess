import chess
import numpy as np
from keras.models import load_model

from project.Custom_Agent.utility import Utility


class BoardUtility(Utility):
    def __init__(self) -> None:
        self.model = None

    def load_chess_model(self, chess_model):
        self.model = load_model(chess_model)

    # hier wordt board info door het neural network gehaald om de utility te berekenen
    def board_value(self, board: chess.Board):
        boardvalue = BoardUtility.get_board_data(board)
        x1 = boardvalue[0][np.newaxis, ...]
        x2 = boardvalue[1][np.newaxis, ...]
        # Predict the qvalue for this state
        qvalue = self.model.predict({"board_data": x1, "feature_data": x2})[0][0]
        return qvalue

    @staticmethod
    def one_hot_board(board: chess.Board):
        board_array = np.zeros(shape=(8, 8, 13), dtype="float32")
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
                if current_piece is None:
                    board_array[i][j] = one_hot_pieces["E"]
                else:
                    board_array[i][j] = one_hot_pieces[current_piece.symbol()]
                current_tile += 1

        return board_array

    @staticmethod
    def get_board_data(board: chess.Board):
        one_hot_data = BoardUtility.one_hot_board(board)
        extracted_features = BoardUtility.check_for_checks(board, 2)
        extracted_features = np.asarray(extracted_features).astype('float32')
        return one_hot_data, extracted_features

    # search for (unavoidable) checkmate in x turns
    @staticmethod
    def mate_in_x(board: chess.Board, move_count, depth=0):
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            # schaakmat is nog niet gevonden
            # checkmate has not yet been found
            # let's go over all possible moves of the opponent

            # als voor elke zet een schaakmat mogelijk is, dan werkt het algoritme wel
            elif move_count > depth + 1:  # mogen nog dieper gaan
                mate_possible = True
                for opponents_move in list(board.legal_moves):
                    board.push(opponents_move)
                    # dieper naar chekmate zoeken
                    new_move = BoardUtility.mate_in_x(
                        board, move_count, depth + 1)
                    board.pop()
                    # no guarantee of checkmate at this depth if no checkmate has been found for this move
                    if new_move is None:
                        # board.pop()
                        mate_possible = False
                        break
                # every move can be considered a checkmate move when the whole list has been completed so we can take the current move.
                if mate_possible:
                    board.pop()
                    return move
            board.pop()
        return None

    @staticmethod
    def check_for_checks(board: chess.Board, move_limit):
        """
            A simple breadth first search to check for checks after a certain amount of moves
            :param board: The board to discover possible checks on.
            :param move_limit: The maximum number of moves (turns) to scout ahead
        """

        board_queue = Queue()
        nr_white_in_check = nr_black_in_check = 0  # The number of possible checks we will find

        board_queue.push((board, 0))
        # keep history of positions, we don't want to count the same position twice
        history = []

        while not board_queue.isEmpty():
            board, depth = board_queue.pop()
            # if we already checked this position we skip it
            if board.fen() in history:
                continue
            history.append(board.fen())
            # is current player is in check
            if board.is_check():
                if board.turn == chess.WHITE:  # it's white's turn
                    nr_white_in_check += 1
                else:
                    nr_black_in_check += 1  # it's black's turn

            # search deeper if max depth is not reached yet
            if not depth + 1 > move_limit:
                for move in board.legal_moves:
                    board.push(move)
                    board_queue.push((board.copy(), depth + 1))
                    board.pop()  # revert the move

        return nr_white_in_check, nr_black_in_check


class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy. Stolen from lab 1"

    def __init__(self):
        self.list = []

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0, item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0
