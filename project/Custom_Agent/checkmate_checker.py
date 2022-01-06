import chess

"""
    A simple breadth first search to check for checkmates after a certain amount of moves
"""


def check_for_checkmate(board: chess.Board, moves_limit, moves_completed=0):
    for move in board.legal_moves:
        if board.is_into_check(move):
            return True

        board.push(move)
        moves_completed += 1
        check_possible = False
        if moves_completed < moves_limit:
            check_possible = check_for_checkmate(board, moves_limit, moves_completed)
        board.pop()  # Revert to the previous situation
        return check_possible

    return False
