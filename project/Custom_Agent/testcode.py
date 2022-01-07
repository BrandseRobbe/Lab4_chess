import chess

from project.Custom_Agent.Board_utility import BoardUtility

# checkmate in one test
# board = chess.Board("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
# best_move = BoardUtility.mate_in_x(board, 1)
# board.push(best_move)
# print(board)


# checkmate in x test
board = chess.Board("8/8/1k6/5R2/6R1/8/8/8 w - - 2 2")
x = 3
print(board)
for n in range(x):
    best_move = BoardUtility.mate_in_x(board, x - n)
    print(best_move)
    board.push(best_move)
    # print(board)
    if board.is_checkmate():
        break

    # move for black
    board.push(list(board.legal_moves)[0])
print()
print(board)
