import chess

from project.Custom_Agent.Board_utility import BoardUtility

# checkmate in one test
# board = chess.Board("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
# best_move = BoardUtility.mate_in_x(board, 1)
# board.push(best_move)
# print(board)


# checkmate in x test
board = chess.Board("8/1k6/8/8/5R2/6R1/1K6/8 b")
x = 4

for n in range(x * 2):
    if board.turn == chess.WHITE:
        print("White's turn")
        best_move = BoardUtility.mate_in_x(board, x - n)
    else:  # zwart heeft geen checkmates hier dus de eerste move selecteren
        print("Black's turn")
        best_move = list(board.legal_moves)[0]
    white_checks_count, black_checks_count = BoardUtility.check_for_checks(board, 2)
    print(white_checks_count, black_checks_count)
    print(board)
    board.push(best_move)

    if board.is_checkmate():
        break

print()
# print(board)
