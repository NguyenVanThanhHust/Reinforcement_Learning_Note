import os
import numpy as np
import itertools

# Define connect 4 https://en.wikipedia.org/wiki/Connect_Four
# Rule is above
class Board():
    def __init__(self, board_shape=[6, 7], current_board=None):
        if current_board is None:
            self.board = np.zeros(board_shape).astype(str)
            self.current_board = self.board
            self.board[self.board == "0.0"] = " "
        else:
            self.board = current_board
            self.board[self.board == "0.0"] = " "
            self.board[self.board == "1.0"] = "X"
            self.board[self.board == "2.0"] = "0"
            self.current_board = self.board

        self.player = 0
        # self.board_width = board_shape[0]
        # self.board_height = board_shape[1]

        self.board_height = board_shape[0]
        self.board_width = board_shape[1]


    def __repr__(self, ):
        print(self.current_board)

    def __str__(self, ):
        print(self.board)
        return "Board height: {}, width: {}".format(self.board_height, self.board_width)

    def drop_piece(self, column):
        if self.current_board[0, column] != " ":
            return "Invalid move"
        else:
            row = 0; pos = " "
            while (pos == " "):
                if row == 6:
                    row += 1
                    break
                pos = self.current_board[row, column]
                row += 1
            if self.player == 0:
                self.current_board[row-2, column] = "O"
                self.player = 1
            elif self.player == 1:
                self.current_board[row-2, column] = "X"
                self.player = 0

            
    def check_winner(self, ):
        if self.player == 1:
            for row in range(6):
                for col in range(7):
                    if self.current_board[row, col] != " ":
                        # rows
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col] == "O" and \
                                self.current_board[row + 2, col] == "O" and self.current_board[row + 3, col] == "O":
                                #print("row")
                                return True
                        except IndexError:
                            next
                        # columns
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row, col + 1] == "O" and \
                                self.current_board[row, col + 2] == "O" and self.current_board[row, col + 3] == "O":
                                #print("col")
                                return True
                        except IndexError:
                            next
                        # \ diagonal
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col + 1] == "O" and \
                                self.current_board[row + 2, col + 2] == "O" and self.current_board[row + 3, col + 3] == "O":
                                #print("\\")
                                return True
                        except IndexError:
                            next
                        # / diagonal
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col - 1] == "O" and \
                                self.current_board[row + 2, col - 2] == "O" and self.current_board[row + 3, col - 3] == "O"\
                                and (col-3) >= 0:
                                #print("/")
                                return True
                        except IndexError:
                            next
        if self.player == 0:
            for row in range(6):
                for col in range(7):
                    if self.current_board[row, col] != " ":
                        # rows
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col] == "X" and \
                                self.current_board[row + 2, col] == "X" and self.current_board[row + 3, col] == "X":
                                return True
                        except IndexError:
                            next
                        # columns
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row, col + 1] == "X" and \
                                self.current_board[row, col + 2] == "X" and self.current_board[row, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        # \ diagonal
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col + 1] == "X" and \
                                self.current_board[row + 2, col + 2] == "X" and self.current_board[row + 3, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        # / diagonal
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col - 1] == "X" and \
                                self.current_board[row + 2, col - 2] == "X" and self.current_board[row + 3, col - 3] == "X"\
                                and (col-3) >= 0:
                                return True
                        except IndexError:
                            next

        # # Check column winner
        # for i in range(self.board_width):
        #     current_column = ""
        #     current_column = current_column.join(self.current_board[:, i])
        #     max_X = max(len(list(y)) for (c,y) in itertools.groupby(current_column) if c=='X')
        #     max_O = max(len(list(y)) for (c,y) in itertools.groupby(current_column) if c=='O')
        #     if max_X >= 4:
        #         return 1
        #     if max_O >= 4:
        #         return 2

        # # Check row winner
        # for i in range(self.board_height):
        #     current_row = ""
        #     current_row = current_row.join(self.current_board[i, :])
        #     max_X = max(len(list(y)) for (c,y) in itertools.groupby(current_row) if c=='X')
        #     max_O = max(len(list(y)) for (c,y) in itertools.groupby(current_row) if c=='O')
        #     if max_X >= 4:
        #         return 1
        #     if max_O >= 4:
        #         return 2

        # # Check first diagonal winner
        # ## for start from most left column
        # for i in range(self.board_height):
        #     current_diag = ""
        #     diag_line = [self.current_board[k, k+i] for k in range(0, min(self.board_height, self.board_width))]
        #     current_diag = current_diag.join(diag_line)
        #     max_X = max(len(list(y)) for (c,y) in itertools.groupby(current_diag) if c=='X')
        #     max_O = max(len(list(y)) for (c,y) in itertools.groupby(current_diag) if c=='O')
        #     if max_X >= 4:
        #         return 1
        #     if max_O >= 4:
        #         return 2

        # ## for start from top row
        # for i in range(self.board_width):
        #     current_diag = ""
        #     diag_line = [self.current_board[k+i, k] for k in range(0, min(self.board_height, self.board_width))]
        #     current_diag = current_diag.join(diag_line)
        #     max_X = max(len(list(y)) for (c,y) in itertools.groupby(current_diag) if c=='X')
        #     max_O = max(len(list(y)) for (c,y) in itertools.groupby(current_diag) if c=='O')
        #     if max_X >= 4:
        #         return 1
        #     if max_O >= 4:
        #         return 2


        # # Check second diagonal winner
        # ## for start from most right column
        # ## Flip the board and do similar as from first diagonal
        # fliplr_board = np.fliplr(self.current_board)
        # for i in range(self.board_width):
        #     current_diag = ""
        #     diag_line = [fliplr_board[k+i, k] for k in range(0, min(self.board_height, self.board_width))]
        #     current_diag = current_diag.join(diag_line)
        #     max_X = max(len(list(y)) for (c,y) in itertools.groupby(current_diag) if c=='X')
        #     max_O = max(len(list(y)) for (c,y) in itertools.groupby(current_diag) if c=='O')
        #     if max_X >= 4:
        #         return 1
        #     if max_O >= 4:
        #         return 2

        # return 0

    def actions(self, ):
        acts = []
        for col in range(self.board_width):
            if self.current_board[0, col] == " ":
                acts.append(col)
        return acts


