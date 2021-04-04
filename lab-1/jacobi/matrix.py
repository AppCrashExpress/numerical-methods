from math import sqrt

def create_matrix(n):
    return [[0] * n for _ in range(n)]

def copy_matrix(matrix):
    return [row[:] for row in matrix]

def check_square(matrix):
    for row in matrix:
        if (len(matrix) != len(row)):
            return False
    return True

def transpose(matrix):
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix[i])):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

def calc_norm(matrix):
    matrix_sum = 0
    for row in matrix:
        for value in row:
            matrix_sum += value * value

    return sqrt(matrix_sum)

def str_matrix(matrix):
    matrix_str = ""
    for row in matrix:
        matrix_str += "\n "
        matrix_str += str(row)

    if matrix_str != "":
        matrix_str = matrix_str [2:]

    return "[" + matrix_str + "]"
