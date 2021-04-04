import math
from matrix import create_matrix, copy_matrix, transpose, str_matrix

MAX_ITER = 1000

def create_identity(n):
    iden = create_matrix(n)
    for i in range(n):
        iden[i][i] = 1
    return iden

def multiply_matrix(a, b):
    b = copy_matrix(b)
    transpose(b)

    return [[sum([x * y for x, y in zip(a_row, b_row)]) for b_row in b] for a_row in a]


def jacobi_iteration(matrix, eps=10e-5):
    def non_diagonal_norm(matrix):
        total = 0
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix[i])):
                total += matrix[i][j] ** 2
        return (math.sqrt(total))

    def max_non_diagonal(matrix):
        max_i, max_j = 0, 1
        max_val = abs(matrix[max_i][max_j])
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix[i])):
                val = abs(matrix[i][j])

                if max_val < val:
                    max_val = val
                    max_i, max_j = i, j
        return (max_i, max_j)

    def create_rotation(matrix, i, j):
        rotation = create_identity(len(matrix))
        if matrix[i][i] == matrix[j][j]:
            angle = math.pi / 4
        else:
            angle = math.atan(2 * matrix[i][j] / (matrix[i][i] - matrix[j][j])) / 2

        s = math.sin(angle)
        c = math.cos(angle)

        rotation[i][j] = -s;
        rotation[j][j] = c;
        rotation[j][i] = s;
        rotation[i][i] = c;
        
        return rotation


    matrix  = copy_matrix(matrix)
    vectors = create_identity(len(matrix))

    iter = 0
    while non_diagonal_norm(matrix) >= eps and iter <= MAX_ITER:
        max_i, max_j = max_non_diagonal(matrix)
        rotation = create_rotation(matrix, max_i, max_j)

        vectors  = multiply_matrix(vectors, rotation)

        matrix = multiply_matrix(matrix, rotation)
        transpose(rotation)
        matrix = multiply_matrix(rotation, matrix)
        
        iter += 1
        
    return matrix, vectors
    return [matrix[i][i] for i in range(len(matrix))], [[vectors[i][j] for j in range(len(vectors[i]))] for i in range(len(vectors))]


if __name__ == "__main__":
    matrix = [[-6,  6, -8],
              [ 6, -4,  9],
              [-8,  9, -2]]

    values, vectors = jacobi_iteration(matrix)
    print(str_matrix(values))
    print(str_matrix(vectors))
