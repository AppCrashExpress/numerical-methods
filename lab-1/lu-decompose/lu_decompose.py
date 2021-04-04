from matrix  import create_matrix, copy_matrix, check_square, str_matrix
from math    import isclose
import logging, json

def decompose_matrix(matrix):
    """Decomposes the matrix into lower and upper matrices"""
    if not check_square(matrix):
        raise IndexError("Recieved matrix is not square")
    
    matrix_size = len(matrix)
    lower = create_matrix(matrix_size)
    upper = copy_matrix(matrix)

    permut = [i for i in range(matrix_size + 1)]
    permut[matrix_size] = 0

    for iter_i in range(matrix_size):
        max_i = iter_i
        max_val = abs(matrix[iter_i][iter_i])

        for i in range(iter_i + 1, matrix_size):
            abs_val = abs(matrix[i][iter_i])
            if abs_val > max_val:
                max_val = abs_val
                max_i = i

        if isclose(max_val, 0):
            raise RuntimeError("Recieved matrix is degenerate")

        if iter_i != max_i:
            permut[max_i], permut[iter_i] = permut[iter_i], permut[max_i]
            upper[max_i], upper[iter_i] = upper[iter_i], upper[max_i]
            lower[max_i], lower[iter_i] = lower[iter_i], lower[max_i]
            permut[matrix_size] += 1

        lower[iter_i][iter_i] = 1
        for coef_i in range(iter_i + 1, matrix_size):
            lower[coef_i][iter_i] = upper[coef_i][iter_i] / upper[iter_i][iter_i]

        for row_i in range(iter_i + 1, matrix_size):
            upper[row_i] = [x - y * lower[row_i][iter_i] for (x, y) in zip(upper[row_i], upper[iter_i])]

    return (lower, upper, permut)

def solve_lu(lower, upper, permutation, constants):
    """Solves the LU decompostion with given constants 
    (equation constraints)"""
    matrix_size = len(lower)

    constants = [constants[p] for p in permutation[:-1]]

    temp_constants = [0] * matrix_size
    for i in range(matrix_size):
        for j in range(i):
            temp_constants[i] -= temp_constants[j] * lower[i][j]
        temp_constants[i] = (temp_constants[i] + constants[i]) / lower[i][i]

    result = [0] * matrix_size
    for i in reversed(range(matrix_size)):
        for j in range(i + 1, matrix_size):
            result[i] -= result[j] * upper[i][j]
        result[i] = (result[i] + temp_constants[i]) / upper[i][i]

    return result

def calc_lu_determinant(lower, upper, p):
    """Calculates determinant from LU decomposition. 
    Permutation array is necessary to calculate sign"""
    result = 1
    
    matrix_size = len(lower)

    for i in range(matrix_size):
        result *= lower[i][i] * upper[i][i]

    result *= 1 if (p[matrix_size] % 2 == 0) else -1

    return result

def calc_lu_inverse(lower, upper, p):
    """Calculates columns of matrice inverse from its decomposition.
    Transposition ensures the same direction of lists as original"""
    def transpose(matrix):
        matrix_size = len(matrix)
        for i in range(matrix_size):
            for j in range(i + 1, matrix_size):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    matrix_size = len(lower)
    result = [0] * matrix_size
    for i in range(matrix_size):
        inverse_column = [0] * i + [1] + [0] * (matrix_size - 1 - i)
        result[i] = solve_lu(lower, upper, p, inverse_column)

    transpose(result)
    return result

if __name__ == "__main__":
    logging.basicConfig(filename='lu-decomp.log', encoding='utf-8', level=logging.INFO)

    with open("task.json", "r") as json_file:
        task = json.load(json_file)

    matrix = task["matrix"]
    constants = task["constants"]

    logging.info("Input matrix:\n%s", str_matrix(matrix))
    logging.info("Matrix constants: %s", str(constants))

    decomposition = decompose_matrix(matrix)

    logging.info("Lower matrix:\n%s", str_matrix(decomposition[0]))
    logging.info("Upper matrix:\n%s", str_matrix(decomposition[1]))
    logging.info("Permutation array: %s", decomposition[2])

    res = solve_lu(*decomposition, constants)

    logging.info("Solution: %s", str(res))

    det = calc_lu_determinant(*decomposition)
    inv = calc_lu_inverse(*decomposition)

    logging.info("Determinant: %i", det)
    logging.info("Matrix inverse:\n%s", str_matrix(inv))

    print("Solution: ", res)
    

