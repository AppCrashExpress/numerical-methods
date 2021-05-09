import numpy as np
import logging, json
from math import isclose

def decompose_matrix(matrix):
    """Decomposes the matrix into lower and upper matrices"""
    if matrix.shape[0] != matrix.shape[1]:
        raise IndexError("Recieved matrix is not square")
    
    matrix_size = len(matrix)
    lower = np.zeros_like(matrix, dtype=np.float32)
    upper = np.array(matrix, dtype=np.float32)

    permut = np.fromfunction(lambda i : i, (matrix_size + 1, ), dtype=np.int16)
    permut[matrix_size] = 0

    for iter_i in range(matrix_size):
        max_i = iter_i
        max_val = abs(matrix[iter_i, iter_i])

        for i in range(iter_i + 1, matrix_size):
            abs_val = abs(matrix[i, iter_i])
            if abs_val > max_val:
                max_val = abs_val
                max_i = i

        if isclose(max_val, 0, abs_tol=10e-5):
            raise RuntimeError("Recieved matrix is degenerate")

        if iter_i != max_i:
            permut[[max_i, iter_i]] = permut[[iter_i, max_i]]
            upper[[max_i, iter_i]] = upper[[iter_i, max_i]]
            lower[[max_i, iter_i]] = lower[[iter_i, max_i]]
            permut[matrix_size] += 1

        lower[iter_i, iter_i] = 1
        lower[iter_i+1:, iter_i] = (upper[iter_i+1:, iter_i] / upper[iter_i, iter_i])
        
        upper[iter_i+1:] = upper[iter_i+1:] - np.outer(lower[iter_i+1:, iter_i], upper[iter_i])

    return (lower, upper, permut)

def solve_lu(lower, upper, permutation, constants):
    """Solves the LU decompostion with given constants 
    (equation constraints)"""
    matrix_size = len(lower)

    constants = constants[permutation[:-1]]

    temp_constants = np.zeros_like(constants, dtype=np.float32)
    for i in range(matrix_size):
        for j in range(i):
            temp_constants[i] -= temp_constants[j] * lower[i][j]
        temp_constants[i] = (temp_constants[i] + constants[i]) / lower[i][i]

    result = np.zeros_like(constants, dtype=np.float32)
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
    matrix_size = len(lower)
    result = np.zeros_like(lower, dtype=np.float32)
    for i in range(matrix_size):
        inverse_column = np.zeros((matrix_size, ))
        inverse_column[i] = 1
        result[i] = solve_lu(lower, upper, p, inverse_column)

    return result.T


if __name__ == "__main__":
    logging.basicConfig(filename='lu-decomp.log', encoding='utf-8', level=logging.INFO)

    with open("task.json", "r") as json_file:
        task = json.load(json_file)

    matrix = np.array(task["matrix"])
    constants = np.array(task["constants"])

    logging.info("Input matrix:\n%s", str(matrix))
    logging.info("Matrix constants: %s", str(constants))

    decomposition = decompose_matrix(matrix)

    logging.info("Lower matrix:\n%s", str(decomposition[0]))
    logging.info("Upper matrix:\n%s", str(decomposition[1]))
    logging.info("Permutation array: %s", str(decomposition[2]))

    res = solve_lu(*decomposition, constants)

    logging.info("Solution: %s", str(res))

    det = calc_lu_determinant(*decomposition)
    inv = calc_lu_inverse(*decomposition)

    logging.info("Determinant: %i", det)
    logging.info("Matrix inverse:\n%s", str(inv))

    print("Solution: ", res)
    

