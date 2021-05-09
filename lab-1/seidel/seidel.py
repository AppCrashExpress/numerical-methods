import numpy as np
import logging, json
from math import sqrt
from numpy.linalg import norm

MAX_ITER_COUNT = 1000

def calc_equivalent(matrix, contants):
    new_matrix = np.zeros_like(matrix)
    new_constants = np.zeros_like(constants)

    diag = matrix.diagonal()

    if not(all(diag != 0)):
            raise ValueError("Diagonal values can not contain 0")

    new_matrix = - matrix / diag[:, np.newaxis]
    for i in range(len(new_matrix)):
        new_matrix[i, i] = 0
    new_constants = constants / diag
    
    return (new_matrix, new_constants)


def test_exit(prev_res, new_res, coef_divident, coef_divisor, eps):
    difference = new_res - prev_res
    diff_norm = norm(difference)

    if coef_divisor < 1:
        return diff_norm * coef_divident / (1 - coef_divisor) <= eps
    else:
        return diff_norm <= eps

def solve_iter(matrix, constants, eps = 10e-5):
    matrix_norm  = norm(matrix)
    x_values     = constants.copy()
    new_x_values = constants.copy()
    
    for iter_no in range(MAX_ITER_COUNT):
        row_sums = np.sum(matrix * x_values, axis=1)
        new_x_values = constants + row_sums

        if test_exit(x_values, new_x_values, matrix_norm, matrix_norm, eps):
            break
        else:
            x_values = new_x_values.copy()

    return new_x_values

def solve_seidel(matrix, constants, eps = 10e-5):
    upper        = np.triu(matrix)
    upper_norm   = norm(upper)
    matrix_norm  = norm(matrix)
    x_values     = constants.copy()
    new_x_values = constants.copy()
    
    for iter_no in range(MAX_ITER_COUNT):
        for i in range(len(matrix)):
            row_sum = sum(matrix[i] * new_x_values)
            new_x_values[i] = constants[i] + row_sum

        if test_exit(x_values, new_x_values, upper_norm, matrix_norm, eps):
            break
        else:
            x_values = new_x_values.copy()

    return new_x_values

if __name__ == "__main__":
    logging.basicConfig(filename='iter-seidel.log', encoding='utf-8', level=logging.INFO)

    with open("task.json", "r") as json_file:
        task = json.load(json_file)

    matrix = np.array(task["matrix"])
    constants = np.array(task["constants"])
    try:
        eps = task["epsilon"]
    except KeyError:
        eps = 10e-5

    logging.info("Input matrix:\n%s", str(matrix))
    logging.info("Matrix constants: %s", str(constants))
    logging.info("Selected epsilon: %f", eps)

    new_a, new_b = calc_equivalent(matrix, constants)

    logging.info("Equivalent matrix:\n%s", str(new_a))
    logging.info("Equivalent constants: %s", str(new_b))

    res_i = solve_iter(new_a, new_b, eps)
    res_s = solve_seidel(new_a, new_b, eps)

    logging.info("Iterative solution: %s", str(res_i))
    logging.info("Seidel solution: %s", str(res_s))

    print("Iterative solution: ", res_i)
    print("Seidel solution: ", res_s)
