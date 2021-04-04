from matrix import create_matrix, str_matrix
from math import sqrt
import logging, json

MAX_ITER_COUNT = 1000

def calc_equivalent(matrix, contants):
    new_matrix = create_matrix(len(matrix))
    new_constants = [0] * len(constants)
    
    for i in range(len(matrix)):
        diag_val = matrix[i][i]
        if diag_val == 0:
            raise ValueError("Diagonal values can not contain 0")

        for j in range(len(matrix[0])):
            if i == j:
                new_matrix[i][j] = 0
            else:
                new_matrix[i][j] = - matrix[i][j] / diag_val

        new_constants[i] = constants[i] / diag_val

    return (new_matrix, new_constants)
    

def calc_vec_norm(x):
    vec_sum = 0
    for val in x:
        vec_sum += val * val

    return sqrt(vec_sum)

def calc_matr_norm(x):
    matr_sum = 0
    for row in x:
        for val in row:
            matr_sum += val * val

    return sqrt(matr_sum)

def test_exit(prev_res, new_res, coef_divident, coef_divisor, eps):
    difference = [x - x0 for x, x0 in zip (new_res, prev_res)]
    diff_norm = calc_vec_norm(difference)

    if coef_divisor < 1:
        return diff_norm * coef_divident / (1 - coef_divisor) <= eps
    else:
        return diff_norm <= eps

def solve_iter(matrix, constants, eps = 10e-5):
    matrix_norm = calc_matr_norm(matrix)
    x_values = constants[:]
    new_x_values = constants[:]
    
    for iter_no in range(MAX_ITER_COUNT):
        for i in range(len(matrix)):
            row_sum = sum([a * x for a, x in zip(matrix[i], x_values)])
            new_x_values[i] = constants[i] + row_sum

        if test_exit(x_values, new_x_values, matrix_norm, matrix_norm, eps):
            break
        else:
            x_values = new_x_values[:]

    return new_x_values

def solve_seidel(matrix, constants, eps = 10e-5):
    def get_upper(matrix):
        upper = create_matrix(len(matrix))
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if j > i:
                    continue
                else:
                    upper[i][j] = matrix[i][j]

        return upper

    upper        = get_upper(matrix)
    upper_norm   = calc_matr_norm(upper)
    matrix_norm  = calc_matr_norm(matrix)
    x_values     = constants[:]
    new_x_values = constants[:]
    
    for iter_no in range(MAX_ITER_COUNT):
        for i in range(len(matrix)):
            row_sum = sum([a * x for a, x in zip(matrix[i], new_x_values)])
            new_x_values[i] = constants[i] + row_sum

        if test_exit(x_values, new_x_values, upper_norm, matrix_norm, eps):
            break
        else:
            x_values = new_x_values[:]

    return new_x_values

if __name__ == "__main__":
    logging.basicConfig(filename='iter-seidel.log', encoding='utf-8', level=logging.INFO)

    with open("task.json", "r") as json_file:
        task = json.load(json_file)

    matrix = task["matrix"]
    constants = task["constants"]
    try:
        eps = task["epsilon"]
    except KeyError:
        eps = 10e-5

    logging.info("Input matrix:\n%s", str_matrix(matrix))
    logging.info("Matrix constants: %s", str(constants))
    logging.info("Selected epsilon: %f", eps)

    new_a, new_b = calc_equivalent(matrix, constants)

    logging.info("Equivalent matrix:\n%s", str_matrix(new_a))
    logging.info("Equivalent constants: %s", str(new_b))

    res_i = solve_iter(new_a, new_b, eps)
    res_s = solve_seidel(new_a, new_b, eps)

    logging.info("Iterative solution: %s", str(res_i))
    logging.info("Seidel solution: %s", str(res_s))

    print("Iterative solution: ", res_i)
    print("Seidel solution: ", res_s)
