import numpy as np
from math import sqrt, copysign
from cmath import sqrt as comp_sqrt
from functools import reduce
import logging, json

def norm(array):
    return sqrt(reduce(lambda x, n: x + n*n, array, 0))

def calc_householder_vector(matrix, iter_no):
    matrix_size = len(matrix)
    vector = np.zeros((matrix_size, ))
    vector[iter_no:] = matrix[iter_no:, iter_no]

    vector[iter_no] += np.sign(vector[iter_no]) * norm(vector)

    return vector

def calc_householder_matrix(matrix, iter_no):
    matrix_size = len(matrix)
    identity    = np.identity(matrix_size)
    house_vec   = calc_householder_vector(matrix, iter_no)
    
    coef = 2 / np.dot(house_vec, house_vec)

    return identity - coef * (np.outer(house_vec, house_vec))


def qr_decomposition(matrix):
    Q = 1

    for i in range(len(matrix)):
        house = calc_householder_matrix(matrix, i)
        Q = np.dot(Q, house)
        matrix = np.dot(house, matrix)

    return Q, matrix

def test_second_eps(matrix, eps):
    # Interested in elements below second diagonal:
    #  a00   a01   a02  a03 
    #  a10   a11   a12  a13 
    # [a20]  a21   a22  a23 
    # [a30] [a31]  a32  a33 
    # ...
    
    for i, col in enumerate(abs(matrix.T)):
        if not all(col[i + 2:] < eps):
            return False

    return True

def square_eigen(matrix):
    if matrix.shape != (2, 2):
        raise ValueError("Matrix should be square")

    m = matrix

    # a is 1
    b = -m[0, 0] - m[1, 1]
    c = m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]

    d = b * b - 4 * c

    if (d >= 0):
        sqrt_d = sqrt(d)
    else:
        sqrt_d = comp_sqrt(d)

    return ( (b + sqrt_d) / 2, (b - sqrt_d) / 2 )

def get_eigenval(matrix, col_i, eps):
    def square_subarray(i):
        return matrix[i:i+2, i:i+2]

    if norm(abs(matrix[col_i+1:, col_i])) <= eps:
        return (True, matrix[col_i, col_i], 0)

    elif norm(abs(matrix[col_i+2:, col_i])) <= eps:
        prev_comp_0, prev_comp_1 = square_eigen(square_subarray(col_i))
        
        Q, R = qr_decomposition(matrix)
        matrix = R @ Q

        comp_0, comp_1 = square_eigen(square_subarray(col_i))

        close = [0, 0]
        close[0] = abs(comp_0 - prev_comp_0) <= eps
        close[1] = abs(comp_1 - prev_comp_1) <= eps

        if all(close):
            return (True, comp_0, comp_1)

    return (False, 0, 0)


def qr_algorithm(matrix, eps):
    matrix_size = len(matrix)

    while not test_second_eps(matrix, eps):
        Q, R = qr_decomposition(matrix)
        matrix = R @ Q

    logging.debug("Reduced matrix:\n%s", str(matrix))

    col_i = 0
    values = np.zeros((matrix_size, ), dtype=np.complex64)

    while col_i < matrix_size - 1:
        exists, val_1, val_2 = get_eigenval(matrix, col_i, eps)
        if exists:
            if np.iscomplex(val_1):
                logging.debug("Complex values found: %s %s", str(val_1), str(val_2))
                values[col_i] = val_1
                values[col_i + 1] = val_2
                col_i += 2
            else:
                logging.debug("Real value found: %f", val_1)
                values[col_i] = val_1
                col_i += 1
        else:
            logging.debug("No values found, improving...")
            Q, R = qr_decomposition(matrix)
            matrix = R @ Q

    if values[-1] == 0:
        values[-1] = matrix[-1, -1]

    return values


def main():
    logging.basicConfig(filename='qr.log', encoding='utf-8', level=logging.INFO)

    with open("task.json", "r") as json_file:
        task = json.load(json_file)

    matrix = np.array(task["matrix"])
    try:
        eps = task["epsilon"]
    except KeyError:
        eps = 10e-5

    logging.info("Input matrix:\n%s", str(matrix))
    logging.info("Selected epsilon: %f", eps)

    eigens = qr_algorithm(matrix, eps)

    logging.info("Eigenvalues: %s", str(eigens))

    print("Eigenvalues: ")
    print(eigens)



if __name__ == "__main__":
    main()
