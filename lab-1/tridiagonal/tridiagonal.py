def check_tridiagonal(matrix):
    matrix_size = len(matrix)
    for i in range(matrix_size - 2):
        for j in range(i + 2, matrix_size):
            if matrix[i][j] != 0 or matrix[j][i] != 0:
                return False
    return True

def get_diagonals(matrix):
    if not check_tridiagonal(matrix):
        raise ValueError("Matrix is not tridiagonal")

    matrix_size = len(matrix)
    
    upper = [0] * (matrix_size - 1)
    for i in range(matrix_size - 1):
        upper[i] = matrix[i][i + 1]

    mid = [0] * matrix_size
    for i in range(matrix_size):
        mid[i] = matrix[i][i]

    lower = [0] * (matrix_size - 1)
    for i in range(matrix_size - 1):
        lower[i] = matrix[i + 1][i]

    return (upper, mid, lower)


def calc_p_coeffs(upper_diag, mid_diag, lower_diag, constraints):
    coef_count = len(constraints)
    coeffs = [0] * coef_count

    coeffs[0] = - upper_diag[0] / mid_diag[0]

    for i in range(1, coef_count - 1):
        coeffs[i] = - upper_diag[i] / (mid_diag[i] + lower_diag[i] * coeffs[i - 1])

    return coeffs

def calc_coeffs(upper_diag, mid_diag, lower_diag, constraints):
    lower_diag = [0] + lower_diag[:]

    coef_count = len(constraints)
    p_coeffs = calc_p_coeffs(upper_diag, mid_diag, lower_diag, constraints)
    q_coeffs = [0] * coef_count

    q_coeffs[0] = constraints[0] / mid_diag[0]

    for i in range(1, coef_count):
        dividend = constraints[i] - lower_diag[i] * q_coeffs[i - 1]
        divisor  = mid_diag[i] + lower_diag[i] * p_coeffs[i - 1]
        q_coeffs[i] = dividend / divisor

    return (p_coeffs, q_coeffs)


def solve_tridiagonal(upper_diag, mid_diag, lower_diag, constraints):
    var_count = len(constraints)
    result = [0] * var_count

    p_coeffs, q_coeffs = calc_coeffs(upper_diag, mid_diag, lower_diag, constraints)

    result[var_count - 1] = q_coeffs[var_count - 1]

    for i in reversed(range(var_count - 1)):
        result[i] = p_coeffs[i] * result[i + 1] + q_coeffs[i]

    return result

if __name__ == "__main__":
    matrix = [[15, 8, 0, 0, 0], [2, -15, 4, 0, 0], [0, 4, 11, 5, 0], [0, 0, -3, 16, -7], [0, 0, 0, 3, 8]]
    constraints = [92, -84, -77, 15, -11]

    res = solve_tridiagonal(*get_diagonals(matrix), constraints)
    print(res)
