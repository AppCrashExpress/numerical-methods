import numpy as np
from math import exp, sqrt

np.set_printoptions(suppress=True)

def calc_poly(x, y, n):
    def calc_matrix(x, n):
        power_sums = np.array([sum(x ** i) for i in range(2 * n + 1)])
        matrix = np.zeros(((n + 1), (n + 1)))
        for i in range(n + 1):
            matrix[i, :] = power_sums[i:i+(n+1)]
            matrix[:, i] = power_sums[i:i+(n+1)]
        return matrix

    def calc_coeffs(x, y, n):
        return np.array([sum(y * x ** i) for i in range(n + 1)])

    matrix = calc_matrix(x, n)
    coeffs = calc_coeffs(x, y, n)
    return np.linalg.solve(matrix, coeffs)

def interpolate_point(x, poly):
    y = 0
    for a in np.flip(poly):
        y = y * x + a
    return y

def main():
    f = lambda x : exp(x)
    points = [(i, f(i)) for i in range(5)]

    xs, ys = map(np.array, [list(t) for t in zip(*points)])

    poly = calc_poly(xs, ys, 1)
    print(interpolate_point(3.4, poly))

if __name__ == "__main__":
    main()
