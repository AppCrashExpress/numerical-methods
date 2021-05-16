import logging
import numpy as np
from math import sqrt

def make_splines(points_x, points_y):
    def get_diffs(x):
        return np.diff(x)

    def get_matrix(dx):
        matrix = np.zeros((len(dx)-1, len(dx)-1))
        
        for i, row in enumerate(matrix[1:-1, :], 1):
            values = [dx[i-1], 2*(dx[i-1]+dx[i]), dx[i]]
            row[i-1:i+2] = values

        rl = len(dx)-2
        matrix[0,:2] = [2*(dx[0]+dx[1]), dx[1]]
        matrix[-1, -2:] = [dx[-2], 2*(dx[-2]+dx[-1])]

        return matrix

    def get_constants(dx, y):
        c_calc = lambda i : 3 * ( (y[i+2]-y[i+1])/dx[i+1] - (y[i+1]-y[i])/dx[i] )
        return np.array([c_calc(i) for i in range(len(y)-2)])

    def get_c_coeffs(dx, y):
        matr = get_matrix(dx)
        consts = get_constants(dx, y)

        logging.debug(f"Matrix:\n{matr}")
        logging.debug(f"Constants:\n{consts}")

        return np.linalg.solve(matr, consts)

    def get_coeffs(x, y):
        dx = get_diffs(x)
        coeffs = np.zeros((len(x)-1, 4))

            # A
        coeffs[:, 0] = y[:-1]
            # C
        coeffs[1:, 2] = get_c_coeffs(dx, y)
            # B
        b_calc = lambda i : (y[i+1]-y[i])/dx[i] - dx[i]*(coeffs[i+1,2]+2*coeffs[i,2])/3
        coeffs[:-1, 1] = np.array([b_calc(i) for i in range(len(dx) - 1)])
        coeffs[-1, 1]  = (y[-1]-y[-2])/dx[-1] - dx[-1]*2*coeffs[-1,2]/3
            # D
        d_calc = lambda i : (coeffs[i+1,2]-coeffs[i,2])/(3*dx[i])
        coeffs[:-1, 3] = np.array([d_calc(i) for i in range(len(dx) - 1)])
        coeffs[-1, 3]  = -coeffs[-1,2]/(3*dx[-1])

        return coeffs
    
    return get_coeffs(points_x, points_y)

def interpolate(x, xs, coefs):
    def find_interval(x, xs):
        for i, x_i in enumerate(xs):
            if x_i <= x:
                return i
        return None

    i = find_interval(x, xs)
    diff = x - xs[i]
    y = 0
    for coef in np.flip(coefs[i, :]):
        y = y * diff + coef

    return y


def main():
    logging.basicConfig(filename='spline.log', encoding='utf-8', level=logging.INFO)

    f = lambda x : sqrt(x)
    points = [(i * 1.7, f(i * 1.7)) for i in range(5)]
    x = 3

    xs, ys = map(np.array, [list(t) for t in zip(*points)])

    logging.info(f"Point x: {xs}")
    logging.info(f"Point y: {ys}")

    coefs = make_splines(xs, ys)
    logging.info(f"Coefficient matrix:\n{coefs}")

    real_y   = f(x)
    interp_y = interpolate(1.5, xs, coefs)
    logging.info(f"Interpolated y: {interp_y}")
    print(f"For point {x} interpolated value is {interp_y}, when expected {f(x)}")
    print(f"Error: {1 - min(interp_y, real_y) / max(interp_y, real_y)}")

if __name__ == "__main__":
    main()
