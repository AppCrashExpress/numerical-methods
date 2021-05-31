import logging
import numpy as np
from tridiagonal import solve_tridiagonal

def make_splines(xs, ys):
    def get_diffs(xs):
        dx = np.zeros_like(xs)
        dx[1:] = np.diff(xs)
        return dx
    
    def get_a_coeffs(ys):
        a = np.zeros_like(ys)
        a[1:] = ys[:-1]
        return a
    
    def get_c_coeffs(dx, ys):
        n = len(ys)
        b = np.array([3 * ( (ys[i]-ys[i-1])/dx[i] - (ys[i-1]-ys[i-2])/dx[i-1] ) for i in range(2,n) ])
        u = dx[2:-1]
        m = 2*( dx[1:-1] + dx[2:] )
        l = dx[2:-1]
        
        c = np.zeros_like(ys)
        c[2:] = solve_tridiagonal(u.tolist(), m.tolist(), l.tolist(), b.tolist())
        return np.array(c)
    
    def get_b_coeffs(dx, ys, c):
        b = np.zeros_like(ys)
        b[1:-1] = [(ys[i]-ys[i-1])/dx[i] - dx[i]*(c[i+1] + 2*c[i])/3 for i in range(1,len(ys)-1)]
        b[-1] = (ys[-1]-ys[-2])/dx[-1] - 2/3*dx[-1]*c[-1]
        return b

    def get_d_coeffs(dx, ys, c):
        d = np.zeros_like(ys)
        l = [(c[i+1]-c[i])/(3*dx[i]) for i in range(1, len(dx) - 1)]
        d[1:-1] = l
        d[-1] = -c[-1]/(3*dx[-1])
        return d
    
    dx = get_diffs(xs)
    a  = get_a_coeffs(ys)
    c  = get_c_coeffs(dx, ys)
    b  = get_b_coeffs(dx, ys, c)
    d  = get_d_coeffs(dx, ys, c)
    
    logging.info(f"a: {a}")
    logging.info(f"b: {b}")
    logging.info(f"c: {c}")
    logging.info(f"d: {d}")
    
    return (a, b, c, d)

def interpolate(x, xs, a, b, c, d):
    def find_interval(x, xs):
        for i in range(1, len(xs)):
            if x >= xs[i-1] and x <= xs[i]:
                return i

    i = find_interval(x, xs)
    diff = x - xs[i-1]
    return a[i] + b[i] * diff + c[i] * (diff ** 2) + d[i] * (diff ** 3)

def main():
    logging.basicConfig(filename='spline.log', encoding='utf-8', level=logging.INFO)

    f = lambda x : np.sqrt(x)
    points = [(i * 1.7, f(i * 1.7)) for i in range(5)]
    x = 3

    xs, ys = map(np.array, [list(t) for t in zip(*points)])

    logging.info(f"Point x: {xs}")
    logging.info(f"Point y: {ys}")

    a, b, c, d = make_splines(xs, ys)

    real_y   = f(x)
    interp_y = interpolate(x, xs, a, b, c, d)
    logging.info(f"Interpolated y: {interp_y}")
    print(f"For point {x} interpolated value is {interp_y}, when expected {f(x)}")
    print(f"Error: {1 - min(interp_y, real_y) / max(interp_y, real_y)}")

if __name__ == "__main__":
    main()

