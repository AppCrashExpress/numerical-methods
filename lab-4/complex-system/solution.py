import numpy as np
from collections import deque as Deque

def f(x, y, dy):
    return (4*y - 4*x*dy) / (2*x + 1)

def g(x, y, z):
    return z

def original_f(x):
    return x + np.exp(-2*x)

def solve_runge(f, g, a, b, h, y0, dy0):
    xs = list(np.arange(a, b+h, h))
    ys = []

    y = y0
    z = dy0
    for x in xs:
        ys.append(y)
        
        k1 = h * g(x, y, z)
        l1 = h * f(x, y, z)

        k2 = h * g(x + h/2, y + k1/2, z + l1/2)
        l2 = h * f(x + h/2, y + k1/2, z + l1/2)
        
        k3 = h * g(x + h/2, y + k2/2, z + l2/2)
        l3 = h * f(x + h/2, y + k2/2, z + l2/2)

        k4 = h * g(x + h, y + k3, z + l3)
        l4 = h * f(x + h, y + k3, z + l3)
        
        y_diff = (k1 + 2*k2 + 2*k3 + k4) / 6
        z_diff = (l1 + 2*l2 + 2*l3 + l4) / 6
        
        y += y_diff
        z += z_diff
    
    return xs, ys

def find_interval(x, xs):
    for i in range(len(xs) - 1):
        if x >= xs[i] and x <= xs[i+1]:
            return i
    return None

def df_num1(x, xs, ys, i = None):
    # For polynomial of first degree
    if i is None:
        i = find_interval(x, xs)
    if i is None:
        return None

    return (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])

def shooting_method(a, b, h, alpha, beta, delta, gamma, y0, y1, eps):
    def calc_phi(b, y1, delta, gamma, xs, ys):
        dy = df_num1(b, xs, ys)
        return delta*ys[-1] + gamma*dy - y1

    n     = Deque(maxlen=3)
    dy    = Deque(maxlen=3)
    integ = Deque(maxlen=3)
    phi   = Deque(maxlen=3)

    n.extend([1.0, 0.8])
    for i in range(2):
        dy.append((y0 - alpha * n[i]) / beta)
        integ.append(solve_runge(f, g, a, b, h, n[i], dy[i]))
        phi.append(calc_phi(b, y1, delta, gamma, integ[i][0], integ[i][1]))

    while abs(phi[-1]) > eps:
        n.append(n[-1] - (n[-1] - n[-2]) / (phi[-1] - phi[-2]) * phi[-1])
        dy.append((y0 - alpha * n[-1]) / beta)
        integ.append(solve_runge(f, g, a, b, h, n[-1], dy[-1]))
        phi.append(calc_phi(b, y1, delta, gamma, integ[-1][0], integ[-1][1]))

    return integ[-1]

def main():
    a, b, step = 0, 1, 0.1
    alpha, beta, delta, gamma = 0, 1, 2, 1
    y0, y1 = -1, 3
    
    print(shooting_method(a, b, step, alpha, beta, delta, gamma, y0, y1, 1e-5))

if __name__ == "__main__":
    main()