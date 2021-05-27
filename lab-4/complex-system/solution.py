import numpy as np
from collections import deque as Deque

from tridiagonal import solve_tridiagonal

def cauchy_f(x, y, dy):
    return (4*y - 4*x*dy) / (2*x + 1)

def g(x, y, z):
    return z

def p(x):
    return 4 * x / (2 * x + 1)

def q(x):
    return -4 / (2 * x + 1)

def f(x):
    return 0

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

def solve_shooting(f, g, a, b, h, alpha, beta, delta, gamma, y0, y1, eps):
    # The spray-and-pray method
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

def solve_finite_diff(f, p, q, a, b, h, alpha, beta, delta, gamma, y0, y1):
    xs = list(np.arange(a, b + h, h))
    u = [beta] + [1 + p(x) * h/2 for x in xs[:-2]]
    m = [alpha * h - beta] + [q(x) * h*h - 2 for x in xs[:-2]] \
        + [delta * h + gamma]
    l = [1 - p(x) * h/2 for x in xs[:-2]] + [-gamma]

    c = [y0 * h] + [f(x) * h*h for x in xs[:-2]] + [y1 * h]
    
    ys = solve_tridiagonal(u, m, l, c)
    return xs, ys

def test_rrr(shooters, finits):
    def get_error(l1, l2, order):
        return [abs(i1 - i2) / (2**order - 1) for i1, i2 in zip(l1, l2)]
    
    return (
        # 4th order? 1st order?
        get_error(shooters[0], shooters[1], 4),
        get_error(finits[0], finits[1], 1)
    )

def test_exact(shooter, finit, exact):
    def get_error(l1, l2):
        return [abs(i1 - i2) for i1, i2 in zip(l1, l2)]

    return (
        get_error(shooter, exact), 
        get_error(finit, exact)
    )
    
def main():
    def print_pairwise(xs, ys):
        for i, (x, y) in enumerate(zip(xs, ys)):
            print(f"{i:2}: {x:.3f} {y}")
    
    a, b, step = 0, 1, 0.1
    alpha, beta, delta, gamma = 0, 1, 2, 1
    y0, y1 = -1, 3
    eps = 1e-5
    
    shooters = []
    finits   = []

    for h in [step, step/2]:
        print("For step", h)

        x, y = solve_shooting(cauchy_f, g, a, b, h, alpha, beta, delta, gamma, y0, y1, eps)
        print("Shooting method:")
        print_pairwise(x, y)
        shooters.append(y)

        x, y = solve_finite_diff(f, p, q, a, b, h, alpha, beta, delta, gamma, y0, y1)
        print("Finite difference:")
        print_pairwise(x, y)
        finits.append(y)

        print()

    exact = [original_f(xi) for xi in x]
    print("Analytical solution:")
    print_pairwise(x, exact)

    p_shooter, p_finit = test_rrr(shooters, finits)
    print("\nPosterior errors:")
    print(" Shooter method      Finite difference")
    for i, (s, fin) in enumerate(zip(p_shooter, p_finit)):
        print(f"{s:12.9f}  {fin:12.9f}")
    
    e_shooter, e_finit = test_exact(shooters[1], finits[1], exact)
    print("\nExact errors:")
    print(" Shooter method      Finite difference")
    for i, (s, fin) in enumerate(zip(e_shooter, e_finit)):
        print(f"{s:12.9f}        {fin:12.9f}")


if __name__ == "__main__":
    main()