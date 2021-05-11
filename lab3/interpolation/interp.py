import logging
from math import sqrt

def lagrange_interpolate(x, pivots):
    pivots.sort()

    for xn, xp in zip(pivots[1:], pivots[:-1]):
        if xn == xp:
            return None

    y = 0
    for i in range(len(pivots)):
        xi, poly = pivots[i]

        for j in range(len(pivots)):
            if i == j:
                continue
            xj, _ = pivots[j]
            poly *= (x - xj) / (xi - xj)
        
        y += poly

    return y

def main():
    f = lambda x : sqrt(x)
    xs = [0, 1.7, 3.4, 5.1]

    points = list(map(lambda x : (x, f(x)), xs))

    real_x = f(3)
    interp_x = lagrange_interpolate(3, points)

    print(real_x)
    print(interp_x)

if __name__ == "__main__":
    main()
