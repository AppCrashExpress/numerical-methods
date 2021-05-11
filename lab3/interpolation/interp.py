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

def newton_interpolate(x, pivots):
    def build_diffs(xs, diffs, i, j):
        if i == j:
            return diffs[i][j]

        if diffs[i][j - 1] is None:
            build_diffs(xs, diffs, i, j - 1)
        if diffs[i + 1][j] is None:
            build_diffs(xs, diffs, i + 1, j)

        diffs[i][j] = (diffs[i + 1][j] - diffs[i][j - 1]) / (xs[j] - xs[i])

    pivots.sort()

    xs, ys = [list(t) for t in zip(*pivots)]

    div_diffs = [[None for _ in pivots] for _ in pivots]
    for i in range(len(pivots)):
        div_diffs[i][i] = ys[i]
    build_diffs(xs, div_diffs, 0, len(pivots) - 1)

    y = 0
    for i in range(len(pivots)):
        coef = 1
        for j in range(i):
            coef *= x - xs[j]
        y += div_diffs[0][i] * coef

    return y

def main():
    f = lambda x : sqrt(x)
    xs = [0, 1.7, 3.4, 5.1]

    points = list(map(lambda x : (x, f(x)), xs))

    real_x = f(3)
    lagrange_x = lagrange_interpolate(3, points)
    newton_x = newton_interpolate(3, points)

    print(real_x)
    print(lagrange_x)
    print(newton_x)

if __name__ == "__main__":
    main()
