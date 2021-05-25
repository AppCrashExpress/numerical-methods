import numpy as np

def f(x):
    # return 1 / ( (3 * x + 4) * x + 2 )
    return x / (3*x + 4) ** 2

def solve_rectangle(f, x, h):
    return h * ( sum([f( (x[i] + x[i+1]) / 2 ) for i in range(len(x) - 1)]) )

def solve_trapezoid(y, h):
    return h * ( y[0]/2 + sum(y[1:-1]) + y[-1]/2 )

def solve_simpson(y, h):
    four_subseq = [4 * y_i for y_i in y[1:-1:2]]
    two_subseq  = [2 * y_i for y_i in y[2:-2:2]]

    return h/3 * ( y[0] + sum(four_subseq) + sum(two_subseq) + y[-1] )

def test_rrr(rect, trape, simps):
    return {
            "rectangle": (rect[0] - rect[1]) / (2 ** 2 - 1),
            "trapezoid": (trape[0] - trape[1]) / (2 ** 2 - 1),
            "simpson":   (simps[0] - simps[1]) / (2 ** 4 - 1)
    }

def main():
    h = [0.5, 0.25]
    x0, xn = -1, 1

    rect  = []
    trape = []
    simps = []

    for h_i in h:
        x = list(np.arange(x0, xn + h_i, h_i))
        y = [f(xi) for xi in x]
        rect.append(solve_rectangle(f, x, h_i))
        trape.append(solve_trapezoid(y, h_i))
        simps.append(solve_simpson(y, h_i))

    error = test_rrr(rect, trape, simps)

    print(rect)
    print(trape)
    print(simps)
    print(error)

if __name__ == "__main__":
    main()
