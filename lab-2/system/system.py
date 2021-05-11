import logging, json
import numpy as np
from math import sqrt, exp, log
from numpy.linalg import solve

MAX_ITER = 1000

f = [
    lambda x : x[0] ** 2 + x[1] ** 2 - 4,
    lambda x : x[0] - exp(x[1]) + 2
]

df = [
    {
        "x0": lambda x : 2 * x[0],
        "x1": lambda x : 2 * x[1]
    },
    {
        "x0": lambda x : 1,
        "x1": lambda x : -exp(x[1])
    }
]

phi = [
    lambda x : sqrt(4 - x[1]**2),
    lambda x : log(x[0] + 2)
]

dphi = [
    {
        "x0": lambda x : 0,
        "x1": lambda x : - x[1] / sqrt(4 - x[1] ** 2)
    },
    {
        "x0": lambda x : 1 / (x[0] + 2),
        "x1": lambda x : 0
    }
]

def norm(x, x_prev):
    return sqrt(sum([(xn - xp) ** 2 for xn, xp in zip(x, x_prev)]))

def iteration_method(a, b, eps):
    def get_phi_norm(x):
        return max(abs(dphi[0]["x0"](x)) + abs(dphi[0]["x1"](x)),
                   abs(dphi[1]["x0"](x)) + abs(dphi[1]["x1"](x)))

    x0_interv = [a[0], b[0]]
    x1_interv = [a[1], b[1]]

    x_prev = [
            (x0_interv[1] + x0_interv[0]) / 2,
            (x1_interv[1] + x1_interv[0]) / 2
    ]

    q = get_phi_norm(x_prev)

    if (q >= 1):
        logging.warning(f"q >= 1 (equals {q}), cannot estimate root")
        return None

    logging.info(f"Inital x0 = {x_prev[0]}, x1 = {x_prev[1]}, q = {q}")
    
    iter_no = 0
    while iter_no <= MAX_ITER:
        iter_no += 1

        x = [func(x_prev) for func in phi]

        logging.info(f"Iteration {iter_no}: x0 = {x[0]}, x1 = {x[1]}")

        error = q / (1 - q) * norm(x, x_prev)
        if (error <= eps):
            break

        logging.info(f"{error} > {eps} , continue...")
        x_prev = x

    logging.info(f"Method ended on iteration {iter_no} with x0 value of {x[0]}, x1 value of {x[1]}")
    return x

def newton_method(a, b, eps):

    def jacobi_matrix(x):
        return [
            [df[0]["x0"](x), df[0]["x1"](x)],
            [df[1]["x0"](x), df[1]["x1"](x)]
        ]

    x0_interv = [a[0], b[0]]
    x1_interv = [a[1], b[1]]

    x_prev = [
            (x0_interv[1] + x0_interv[0]) / 2,
            (x1_interv[1] + x1_interv[0]) / 2
    ]

    logging.info(f"Inital x0 = {x_prev[0]}, x1 = {x_prev[1]}")

    iter_no = 0
    while iter_no <= MAX_ITER:
        iter_no += 1

        jacobi = np.array(jacobi_matrix(x_prev))
        b = np.array([-f[0](x_prev), -f[1](x_prev)])
        delta_x = solve(jacobi, b).tolist()

        x = [px + dx for px, dx in zip(x_prev, delta_x)]

        logging.info(f"Iteration {iter_no}: x0 = {x[0]}, x1 = {x[1]}")

        error = norm(x, x_prev)
        if (error <= eps):
            break

        logging.info(f"{error} > {eps} , continue...")
        x_prev = x

    logging.info(f"Method ended on iteration {iter_no} with x0 value of {x[0]}, x1 value of {x[1]}")
    return x
        


def main():
    logging.basicConfig(filename='system.log', encoding='utf-8', level=logging.INFO)

    a = [0, 0]
    b = [0.7, 0.7]
    eps = 0.01
    print("Iteration method:", iteration_method(a, b, eps))
    print("Newton method:", newton_method(a, b, eps))
    


if __name__ == "__main__":
    main()
