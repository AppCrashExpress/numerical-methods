import logging, json
from math import sqrt, exp, log

MAX_ITER = 1000

f = [
    lambda x : x[0] ** 2 + x[1] ** 2 - 4,
    lambda x : x[0] - exp(x[1]) + 2
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

def iteration_method(a, b, eps):
    def norm(x, x_prev):
        return sqrt(sum([(xn - xp) ** 2 for xn, xp in zip(x, x_prev)]))
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
        x = [func(x_prev) for func in phi]

        logging.info(f"Iteration {iter_no}: x0 = {x[0]}, x1 = {x[1]}")

        error = q / (1 - q) * norm(x, x_prev)
        if (error <= eps):
            break

        logging.info(f"{error} > {eps} , continue...")
        x_prev = x

    return x

def main():
    logging.basicConfig(filename='system.log', encoding='utf-8', level=logging.INFO)

    a = [0, 0]
    b = [0.7, 0.7]
    eps = 0.01
    print("Iteration method:", iteration_method(a, b, eps))
    


if __name__ == "__main__":
    main()
