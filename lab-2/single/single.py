import logging, json
from math import sqrt, log

MAX_ITER = 1000

def f(x):
    return 2 ** x + x ** 2 - 2

def df(x):
    return log(2) * (2 ** x) + 2 * x

def ddf(x):
    return log(2) * log(2) * (2 ** x) + 2

def phi(x):
    return sqrt(2 - 2 ** x)

def dphi(x):
    return -(1/2) / sqrt(2 - 2 ** x) * (log(2) * (2 ** x))

def iteration_method(a, b, eps):
    def get_max_change(a, b):
        # Depends on a function being monotone
        return max( abs(dphi(a)), abs(dphi(b)) )

    x = x_prev = (a + b) / 2
    q = get_max_change(a, b)

    if (q >= 1):
        logging.warning(f"q >= 1 (equals {q}), cannot estimate root")
        return None

    logging.info(f"Inital x = {x}, q = {q}")

    iter_no = 0
    while iter_no <= MAX_ITER:
        iter_no += 1

        x = phi(x_prev)

        logging.info(f"Iteration {iter_no}: x = {x}")

        error = q / (1 - q) * abs(x - x_prev)
        if (error <= eps):
            break

        logging.info(f"{error} > {eps} , continue...")
        x_prev = x

    return x


def main():
    logging.basicConfig(filename='single.log', encoding='utf-8', level=logging.INFO)

    a, b, eps = 0, 0.7, 0.01
    print(iteration_method(a, b, eps))
    


if __name__ == "__main__":
    main()
