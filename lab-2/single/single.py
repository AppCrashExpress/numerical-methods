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

    logging.info("Iteration method")

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

    logging.info(f"Method ended on iteration {iter_no} with x value of {x}")
    return x

def newton_method(a, b, eps):
    # All functions also depend on monotone
    def get_max_change(a, b):
        return max( abs(df(a)), abs(df(b)) )

    def get_max_rate(a, b):
        return max( abs(ddf(a)), abs(ddf(b)) )

    logging.info("\nNewton method")

    x = x_prev = (a + b) / 2
    M = get_max_rate(a, b)
    m = get_max_change(a, b)
    c = M / (2 * m)

    logging.info(f"Inital x = {x}, c = {c}")

    iter_no = 0
    while iter_no <= MAX_ITER:
        iter_no += 1

        x = x_prev - f(x_prev)/df(x_prev)

        logging.info(f"Iteration {iter_no}: x = {x}")

        error = (x - x_prev)**2 * c
        if (error <= eps):
            break

        logging.info(f"{error} > {eps} , continue...")
        x_prev = x

    logging.info(f"Method ended on iteration {iter_no} with x value of {x}")
    return x

def main():
    logging.basicConfig(filename='single.log', encoding='utf-8', level=logging.INFO)

    a, b, eps = 0, 0.7, 0.01
    print("Iteration method:", iteration_method(a, b, eps))
    print("Newton method:", newton_method(a, b, eps))
    


if __name__ == "__main__":
    main()
