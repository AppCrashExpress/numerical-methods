import numpy as np

def f(x):
    return np.arccos(x)

def df(x):
    return -1 / np.sqrt(1 - x*x)

def ddf(x):
    return -x / np.power((1 - x*x), 3/2)

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

def df_num2(x, xs, ys):
    # For polynomial of second degree
    i1 = find_interval(x, xs)
    if i1 is None:
        return None
    i2 = i1 + 1

    l = df_num1(x, xs, ys, i1)
    r = df_num1(x, xs, ys, i2)

    return l + (r - l) / (xs[i2+1] - xs[i1]) * (2 * x - xs[i1] - xs[i1+1])


def ddf_num(x, xs, ys):
    i1 = find_interval(x, xs)
    if i1 is None:
        return None
    i2 = i1 + 1

    l = df_num1(x, xs, ys, i1)
    r = df_num1(x, xs, ys, i2)
    
    return (r - l) / (xs[i2+1] - xs[i1]) * 2
    
def main():
    x = 0.2
    xs = [0.2 * (i - 1) for i in range(5)]
    ys = [f(x_i) for x_i in xs]

    print(f"First derivative is {df(x)}, "
          f"approximately {df_num2(x, xs, ys)}")
    print(f"Second derivative is {ddf(x)}, "
          f"approximately {ddf_num(x, xs, ys)}")

if __name__ == "__main__":
    main()
