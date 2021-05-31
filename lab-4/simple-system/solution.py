import numpy as np

def f(x, y, dy):
    # Returns ddy
    return 4*x*dy - (4*x*x - 2) * y

def g(x, y, z):
    return z

def original_f(x):
    return (1 + x) * np.exp(x*x)

def solve_euler(a, b, h, y0, dy0):
    # The order of ODE is lowered with z = y', 
    # so both equations have to be solved:
    # z = z + h*f(x, y, z)
    # y = y + h*g(x, y, z) = y + h*z
    xs = list(np.arange(a, b+h, h))
    ys = []
    y = y0
    z = dy0
    for x in xs:
        ys.append(y)
        z += h * f(x, y, z)
        y += h * z

    return xs, ys

def solve_runge(a, b, h, y0, dy0):
    xs = list(np.arange(a, b+h, h))
    ys = []
    zs = []
    y = y0
    z = dy0
    for x in xs:
        ys.append(y)
        zs.append(z)
        
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
    
    return xs, ys, zs

def solve_adam(xs, ys, zs, h):
    order = 4
    ys = ys[:order]
    zs = zs[:order]
    y = ys[-1]
    z = zs[-1]
    for i in range(3, len(xs) - 1):
        z += h/24 * (55 * f(xs[ i ], ys[ i ], zs[ i ]) -
                     59 * f(xs[i-1], ys[i-1], zs[i-1]) +
                     37 * f(xs[i-2], ys[i-2], zs[i-2]) -
                      9 * f(xs[i-3], ys[i-3], zs[i-3]))
        
        y += h/24 * (55 * g(xs[ i ], ys[ i ], zs[ i ]) -
                     59 * g(xs[i-1], ys[i-1], zs[i-1]) +
                     37 * g(xs[i-2], ys[i-2], zs[i-2]) -
                      9 * g(xs[i-3], ys[i-3], zs[i-3]))
        
        ys.append(y)
        zs.append(z)
        
    return xs, ys

def test_rrr(eulers, runges, adams):
    def get_error(l1, l2, order):
        return [abs(i1 - i2) / (2**order - 1) for i1, i2 in zip(l1, l2)]
    
    return (
        get_error(eulers[0], eulers[1], 1),
        get_error(runges[0], runges[1], 4),
        get_error(adams[0],  adams[1],  4)
    )
    

def test_exact(euler, runge, adam, exact):
    def get_error(l1, l2):
        return [abs(i1 - i2) for i1, i2 in zip(l1, l2)]
    
    return (
        get_error(euler, exact), 
        get_error(runge, exact), 
        get_error(adam, exact)
    )

def main():
    def print_pairwise(xs, ys):
        for i, (x, y) in enumerate(zip(xs, ys)):
            print(f"{i}: {x:.3f} {y}")
            
    a, b, step = 0, 1, 0.1
    y0, dy0 = 1, 1
    
    eulers = []
    runges = []
    adams  = []
    
    for h in [step, step/2]:
        print("For step", h)

        x, y = solve_euler(a, b, h, y0, dy0)
        print("Euler:")
        print_pairwise(x, y)
        eulers.append(y)

        x, y, z = solve_runge(a, b, h, y0, dy0)
        print("Runge:")
        print_pairwise(x, y)
        runges.append(y)

        x, y = solve_adam(x, y, z, h)
        print("Adam:")
        print_pairwise(x, y)
        adams.append(y)
        
        print()
        
    exact = [original_f(xi) for xi in x]
    print("Analytical solution:")
    print_pairwise(x, exact)
    
    p_euler, p_runge, p_adam = test_rrr(eulers, runges, adams)
    print("\nPosterior errors:")
    print(" Euler         Runge        Adam       ")
    for i, (e, r, a) in enumerate(zip(p_euler, p_runge, p_adam)):
        print(f"{e:12.9f}  {r:12.9f} {a:12.9f}")
    
    e_euler, e_runge, e_adam = test_exact(eulers[1], runges[1], adams[1], exact)
    print("\nExact errors:")
    print(" Euler         Runge        Adam       ")
    for i, (e, r, a) in enumerate(zip(e_euler, e_runge, e_adam)):
        print(f"{e:12.9f}  {r:12.9f} {a:12.9f}")

if __name__ == "__main__":
    main()
