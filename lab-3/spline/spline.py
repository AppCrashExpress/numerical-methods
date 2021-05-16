import numpy as np

def make_splines(points):
    def get_diffs(x):
        return np.diff(x)

    def get_matrix(dx):
        matrix = np.zeros((len(dx)-1, len(dx)-1))
        
        for i, row in enumerate(matrix[1:-1, :], 1):
            values = [dx[i-1], 2*(dx[i-1]+dx[i]), dx[i]]
            row[i-1:i+2] = values

        rl = len(dx)-2
        matrix[0,:2] = [2*(dx[0]+dx[1]), dx[1]]
        matrix[-1, -2:] = [dx[-2], 2*(dx[-2]+dx[-1])]

        return matrix

    def get_constants(dx, y):
        c_calc = lambda i : 3 * ( (y[i+2]-y[i+1])/dx[i+1] - (y[i+1]-y[i])/dx[i] )
        return np.array([c_calc(i) for i in range(len(y)-2)])

    def get_c_coeffs(dx, y):
        matr = get_matrix(dx)
        consts = get_constants(dx, ys)

        return np.linalg.solve(matr, consts)

    def get_coeffs(x, y):
        dx = get_diffs(x)
        coeffs = np.zeros((len(x)-1, 4))

            # A
        coeffs[:, 0] = y[:-1]
            # C
        coeffs[1:, 2] = get_c_coeffs(dx, y)
            # B
        b_calc = lambda i : (y[i+1]-y[i])/dx[i] - dx[i]*(coeffs[i+1,2]+2*coeffs[i,2])/3
        coeffs[:-1, 1] = np.array([b_calc(i) for i in range(len(dx) - 1)])
        coeffs[-1, 1]  = (y[-1]-y[-2])/dx[-1] - dx[-1]*2*coeffs[-1,2]/3
            # D
        d_calc = lambda i : (coeffs[i+1,2]-coeffs[i,2])/(3*dx[i])
        coeffs[:-1, 3] = np.array([d_calc(i) for i in range(len(dx) - 1)])
        coeffs[-1, 3]  = -coeffs[-1,2]/(3*dx[-1])

        return coeffs
    
    xs, ys = map(np.array, [list(t) for t in zip(*points)])

    print(get_coeffs(xs, ys))

def main():
    points = [(0,0), (1, 1.8415), (2,2.9093), (3,3.1411), (4,3.2432)]

    make_splines(points)


if __name__ == "__main__":
    main()
