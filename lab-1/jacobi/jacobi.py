import math
import numpy as np
import logging, json

MAX_ITER = 1000

def jacobi_iteration(matrix, eps=10e-5):
    def non_diagonal_norm(matrix):
        total = 0
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix[i])):
                total += matrix[i][j] ** 2
        return (math.sqrt(total))

    def max_non_diagonal(matrix):
        max_i, max_j = 0, 1
        max_val = abs(matrix[max_i][max_j])
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix[i])):
                val = abs(matrix[i][j])

                if max_val < val:
                    max_val = val
                    max_i, max_j = i, j
        return (max_i, max_j)

    def create_rotation(matrix, i, j):
        rotation = np.identity(len(matrix))
        if matrix[i, i] == matrix[j, j]:
            angle = math.pi / 4
        else:
            angle = math.atan(2 * matrix[i, j] / (matrix[i, i] - matrix[j, j])) / 2

        s = math.sin(angle)
        c = math.cos(angle)

        rotation[i, j] = -s;
        rotation[j, j] = c;
        rotation[j, i] = s;
        rotation[i, i] = c;
        
        return rotation


    matrix  = matrix.copy()
    vectors = np.identity(len(matrix))

    iter = 0
    while non_diagonal_norm(matrix) >= eps and iter <= MAX_ITER:
        max_i, max_j = max_non_diagonal(matrix)
        rotation = create_rotation(matrix, max_i, max_j)

        vectors  = vectors @ rotation

        matrix = matrix @ rotation
        rotation = rotation.T
        matrix = rotation @ matrix
        
        iter += 1
        
    return np.diag(matrix), vectors.T


if __name__ == "__main__":
    logging.basicConfig(filename='jacobi.log', encoding='utf-8', level=logging.INFO)

    with open("task.json", "r") as json_file:
        task = json.load(json_file)

    matrix = np.array(task["matrix"])
    try:
        eps = task["epsilon"]
    except KeyError:
        eps = 10e-5

    logging.info("Input matrix:\n%s", str(matrix))
    logging.info("Selected epsilon: %f", eps)

    values, vectors = jacobi_iteration(matrix)

    logging.info("Eigenvalues: %s", str(values))
    logging.info("Eigenvectors:\n%s", str(vectors))

    print("Eigenvalues: ", values)
    print("Eigenvectors: ")
    print(vectors)
