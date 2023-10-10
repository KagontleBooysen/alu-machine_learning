#!/usr/bin/env python3

def cofactor(matrix):
    """
    Calculate the cofactor matrix of a square matrix.

    Args:
        matrix (list of list of int): The input square matrix.

    Returns:
        list of list of int: The cofactor matrix of the input matrix.

    Raises:
        TypeError: If the input is not a list of lists.
        ValueError: If the input matrix is not square or is empty.
    """
    # Check if the input is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix is not a list of lists")

    # Check if the matrix is square (number of rows == number of columns)
    num_rows = len(matrix)
    if num_rows == 0 or any(len(row) != num_rows for row in matrix):
        raise ValueError("matrix is not square or is empty")

    # Helper function to calculate the determinant of a 2x2 matrix
    def determinant_2x2(mat):
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

    # Initialize an empty cofactor matrix
    cofactor_matrix = []

    for i in range(num_rows):
        cofactor_row = []
        for j in range(num_rows):
            # Calculate the determinant of the submatrix without row i and column j
            submatrix = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]

            # Calculate the cofactor as (-1)^(i+j) times the determinant of the submatrix
            cofactor_value = (-1) ** (i + j) * determinant_2x2(submatrix)
            cofactor_row.append(cofactor_value)

        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix

# Example usage:
mat1 = [[5]]
mat2 = [[1, 2], [3, 4]]
mat3 = [[1, 1], [1, 1]]
mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
mat5 = []
mat6 = [[1, 2, 3], [4, 5, 6]]

print("Cofactor of mat1 (1x1 matrix):")
print(cofactor(mat1))
print("Cofactor of mat2 (2x2 matrix):")
print(cofactor(mat2))
print("Cofactor of mat3 (2x2 matrix with equal elements):")
print(cofactor(mat3))
print("Cofactor of mat4 (3x3 matrix):")
print(cofactor(mat4))
try:
    print("Cofactor of mat5 (empty matrix):")
    cofactor(mat5)
except Exception as e:
    print(e)
try:
    print("Cofactor of mat6 (non-square matrix):")
    cofactor(mat6)
except Exception as e:
    print(e)

