from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt

class Least_Squares_Solver:

    '''
        Initially written by Ming Hsiao in MATLAB
        Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
    '''
    @staticmethod
    def solve_default(A, b):
        from scipy.sparse.linalg import spsolve
        x = spsolve(A.T @ A, A.T @ b)
        return x, None

    @staticmethod
    def solve_pinv(A, b):
        # return x s.t. Ax = b using pseudo inverse.
        N = A.shape[1]
        x = inv(A.T @ A) @ A.T @ b
        return x, None

    @staticmethod
    def solve_lu(A, b):
        # return x, U s.t. Ax = b, and A = LU with LU decomposition.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
        N = A.shape[1]
        LU = splu(A.T @ A, permc_spec='NATURAL')
        x = LU.solve(A.T @ b)
        U = LU.U.A
        return x, U

    @staticmethod
    def solve_lu_colamd(A, b):
        # return x, U s.t. Ax = b, and Permutation_rows A Permutration_cols = LU with reordered LU decomposition.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
        N = A.shape[1]
        LU = splu(A.T @ A, permc_spec='COLAMD')
        x = LU.solve(A.T @ b)
        U = LU.U.A
        return x, U

    @staticmethod
    def solve_qr(A, b):
        # return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
        # https://github.com/theNded/PySPQR
        N = A.shape[1]
        z, R, E, rank = rz(A, b, permc_spec='NATURAL')
        x = spsolve_triangular(R, z, lower=False)
        return x, R

    @staticmethod
    def solve_qr_colamd(A, b):
        # return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
        # https://github.com/theNded/PySPQR
        N = A.shape[1]
        z, R, E, rank = rz(A, b, permc_spec='COLAMD')
        x = permutation_vector_to_matrix(E) @ spsolve_triangular(R, z, lower=False)
        return x, R

    '''
        Initially written by Ming Hsiao in MATLAB
        Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
    '''
    @staticmethod
    def solve(A, b, method='default'):
        '''
        \param A (M, N) Jacobian matirx
        \param b (M, 1) residual vector
        \return x (N, 1) state vector obtained by solving Ax = b.
        '''
        M, N = A.shape

        fn_map = {
            'default': Least_Squares_Solver.solve_default,
            'pinv': Least_Squares_Solver.solve_pinv,
            'lu': Least_Squares_Solver.solve_lu,
            'qr': Least_Squares_Solver.solve_qr,
            'lu_colamd': Least_Squares_Solver.solve_lu_colamd,
            'qr_colamd': Least_Squares_Solver.solve_qr_colamd,
        }

        return fn_map[method](A, b)
