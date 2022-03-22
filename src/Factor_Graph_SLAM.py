import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.sparse import csr_matrix

from Least_Squares_Solver import Least_Squares_Solver as Solver
from Landmark_Associator import Landmark_Associator as Associator

class Factor_Graph_SLAM:

    # Parameters
    sigma_odom = np.eye(2)
    sigma_landmark = np.eye(2)

    # Class variables
    method = ''
    list_of_trajs = [] # stores the optimized traj for each frame
    list_of_landmarks = [] # stores the optimized landmark positions for each frame

    def __init__(self, method):
        '''
        \param method: the method to be used to solve the least squares
        optimization problem
        '''
        self.method = method

    def run(self, odom_measurements, landmarks, p0):
        '''
        Solves the factor graph SLAM problem.

        \param odom_measurements: [delta_x, delta_y, delta_theta] of the robot's
        pose global in coordinates between each frame. Shape (n_frames,3)
        \param landmarks: [distance_x, distance_y] of the landmark to the robot
        for each landmark in a given frame. Each row of the list corresponds to
        a single frame. A given frame may have 0 or multiple landmarks
        \param p0: the initial pose of the robot

        \return traj: the optimized trajectory that the robot has followed
        \return landmarks: the optimized positions of the landmarks
        \return R: the matrix factorization from the least squares solver. Not
        every method produces a matrix factorization
        \return A: the A matrix for the least squares problem, Ax-b=0
        \return b: the b vector for the least squares problem, Ax-b=0
        '''
        if len(self.list_of_trajs) > 0 and len(self.list_of_landmarks) > 0:
            prev_landmarks = self.list_of_landmarks[-1]
            traj_estimate = self.list_of_trajs[-1]
        else:
            prev_landmarks = []
            traj_estimate = p0

        # Associate landmark measurements with previously seen landmarks
        landmark_measurements, n_landmarks = Associator.associate_landmarks(prev_landmarks,
                                                                            landmarks,
                                                                            traj_estimate,
                                                                            odom_measurements[-1],
                                                                            self.sigma_landmark)

        # Build a linear system
        n_poses = len(odom_measurements)
        A, b = self.create_linear_system(odom_measurements, landmark_measurements,
                                         p0, n_poses, n_landmarks)

        # Solve with the selected method
        x, R = Solver.solve(A, b, self.method)
        traj, landmarks = Factor_Graph_SLAM.devectorize_state(x, n_poses)

        # Store the optimized traj and landmark positions
        self.list_of_trajs.append(traj)
        self.list_of_landmarks.append(landmarks)

        return traj, landmarks, R, A, b

    def create_linear_system(self, odom_measurements, landmark_measurements,
                             p0, n_poses, n_landmarks):
        '''
        Creates the linear system to solve the least squares problem, Ax-b=0.

        \param odom_measurements: Odometry measurements between i and i+1 in the
        global coordinate system. Shape: (n_odom, 3).
        \param landmark_measurements: Landmark measurements between pose i and
        landmark j in the global coordinate system. Shape: (n_obs, 4).
        \param p0: the initial pose of the robot

        \return A: (M, N) Jacobian matrix.
        \return b: (M, ) Residual vector.
        where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
              N = n_poses * 2 + n_landmarks * 2, length of the state vector.
        '''

        n_odom = len(odom_measurements)
        n_obs = len(landmark_measurements)

        M = (n_odom + 1) * 2 + n_obs * 2
        N = n_poses * 2 + n_landmarks * 2

        A = np.zeros((M, N))
        b = np.zeros((M, ))

        # Prepare Sigma^{-1/2}.
        sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(self.sigma_odom))
        sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(self.sigma_landmark))

        # TODO(jonathan): build the A matrix and b vector

        return csr_matrix(A), b

    '''
        Initially written by Ming Hsiao in MATLAB
        Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
    '''
    @staticmethod
    def vectorize_state(traj, landmarks):
        x = np.concatenate((traj.flatten(), landmarks.flatten()))
        return x

    '''
        Initially written by Ming Hsiao in MATLAB
        Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
    '''
    @staticmethod
    def devectorize_state(x, n_poses):
        traj = x[:n_poses * 2].reshape((-1, 2))
        landmarks = x[n_poses * 2:].reshape((-1, 2))
        return traj, landmarks

    '''
        Initially written by Ming Hsiao in MATLAB
        Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
    '''
    @staticmethod
    def plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks):
        plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='gt poses')
        plt.scatter(gt_landmarks[:, 0],
                    gt_landmarks[:, 1],
                    c='b',
                    marker='+',
                    label='gt landmarks')

        plt.plot(traj[:, 0], traj[:, 1], 'r-', label='poses')
        plt.scatter(landmarks[:, 0],
                    landmarks[:, 1],
                    s=30,
                    facecolors='none',
                    edgecolors='r',
                    label='landmarks')

        plt.legend()
        plt.show()
