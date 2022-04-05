import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.sparse import csr_matrix

from Least_Squares_Solver import Least_Squares_Solver as Solver
from Landmark_Associator import Landmark_Associator as Associator
from Pose import Pose

class Factor_Graph_SLAM:

    # Class variables
    method = ''
    list_of_trajs = [] # stores the optimized traj for each frame
    list_of_landmarks = [] # stores the optimized landmark positions for each frame

    def __init__(self, method, dimensions=2):
        '''
        \param method: the method to be used to solve the least squares
        optimization problem
        \param dimensions: number of pose/landmark dimensions to use when
                           solving SLAM problem
        '''
        self.method = method

        assert(isinstance(dimensions, int))
        self.dimensions = dimensions

        # Uncertainty Values (maybe TODO - shouldn't just be ones)
        if self.dimensions == 2:
            self.sigma_odom = np.eye(self.dimensions+1)
            self.sigma_landmark = np.eye(self.dimensions)
        else:
            raise NotImplementedError

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
            traj_estimate = [p0]

        # print("before ass:\n", landmarks)
        # Associate landmark measurements with previously seen landmarks
        landmark_measurements, n_landmarks = Associator.associate_landmarks(prev_landmarks,
                                                                            landmarks,
                                                                            traj_estimate,
                                                                            odom_measurements[-1],
                                                                            self.sigma_landmark)
        # print("after:\n", landmarks)
        # Build a linear system
        n_poses = len(odom_measurements)+1
        traj, landmarks = Factor_Graph_SLAM.init_states(p0, odom_measurements, landmark_measurements, n_poses, n_landmarks)

        # print(f"Init states: \n\tTraj: {traj} \n\t lmarks: {landmarks}")
        # Iterative optimization
        x = Factor_Graph_SLAM.vectorize_state(traj, landmarks)
        R = None
        for i in range(10):
            A, b = self.create_linear_system(x, odom_measurements, landmark_measurements,
                                             p0, n_poses, n_landmarks)
            dx, R = Solver.solve(A, b, self.method)
            # print(f"\t iter {i} .. norm: {np.linalg.norm(dx)}")
            x = x + dx

        traj, landmarks = self.devectorize_state(x, n_poses)
        # Store the optimized traj and landmark positions
        self.list_of_trajs.append(traj)
        self.list_of_landmarks.append(landmarks)

        # print("\n\nlandmarks returned:\n", landmarks)
        return traj, landmarks, R, A, b

    @staticmethod
    def init_states(p0, odoms, observations, n_poses, n_landmarks):
        '''
        Initialize the state vector given odometry and observations.
        '''
        traj = np.zeros((n_poses, 3))
        traj[0] = p0
        landmarks = np.zeros((n_landmarks, 2))
        landmarks_mask = np.zeros((n_landmarks), dtype=np.bool)

        # print(f"obs size: {observations.shape}\t n_landys: {n_landmarks}")

        for i in range(len(odoms)):
            traj[i + 1, :] = traj[i, :] + odoms[i, :]

        for i in range(len(observations)):
            pose_idx = int(observations[i, 0])
            landmark_idx = int(observations[i, 1])

            if not landmarks_mask[landmark_idx]:
                landmarks_mask[landmark_idx] = True

                pose = traj[pose_idx, :]
                observation = observations[i, 2:]

                landmarks[landmark_idx,:] = Associator.transform_to_global_frame(observation, pose)

        return traj, landmarks

    @staticmethod
    def get_meas_jacob_pose(pose, lmark):
        '''
        Derivative of measurement function in terms of pose

        \param pose (3,): pose of robot [x,y,theta]
        \param lmark (2,): position of landmark in space [x,y]
        \return jacob (2,3): derivatives of measurement in terms of robot pose
        '''

        rx, ry, r_theta = pose[0], pose[1], pose[2]
        lx, ly = lmark[0], lmark[1]
        dx = lx - rx
        dy = ly - ry

        jacob = np.array([[-np.cos(r_theta), -np.sin(r_theta), -np.sin(r_theta)*dx + np.cos(r_theta)*dy],
                          [np.sin(r_theta), -np.cos(r_theta), -np.cos(r_theta)*dx - np.sin(r_theta)*dy]])

        return jacob

    @staticmethod
    def get_meas_jacob_landmark(pose):
        '''
        Derivative of measurement function in terms of landmark

        \param pose (3,): pose of robot [x,y,theta]
        \return jacob (2,2): derivatives of measurement in terms of landmark position
        '''

        _, _, r_theta = pose[0], pose[1], pose[2]

        jacob = np.array([[np.cos(r_theta), np.sin(r_theta)],
                          [-np.sin(r_theta), np.cos(r_theta)]])

        return jacob

    def estimate_landmark_measurement(self, pose, global_lmark_pos):

        # print(f"ELM inputs:\t p: {pose}\t glp: {global_lmark_pos}")
        obs = global_lmark_pos - pose[:self.dimensions]
        th = pose[2]
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])

        # just rotation
        # vehicle_pose = Pose(np.array([0, 0, pose[2]]))
        # R = vehicle_pose.getTransformationMatrix2D()[:self.dimensions,:self.dimensions]

        # meas_pre_rot = np.array([obs[0], obs[1], 1])
        measurement = (R @ obs.reshape(self.dimensions,1))

        return measurement

    def create_linear_system(self, x, odom_measurements, landmark_measurements,
                             p0, n_poses, n_landmarks):
        '''
        Creates the linear system to solve the least squares problem, Ax-b=0.

        \param odom_measurements: Odometry measurements between i and i+1 in the
        global coordinate system. Shape: (n_odom, 3).
        \param landmark_measurements: Landmark measurements between pose i and
                landmark j in the global coordinate system.
                Shape: (n_lmark_meas, 2+self.dimensions).
        \param p0: the initial pose of the robot

        \return A: (M, N) Jacobian matrix.
        \return b: (M, ) Residual vector.
        where M = (n_odom + 1) * self.dimensions + n_lmark_meas * self.dimensions
                :total rows of measurements.
              N = n_poses * self.dimensions + n_landmarks * self.dimensions
                :length of the state vector.
        '''

        # print("STARTING OFF WITH x:\n", x)
        n_odom = len(odom_measurements)
        n_lmark_meas = len(landmark_measurements)

        pose_dims = 0
        if self.dimensions == 2:
            pose_dims = 3 # dims are x,y,theta
        elif self.dimensions == 3:
            print("3D NOT IMPLEMENTED IN create_linear_system()!!!")
            raise NotImplementedError

        M = (n_odom + 1) * (pose_dims) + n_lmark_meas * (self.dimensions)
        N = n_poses * (pose_dims) + n_landmarks * (self.dimensions)

        A = np.zeros((M, N))
        b = np.zeros((M, ))
        # Prepare Sigma^{-1/2}.
        sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(self.sigma_odom))
        sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(self.sigma_landmark))
        # TODO: special jacob for pose (angles diff than poses)

        # anchor initial state at p0
        A[:pose_dims,:pose_dims] = np.eye(pose_dims)
        b[:pose_dims] = p0[:pose_dims] - x[:pose_dims]

        # Fill in odometry measurements
        for odom_idx in range(n_odom):
            col = odom_idx*pose_dims
            row = pose_dims + col # add space from anchoring initial point

            # -I for current pose rows
            A[row:row+pose_dims, col:col+pose_dims] = \
                                -np.eye(pose_dims) @ sqrt_inv_odom
            # I for next pose rows
            A[row:row+pose_dims, \
              col+pose_dims:col+2*pose_dims] = \
                                np.eye(pose_dims) @ sqrt_inv_odom
            # add odom measurement to b
            b[row:row+pose_dims] = \
                                odom_measurements[odom_idx,:pose_dims] @ sqrt_inv_odom

        # Fill in landmark measurements
        for meas_idx in range(n_lmark_meas):
            pose_idx = int(landmark_measurements[meas_idx,0])
            landmark_idx = int(landmark_measurements[meas_idx,1])
            row = pose_dims*(1+n_odom) + self.dimensions*meas_idx

            pose = x[pose_dims*pose_idx:pose_dims*(pose_idx+1)]

            # Put Jacobian (d-meas/d-landmark) into landmark rows
            lmark_jacob = Factor_Graph_SLAM.get_meas_jacob_landmark(pose)

            A[row:row+self.dimensions, \
              (pose_dims*n_poses + self.dimensions*landmark_idx):(pose_dims*n_poses + self.dimensions*(landmark_idx+1))] = \
                                lmark_jacob @ sqrt_inv_obs

            # Put Jacobian (d-meas/d-pose) into pose rows
            lmark_x_idx = pose_dims*n_poses + landmark_idx*self.dimensions
            global_lmark_pos = x[lmark_x_idx:lmark_x_idx+self.dimensions]
            est_lmark_meas = self.estimate_landmark_measurement(pose, global_lmark_pos).flatten()

            pose_jacob = Factor_Graph_SLAM.get_meas_jacob_pose(pose, global_lmark_pos)
            A[row:row+self.dimensions, pose_dims*pose_idx:pose_dims*(pose_idx+1)] = \
                                sqrt_inv_obs @ pose_jacob

            # print(f"uh oh: {landmark_measurements[meas_idx,2:2+self.dimensions].shape}\t {est_lmark_meas.shape}")
            # add measurement to b
            b[row:row+self.dimensions] = \
                                (landmark_measurements[meas_idx,2:2+self.dimensions] - est_lmark_meas) @ sqrt_inv_obs

        # print("b vec:\n", b)
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
    def devectorize_state(self, x, n_poses):
        traj = x[:n_poses * (self.dimensions+1)].reshape((-1, self.dimensions+1))
        landmarks = x[n_poses * (self.dimensions+1):].reshape((-1, self.dimensions))
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
