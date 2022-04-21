import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.sparse import csr_matrix
import copy

from Least_Squares_Solver import Least_Squares_Solver as Solver
from Landmark_Associator import Landmark_Associator as Associator
from Pose import Pose

class Factor_Graph_SLAM:

    # Class variables
    method = ''
    list_of_trajs = [] # stores the optimized traj for each frame
    list_of_landmarks = [] # stores the optimized landmark positions for each frame

    # Tune these variables
    sigma_gps = [1.0, 1.0] # x,y
    sigma_odom = [0.1**2, 0.1**2, (np.pi/180.)**2] # x,y,theta
    sigma_landmark = [0.01**2, 0.01**2] # x,y

    def __init__(self, method, dimensions=2):
        '''
        \param method: the method to be used to solve the least squares
        optimization problem
        \param dimensions: number of pose/landmark dimensions to use when
                           solving SLAM problem
        \param sigma_odom: standard deviation of odometry measurements
        \param sigma_landmark: standard deviation of landmark measurements
        '''
        self.method = method

        assert(isinstance(dimensions, int))
        self.dimensions = dimensions

        if self.dimensions == 2:
            self.sigma_odom = np.diag(Factor_Graph_SLAM.sigma_odom)
            self.sigma_gps_pose = np.diag(Factor_Graph_SLAM.sigma_gps)
            self.sigma_landmark = np.diag(Factor_Graph_SLAM.sigma_landmark)
        else:
            raise NotImplementedError

    def run(self, odom_measurements, landmarks, gps_measurements, p0,
            step_convergence_thresh=1e-8, max_iters=25):
        '''
        Solves the factor graph SLAM problem.

        \param odom_measurements: [delta_x, delta_y, delta_theta] of the robot's
        pose global in coordinates between each frame. Shape (n_frames,3)
        \param landmarks: [distance_x, distance_y] of the landmark to the robot
        for each landmark in a given frame. Each row of the list corresponds to
        a single frame. A given frame may have 0 or multiple landmarks
        \param gps_measurements:
        \param p0: the initial pose of the robot
        \param step_convergence_thresh: threshold for breaking out of the
                    optimization loop (compared with norm of dx)
        \param max_iters: maximum number of steps the algorithm will take by
                    creating and solving a linear system for state vector
                    updates (dx)

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

        assert len(odom_measurements)+1 == len(landmarks)

        # Associate landmark measurements with previously seen landmarks
        landmark_measurements, n_landmarks = Associator.associate_landmarks(prev_landmarks,
                                                                            landmarks,
                                                                            traj_estimate,
                                                                            odom_measurements[-1],
                                                                            self.sigma_landmark)
        # Build a linear system
        n_poses = len(odom_measurements)+1
        traj, landmarks = Factor_Graph_SLAM.init_states(p0, odom_measurements, landmark_measurements, n_poses, n_landmarks)
        init_traj = copy.deepcopy(traj)

        # Iterative optimization
        x = Factor_Graph_SLAM.vectorize_state(traj, landmarks)
        R = None

        for i in range(max_iters):
            A, b = self.create_linear_system(x, odom_measurements, landmark_measurements, gps_measurements,
                                             p0, n_poses, n_landmarks)
            dx, R = Solver.solve(A, b, self.method)
            x = x + dx

            if np.linalg.norm(dx) < step_convergence_thresh:
                print(f'\tHit convergence threshold! Iters: {i}  (final dx norm: {np.linalg.norm(dx):.4E})')
                break
            elif i+1 == max_iters:
                print(f'\tHit iteration limit! Iters: {i+1}  (final dx norm: {np.linalg.norm(dx):.4E})')


        traj, landmarks = self.devectorize_state(x, n_poses)
        # Store the optimized traj and landmark positions
        self.list_of_trajs.append(traj)
        self.list_of_landmarks.append(landmarks)

        return traj, landmarks, R, A, b, init_traj

    def create_linear_system(self, x, odom_measurements, landmark_measurements,
                             gps_measurements, p0, n_poses, n_landmarks):
        '''
        Creates the linear system to solve the least squares problem, Ax-b=0.

        \param x: state vector of poses stacked with landmarks (all global frame)
        \param odom_measurements: Odometry measurements between i and i+1 in the
        global coordinate system. Shape: (n_odom, 3).
        \param landmark_measurements: Landmark measurements between pose i and
                landmark j in the global coordinate system.
                Shape: (n_lmark_meas, 2+self.dimensions).
        \param gps_measurements:
        \param p0: the initial pose of the robot
        \param n_poses: the number of poses in the state vector
        \param n_landmarks: the number of landmarks in the state vector

        \return A: (M, N) Jacobian matrix.
        \return b: (M, ) Residual vector.
        where M = (n_odom + 1) * self.dimensions + n_lmark_meas * self.dimensions
                :total rows of measurements.
              N = n_poses * self.dimensions + n_landmarks * self.dimensions
                :length of the state vector.
        '''

        n_odom = len(odom_measurements)
        n_lmark_meas = len(landmark_measurements)
        n_gps = len(gps_measurements)

        # how many elements are in a pose
        pose_dims = 0
        if self.dimensions == 2:
            pose_dims = 3 # (x,y,theta)
        elif self.dimensions == 3:
            print("3D NOT IMPLEMENTED IN create_linear_system()!!!")
            raise NotImplementedError

        M = n_gps * self.dimensions + n_odom * pose_dims + n_lmark_meas * self.dimensions
        N = n_poses * (pose_dims) + n_landmarks * (self.dimensions)

        A = np.zeros((M, N))
        b = np.zeros((M, ))

        # Prepare Sigma^{-1/2}.
        sqrt_gps_pose = np.linalg.inv(scipy.linalg.sqrtm(self.sigma_gps_pose))
        sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(self.sigma_odom))
        sqrt_inv_landmark = np.linalg.inv(scipy.linalg.sqrtm(self.sigma_landmark))

        # apply GPS measurements
        for gps_idx in range(n_gps):
            gps_pose = gps_measurements[gps_idx][1:]

            # traj estimate pose corresponding to gps measurement
            pose_idx = int(gps_measurements[gps_idx][0])
            pose = x[pose_dims*pose_idx:pose_dims*(pose_idx+1)]

            # I for rows corresponding to traj estimate pose
            A[self.dimensions*gps_idx:self.dimensions*(gps_idx+1), pose_dims*pose_idx:pose_dims*pose_idx + self.dimensions] = \
                        np.eye(self.dimensions) @ sqrt_gps_pose
            b[self.dimensions*gps_idx:self.dimensions*(gps_idx+1)] = (gps_pose[:self.dimensions] - pose[:self.dimensions]) @ sqrt_gps_pose

        # Fill in odometry measurements
        for odom_idx in range(n_odom):
            col = odom_idx*pose_dims
            row = col + n_gps*self.dimensions # add space from anchoring gps points

            # -I for current pose rows
            A[row:row+pose_dims, col:col+pose_dims] = \
                                -np.eye(pose_dims) @ sqrt_inv_odom
            # I for next pose rows
            A[row:row+pose_dims, col+pose_dims:col+2*pose_dims] = \
                                np.eye(pose_dims) @ sqrt_inv_odom

            # add odom measurement to b
            odom_est = Factor_Graph_SLAM.odometry_estimation(x, odom_idx, pose_dims)
            b[row:row+pose_dims] = \
                                (odom_measurements[odom_idx,:pose_dims] - odom_est) @ sqrt_inv_odom

        # Fill in landmark measurements
        for meas_idx in range(n_lmark_meas):
            pose_idx = int(landmark_measurements[meas_idx,0])
            landmark_idx = int(landmark_measurements[meas_idx,1])
            row = self.dimensions*n_gps + pose_dims*n_odom + self.dimensions*meas_idx

            x_vec_pose_row = pose_dims*pose_idx
            pose = x[x_vec_pose_row:x_vec_pose_row + pose_dims]
            # Put Jacobian (d-meas/d-landmark) into landmark rows
            lmark_jacob = Factor_Graph_SLAM.get_meas_jacob_landmark(pose)

            A[row:row+self.dimensions, \
              (pose_dims*n_poses + self.dimensions*landmark_idx):(pose_dims*n_poses + self.dimensions*(landmark_idx+1))] = \
                                lmark_jacob @ sqrt_inv_landmark

            # Put Jacobian (d-meas/d-pose) into pose rows
            lmark_x_idx = pose_dims*n_poses + landmark_idx*self.dimensions
            global_lmark_pos = x[lmark_x_idx:lmark_x_idx+self.dimensions]

            pose_jacob = Factor_Graph_SLAM.get_meas_jacob_pose(pose, global_lmark_pos)
            A[row:row+self.dimensions, pose_dims*pose_idx:pose_dims*(pose_idx+1)] = \
                                sqrt_inv_landmark @ pose_jacob

            # add measurement to b
            est_lmark_meas = Factor_Graph_SLAM.estimate_landmark_measurement(pose, global_lmark_pos).flatten()
            b[row:row+self.dimensions] = \
                                (landmark_measurements[meas_idx,2:2+self.dimensions] - est_lmark_meas) @ sqrt_inv_landmark

        return csr_matrix(A), b

    @staticmethod
    def init_states(p0, odoms, observations, n_poses, n_landmarks):
        '''
        Initialize the state vector given odometry and observations.

        \param p0 (3,): initial pose of robot [x, y, theta]
        \param odoms (n_poses, 3): odometry measurements [dx, dy, dtheta]
        \param observations (M, 4): observations of landmarks during trajectory
                    [pose_idx, landmark_idx, x, y]
        \return traj: trajectory predicted given initial pose and odometry measurements
        '''
        traj = np.zeros((n_poses, 3))
        traj[0] = p0 # trajectory starts at p0

        landmarks = np.zeros((n_landmarks, 2))
        landmarks_mask = np.zeros((n_landmarks), dtype=np.bool)

        for i in range(len(odoms)):
            traj[i + 1, :] = traj[i, :] + odoms[i, :]

        for i in range(len(observations)):
            pose_idx = int(observations[i, 0])
            landmark_idx = int(observations[i, 1])

            # if we haven't estiamted this landmarks global pos yet
            if not landmarks_mask[landmark_idx]:

                pose = traj[pose_idx, :]
                observation = observations[i, 2:]

                landmarks[landmark_idx,:] = Associator.transform_to_global_frame(observation, pose)

                landmarks_mask[landmark_idx] = True # we have an estimate for this landmark's global position

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

    @staticmethod
    def odometry_estimation(x, i, pose_dim):
        '''
        \param x: State vector containing both the pose and landmarks
        \param i: Index of the pose to start from (odometry between pose i and i+1)
        \param pose_dim: Number of elements in each pose
        \return odom Odometry (\Delta x, \Delta y) in the shape (2, )
        '''
        odom = x[pose_dim*(i+1):pose_dim*(i+2)] - x[pose_dim*i:pose_dim*(i+1)]

        return odom

    @staticmethod
    def estimate_landmark_measurement(pose, global_lmark_pos):
        '''
        Estimate a landmark measurement given the current estimate for the global
        position of the corresponding pose and landmark

        \param pose (3,): pose of the robot [x,y,theta]
        \param global_lmark_pos (2,): estimated position in global frame of landmark
        \return measurement (2,): estimated landmark measurement in robot frame
        '''
        dims = global_lmark_pos.size

        obs = global_lmark_pos - pose[:dims]

        # rotate observation into robot frame
        vehicle_pose = Pose(pose)
        R = vehicle_pose.getRotationMatrix2D()
        measurement = (R.T @ obs.reshape(dims,1))

        return measurement

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
    def plot_traj_and_landmarks(traj, landmarks, gps, gt_traj, gt_landmarks,
                                init_traj, p_init=None):
        plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='gt poses')
        plt.scatter(gt_landmarks[:, 0],
                    gt_landmarks[:, 1],
                    c='b',
                    marker='+',
                    label='gt landmarks')

        if p_init is not None:
            plt.scatter(p_init[0],
                        p_init[1],
                        s=30,
                        facecolors='orange',
                        edgecolors='g',
                        label='p0')

        plt.scatter(gps[:, 1],
                    gps[:, 2],
                    s=30,
                    facecolors='purple',
                    edgecolors='purple',
                    label='gps')

        plt.plot(traj[:, 0], traj[:, 1], 'r-', label='poses')
        plt.scatter(landmarks[:, 0],
                    landmarks[:, 1],
                    s=30,
                    facecolors='none',
                    edgecolors='r',
                    label='landmarks')

        plt.plot(init_traj[:, 0], init_traj[:, 1], 'g-', label='dead-reckoning')

        plt.legend()
        plt.show()
