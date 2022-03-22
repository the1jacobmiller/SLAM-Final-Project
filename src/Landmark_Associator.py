class Landmark_Associator:

    @staticmethod
    def associate_landmarks(prev_landmarks, new_landmarks,
                            traj_estimate, odom_measurement, sigma_landmark):
        '''
        Associates new landmark observations with previously seen landmarks.

        \param prev_landmarks: [x, y] in the global frame for each previously
        observed landmark
        \param new_landmarks: [distance_x, distance_y] of the landmark to the robot
        for each landmark in a given frame. Each row of the list corresponds to
        a single frame. A given frame may have 0 or multiple landmarks
        \param traj_estimate: the estimated trajectory of the robot in the
        global frame for each frame
        \param: odom_measurement: [delta_x, delta_y, delta_theta] of the robot's
        pose in the global frame for the latest frame
        \param sigma_landmark: Shared covariance matrix of landmark measurements
        with shape (2, 2)

        \return landmark_measurements: Landmark measurements between pose i and
        landmark j in the global coordinate system. Shape: (n_obs, 4).
        \return n_landmarks: the number of unique landmarks
        '''

        landmark_measurements = []
        n_landmarks = 0
        # TODO(corinne): associate new_landmarks with prev_landmarks

        return landmark_measurements, n_landmarks
