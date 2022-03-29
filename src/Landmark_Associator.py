import numpy as np


class Landmark_Associator:
    @staticmethod
    def get_mahalanobis_distance(prev_landmark, landmark, cov):
        x_diff_mean = np.array([prev_landmark[0] - landmark[0], prev_landmark[1] -  landmark[1]])
        mahal = np.sqrt(x_diff_mean.T @ cov @ x_diff_mean)
        print("Mahalanobis distance", mahal)
        return mahal

    @staticmethod
    def get_euclidean_distance(prev_landmark, landmark):
        print(prev_landmark[0] - landmark[0])
        print(prev_landmark[1] - landmark[1])
        euclid = np.sqrt((prev_landmark[0] - landmark[0])**2 +(prev_landmark[1] - landmark[1])**2)
        print("euclidean distance", euclid)
        return euclid
    @staticmethod
    def create_landmark_measurement(pose_id, landmark_id,pose, landmark ):
        return [pose_id, landmark_id, landmark[0] - pose[0], landmark[1] - pose[1]]

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
        pose_id, landmark_id,
        \return n_landmarks: the number of unique landmarks
        '''


        '''
        Thoughts
        we need to get the pose id & landmark id where we create landmark id
        
        '''
        #TODO: TUNE ME
        association_thresh = 0.001
        landmark_measurements = []
        pose_id = np.array(new_landmarks).shape[0] - 1
        print("pose_id", pose_id)
        n_landmarks = 0
        # TODO(corinne): associate new_landmarks with prev_landmarks
        landmark_measurements = prev_landmarks
        n_landmarks = len(prev_landmarks)
        print("prev_landmarks first ", prev_landmarks)
        print("new_landmarks first ",  np.array(new_landmarks).shape)
        next_landmarks = np.array(new_landmarks)[-1]
        for landmark in next_landmarks:
            landmark_associated = False
            print("prev_landmarks",prev_landmarks)
            for prev in prev_landmarks:
                print("prev",prev)
                print("landmark",landmark)
                if Landmark_Associator.get_euclidean_distance(prev, landmark) < association_thresh:
                    landmark_associated = True
                    #TODO: do we need to update the pose number if we see a landmark again
                    #prev[0] = pose_id
                    break
            if not landmark_associated:
                print(np.array(traj_estimate).shape, np.array(landmark).shape)

                new_landmark = Landmark_Associator.create_landmark_measurement(pose_id, n_landmarks, traj_estimate, landmark)
                n_landmarks += 1
                print("Added", new_landmark)
                landmark_measurements.append(new_landmark)

        print("returned n_landmarks", n_landmarks)
        print("returned landmark_measurements", landmark_measurements)
        return landmark_measurements, n_landmarks
