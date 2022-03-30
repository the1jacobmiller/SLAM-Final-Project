import numpy as np
from Pose import Pose
# from src.Pose import Pose

class Landmark_Associator:
    @staticmethod
    def get_mahalanobis_distance(prev_landmark, landmark, cov):
        x_diff_mean = np.array([prev_landmark[0] - landmark[0], prev_landmark[1] - landmark[1]])
        mahal = np.sqrt(x_diff_mean.T @ cov @ x_diff_mean)
        print("Mahalanobis distance", mahal)
        return mahal

    @staticmethod
    def get_euclidean_distance(prev_landmark, landmark):
        euclid = np.sqrt((prev_landmark[0] - landmark[0])**2 +(prev_landmark[1] - landmark[1])**2)
        return euclid

    @staticmethod
    def apply_odom_step_2d(odom_measurement, pose):
        return np.array([pose[0] + odom_measurement[0], pose[1] + odom_measurement[1], pose[2]])
    @staticmethod
    def transform_to_global_frame(observation, pose):
        '''
        Gets the 2D pose of the robot in global coordinates from the designated
        frame.

        \param frame: the frame to retrieve the landmarks from, of type protobuf Frame

        \return pose2D: [x, y, theta] of the robot in the designated frame
        '''

        vehicle_pose = Pose(pose)
        H = vehicle_pose.getTransformationMatrix2D()
        landmark_rel_pos = np.array([observation[0], observation[1], 1])

        landmark_global_pos = (H @ landmark_rel_pos)[:2]
        return landmark_global_pos

    @staticmethod
    def associate_with_prev_landmarks(observation, pose, prev_landmarks):
        # TODO: TUNE ME
        association_thresh = 10

        observation_global_frame = Landmark_Associator.transform_to_global_frame(observation, pose)
        for prev_landmark_id in range(len(prev_landmarks)):
            prev_landmark = Landmark_Associator.transform_to_global_frame( prev_landmarks[prev_landmark_id], pose)

            # prev_landmark = prev_landmarks[prev_landmark_id]
            # print(prev_landmark)
            # print( observation_global_frame)
            dist = Landmark_Associator.get_euclidean_distance(prev_landmark, observation_global_frame)
            if dist < association_thresh:
                return prev_landmark_id
        return -1

    @staticmethod
    def create_landmark_measurement(pose_id, landmark_id, pose, landmark ):
        return np.array([pose_id, landmark_id, landmark[0] - pose[0], landmark[1] - pose[1]])
        # return np.array([pose_id, landmark_id, landmark[0] - pose[0], landmark[1] - pose[1]]).reshape(1,4)

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
        #todo: len(prev_landmarks)
        n_landmarks = len(prev_landmarks)
        landmark_measurements = []
        #Note this will iterate through the poses. We will have an extra landmark for the current step where
        #the pose has yet to be calculated. We will need to estimate odom and compute the associations from there
        for pose_id in range(len(traj_estimate)):
            print("pose",traj_estimate[pose_id])
            pose = traj_estimate[pose_id]

            landmarks = new_landmarks[pose_id]
            for observation in landmarks:
                landmark_id = Landmark_Associator.associate_with_prev_landmarks(observation, pose, prev_landmarks)
                if landmark_id == -1:
                    landmark_measurements.append(Landmark_Associator.create_landmark_measurement(pose_id, n_landmarks, observation, pose))
                    n_landmarks += 1
                else:
                    landmark_measurements.append(Landmark_Associator.create_landmark_measurement(pose_id, landmark_id, observation, pose))
        #TODO: THIS CURRENTLY ADDS THE LANDMARKS 2x.
        new_pose_estimate = Landmark_Associator.apply_odom_step_2d(odom_measurement, traj_estimate[-1])
        pose_id = len(traj_estimate)
        landmarks = new_landmarks[-1]
        for observation in landmarks:
            landmark_id = Landmark_Associator.associate_with_prev_landmarks(observation, new_pose_estimate, prev_landmarks)
            if landmark_id == -1:
                landmark_measurements.append(
                    Landmark_Associator.create_landmark_measurement(pose_id, n_landmarks, observation, new_pose_estimate))
                n_landmarks += 1

            else:
                landmark_measurements.append(
                    Landmark_Associator.create_landmark_measurement(pose_id, landmark_id, observation, new_pose_estimate))
        landmark_measurements = np.array(landmark_measurements)
        print("returned n_landmarks", n_landmarks)
        print("returned landmark_measurements", landmark_measurements)
        print("returned landmark_measurements", landmark_measurements.shape)
        return landmark_measurements, n_landmarks


















    
# # landmark_measurements = []
        # pose_id = np.array(new_landmarks).shape[0] - 1
        # print("pose_id", pose_id)
        # n_landmarks = 0
        # # TODO(corinne): associate new_landmarks with prev_landmarks
        # landmark_measurements = np.array(prev_landmarks)
        # n_landmarks = len(prev_landmarks)
        # print("new_landmarks",np.array(new_landmarks).shape)
        # next_landmarks = np.array(new_landmarks)[-1]
        # print("landmark_measurements",landmark_measurements.shape)
        # if len(prev_landmarks) != 0:
        #     prev_landmarks = np.array(prev_landmarks).squeeze(0)
        # for landmark in next_landmarks:
        #     landmark_associated = False
        #     print("prev_landmarks",np.array(prev_landmarks).shape)
        #     for prev in prev_landmarks:
        #         print("prev",np.array(prev).shape)
        #         print("landmark",np.array(landmark).shape)
        #         if Landmark_Associator.get_euclidean_distance(prev, landmark) < association_thresh:
        #             landmark_associated = True
        #             #TODO: do we need to update the pose number if we see a landmark again
        #             #prev[0] = pose_id
        #             break
        #     if not landmark_associated:
        #         print(np.array(traj_estimate).shape, np.array(landmark).shape)
        #
        #         new_landmark = Landmark_Associator.create_landmark_measurement(pose_id, n_landmarks, traj_estimate, landmark)
        #         n_landmarks += 1
        #         print("Added", new_landmark.shape)
        #         if landmark_measurements.size == 0:
        #             landmark_measurements = new_landmark
        #         else:
        #             print("landmark_measurements", landmark_measurements.shape)
        #             landmark_measurements = np.concatenate((landmark_measurements, new_landmark), axis=0)
        #         print("during landmark_measurements", landmark_measurements.shape)
        #
        #         # landmark_measurements.append(new_landmark)
        #
        # print("returned n_landmarks", n_landmarks)
        # print("returned landmark_measurements", landmark_measurements)
        # print("returned landmark_measurements", landmark_measurements.shape)
        # return landmark_measurements, n_landmarks
