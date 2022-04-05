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
        return np.array([pose[0] + odom_measurement[0], pose[1] + odom_measurement[1], pose[2] + odom_measurement[2]])

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
        association_thresh = 0.5 # tuned for euclidean dist with no p0 noise

        for prev_landmark_id in range(len(prev_landmarks)):
            prev_landmark = prev_landmarks[prev_landmark_id]
            dist = Landmark_Associator.get_euclidean_distance(prev_landmark, observation)

            if dist < association_thresh:
                return prev_landmark_id
        return -1

    @staticmethod
    def create_landmark_measurement(pose_id, landmark_id, observation):
        return np.array([pose_id, landmark_id, observation[0], observation[1]])

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

        n_landmarks = len(prev_landmarks)

        # print(f"prev landmarks:\n {prev_landmarks}")

        landmark_measurements = []
        #CA: Note this will iterate through the poses. We will have an extra landmark for the current step where
        #     the pose has yet to be calculated. We will need to estimate odom and compute the associations from there

        # JS: re:above comment
        #         the way this works now, that is not the case, but if we use the odom_measurements (which we def should),
        #         this is something we need to think about. Seems like we get 1 more than we need? or maybe p0 is
        #         behind the first pose measurements are taken from?

        # iterate through poses in trajectory
        for pose_id in range(len(traj_estimate)):
            pose = traj_estimate[pose_id]
            landmarks = new_landmarks[pose_id]

            # loop through landmark measurements corresponding with pose
            for lmark_local_frame in landmarks:
                lmark_global_frame = Landmark_Associator.transform_to_global_frame(lmark_local_frame, pose)
                # observation = lmark_global_frame - pose[:2]

                landmark_id = Landmark_Associator.associate_with_prev_landmarks(lmark_global_frame, pose, prev_landmarks)

                if landmark_id == -1: # no match
                    landmark_measurements.append(Landmark_Associator.create_landmark_measurement(pose_id, n_landmarks, lmark_local_frame))
                    # add new landmark to prev_landmarks so we can (potentially) match new landmarks to it
                    if len(prev_landmarks) > 0:
                        prev_landmarks = np.vstack([prev_landmarks, lmark_global_frame])
                    else:
                        prev_landmarks = lmark_global_frame.reshape(1,2)
                    n_landmarks += 1
                else:
                    # found a match
                    landmark_measurements.append(Landmark_Associator.create_landmark_measurement(pose_id, landmark_id, lmark_local_frame))

        landmark_measurements = np.array(landmark_measurements)
        return landmark_measurements, n_landmarks
