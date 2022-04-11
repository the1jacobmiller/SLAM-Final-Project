import numpy as np
from Pose import Pose
# from src.Pose import Pose
import cv2 as cv

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
    def add_new_cropped_bb_to_dict(cropped_bb,landmark_id, associated_cropped_bbox):
        if landmark_id in associated_cropped_bbox:
            associated_cropped_bbox[landmark_id].append(cropped_bb)
        else:
            associated_cropped_bbox[landmark_id] = [cropped_bb]
        return associated_cropped_bbox

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
    def associate_with_global_landmarks_features(new_cropped_bbox, prev_cropped_bboxes):
        if len(prev_cropped_bboxes) == 0:
            return -1
        # Initiate ORB detector
        orb = cv.ORB_create()
        association_thresh = 2.0 # tuned p0 noise
        max_num_matches = 0
        closest_idx = -1
        # find the keypoints and descriptors with ORB
        _, obs_des = orb.detectAndCompute(np.array(new_cropped_bbox), None)
        if obs_des is None:
            return -1
        for idx in range(len(prev_cropped_bboxes)):
            lmark_cropped_bbox = prev_cropped_bboxes[idx][0]
            assert(len(np.array(lmark_cropped_bbox).shape) == 3) #making sure shape is correct
            # find the keypoints and descriptors with ORB
            _, landmark_des = orb.detectAndCompute(np.array(lmark_cropped_bbox), None)
            if landmark_des is None:
                continue
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(obs_des, landmark_des)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > max_num_matches:
                closest_idx = idx
                max_num_matches = len(matches)

        if max_num_matches > association_thresh:
            return closest_idx
        return -1
    @staticmethod
    def associate_with_global_landmarks(observation, global_landmarks):
        # TODO: TUNE ME
        association_thresh = 2.0 # tuned for euclidean dist with no p0 noise
        min_dist = np.inf
        closest_idx = -1
        for idx in range(len(global_landmarks)):
            global_landmark = global_landmarks[idx][0:2]
            dist = Landmark_Associator.get_euclidean_distance(global_landmark, observation)
            if dist < min_dist:
                closest_idx = idx
                min_dist = dist

        if min_dist < association_thresh:
            return closest_idx
        return -1

    @staticmethod
    def create_landmark_measurement(pose_id, landmark_id, observation):
        return np.array([pose_id, landmark_id,  observation[0], observation[1]])

    @staticmethod
    def process_landmarks_at_pose(pose, pose_id, landmarks, landmark_measurements, global_landmarks, associated_cropped_bbox, n_landmarks):
        # Keeps track of global landmark ids already associated with this
        # pose.
        matched_lmark_ids = []
        # loop through landmark measurements corresponding with pose
        for lmark_local_frame in landmarks:
            lmark_rel_pos = lmark_local_frame[0:2]
            cropped_bbox = lmark_local_frame[2]
            lmark_global_frame = Landmark_Associator.transform_to_global_frame(lmark_rel_pos, pose)
            landmark_idx = Landmark_Associator.associate_with_global_landmarks_features(cropped_bbox, associated_cropped_bbox)

            if landmark_idx == -1:
                # no match - assign a new id to this landmark
                new_global_landmark = np.array([lmark_global_frame[0], lmark_global_frame[1], n_landmarks]).reshape((1,3))
                n_landmarks += 1

                # add new landmark to prev_landmarks so we can (potentially) match new landmarks to it

                if len(global_landmarks) > 0:
                    global_landmarks = np.vstack([global_landmarks, new_global_landmark])
                else:
                    global_landmarks = new_global_landmark
                landmark_measurements.append(Landmark_Associator.create_landmark_measurement(pose_id, new_global_landmark[0,2], lmark_local_frame))
                matched_lmark_ids.append(new_global_landmark[0,2])
                associated_cropped_bbox = Landmark_Associator.add_new_cropped_bb_to_dict(cropped_bbox, new_global_landmark[0,2], associated_cropped_bbox)
            else:
                # found a match
                if global_landmarks[landmark_idx,2] == -1:
                    # We haven't seen this landmark yet in this iteration.
                    # Assign this global landmark an id
                    global_landmarks[landmark_idx,2] = n_landmarks
                    n_landmarks += 1

                if global_landmarks[landmark_idx,2] not in matched_lmark_ids:
                    landmark_measurements.append(Landmark_Associator.create_landmark_measurement(pose_id, global_landmarks[landmark_idx,2], lmark_local_frame))
                    matched_lmark_ids.append(global_landmarks[landmark_idx,2])
                    associated_cropped_bbox = Landmark_Associator.add_new_cropped_bb_to_dict(cropped_bbox,
                                                                                             global_landmarks[landmark_idx, 2],
                                                                                             associated_cropped_bbox)

        return landmark_measurements, global_landmarks, n_landmarks, associated_cropped_bbox

    @staticmethod
    def associate_landmarks(prev_landmarks,associated_cropped_bb, new_landmarks,
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

        n_landmarks = 0
        prev_landmarks = np.array(prev_landmarks)
        landmark_measurements = [] # pose_id,landmark_id,x,y,bb
        global_landmarks = [] # x,y,id
        if len(prev_landmarks) > 0:
            global_landmarks = np.hstack((prev_landmarks, -np.ones((len(prev_landmarks),1))))
        # iterate through poses in trajectory
        for pose_id in range(len(traj_estimate)):
            pose = traj_estimate[pose_id]
            landmarks = new_landmarks[pose_id]

            landmark_measurements,\
            global_landmarks,\
            n_landmarks,associated_cropped_bb = Landmark_Associator.process_landmarks_at_pose(pose,
                                                                        pose_id,
                                                                        landmarks,
                                                                        landmark_measurements,
                                                                        global_landmarks,
                                                                        associated_cropped_bb,
                                                                        n_landmarks)

            # new_cropped_bboxes.append(new_cropped_bbox)

        # Take the most recent odom step and associate the most recent landmark
        # observations.
        pose_id = len(traj_estimate)
        pose_f = Landmark_Associator.apply_odom_step_2d(odom_measurement,
                                                        traj_estimate[-1])
        landmark_measurements,\
        global_landmarks,\
        n_landmarks,associated_cropped_bb = Landmark_Associator.process_landmarks_at_pose(pose_f,
                                                                    pose_id,
                                                                    landmarks,
                                                                    landmark_measurements,
                                                                    global_landmarks,
                                                                    associated_cropped_bb,
                                                                    n_landmarks)

        # new_cropped_bboxes.append(new_cropped_bbox)
        landmark_measurements = np.array(landmark_measurements)
        return landmark_measurements, n_landmarks, associated_cropped_bb
