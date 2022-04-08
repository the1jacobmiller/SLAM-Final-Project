import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
from scipy.spatial.transform import Rotation
from Pose import Pose

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

class WaymoOD_Parser:
    # Tune/change these
    p0_noise = [0.0, 0.0, 0.0] # std dev of x,y,theta
    odom_noise = [0.1, 0.1, np.pi/180.] # std dev of x,y,theta
    landmark_noise = [0.01, 0.01] # std dev of x,y
    gps_noise = [1.0, 1.0] # x,y
    gps_update_freq = 1.0 # Hz

    @staticmethod
    def parse(file, max_frames=np.inf):
        '''
        Parses the Waymo Open Dataset for factor graph SLAM inputs.

        \param file: the file name to retrieve data from
        \param max_frames: the maximum number of frames to return

        \return odom_measurements: [delta_x, delta_y, delta_theta] of the robot's
        pose global in coordinates between each frame. Shape (n_frames,3)
        \return landmarks: [distance_x, distance_y] of the landmark to the robot
        for each landmark in a given frame. Each row of the list corresponds to
        a single frame. A given frame may have 0 or multiple landmarks
        \return gt_traj: the ground truth trajectory that the robot follows
        \return gt_landmarks: the ground truth positions of the landmarks
        '''
        tf.enable_eager_execution()
        dataset = tf.data.TFRecordDataset(file, compression_type='')

        odom_measurements = []
        landmarks = []
        labeled_landmarks = []
        gps_measurements = []
        gt_traj = []
        gt_landmarks = []

        prev_frame = None
        prev_gps_time = None
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            if prev_frame is not None:
                odom_measurements.append(WaymoOD_Parser.getOdomMeasurement(frame,
                                                                           prev_frame))
            landmarks.append(WaymoOD_Parser.getLandmarks(frame))
            labeled_landmarks.append(WaymoOD_Parser.getLabeledLandmarks(frame))

            gt_pose = WaymoOD_Parser.getGroundTruthPose2D(frame)
            gt_traj.append(gt_pose)

            frame_time = frame.timestamp_micros/1e6
            if prev_gps_time is None or \
               frame_time - prev_gps_time >= (1./WaymoOD_Parser.gps_update_freq):
                pose_id = len(gt_traj)-1
                gps_position = gt_pose[:2] + np.random.normal([0.,0.], WaymoOD_Parser.gps_noise)
                gps_measurements.append([pose_id, gps_position[0], gps_position[1]])
                prev_gps_time = frame_time

            prev_frame = frame
            if len(gt_traj) >= max_frames:
                break

        gt_landmarks = WaymoOD_Parser.getGroundTruthLandmarks(gt_traj, labeled_landmarks)
        p0 = np.asarray(gt_traj[0]) + np.random.normal([0.,0.,0.], WaymoOD_Parser.p0_noise)

        return p0, np.asarray(odom_measurements), landmarks, \
               np.asarray(gps_measurements), np.asarray(gt_traj), \
               np.asarray(gt_landmarks)

    @staticmethod
    def getOdomMeasurement(frame, prev_frame):
        pose_t1 = WaymoOD_Parser.getGroundTruthPose2D(frame)
        pose_t0 = WaymoOD_Parser.getGroundTruthPose2D(prev_frame)

        delta_pose = pose_t1 - pose_t0
        delta_pose[2] = WaymoOD_Parser.warp2pi(delta_pose[2])

        # Add noise to the measurement
        odom = np.asarray(delta_pose) + np.random.normal([0.,0.,0.], WaymoOD_Parser.odom_noise)

        return odom

    @staticmethod
    def getLandmarks(frame):
        '''
        Gets the relative positions of the landmarks in the designated frame.

        \param frame: the frame to retrieve the landmarks from, of type protobuf Frame

        \return landmarks: [distance_x, distance_y] of the landmark to the robot
        for each landmark in a given frame. Each row of the list corresponds to
        a single frame. A given frame may have 0 or multiple landmarks
        '''
        landmarks = []

        for laser_label in frame.laser_labels:
            if laser_label.type == laser_label.TYPE_SIGN:
                landmark_rel_pos = np.array([laser_label.box.center_x,
                                             laser_label.box.center_y])

                # Add noise to the measurement
                landmark_rel_pos = landmark_rel_pos + \
                                   np.random.normal([0.,0.], WaymoOD_Parser.landmark_noise)

                landmarks.append(landmark_rel_pos)

        return landmarks

    @staticmethod
    def getLabeledLandmarks(frame):
        '''
        Gets the relative positions of the landmarks in the designated frame,
        with their associated ID.

        \param frame: the frame to retrieve the landmarks from, of type protobuf Frame

        \return labeled_landmarks: [distance_x, distance_y, ID] of the landmark
        to the robot for each landmark in a given frame. A given frame may have
        0 or multiple landmarks
        '''
        labeled_landmarks = []

        for laser_label in frame.laser_labels:
            if laser_label.type == laser_label.TYPE_SIGN:
                labeled_landmarks.append(laser_label)

        return labeled_landmarks

    @staticmethod
    def getGroundTruthPose2D(frame):
        '''
        Gets the 2D pose of the robot in global coordinates from the designated
        frame.

        \param frame: the frame to retrieve the landmarks from, of type protobuf Frame

        \return pose2D: [x, y, theta] of the robot in the designated frame
        '''
        pose2D = Pose(frame.pose.transform)
        return pose2D.getPose2D()

    @staticmethod
    def getGroundTruthLandmarks(gt_traj, labeled_landmarks):
        '''
        Gets the global positions of each unique landmark.

        \param gt_traj: the ground truth trajectory that the robot follows
        \param labeled_landmarks: [distance_x, distance_y] of the landmark
        to the robot for each landmark in a given frame. A given frame may have
        0 or multiple landmarks

        \return gt_landmarks: the ground truth positions of the landmarks
        '''
        n_frames = len(gt_traj)
        landmark_ids = []
        gt_landmarks = [] # format: [x, y] in global frame

        for i in range(n_frames):
            landmarks_frame_i = labeled_landmarks[i]
            vehicle_pose = Pose(gt_traj[i])

            unique_landmarks = []
            for landmark in landmarks_frame_i:
                if landmark.id not in landmark_ids:
                    unique_landmarks.append(landmark)

            for landmark in unique_landmarks:
                landmark_ids.append(landmark.id)

                landmark_rel_pos = np.array([landmark.box.center_x, landmark.box.center_y, 1])
                H = vehicle_pose.getTransformationMatrix2D()
                landmark_global_pos = (H @ landmark_rel_pos)[:2]
                gt_landmarks.append(landmark_global_pos)

        return gt_landmarks

    @staticmethod
    def warp2pi(angle_rad):
        '''
        \param angle_rad Input angle in radius
        \return angle_rad_warped Warped angle to [-\pi, \pi].
        '''
        return (angle_rad + np.pi) % (2.*np.pi) - np.pi
