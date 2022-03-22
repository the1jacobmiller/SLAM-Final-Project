import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

class WaymoOD_Parser:

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
        gt_traj = []
        gt_landmarks = []

        prev_frame = None
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            odom_measurements.append(WaymoOD_Parser.getOdomMeasurement(frame, prev_frame))
            landmarks.append(WaymoOD_Parser.getLandmarks(frame))
            labeled_landmarks.append(WaymoOD_Parser.getLabeledLandmarks(frame))
            gt_traj.append(WaymoOD_Parser.getGroundTruthPose2D(frame))

            if len(gt_traj) >= max_frames:
                break

        gt_landmarks = WaymoOD_Parser.getGroundTruthLandmarks(gt_traj, labeled_landmarks)

        return odom_measurements, landmarks, gt_traj, gt_landmarks

    @staticmethod
    def getOdomMeasurement(frame, prev_frame):
        if prev_frame is None:
            return np.array([0., 0., 0.])

        pose_t1 = WaymoOD_Parser.getGroundTruthPose(frame)
        pose_t0 = WaymoOD_Parser.getGroundTruthPose(prev_frame)

        delta_pose = pose_t1 - pose_t0
        delta_pose[2] = WaymoOD_Parser.warp2pi(delta_pose[2])

        return delta_pose

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

        # TODO(jacob): get landmarks from frame

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

        # TODO(jacob): get landmarks with labels from frame

        return labeled_landmarks

    @staticmethod
    def getGroundTruthPose2D(frame):
        '''
        Gets the 2D pose of the robot in global coordinates from the designated
        frame.

        \param frame: the frame to retrieve the landmarks from, of type protobuf Frame

        \return pose2d: [x, y, theta] of the robot in the designated frame
        '''
        return WaymoOD_Parser.transform3DtoPose2D(frame.pose.transform)

    @staticmethod
    def getGroundTruthLandmarks(gt_traj, labeled_landmarks):
        '''
        Gets the global positions of each unique landmark.

        \param gt_traj: the ground truth trajectory that the robot follows
        \param labeled_landmarks: [distance_x, distance_y, ID] of the landmark
        to the robot for each landmark in a given frame. A given frame may have
        0 or multiple landmarks

        \return gt_landmarks: the ground truth positions of the landmarks
        '''
        gt_landmarks = []

        # TODO(jacob): get the ground truth landmark positions

        return gt_landmarks

    @staticmethod
    def transform3DtoPose2D(transform3D):
        '''
        Converts a 3D transform protobuf to a 2D pose.

        \param transform3D: 3D transform protobuf

        \return pose2d: [x, y, theta] of the robot
        '''

        pose2d = [0., 0., 0.]

        # TODO(jacob): calculate the 2D pose

        return pose2d

    @staticmethod
    def warp2pi(angle_rad):
        '''
        \param angle_rad Input angle in radius
        \return angle_rad_warped Warped angle to [-\pi, \pi].
        '''
        return (angle_rad + np.pi) % (2.*np.pi) - np.pi
