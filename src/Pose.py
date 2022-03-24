import math
import numpy as np
from scipy.spatial.transform import Rotation

from waymo_open_dataset import dataset_pb2 as open_dataset

class Pose:
    pose = np.zeros((7,))

    def __init__(self, transform):
        if isinstance(transform, np.ndarray):
            transform = transform.flatten()

        # The input is a 3D transformation matrix
        if len(transform) == 16:
            # Get the position
            self.pose[0] = transform[3]
            self.pose[1] = transform[7]
            self.pose[2] = transform[11]

            # Get the orientation
            R = np.array([[transform[0], transform[1], transform[2]],
                          [transform[4], transform[5], transform[6]],
                          [transform[8], transform[9], transform[10]]])
            orientation = Rotation.from_matrix(R)
            self.pose[3:] = orientation.as_quat()

        # The input is a 3D pose with a quaternion representation
        elif len(transform) == 7:
            self.pose = np.asarray(transform)

        # The input is a 2D transformation matrix
        elif len(transform) == 9:
            self.pose[0] = transform[2]
            self.pose[1] = transform[5]

            R = np.array([[transform[0], transform[1], 0.],
                          [transform[3], transform[4], 0.],
                          [0.,           0.,           1.]])
            self.pose[3:] = Rotation.from_matrix(R).as_quat()

        # The input is a 2D pose
        elif len(transform) == 3:
            self.pose[0] = transform[0]
            self.pose[1] = transform[1]
            self.pose[3:] = Rotation.from_euler('z', transform[2]).as_quat()

        else:
            print("Pose input has invalid length!")

    def getPose(self):
        return self.pose

    def getPosition(self):
        return self.pose[:3]

    def getQuaternion(self):
        return self.pose[3:]

    def getRotationMatrix(self):
        return Rotation.from_quat(self.getQuaternion()).as_matrix()

    def getEulerAngles(self):
        return Rotation.from_quat(self.getQuaternion()).as_euler('zyx')

    def getTransformationMatrix(self):
        R = self.getRotationMatrix()
        pos = self.getPosition()
        H = np.array([[R[0,0], R[0,1], R[0,2], pos[0]],
                      [R[1,0], R[1,1], R[1,2], pos[1]],
                      [R[2,0], R[2,1], R[2,2], pos[2]],
                      [0.,     0.,     0.,     1.]])
        return H

    def getPose2D(self):
        pos = self.getPosition2D()
        theta = self.getOrientation2D()
        return np.array([pos[0], pos[1], theta])

    def getPosition2D(self):
        return self.pose[:2]

    def getOrientation2D(self):
        return self.getEulerAngles()[0]

    def getRotationMatrix2D(self):
        theta = self.getOrientation2D()
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return R

    def getTransformationMatrix2D(self):
        R = self.getRotationMatrix2D()
        pos = self.getPosition2D()
        H = np.array([[R[0,0], R[0,1], pos[0]],
                      [R[1,0], R[1,1], pos[1]],
                      [0.,     0.,     1.]])
        return H
