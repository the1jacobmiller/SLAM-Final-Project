import unittest

import numpy as np

import src.Landmark_Associator as la


class TestDistanceFunctions(unittest.TestCase):
    def test_euclidean(self):
        landmark = [2, -1]
        prev_landmark = [-2, 2]
        self.assertEqual(la.Landmark_Associator.get_euclidean_distance(prev_landmark, landmark), 5)

class TestLandmarkAssociator(unittest.TestCase):
    def test_first_iteration(self):
        landmarks = [[[1,1],[2,2],[3,3]]]
        prev_landmarks = []
        traj_estimate = [[0,0,np.pi]]
        landmark_measurements, n_landmarks = la.Landmark_Associator.associate_landmarks(prev_landmarks, landmarks,traj_estimate, odom_measurement=None, sigma_landmark=None)
        self.assertEqual(n_landmarks,3)
        self.assertTrue((landmark_measurements == [[0, 0, -1, -1],[0, 1, -2, -2],[0, 2, -3, -3]]).all())

    def test_second_iteration(self):
        #run first iteration same as above
        landmarks = [[[1, 1], [2, 2], [3, 3]]]
        prev_landmarks = []
        traj_estimate = [[0, 0, np.pi]]
        landmark_measurements, n_landmarks = la.Landmark_Associator.associate_landmarks(prev_landmarks, landmarks,
                                                                                        traj_estimate,
                                                                                        odom_measurement=None,
                                                                                        sigma_landmark=None)
        self.assertEqual(n_landmarks, 3)
        self.assertTrue((landmark_measurements == [[0, 0, -1, -1], [0, 1, -2, -2], [0, 2, -3, -3]]).all())

        #We should only add one onto the list here
        prev_landmarks = [[1, 1], [2, 2], [3, 3]]
        landmarks = [[[1, 1], [5,5]]]
        traj_estimate = [[0, 0, np.pi],[6, 6, np.pi]]

        landmark_measurements, n_landmarks = la.Landmark_Associator.associate_landmarks(prev_landmarks, landmarks,
                                                                                        traj_estimate,
                                                                                        odom_measurement=None,
                                                                                        sigma_landmark=None)
        self.assertEqual(n_landmarks, 2)
        self.assertTrue((landmark_measurements == [[0, 0, -1, -1],[1,1,-5,-5]]).all())




if __name__ == '__main__':
    unittest.main()