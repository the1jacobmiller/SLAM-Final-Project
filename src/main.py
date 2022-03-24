import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

from WaymoOD_Parser import WaymoOD_Parser
from Factor_Graph_SLAM import Factor_Graph_SLAM

sigma_p0 = 2.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path to tfrecord file')
    parser.add_argument(
        '--method',
        default='default',
        help='choices are default, pinv, qr, lu, qr_colamd, lu_colamd')
    parser.add_argument('--plot_R', action='store_true')
    parser.add_argument('--plot_traj_and_landmarks', action='store_true')
    args = parser.parse_args()

    # Parse the data to retrieve odom and landmark measurements for all frames
    # along with the ground truth trajctory and landmarks.
    odom_measurements, landmark_measurements, gt_traj, gt_landmarks = WaymoOD_Parser.parse(args.data)
    n_frames = len(gt_traj)

    # Make sure the data is of the correct type and shape
    assert len(landmark_measurements) == n_frames
    assert odom_measurements.shape == (n_frames-1,3)
    assert gt_traj.shape == (n_frames,3)
    assert gt_landmarks.shape[1] == 2

    SLAM = Factor_Graph_SLAM(args.method)
    p0 = gt_traj[0] + np.random.normal(0, sigma_p0)

    traj, landmarks, R = None, None, None
    for i in range(1,n_frames):
        start_time = time.time()

        # Solve the factor graph SLAM problem with frames 0 to i
        traj, landmarks, R, A, b = SLAM.run(odom_measurements[:i],
                                            landmark_measurements[:i],
                                            p0)

        runtime = time.time() - start_time
        print('Iteration', i, 'took', runtime, 's')

    if args.plot_R and R is not None:
        plt.spy(R)
        plt.show()

    # Visualize the final result
    if args.plot_traj_and_landmarks:
        SLAM.plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
