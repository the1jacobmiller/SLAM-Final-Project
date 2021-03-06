import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

from WaymoOD_Parser import WaymoOD_Parser
from Factor_Graph_SLAM import Factor_Graph_SLAM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path to tfrecord file')
    parser.add_argument(
        '--method',
        default='default',
        help='choices are default, pinv, qr, lu, qr_colamd, lu_colamd')
    parser.add_argument('--n_dims', type=int, default=2,
        help='[int] number of pose/landmark dimensions to use for SLAM')
    parser.add_argument('--n_frames', default=np.inf,
        help='[int] number of frames to process for SLAM')
    parser.add_argument('--plot_R', action='store_true')
    parser.add_argument('--plot_traj_and_landmarks', action='store_true',
                        default=True)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    np.random.seed(1)

    # check argument values
    assert(0 < args.n_dims < 4)

    # Parse the data to retrieve odom and landmark measurements for all frames
    # along with the ground truth trajctory and landmarks.
    p0, odom_measurements, landmark_measurements, gps_measurements, \
        gt_traj, gt_landmarks = WaymoOD_Parser.parse(args.data,
                                                     max_frames=float(args.n_frames))
    n_frames = len(gt_traj)

    # Make sure the data is of the correct type and shape
    assert len(landmark_measurements) == n_frames
    assert odom_measurements.shape == (n_frames-1,3)
    assert gt_traj.shape == (n_frames,3)
    assert gt_landmarks.shape[1] == 2

    SLAM = Factor_Graph_SLAM(args.method, dimensions=args.n_dims)

    traj, landmarks, R = None, None, None
    iter_runtimes = []
    total_traj_start = time.time()
    for i in range(2,n_frames):
        start_time = time.time()

        # only want GPS data up until current frame
        gps_indices = np.where(gps_measurements[:,0] < i)[0]

        # Solve the factor graph SLAM problem with frames 0 to i
        # Note: there are landmark measurements at p0, but the first odom
        # measurement is between p0 and p1. Because of this, there should be
        # row in odom_measurements than landmark_measurements.
        traj, landmarks, R, A, b, init_traj = SLAM.run(odom_measurements[:i-1],
                                                       landmark_measurements[:i],
                                                       gps_measurements[gps_indices],
                                                       p0)
        runtime = time.time() - start_time
        iter_runtimes.append(runtime)
        print(f'Iteration {i} took {runtime:.5f} s')

        if args.debug:
            SLAM.plot_traj_and_landmarks(traj, landmarks, gps_measurements, gt_traj,
                                         gt_landmarks, init_traj, p_init=p0)

    runtime = time.time() - total_traj_start
    print('The full trajectory took', runtime, 's')
    print('The avg. runtime per frame was', runtime/n_frames)

    plt.plot(np.arange(2,n_frames), iter_runtimes, 'b-')
    plt.xlabel("Frame")
    plt.ylabel("Runtime (s)")
    plt.show()

    if args.plot_R and R is not None:
        plt.spy(R)
        plt.show()

    # Visualize the final result
    if args.plot_traj_and_landmarks:
        SLAM.plot_traj_and_landmarks(traj, landmarks, gps_measurements, gt_traj,
                                     gt_landmarks, init_traj, p_init=p0)
