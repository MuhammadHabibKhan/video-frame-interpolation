import cv2
import numpy as np
import os

# Function to interpolate a frame using Lucas-Kanade Optical Flow & remap
def warp_frame_lk(prev_frame, next_frame, flow, alpha):
    h, w = flow.shape[:2]
    coords_x, coords_y = np.meshgrid(np.arange(w), np.arange(h))

    # Compute new coordinates using optical flow displacement
    map_x = (coords_x + flow[..., 0] * alpha).astype(np.float32)
    map_y = (coords_y + flow[..., 1] * alpha).astype(np.float32)

    # Warp the frame using optical flow
    interpolated_frame = cv2.remap(prev_frame, map_x, map_y, interpolation=cv2.INTER_LANCZOS4)
    return interpolated_frame

# Function to process frames from a folder and generate interpolated frames
def interpolate_frame_lk(input_folder, output_folder, n_intermediate_frames=1):
    os.makedirs(output_folder, exist_ok=True)
    frame_files = sorted(os.listdir(input_folder))

    # Lucas-Kanade parameters
    lk_params = dict(
        winSize=(21, 21),  
        maxLevel=3,        
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    for i in range(len(frame_files) - 1):
        frame1 = cv2.imread(os.path.join(input_folder, frame_files[i]))
        frame2 = cv2.imread(os.path.join(input_folder, frame_files[i + 1]))
        gray1, gray2 = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in [frame1, frame2]]

        # Detect good features to track
        prev_points = cv2.goodFeaturesToTrack(gray1, maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)

        # Calculate optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, prev_points, None, **lk_params)

        # Generate dense flow field
        flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
        for j, (pt, st) in enumerate(zip(prev_points, status)):
            if st:
                x1, y1 = pt.ravel()
                x2, y2 = next_points[j].ravel()
                flow[int(y1), int(x1)] = [x2 - x1, y2 - y1]

        # Save original frame
        cv2.imwrite(os.path.join(output_folder, f"frame_{2*i:04d}.png"), frame1)

        # Generate and save intermediate frames
        alphas = [k / (n_intermediate_frames + 1) for k in range(1, n_intermediate_frames + 1)]
        for alpha in alphas:
            interpolated_frame = warp_frame_lk(frame1, frame2, flow, alpha)
            cv2.imwrite(os.path.join(output_folder, f"frame_{2*i+1:04d}.png"), interpolated_frame)

    # Save the last frame
    last_frame = cv2.imread(os.path.join(input_folder, frame_files[-1]))
    cv2.imwrite(os.path.join(output_folder, f"frame_{2*(len(frame_files)-1):04d}.png"), last_frame)

    print(f"Frames and interpolated frames saved in '{output_folder}'.")


# Usage
input_folder = "../data/low_12_fps"
output_folder = "../data/lucas-kanade/high_24_fps"

interpolate_frame_lk(input_folder, output_folder)