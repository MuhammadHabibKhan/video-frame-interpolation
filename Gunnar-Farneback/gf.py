import cv2
import numpy as np
import os

def warp_frame(prev_frame, flow, alpha):
    h, w = flow.shape[:2]
    coords_x, coords_y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    # Compute interpolated coordinates
    map_x = (coords_x + flow[..., 0] * alpha).astype(np.float32)
    map_y = (coords_y + flow[..., 1] * alpha).astype(np.float32)

    # Ensure mappings stay within bounds to avoid black artifacts
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)
    
    # Warp the frame using remap with BORDER_REPLICATE
    interpolated_frame = cv2.remap(prev_frame, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return interpolated_frame

def interpolate_frame_gf(input_folder, output_folder, n_intermediate_frames=1):
    os.makedirs(output_folder, exist_ok=True)
    
    # Sort files numerically to avoid misordering
    frame_files = sorted(os.listdir(input_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    for i in range(len(frame_files) - 1):
        frame1 = cv2.imread(os.path.join(input_folder, frame_files[i]))
        frame2 = cv2.imread(os.path.join(input_folder, frame_files[i + 1]))
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 
            pyr_scale=0.3, levels=5, winsize=25, 
            iterations=5, poly_n=7, poly_sigma=1.5, 
            flags=0
        )

        # # Save the original frame
        # cv2.imwrite(os.path.join(output_folder, f"frame_{2*i:04d}.png"), frame1)

        # Generate and save intermediate frames
        alphas = [k / (n_intermediate_frames + 1) for k in range(1, n_intermediate_frames + 1)]
        for j, alpha in enumerate(alphas):
            interpolated_frame = warp_frame(frame1, flow, alpha)
            cv2.imwrite(os.path.join(output_folder, f"frame_{2*i:04d}.png"), interpolated_frame)

        # Save the original frame
        cv2.imwrite(os.path.join(output_folder, f"frame_{2*i+1:04d}.png"), frame1)

    # Save the last frame
    last_frame = cv2.imread(os.path.join(input_folder, frame_files[-1]))
    cv2.imwrite(os.path.join(output_folder, f"frame_{2*(len(frame_files)-1):04d}.png"), last_frame)

    print(f"Frames and interpolated frames saved in '{output_folder}'.")

# Usage
input_folder = "../data/low_12_fps"
output_folder = "../data/gunnar-farneback/high_24_fps"
interpolate_frame_gf(input_folder, output_folder)
