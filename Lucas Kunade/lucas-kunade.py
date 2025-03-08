import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin')

import cv2
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time

# Function to calculate Temporal SSIM (TSSIM)
def calculate_tssim(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    tssim_value, _ = ssim(gray1, gray2, full=True)
    return tssim_value

# Function to interpolate frames using Lucas-Kanade Optical Flow
def interpolate_frame_lk(prev_frame, next_frame, flow, alpha):
    h, w = flow.shape[:2]
    coords_x, coords_y = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate forward remap coordinates
    map_x = (coords_x + flow[..., 0] * alpha).astype(np.float32)
    map_y = (coords_y + flow[..., 1] * alpha).astype(np.float32)

    # Remap using bilinear interpolation
    interpolated_frame = cv2.remap(prev_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return interpolated_frame

# Initialize video capture
capture = cv2.VideoCapture("bridgerton_dance_12_fps_35_sec.mp4")

# Get video properties
fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create output video writer with increased FPS
n_intermediate_frames = 1  # Number of frames to generate between each pair of frames
output_fps = fps * (n_intermediate_frames + 1)
video_file_name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + "_output.avi"
out = cv2.VideoWriter(f"C:/vfi_video_outputs/{video_file_name}", fourcc, output_fps, (width, height))

# Read the first frame
ret, frame1 = capture.read()
if not ret:
    print("Failed to read video.")
    exit()

# Convert to grayscale
gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(21, 21),  # Size of the search window
    maxLevel=3,        # Number of pyramid levels
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)  # Termination criteria
)

# Metrics
psnr_values = []
ssim_values = []
mse_values = []
tssim_values = []

# Start measuring execution time
start_time = time.time()

while True:
    ret, frame2 = capture.read()
    if not ret:
        break

    # Convert to grayscale
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect good features to track
    prev_points = cv2.goodFeaturesToTrack(gray_frame1, maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)

    # Calculate optical flow using Lucas-Kanade
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(gray_frame1, gray_frame2, prev_points, None, **lk_params)

    # Generate dense flow from sparse points
    flow = np.zeros((height, width, 2), dtype=np.float32)
    for i, (pt, st) in enumerate(zip(prev_points, status)):
        if st:  # Only use valid points
            x1, y1 = pt.ravel()
            x2, y2 = next_points[i].ravel()
            flow[int(y1), int(x1)] = [x2 - x1, y2 - y1]

    # Write the original frame1 to the output video
    out.write(frame1)

    # Generate and write intermediate frames
    alphas = [i / (n_intermediate_frames + 1) for i in range(1, n_intermediate_frames + 1)]
    for alpha in alphas:
        # Interpolate frame using Lucas-Kanade optical flow
        interpolated_frame = interpolate_frame_lk(frame1, frame2, flow, alpha)

        # Calculate Temporal SSIM
        tssim = calculate_tssim(frame1, interpolated_frame)

        # Calculate performance metrics
        interpolated_gray = cv2.cvtColor(interpolated_frame, cv2.COLOR_BGR2GRAY)
        mse_value = np.mean((interpolated_gray - gray_frame2) ** 2)
        psnr_value = psnr(gray_frame2, interpolated_gray, data_range=255)
        ssim_value, _ = ssim(gray_frame2, interpolated_gray, full=True)

        # Append metrics to the lists
        tssim_values.append(tssim)
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

        # Write interpolated frame to output video
        out.write(interpolated_frame)

    # Update previous frame
    frame1 = frame2
    gray_frame1 = gray_frame2

# Stop measuring execution time
end_time = time.time()

# Release resources
capture.release()
out.release()
cv2.destroyAllWindows()

# Print average metrics
print(f"Average MSE: {np.mean(mse_values):.4f}")
print(f"Average PSNR: {np.mean(psnr_values):.4f} dB")
print(f"Average SSIM: {np.mean(ssim_values):.4f}")
print(f"Average TSSIM: {np.mean(tssim_values):.4f}")
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
