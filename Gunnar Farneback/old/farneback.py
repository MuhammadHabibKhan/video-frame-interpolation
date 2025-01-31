import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin')

import cv2
import time
import numpy as np
from datetime import datetime

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Function to calculate Temporal SSIM (TSSIM)
def calculate_tssim(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    tssim_value, _ = ssim(gray1, gray2, full=True)
    return tssim_value

# Function to calculate Flow Consistency Error
def calculate_flow_consistency(forward_flow, backward_flow, threshold=1.0):
    h, w = forward_flow.shape[:2]
    
    # Create mesh grid for pixel coordinates
    coords_x, coords_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Warp pixel coordinates using forward flow
    map_x_fwd = (coords_x + forward_flow[..., 0]).astype(np.float32)
    map_y_fwd = (coords_y + forward_flow[..., 1]).astype(np.float32)
    
    # Warp pixel coordinates using backward flow
    map_x_bwd = (map_x_fwd + backward_flow[..., 0]).astype(np.float32)
    map_y_bwd = (map_y_fwd + backward_flow[..., 1]).astype(np.float32)
    
    # Compute flow consistency error
    error_x = map_x_bwd - coords_x
    error_y = map_y_bwd - coords_y
    error = np.sqrt(error_x**2 + error_y**2)
    
    # Calculate flow consistency error (mean error below threshold)
    flow_error = np.mean(error[error < threshold]) if np.any(error < threshold) else np.inf
    return flow_error

# Check if CUDA is available
if not cv2.cuda.getCudaEnabledDeviceCount():
    print("No GPU found or OpenCV is not compiled with CUDA support.")
    exit()

# Initialize video capture
capture = cv2.VideoCapture("bridgerton_dance_12_fps_35_sec.mp4")

# Get video properties
fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create output video writer with increased FPS
n_intermediate_frames = 2  # Number of frames to generate between each pair of frames
output_fps = fps * (n_intermediate_frames + 1)

video_file_name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + "_output.avi"
out = cv2.VideoWriter(f"C:/vfi_video_outputs/{video_file_name}", fourcc, output_fps, (width, height))

# Read the first frame
ret, frame1 = capture.read()
if not ret:
    print("Failed to read video.")
    exit()

# Convert to grayscale and upload to GPU
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gpu_prvs = cv2.cuda_GpuMat()
gpu_prvs.upload(prvs)

# Create mask
hsv_mask = np.zeros_like(frame1)
hsv_mask[..., 1] = 255

# Create GPU Farneback optical flow object
farneback_gpu = cv2.cuda_FarnebackOpticalFlow.create(
    numLevels=6,  # Number of pyramid levels
    pyrScale=0.7,  # Pyramid scale
    fastPyramids=False,  # Use fast pyramids or not
    winSize=25,  # Averaging window size
    numIters=15,  # Number of iterations at each pyramid level
    polyN=7,  # Size of pixel neighborhood for polynomial expansion
    polySigma=1.5,  # Gaussian std deviation for expansion
    flags=0  # Operation flags (e.g., for flow initialization)
)

# Function to interpolate frame
def interpolate_frame(frame1, frame2, flow, alpha):
    h, w = frame1.shape[:2]

    # Scale flow by alpha
    scaled_flow = flow * alpha

    # Generate remap coordinates
    coords_x, coords_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (coords_x + scaled_flow[..., 0]).astype(np.float32)
    map_y = (coords_y + scaled_flow[..., 1]).astype(np.float32)

    # Warp frame1 towards frame2 using flow
    interpolated_frame = cv2.remap(frame1, map_x, map_y, interpolation=cv2.INTER_CUBIC)

    return interpolated_frame

# Process video

psnr_values = []
ssim_values = []
mse_values = []
tssim_values = []
flow_consistency_errors = []

# Start measuring execution time
start_time = time.time()

while True:
    ret, frame2 = capture.read()
    if not ret:
        break

    # Convert to grayscale and upload to GPU
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gpu_next = cv2.cuda_GpuMat()
    gpu_next.upload(next)

    # Calculate optical flow on GPU
    gpu_flow = farneback_gpu.calc(gpu_prvs, gpu_next, None)

    # Download flow back to CPU
    flow = gpu_flow.download()

    # Compute backward flow for flow consistency
    backward_flow = farneback_gpu.calc(gpu_next, gpu_prvs, None).download()  # Backward flow

    # Write the original frame1 to the output video
    out.write(frame1)

    # Generate and write intermediate frames
    alphas = [i / (n_intermediate_frames + 1) for i in range(1, n_intermediate_frames + 1)]
    
    for alpha in alphas:
        interpolated_frame = interpolate_frame(frame1, frame2, flow, alpha=alpha)
        
        # Calculate Temporal SSIM
        tssim = calculate_tssim(frame1, interpolated_frame)

        # Calculate Flow Consistency Error
        flow_error = calculate_flow_consistency(flow, backward_flow)

        # Convert interpolated frame and original frame to grayscale
        interpolated_gray = cv2.cvtColor(interpolated_frame, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate performance metrics
        mse_value = np.mean((interpolated_gray - frame2_gray) ** 2)
        psnr_value = psnr(frame2_gray, interpolated_gray, data_range=255)
        ssim_value, _ = ssim(frame2_gray, interpolated_gray, full=True)

        # Append metrics to the lists
        tssim_values.append(tssim)
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        flow_consistency_errors.append(flow_error)

        # Write interpolated frame to output video
        out.write(interpolated_frame)

    # Update previous frame
    frame1 = frame2
    gpu_prvs = gpu_next

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
print(f"Average Flow Consistency Error: {np.mean(flow_consistency_errors):.4f}")
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")