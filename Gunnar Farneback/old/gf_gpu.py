import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin')

import cv2
import time
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

psnr_values = []
ssim_values = []
mse_values = []

# Function to calculate Temporal SSIM (TSSIM) on GPU
def calculate_tssim_gpu(gpu_frame1, gpu_frame2, stream):

    # Convert frames to grayscale on GPU
    gpu_gray1 = cv2.cuda.cvtColor(gpu_frame1, cv2.COLOR_BGR2GRAY, stream=stream)
    gpu_gray2 = cv2.cuda.cvtColor(gpu_frame2, cv2.COLOR_BGR2GRAY, stream=stream)
    
    # Download grayscale frames to CPU for SSIM calculation (if GPU SSIM isn't available)
    gray1 = gpu_gray1.download(stream=stream)
    gray2 = gpu_gray2.download(stream=stream)

    mse_value = np.mean((gpu_gray1 - gpu_gray2) ** 2)
    psnr_value = psnr(gpu_gray2, gpu_gray1, data_range=255)
    ssim_value, _ = ssim(gpu_gray2, gpu_gray1, full=True)

    mse_values.append(mse_value)
    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)
    
    # Compute SSIM
    # tssim_value, _ = ssim(gray1, gray2, full=True)
    # return tssim_value

# Function to calculate Flow Consistency Error on GPU
def calculate_flow_consistency_gpu(gpu_forward_flow, gpu_backward_flow, stream, threshold=1.0):
    h, w = gpu_forward_flow.size()
    
    # Create a mesh grid directly on GPU
    coords_x, coords_y = np.meshgrid(np.arange(w), np.arange(h))
    coords_x = cv2.cuda_GpuMat()
    coords_y = cv2.cuda_GpuMat()
    coords_x.upload(coords_x.astype(np.float32), stream=stream)
    coords_y.upload(coords_y.astype(np.float32), stream=stream)
    
    # Warp pixel coordinates using forward and backward flow
    map_x_fwd = coords_x + gpu_forward_flow[..., 0]
    map_y_fwd = coords_y + gpu_forward_flow[..., 1]
    map_x_bwd = map_x_fwd + gpu_backward_flow[..., 0]
    map_y_bwd = map_y_fwd + gpu_backward_flow[..., 1]
    
    # Compute flow consistency error
    error_x = map_x_bwd - coords_x
    error_y = map_y_bwd - coords_y
    error = cv2.cuda.magnitude(error_x, error_y, stream=stream)
    flow_error = np.mean(error.download(stream=stream)[error.download() < threshold])
    return flow_error

# Function to interpolate frame directly on GPU
def interpolate_frame_gpu(gpu_frame1, gpu_flow, alpha, stream):
    h, w = gpu_frame1.size()

    # Download the flow to CPU to extract flow components (needed for remapping)
    flow_cpu = gpu_flow.download(stream=stream)
    flow_x = flow_cpu[..., 0]
    flow_y = flow_cpu[..., 1]

    # Generate remap coordinates
    coords_x, coords_y = np.meshgrid(np.arange(h), np.arange(w))
    map_x = (coords_x + flow_x * alpha).astype(np.float32)
    map_y = (coords_y + flow_y * alpha).astype(np.float32)

    # Upload remap coordinates back to GPU
    gpu_map_x = cv2.cuda_GpuMat()
    gpu_map_y = cv2.cuda_GpuMat()
    gpu_map_x.upload(map_x, stream=stream)
    gpu_map_y.upload(map_y, stream=stream)

    # Perform remapping on the GPU
    gpu_interpolated_frame = cv2.cuda.remap(
        gpu_frame1, gpu_map_x, gpu_map_y, interpolation=cv2.INTER_CUBIC, stream=stream
    )

    return gpu_interpolated_frame


# Check for CUDA support
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

# Output video settings
n_intermediate_frames = 1  # Number of intermediate frames to generate
output_fps = fps * (n_intermediate_frames + 1)
video_file_name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + "_output.avi"
out = cv2.VideoWriter(f"C:/vfi_video_outputs/{video_file_name}", fourcc, output_fps, (width, height))

# Read the first frame
ret, frame1 = capture.read()
if not ret:
    print("Failed to read video.")
    exit()

# Upload first frame to GPU
gpu_frame1 = cv2.cuda_GpuMat()
gpu_frame1.upload(frame1)

# Convert to grayscale for optical flow
gpu_prvs = cv2.cuda.cvtColor(gpu_frame1, cv2.COLOR_BGR2GRAY)

# Create Farneback optical flow object on GPU
farneback_gpu = cv2.cuda_FarnebackOpticalFlow.create(
    numLevels=6, pyrScale=0.7, fastPyramids=False, winSize=25, numIters=15, polyN=7, polySigma=1.5, flags=0
)

# Create CUDA streams
n_streams = 4
streams = [cv2.cuda_Stream() for _ in range(n_streams)]

# Process video
flow_consistency_errors = []
start_time = time.time()

while True:
    ret, frame2 = capture.read()
    if not ret:
        break

    # Upload next frame to GPU
    gpu_frame2 = cv2.cuda_GpuMat()
    gpu_frame2.upload(frame2)

    # Convert to grayscale on GPU
    gpu_next = cv2.cuda.cvtColor(gpu_frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    gpu_flow = farneback_gpu.calc(gpu_prvs, gpu_next, None)

    # Download flow back to CPU
    # flow = gpu_flow.download()

    # Compute backward flow for flow consistency
    # backward_flow = farneback_gpu.calc(gpu_next, gpu_prvs, None).download()  # Backward flow

    # Write original frame to output
    out.write(frame1)

    # Generate intermediate frames
    alphas = [i / (n_intermediate_frames + 1) for i in range(1, n_intermediate_frames + 1)]
    for alpha in alphas:
        gpu_interpolated_frame = interpolate_frame_gpu(gpu_frame1, gpu_flow, alpha, streams[0])
        interpolated_frame = gpu_interpolated_frame.download()

        # Calculate flow consistency
        # flow_error = calculate_flow_consistency_gpu(flow, backward_flow, streams[0])
        # flow_consistency_errors.append(flow_error)

        # Temporal SSIM (TSSIM)
        tssim = calculate_tssim_gpu(gpu_frame1, gpu_interpolated_frame, streams[0])
        # tssim_values.append(tssim)

        # Write interpolated frame to output
        out.write(interpolated_frame)

    # Update previous frame
    frame1 = frame2
    gpu_frame1 = gpu_frame2
    gpu_prvs = gpu_next

# Release resources
capture.release()
out.release()
cv2.destroyAllWindows()

end_time = time.time()

# Output metrics

# print(f"Average TSSIM: {np.mean(tssim_values):.4f}")

print(f"Average MSE: {np.mean(mse_values):.4f}")
print(f"Average PSNR: {np.mean(psnr_values):.4f} dB")
print(f"Average SSIM: {np.mean(ssim_values):.4f}")
# print(f"Average Flow Consistency Error: {np.mean(flow_consistency_errors):.4f}")

print(f"Total Execution Time: {end_time - start_time:.2f} seconds")