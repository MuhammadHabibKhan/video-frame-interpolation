import os
# Setup CUDA for OpenCV
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin')

import cv2
import numpy as np


# Check if CUDA is available
if not cv2.cuda.getCudaEnabledDeviceCount():
    print("No GPU found or OpenCV is not compiled with CUDA support.")
    exit()

# Initialize video capture
capture = cv2.VideoCapture("bridgerton_dance.mp4")

# Get video properties
fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Output video writer
out = cv2.VideoWriter("output_3.avi", fourcc, fps * 2, (width, height))  # Increase FPS by 2x

# Read the first frame
ret, frame1 = capture.read()
if not ret:
    print("Failed to read video.")
    exit()

# Convert to grayscale and upload to GPU
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gpu_prvs = cv2.cuda_GpuMat()
gpu_prvs.upload(prvs)

# Create GPU Farneback optical flow object
farneback_gpu = cv2.cuda_FarnebackOpticalFlow.create(
    numLevels=5,
    pyrScale=0.5,
    fastPyramids=False,
    winSize=20,
    numIters=10,
    polyN=5,
    polySigma=1.1,
    flags=0
)

# Function to interpolate a middle frame
def interpolate_frame(frame1, frame2, flow, alpha=0.5):
    h, w = frame1.shape[:2]

    # Generate a grid of pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    # Calculate forward and backward flow for interpolation
    flow_x = alpha * flow[..., 0]
    flow_y = alpha * flow[..., 1]

    # Warp the frame using flow
    map_x = (grid_x + flow_x).astype(np.float32)
    map_y = (grid_y + flow_y).astype(np.float32)
    mid_frame = cv2.remap(frame1, map_x, map_y, interpolation=cv2.INTER_CUBIC)

    return mid_frame

# Process video and generate output
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

    # Write the original frame1 to the output video
    out.write(frame1)

    # Example: Generate n intermediate frames between frame1 and frame2
    n = 2  # Number of intermediate frames
    alphas = [i / (n + 1) for i in range(1, n + 1)]  # Generate alpha values

    # Generate and save intermediate frames
    for alpha in alphas:
        mid_frame = interpolate_frame(frame1, frame2, flow, alpha=alpha)
        out.write(mid_frame)

    # Interpolate middle frame
    # mid_frame = interpolate_frame(frame1, frame2, flow)

    # Write original frame and interpolated frame to output
    # out.write(frame1)
    # out.write(mid_frame)

    # Update variables
    frame1 = frame2
    gpu_prvs = gpu_next

# Write the last frame to the video
out.write(frame1)

# Release resources
capture.release()
out.release()
cv2.destroyAllWindows()
