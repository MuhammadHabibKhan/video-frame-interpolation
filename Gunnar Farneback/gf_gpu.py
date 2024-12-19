import cv2
import numpy as np

# Check if CUDA is available
if not cv2.cuda.getCudaEnabledDeviceCount():
    print("No GPU found or OpenCV is not compiled with CUDA support.")
    exit()

# Initialize video capture
capture = cv2.VideoCapture("bridgerton_dance.mp4")

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
    numLevels=5,  # Number of pyramid levels
    pyrScale=0.5,  # Pyramid scale
    fastPyramids=False,  # Use fast pyramids or not
    winSize=15,  # Averaging window size
    numIters=3,  # Number of iterations at each pyramid level
    polyN=5,  # Size of pixel neighborhood for polynomial expansion
    polySigma=1.1,  # Gaussian std deviation for expansion
    flags=0  # Operation flags (e.g., for flow initialization)
)

# Process video
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

    # Compute magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Update HSV mask
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to RGB
    rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    # Display the result
    cv2.imshow('Optical Flow (GPU)', rgb_representation)

    # Handle keypresses
    key = cv2.waitKey(20) & 0xFF
    if key == ord('e'):
        break
    elif key == ord('s'):
        cv2.imwrite('Optical_image_gpu.png', frame2)
        cv2.imwrite('HSV_converted_image_gpu.png', rgb_representation)

    # Update previous frame
    gpu_prvs = gpu_next

# Release resources
capture.release()
cv2.destroyAllWindows()
