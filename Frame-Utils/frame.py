import os
# os.add_dll_directory('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin')

import cv2
import shutil

# Input video file path | MAKE SURE YOU'RE IN FRAME-UTILS workspace and not in any parent directory for the paths to work correctly
video_path = "../data/videos/car_original.mp4"
interpolated_video = "../data/videos/car_high.mp4"

# Folder to save frames / take data from

low_fps_folder = "../data/low_12_fps"
original_frames_folder = "../data/original_frames"

gan_folder = "../data/cnn-gan/results_gan"
high_fps_folder = "../data/cnn-gan/high_24_fps"
gan_folder_no_upscale = "../data/cnn-gan/results_gan_no_upscale"
interpolated_frames_folder = "../data/cnn-gan/interpolated_frames" # cnn interpolated frames
interpolated_gan_folder = "../data/cnn-gan/interpolated_gan_frames" # after gan applied
interpolated_gan_folder_no_upscale = "../data/cnn-gan/interpolated_gan_frames_no_upscale" # after gan applied , no upscaling for analysis and metric evaluation

lucas_kunade_high_folder = "../data/lucas-kanade/high_24_fps"
interpolated_frames_folder_lk = "../data/lucas-kanade/interpolated_frames"

gf_high_folder = "../data/gunnar-farneback/high_24_fps"
interpolated_frames_folder_gf = "../data/gunnar-farneback/interpolated_frames"

# Extract frames from video
def extractFrames(video_path, frames_folder):
    os.makedirs(frames_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        frame_filename = os.path.join(frames_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to '{frames_folder}'.")


def filterFrames(frames_folder, filtered_frames_folder, flag):
    # Folder to save selected frames
    os.makedirs(filtered_frames_folder, exist_ok=True)

    frame_files = sorted(os.listdir(frames_folder))  # Ensure proper order

    for index, frame_file in enumerate(frame_files):

        if (flag == "even"):
            
            if index % 2 == 0:  # Keep only even-indexed frames (0, 2, 4, ...) for interpolation
                src_path = os.path.join(frames_folder, frame_file)
                dst_path = os.path.join(filtered_frames_folder, frame_file)
                shutil.copy(src_path, dst_path)

        if (flag == "odd"):

            if index % 2 != 0:  # Keep only odd-indexed frames (1, 3, 5, ...) for ground truth
                src_path = os.path.join(frames_folder, frame_file)
                dst_path = os.path.join(filtered_frames_folder, frame_file)
                shutil.copy(src_path, dst_path)

    print(f"Filtered frames saved to '{filtered_frames_folder}'.")


# Reconstruct video from frames
def generateVideo(frames_folder, output_video, fps):

    frame_files = sorted(os.listdir(frames_folder))  # Ensure proper order
    
    if not frame_files:
        print("No frames found!")
        return

    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_folder, frame_file))
        out.write(frame)

    out.release()
    print(f"Reconstructed video saved as '{output_video}'.")


# Run the functions

# Extracting original frames, creating low fps video and seperating ground truth for comparison
# extractFrames(video_path, original_frames_folder)
# filterFrames(original_frames_folder, "low_12_fps", "even")
# filterFrames(original_frames_folder, "ground_truth", "odd")
# generateVideo(low_fps_folder, "car_low.mp4", fps=12.4286)

## CNN_GAN

# Extract interpolated frames to compare to ground truth
# extractFrames(interpolated_video, high_fps_folder)
# filterFrames(high_fps_folder, "interpolated_frames", "odd")

# Compare again but against gan improved interpolated frames
# generateVideo(gan_folder, "car_high_gan.mp4", 25)
# filterFrames(gan_folder, interpolated_gan_folder, "odd")

# Need non-upscaled version to compare metrics
# generateVideo(gan_folder_no_upscale, "car_high_gan_no_upscale.mp4", 25)
# filterFrames(gan_folder_no_upscale, interpolated_gan_folder_no_upscale, "odd")


## Lucas Kunade

# generateVideo(lucas_kunade_high_folder, "car_high_lk.mp4", 25)
# filterFrames(lucas_kunade_high_folder, interpolated_frames_folder_lk, "odd")


## Gunnar Farneback

# generateVideo(gf_high_folder, "car_high_gf.mp4", 25)

# using even to extract interpolated frames for gunnar farneback as for the video I had to place interpolated frames first because gf for some reason produced results that were off and made video jittery i.e car moving back and forth
# this will definetly produce metrics that would not represent the actual video as placing interpolated frames first resulted in a fairly decent output.
# filterFrames(gf_high_folder, interpolated_frames_folder_gf, "even")