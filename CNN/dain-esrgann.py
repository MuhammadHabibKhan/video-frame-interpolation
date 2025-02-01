import cv2
import numpy as np
import torch
from torchvision import transforms
from DAIN import DAIN  # Assuming DAIN is available as a module
from ESRGAN import ESRGAN  # Assuming ESRGAN is available as a module

# Frame Extraction from Video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# Interpolating Frames Using Pre-trained DAIN Model
def interpolate_frames(frames, dain_model):
    interpolated_frames = []
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    for i in range(len(frames) - 1):
        frame_a = transform(frames[i]).unsqueeze(0)
        frame_b = transform(frames[i + 1]).unsqueeze(0)
        with torch.no_grad():
            interpolated = dain_model(frame_a, frame_b)
        interpolated_image = interpolated.squeeze().permute(1, 2, 0).numpy() * 255
        interpolated_frames.append(interpolated_image.astype(np.uint8))
    return interpolated_frames

# Enhancing Frames Using Pre-trained ESRGAN Model
def enhance_frames(frames, esrgan_model):
    enhanced_frames = []
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    for frame in frames:
        input_frame = transform(frame).unsqueeze(0)
        with torch.no_grad():
            enhanced = esrgan_model(input_frame)
        enhanced_image = enhanced.squeeze().permute(1, 2, 0).numpy() * 255
        enhanced_frames.append(enhanced_image.astype(np.uint8))
    return enhanced_frames

# Saving the Output Video
def save_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

# Main Workflow
def main(video_path, output_path):
    # Load Pre-trained Models
    dain_model = DAIN(pretrained=True)
    esrgan_model = ESRGAN(pretrained=True)

    # Inference
    frames = extract_frames(video_path)
    interpolated_frames = interpolate_frames(frames, dain_model)
    enhanced_frames = enhance_frames(interpolated_frames, esrgan_model)

    save_video(enhanced_frames, output_path)

# Example usage
# main('input_video.mp4', 'output_video.mp4')