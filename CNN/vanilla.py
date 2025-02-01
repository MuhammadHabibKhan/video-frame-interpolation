import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
from glob import glob

# CNN Model for Frame Interpolation
def create_interpolation_cnn():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 6)),  # Two consecutive frames concatenated
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')  # Output interpolated frame
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Simple GAN Model to Enhance Frame Quality
def create_gan():
    # Generator
    generator = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])

    # Discriminator
    discriminator = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Combined GAN
    discriminator.trainable = False
    gan_input = layers.Input(shape=(128, 128, 3))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = models.Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    return generator, discriminator, gan

# Data Loader for Vimeo-90K
def load_vimeo90k_data(dataset_path, limit=1000):
    triplets = []
    video_dirs = glob(os.path.join(dataset_path, '*'))[:limit]
    for vid_dir in video_dirs:
        frames = sorted(glob(os.path.join(vid_dir, '*.png')))
        if len(frames) >= 3:
            triplets.append((frames[0], frames[1], frames[2]))  # Frame A, Intermediate, Frame B
    return triplets

# Preprocessing Images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    return img / 255.0

# CNN Training
def train_cnn(model, triplets, epochs=10, batch_size=8):
    for epoch in range(epochs):
        np.random.shuffle(triplets)
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i+batch_size]
            X, Y = [], []
            for a, b, gt in batch:
                frame_a = preprocess_image(a)
                frame_b = preprocess_image(b)
                ground_truth = preprocess_image(gt)

                input_pair = np.concatenate((frame_a, frame_b), axis=2)
                X.append(input_pair)
                Y.append(ground_truth)

            X, Y = np.array(X), np.array(Y)
            loss = model.train_on_batch(X, Y)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

# GAN Training
def train_gan(generator, discriminator, gan, real_frames, epochs=10, batch_size=8):
    for epoch in range(epochs):
        for i in range(0, len(real_frames), batch_size):
            batch = real_frames[i:i+batch_size]
            real_images = np.array([preprocess_image(img) for img in batch])

            # Generate fake images
            noise = np.random.normal(0, 1, (len(real_images), 128, 128, 3))
            fake_images = generator.predict(noise)

            # Labels for real and fake images
            real_labels = np.ones((len(real_images), 1))
            fake_labels = np.zeros((len(real_images), 1))

            # Train discriminator
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

            # Train generator (via GAN)
            g_loss = gan.train_on_batch(noise, real_labels)

        print(f"Epoch {epoch+1}/{epochs}, D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, G Loss: {g_loss}")

# Extracting Frames from Video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (128, 128))
        frames.append(frame)
    cap.release()
    return frames

# Interpolating Frames Using CNN
def interpolate_frames(frames, model):
    interpolated_frames = []
    for i in range(len(frames) - 1):
        input_pair = np.concatenate((frames[i], frames[i + 1]), axis=2)
        input_pair = np.expand_dims(input_pair, axis=0) / 255.0
        interpolated = model.predict(input_pair)[0] * 255
        interpolated_frames.append(interpolated.astype(np.uint8))
    return interpolated_frames

# Enhancing Frames Using GAN
def enhance_frames(frames, generator):
    enhanced_frames = []
    for frame in frames:
        input_frame = np.expand_dims(frame, axis=0) / 255.0
        enhanced = generator.predict(input_frame)[0] * 255
        enhanced_frames.append(enhanced.astype(np.uint8))
    return enhanced_frames

# Saving the Output Video
def save_video(frames, output_path, fps=30):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

# Main Workflow
def main(video_path, output_path, dataset_path):
    # Load Vimeo-90K Dataset
    triplets = load_vimeo90k_data(dataset_path)

    # Train CNN Model
    cnn_model = create_interpolation_cnn()
    train_cnn(cnn_model, triplets, epochs=5)

    # Train GAN Model
    real_frames = [t[1] for t in triplets]  # Intermediate ground truth frames
    generator, discriminator, gan = create_gan()
    train_gan(generator, discriminator, gan, real_frames, epochs=5)

    # Inference
    frames = extract_frames(video_path)
    interpolated_frames = interpolate_frames(frames, cnn_model)
    enhanced_frames = enhance_frames(interpolated_frames, generator)

    save_video(enhanced_frames, output_path)

# Example usage
# main('input_video.mp4', 'output_video.mp4', '/path/to/vimeo90k/dataset')
