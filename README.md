# Enhanced Video Frame Interpolation using a Hybrid CNN-GAN Framework

Research-based Final Year Project for Bachelor's degree in Computer Science

------------------------------------------------------------------------------------------------------------

![image](https://github.com/user-attachments/assets/257740a8-4d30-453e-bf06-3e7f1d287de0)
Illustration of Frame Interpolation | [Source](https://commons.wikimedia.org/wiki/File:Motion_interpolation_example.jpg)

## Abstract

Video Frame Interpolation (VFI) is a critical task in computer vision, enabling smoother motion in videos by generating intermediate frames. Traditional optical flow-based methods, such as Lucas-Kanade and Gunnar Farneback, often fail to handle complex motion, occlusions, and fine details effectively, leading to noticeable artifacts. To overcome these limitations, I propose an enhanced VFI approach using a hybrid deep learning framework that combines Convolutional Neural Networks (CNN) and Generative Adversarial Networks (GANs).

In this project, I first implemented frame interpolation using traditional optical flow techniques to analyze their effectiveness and limitations. I then developed a deep learning-based solution incorporating the Super SloMo model, which utilizes CNN for more accurate motion estimation, and Real-ESRGAN, a GAN-based model for enhancing frame quality and restoring details. By integrating these approaches, I aim to significantly improve frame accuracy, temporal coherence, and overall visual fidelity.

This project is not intended to be the best VFI solution available but rather to demonstrate how a relatively lightweight deep learning-based framework can significantly outperform traditional optical flow methods while remaining computationally efficient. The performance of both approaches was compared using qualitative and quantitative metrics, highlighting the advantages of deep learning in video enhancement. The expected outcome of this project is a more robust and accessible VFI model that can be applied to slow-motion video generation, frame rate upscaling, and video restoration, showcasing the potential of deep learning in advancing video processing techniques.

## Results

#### Input Video @ 12.5 fps

https://github.com/user-attachments/assets/72ead80a-cfcb-4ab2-979e-773358ffa273


#### Output Video @ 25 fps 4K Upscaled

https://github.com/user-attachments/assets/bd55b3d6-3786-4aee-8528-c4ca5222b407


Note: The videos shown here are trimmed out of the original input and output vidoes present in the drive folder


| Method                     | PSNR   | SSIM  |
| -------------------------- | ------ | ----- |
| Lucas-Kanade               | 30.49  | 0.56  |
| Gunnar-Farneback           | 29.94  | 0.48  |
| Super-SloMo                | 30.92  | 0.83  |
| Super-SloMo + Real-ESRGAN  | 30.95  | 0.81  |

For more details on the results, refer to the documentation present in the Google Drive folder

## Links

[Demo Video](https://www.youtube.com/watch?v=W2cAjZULx2U) | [Google Drive](https://drive.google.com/drive/folders/1CQSXuuCh6Pmf6hjtWDWnntrJCrHuCaUr?usp=sharing)  

Note: The drive link contains a 'data' folder which has all the input and outputs of the 3 implementations done in this project. Refer to the 'Readme.txt' in the data folder for more details

## Pre-Requisites

- I recommend creating two environments, one for Lucas-Kanade, Gunnar-Farneback & Super-SloMo implementation and install the packages in requirements.txt using the command<br/>

   ``` pip install -r requirements.txt```

- The second enviroment should be created to run Real-ESRGAN as it has a dependency on an older version of pytorch which in turns requires an older version of numpy. Install the packages in requirements-gan.txt using the command<br/>

   ``` pip install -r requirements-gan.txt ```

- After installing the packages for Real-ESRGAN, develop the setup.py using the command<br/>

   ``` pip setup.py develop ```


Please refer to the offical repo of Real-ESRGAN linked at the bottom if any issue arises.

## System Specifications

- CPU: Intel Xeon E5-1680v4 | 8c 16t @ max 3.4ghz when all cores active
- RAM: 32GB DDR4 @ Quad Channel
- GPU: RTX 2060 6GB

## Directory Structure

1) Folder: Frame Utils

- frame.py --> frame and video handling code
- metrics.py --> metrics calculation code (psnr and ssim)
---------------------------------------------------------------------------------------------------------------

2) Folder: Gunnar Farneback

- gf.py --> Implementation of VFI using GF Optical Flow Estimation
---------------------------------------------------------------------------------------------------------------

3) Folder: Lucas Kanade

- optk_flow.py --> Implementation of VFI using LK Optical Flow Estimation
---------------------------------------------------------------------------------------------------------------

4) Folder: Super-SloMo

- eval.py --> Run Super-SloMo on provided video

   ##### Usage:

  ``` python eval.py data/input.mp4 --checkpoint=data/SuperSloMo.ckpt --output=data/output.mp4 --scale=4 ```

  Use ```python eval.py --help ``` for more details

- Download the model checkpoint [SuperSloMo.ckpt](https://drive.google.com/file/d/1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF/view) and place it in "data/"
---------------------------------------------------------------------------------------------------------------

5) Folder: Real-ESRGAN

- inference_realesrgan.py --> Run Real-ESRGAN on provided frames

   ##### Usage: 

        python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

   A common command: 

     ```python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance ```

- Following is the list of options available:

   >  -h                   show this help <br/>
   >  -i --input           Input image or folder. Default: inputs <br/>
   >  -o --output          Output folder. Default: results <br/>
   >  -n --model_name      Model name. Default: RealESRGAN_x4plus <br/>
   >  -s, --outscale       The final upsampling scale of the image. Default: 4 <br/>
   >  --suffix             Suffix of the restored image. Default: out <br/>
   >  -t, --tile           Tile size, 0 for no tile during testing. Default: 0 <br/>
   >  --face_enhance       Whether to use GFPGAN to enhance face. Default: False <br/>
   >  --fp32               Use fp32 precision during inference. Default: fp16 (half precision). <br/>
   >  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs.
  
   > Default: auto

- Download the model weights [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) that were used and place it under "weights/"
---------------------------------------------------------------------------------------------------------------

Note: Folder 4 and 5 only mentions the main file that was run to generate the results
      It also contains only the necessary files required to replicate the results of my project
      For other features, visit the Github repository for each model

  [SuperSloMo](https://github.com/avinashpaliwal/Super-SloMo) | 
  [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
