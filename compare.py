import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def psnr(image1, image2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compute_ssim(image1, image2):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    """
    # Convert BGR to RGB
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Return SSIM with a smaller window size (3) for better handling of small images
    return ssim(image1_rgb, image2_rgb, multichannel=True, win_size=3)

def compare_specific_image(sample_image_name, output_image_name, samples_dir='./samples', output_dir='./output'):
    """
    Compare a specific image from the samples directory with the image from the output directory.
    """
    # Construct the paths for both sample and output images
    sample_path = os.path.join(samples_dir, sample_image_name)
    output_path = os.path.join(output_dir, output_image_name)

    # Read the images
    sample_image = cv2.imread(sample_path)
    output_image = cv2.imread(output_path)

    # Check if the images were loaded successfully
    if sample_image is None or output_image is None:
        raise FileNotFoundError(f"Could not read image {sample_image_name} or {output_image_name}. Ensure the images exist.")

    # Resize images if they are of different sizes
    if sample_image.shape != output_image.shape:
        output_image = cv2.resize(output_image, (sample_image.shape[1], sample_image.shape[0]))

    # Calculate SSIM and PSNR
    ssim_value = compute_ssim(sample_image, output_image)
    psnr_value = psnr(sample_image, output_image)

    # Return or print the comparison results
    print(f"Comparison for image: {sample_image_name}")
    print(f"PSNR: {psnr_value:.4f}")
    print(f"SSIM: {ssim_value:.4f}")
    return psnr_value, ssim_value

if __name__ == "__main__":
    compare_specific_image("0012(synthetic).png", "B_0012.png")
