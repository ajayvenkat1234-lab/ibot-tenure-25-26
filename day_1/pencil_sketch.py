import cv2
import numpy as np
import matplotlib.pyplot as plt

def pencil_sketch(image_path, blur_kernel=21):
    """
    Convert an image to pencil sketch effect.
    
    Args:
        image_path (str): Path to input image
        blur_kernel (int): Gaussian blur kernel size (must be odd)
    
    Returns:
        tuple: (original_rgb, sketch) or (None, None) if error
    """
    # TODO: Implement the algorithm
    # Step 1: Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Original image not found in given path")
        return None,None
    
    # Step 2: Convert to grayscale
    try:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    except Exception:
        print("grayscale conversion error")
        return None,None
    
    # Step 3: Invert grayscale
    inverted = 255 - gray

    
    # Step 4: Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted,(blur_kernel,blur_kernel),0)
    
    # Step 5: Invert blurred image
    inverted_blur = 255 - blurred
    
    # Step 6: Divide and scale
    sketch = cv2.divide(gray, inverted_blur, scale=256)
    try:
        original_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    except Exception:
        print("BGR2RGB conversion on original image failed")
        return None,None
    
    return original_rgb,sketch


def display_result(original, sketch, save_path=None):
    """
    Display original and sketch side-by-side.
    
    Args:
        original: Original image (RGB)
        sketch: Sketch image (grayscale)
        save_path: Optional path to save the sketch
    """
    # TODO: Create matplotlib figure with 1 row, 2 columns
    fig,axs = plt.subplots(1,2)
    # Display original on left, sketch on right
    axs[0].imshow(original)
    axs[1].imshow(sketch,cmap='gray')
    # Add titles and remove axes
    axs[0].set_title("Original Image")
    axs[1].set_title("Sketch")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    # If save_path provided, save the sketch
    if save_path is None:
        print("save path not mentioned")
    else:
        cv2.imwrite(save_path,sketch)


def main():
    """Main function to run the pencil sketch converter."""
    # TODO: Get image path from user or command line
    image_path = input("Enter the image path:")
    save_path = input("Enter save path:")
    blur_kernel = int(input("blur kernel size (it must be an odd number)"))
    while blur_kernel % 2 == 0:
        print("blur_kernel must be an odd number")
        blur_kernel=int(input("Enter blur kernel size"))
    # Call pencil_sketch function
    original_rgb,sketch = pencil_sketch(image_path,blur_kernel)
    # Call display_result function
    if original_rgb is not None:       
        display_result(original_rgb,sketch,save_path)
    # Handle any errors
    



if __name__ == '__main__':
    main()