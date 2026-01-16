import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image ( image_path ) :
    """
    Load and preprocess image for circle detection .

    Args :
    image_path : Path to input image

    Returns :
    tuple : ( original_color , preprocessed_gray ) or (None , None )
    """
    # TODO : Implement preprocessing
    image=cv2.imread("image_path")
    if image is None:
       print("Original Image not found in given path")
       return None,None
    try:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    except Exception:
       print("grayscale conversion error")
       return None,None
    blurred=cv2.GaussianBlur(gray,(21,21),0)
    return image,blurred
    


 


def detect_circles ( gray_image , dp =1 , minDist =50 , param1 =50 ,
    param2 =30 , minRadius =10 , maxRadius =100) :
    """
    Detect circles using Hough Circle Transform .

    Args :
    gray_image : Preprocessed grayscale image
    dp: Inverse accumulator resolution ratio
    minDist : Minimum distance between circle centers
    param1 : Upper Canny threshold
    param2 : Accumulator threshold
    minRadius : Minimum circle radius
    maxRadius : Maximum circle radius

    Returns :
    numpy array of circles (x, y, radius ) or None
    """
    # TODO : Apply HoughCircles
    circles=cv2.HoughCircles(gray_image,cv2.HOUGH_GRADIENT,dp,minDist,
                             param1,param2,minRadius,maxRadius)
    return circles
       

def visualize_circles ( image , circles , save_path = None ) :
    """
    Draw detected circles on image and display .

    Args :
    image : Original color image
    circles : Array of detected circles
    save_path : Optional path to save annotated image
    """
    # TODO : Draw circles and labels
    result_image=image.copy()
    ID=1
    if circles is not None:
        circles=np.uint16(np.around(circles))
        for circle in circles[0,:]:
            cx,cy,radius=circle
            cv2.circle(result_image,(cx,cy),radius,(0,255,0),2)
            cv2.circle(result_image,(cx,cy),2,(0,0,255),3)
            label=f"ID={ID},radius={radius}"
            cv2.putText(result_image,label,(cx-radius,cy-radius-10),cv2.FONT_HERSHEY_COMPLEX,
                                            0.5,(255,0,0),1,cv2.LINE_AA)
            ID+=1
    
    #DISPLAY
    try:
        original_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    except Exception:
        print("BGR2RGB conversion on original image failed")
        return None,None
    
    fig,axs=plt.subplots(1,2)
    axs[0].imshow(original_rgb)
    axs[1].imshow(result_image)
    axs[0].set_title("ORIGINAL IMAGE")
    axs[1].set_title("RESULT IMAGE")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    if save_path is None:
       print("save path not mentioned")
    else:
       cv2.imwrite(save_path,result_image)


def calculate_statistics ( circles ) :
    """
    Calculate and display statistics about detected circles .
    
    Args :
    circles : Array of detected circles

    Returns :
    dict : Statistics dictionary
    """
    # TODO : Compute statistics
    statistics={}
    if circles is not None:
        min_rad=min(circles[0,:,2])
        max_rad=max(circles[0,:,2])
        avg_rad=circles[0,:,2].mean()
        statistics["Number of circles"]=len(circles[0])
        statistics["Min radius"]=min_rad
        statistics["Max radius"]=max_rad
        statistics["Avg radius"]=avg_rad
        return statistics




def main () :
    """ Main function ."""
    # TODO : Implement main workflow
    image_path=input("Enter image path:")
    original,blurred=preprocess_image(image_path)
    circles=detect_circles(blurred)
    visualize_circles(original,circles)
    print(calculate_statistics(circles))

 


if __name__ == '__main__':
    main ()