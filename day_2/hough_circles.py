import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog,messagebox
from PIL import Image,ImageTk
from tqdm import tqdm

#Interactive GUI
class SketchGUI:
    '''
    The Interactive GUI class
    
    Purpose:
    To create a window with 7 sliders and 2 buttons:
    1) Sliders - To control the parameters of HOUGH CIRCLES
    2)Buttons - Load and Save button 

    '''
    def __init__(self,root):
        self.root = root
        self.root.title("Hough Circle Transform Interactive GUI")

        self.original_img = None
        self.processed_img = None

        #Making window size = screen size
        self.root.state("zoomed")

        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=20, pady=1)

        #Creating a container in main frame called row1
        row1 = tk.Frame(main_frame)
        row1.pack()

        #Adding the labels of the widgets
        tk.Label(row1, text="dp").grid(row=0, column=0, padx=8)
        tk.Label(row1, text="minDist").grid(row=0, column=1, padx=8)
        tk.Label(row1, text="param1").grid(row=0, column=2, padx=8)
        tk.Label(row1, text="param2").grid(row=0, column=3, padx=8)
        tk.Label(row1, text="minRadius").grid(row=0, column=4, padx=8)
        tk.Label(row1, text="maxRadius").grid(row=0, column=5, padx=8)
        tk.Label(row1,text="blur_kernel").grid(row=0,column=6,padx=8)

        #Adding the widgets
        self.slider_dp = tk.Scale(row1, from_=1, to=2, resolution=0.1,orient=tk.HORIZONTAL, command=self.update_preview)
        self.slider_dp.set(1)

        self.slider_minDist = tk.Scale(row1, from_=10, to=200,orient=tk.HORIZONTAL, command=self.update_preview)
        self.slider_minDist.set(50)

        self.slider_param1 = tk.Scale(row1, from_=10, to=300,orient=tk.HORIZONTAL, command=self.update_preview)
        self.slider_param1.set(100)

        self.slider_param2 = tk.Scale(row1, from_=10, to=150, orient=tk.HORIZONTAL, command=self.update_preview)
        self.slider_param2.set(30)

        self.slider_minRadius = tk.Scale(row1, from_=0, to=100,orient=tk.HORIZONTAL, command=self.update_preview)
        self.slider_minRadius.set(10)

        self.slider_maxRadius = tk.Scale(row1, from_=20, to=300,orient=tk.HORIZONTAL, command=self.update_preview)
        self.slider_maxRadius.set(100)

        #resolution kept at 2 to ensure odd numbers
        self.slider_blur=tk.Scale(row1,from_=1,to=100,resolution =2,orient=tk.HORIZONTAL, command=self.update_preview)
        self.slider_blur.set(21)

        self.slider_dp.grid(row=1, column=0, padx=10)
        self.slider_minDist.grid(row=1, column=1, padx=10)
        self.slider_param1.grid(row=1, column=2, padx=10)
        self.slider_param2.grid(row=1, column=3, padx=10)
        self.slider_minRadius.grid(row=1, column=4, padx=10)
        self.slider_maxRadius.grid(row=1, column=5, padx=10)
        self.slider_blur.grid(row=1,column=6,padx=10)
        
        self.preview_label = tk.Label(self.root)
        self.preview_label.pack(pady=1)

        #Adding another container for the buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=1)

        #Load button
        self.btn_load = tk.Button(
            btn_frame,
            text="Select Image",
            command=self.load_image,
            font=("Arial", 14),
            padx=20,
            pady=1
        )
        self.btn_load.pack(side=tk.LEFT, padx=10)

        #Save button
        self.btn_save = tk.Button(
            btn_frame,
            text="Save Image",
            command=self.save_image,
            font=("Arial", 14),
            padx=20,
            pady=1
        )
        self.btn_save.pack(side=tk.RIGHT, padx=10)
    
    def load_image(self):
        #Filter the files by filetypes for easy selection
        file_path = filedialog.askopenfilename(title="Select image",
                                               filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            loaded_img=cv2.imread(file_path)
            #Error handling for choosing wrong type of file
            if loaded_img is None:
                messagebox.showerror("error","image loading failed")
                return
            self.original_img = loaded_img
            self.update_preview()

    def update_preview(self, event=None):
        if self.original_img is None:
            return
        
        k = self.slider_blur.get()       
        dp=self.slider_dp.get()
        minDist=self.slider_minDist.get()
        param1=self.slider_param1.get()
        param2=self.slider_param2.get()
        minRadius=self.slider_minRadius.get()
        maxRadius=self.slider_maxRadius.get()

        #Parameter Validation:
        if minRadius >= maxRadius:
            messagebox.showerror("Parameter Error", "minRadius must be less than maxRadius")
            return

        if dp <= 0:
            messagebox.showerror("Parameter Error", "dp must be positive")
            return

        if param1 <= 0 or param2 <= 0:
            messagebox.showerror("Parameter Error", "param1 and param2 must be positive")
            return

        image,blurred = preprocess_image(self.original_img,k)
        self.processed_img , circles = detect_circles(blurred,image,dp,minDist,param1,param2,minRadius,maxRadius)
 
        rgb = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img_w, img_h = img.size

        # Resize keeping the aspect ratio same
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        target_w = int(screen_w * 0.9)
        target_h = int(screen_h * 0.9)
        ratio = min((target_w - 40) / img_w, (target_h - 150) / img_h)
        new_size = ( int(img_w * ratio) , int(img_h * ratio) )

        img = img.resize(new_size, Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        pad_x = max(0, (target_w - new_size[0]) // 2)
        pad_y = max(0, (target_h - new_size[1] - 150) // 2)

        self.preview_label.config(image=img_tk)
        self.preview_label.image = img_tk
        self.preview_label.pack(padx=pad_x, pady=pad_y)
    
    def save_image(self):
        if self.processed_img is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
            if save_path:
                cv2.imwrite(save_path, self.processed_img)
                print(f"saved to {save_path}")

def preprocess_image ( image ,blur_kernel ) :
    """
    Load and preprocess image for circle detection .

    Args :
    image : original image
    blur_kernel : Blur kernel size for gaussian blur

    Returns :
    tuple : ( original_image , blurred_gray_image )

    """
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray,(blur_kernel,blur_kernel),0)
    return image,blurred
    
def detect_circles ( blurred ,image, dp =1 , minDist =50 , param1 =50 ,
    param2 =30 , minRadius =10 , maxRadius =100) :
    """
    Detect circles using Hough Circle Transform and draw them in result_image .

    Args :
    blurred : Preprocessed grayscale image
    image : original image
    dp: Inverse accumulator resolution ratio
    minDist : Minimum distance between circle centers
    param1 : Upper Canny threshold
    param2 : Accumulator threshold
    minRadius : Minimum circle radius
    maxRadius : Maximum circle radius

    Returns :
    tuple : ( result_image , circles )
    """
    # TODO : Apply HoughCircles
    circles=cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,dp=dp,minDist=minDist,
                             param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    result_image=image.copy()
    ID=1
    if circles is not None:
        circles=np.uint16(np.around(circles))
        for circle in circles[0,:]:
            cx,cy,radius=circle
            size = classify_size(circles,radius)
            if size == "small":
                color = (0,255,0)
            elif size == "medium":
                color = (255,0,0)
            elif size == "large":
                color = (0,0,255)
            cv2.circle(result_image,(cx,cy),radius,color,2)
            cv2.circle(result_image,(cx,cy),2,(0,0,255),3)
            label=f"ID={ID},radius={radius},{size}"
            cv2.putText(result_image,label,(cx-radius+5,cy-radius-5),cv2.FONT_HERSHEY_COMPLEX,
                                            0.35,(255,0,0),1,cv2.LINE_AA)
            ID+=1
    return result_image , circles
       
def visualize_circles ( image ,result_image , save_path = None ) :
    """
    Display original image and result image side by side using matplotlib .

    Args :
    image : Original color image
    result_image : result image
    save_path : Optional path to save annotated image
    """
    #DISPLAY
    original_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result_rgb=cv2.cvtColor(result_image,cv2.COLOR_BGR2RGB)
    fig,axs=plt.subplots(1,2)
    axs[0].imshow(original_rgb)
    axs[1].imshow(result_rgb)
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
    statistics={}
    if circles is not None:
        min_rad=min(circles[0,:,2])
        max_rad=max(circles[0,:,2])
        avg_rad=circles[0,:,2].mean()
        statistics["Number of circles"]=len(circles[0])
        statistics["Min radius"]=min_rad
        statistics["Max radius"]=max_rad
        statistics["Avg radius"]=avg_rad
        for ID in range(len(circles[0])):
            statistics[ID+1] = circles[0][ID]
        return statistics
    else:
        return {"Number of circles": 0}

def classify_size(circles,radius):
    '''
    Classify sizes for the circles detected

    Args:
    circles : numpy array of the detected circle's position and radius
    radius : radius of the circle under consideration

    Returns:
    Classified size (small,medium or large)

    '''
    size = None
    if circles is not None:
        min_rad=min(circles[0,:,2])
        max_rad=max(circles[0,:,2])
    divider = ( max_rad - min_rad ) / 3

    if radius < min_rad + divider:
        size = "small"
    elif radius < min_rad +(2 * divider ):
        size = "medium"
    elif radius <= min_rad +(3 * divider ):
        size = "large"

    return size

def video_processing(video_path,save_path,blur_kernel):
    '''
    Shows Preview of the video and saves it in the path mentioned
    Shows the progress bar for the processing of the video 
    Kill Switch to stop processing : 'q' 
    NOTE: Pressing q stops processing but still saves the processed video to the mentioned save_path

    Args:
    video_path : Path where the inpuot video is stored
    save_path : path where the sketch video has to be stored
    blur_kernel : Gaussian kernel size for blurring(must be odd)

    '''
    cap = cv2.VideoCapture(video_path)
    if not save_path.lower().endswith('.mp4'):
        save_path += '.mp4'
    if not cap.isOpened():
        print("video is not found in given path")
        return
    
    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer=cv2.VideoWriter(save_path,fourcc,fps,(frame_width,frame_height))

    cv2.namedWindow("HOUGH CIRCLES video", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("HOUGH CIRCLES video",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    #progress bar is shown using tqdm
    with tqdm(total=total_frames,desc="processing video",leave=False) as pbar:
        while True:
            ret,frame = cap.read()          
            if not ret:
                break
            image , blurred = preprocess_image(frame,blur_kernel)
            result_image , circles = detect_circles(blurred,image)
            writer.write(result_image)
            cv2.imshow("HOUGH CIRCLES video", result_image)
            key = cv2.waitKey(1) & 0xFF

            #killswitch
            if key == ord('q'):
                tqdm.write("Processing stopped by user")
                break
            pbar.update(1)
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    if save_path is not None:
        print(f"video saved to: {save_path}")


def main():
    try:
        choice=int(input('''choose a number:
                        1.Interactive GUI
                        2.video_processing
                        3.Hough Circles Analysis
                        '''))
    except ValueError:
        print("Enter a number between 1-3")
        return
    
    if choice == 1:
        root=tk.Tk()

        #Popup
        root.lift()
        root.attributes('-topmost', True)
        root.after(100, lambda: root.attributes('-topmost', False))
        root.focus_force()

        gui=SketchGUI(root)
        root.mainloop()
        return
    input_path=input("Enter image path:")
    save_path=input("enter save path")

    try:
        blur_kernel=int(input("enter blur kernel size - it must be an odd number"))
    except ValueError:
        print("blur kernel must be an integer")
        return

    while blur_kernel % 2 == 0 or blur_kernel < 0 :
        print("blur_kernel must be an odd positive number")
        blur_kernel=int(input("Enter blur kernel size"))

    if choice == 2:
        if not save_path.lower().endswith(('.mp4')):
            save_path += '.mp4'
        video_processing(input_path,save_path,blur_kernel)
        

    elif choice == 3:
        save_path_statistics = input("enter save path for statistics(text file):")
        if not save_path_statistics.lower().endswith('.txt'):
            save_path_statistics += '.txt'
        if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            save_path += '.jpg'
        original=cv2.imread(input_path)
        if original is None:
            print("original image not found in given path")
            return
        original,blurred=preprocess_image(original,blur_kernel)
        result_image,circles=detect_circles(blurred,original)
        visualize_circles(original,result_image,save_path)
        statistics = calculate_statistics(circles)
        with open(save_path_statistics ,"w") as f:
            for k,v in statistics.items():
                if isinstance(v,np.ndarray):
                    f.write(f" {k} : x_coord={v[0]} , y_coord = {v[1]} , radius = {v[2]} \n")
                else:
                    f.write(f"{k} : {v} \n")
        


if __name__ == '__main__':
    main ()



