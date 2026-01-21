import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog,messagebox
from PIL import Image,ImageTk

#Interactive_GUI 
class Sketch_GUI:
    '''
    The Interactive GUI class
    
    Purpose:
    To create a window with 2 buttons:
    1.Select Image : Allows the user to select image from the computer
    2.Save Sketch : Allows the user to save the sketch  in the computer
    NOTE : This works only for pencil sketch (image must be given in BGR format as input)

    '''
    def __init__(self,root):
        self.root=root
        self.root.title("pencil sketch interactive gui")

        self.original_img=None
        self.processed_img=None

        #blur slider
        self.label_blur = tk.Label(root, text="Blur Kernel Size:",font=("Arial",20))
        self.label_blur.pack()
        self.slider_blur = tk.Scale(root, from_=1, to=99, orient=tk.HORIZONTAL, command=self.update_preview,length=400)
        self.slider_blur.set(21) 
        self.slider_blur.pack(pady=2)

        self.preview_label = tk.Label(root)
        self.preview_label.pack(pady=2)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=2)

        #load button
        self.btn_load = tk.Button(
            btn_frame,
            text="Select Image",
            command=self.load_image,
            font=("Arial", 14),
            padx=20,
            pady=2
        )
        self.btn_load.pack(side=tk.LEFT, padx=10)

        #save button
        self.btn_save = tk.Button(
            btn_frame,
            text="Save Sketch",
            command=self.save_image,
            font=("Arial", 14),
            padx=20,
            pady=2
        )
        self.btn_save.pack(side=tk.LEFT, padx=10)

    def load_image(self):
        #allows only files of certain type to open up
        file_path = filedialog.askopenfilename(title="Select image",
                                               filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            loaded_img=cv2.imread(file_path)
            if loaded_img is None:
                messagebox.showerror("error","image loading failed")
                return
            self.original_img = loaded_img
            self.update_preview()

    def update_preview(self, event=None):
        if self.original_img is None:
            return

        #blur_kernel    
        k = self.slider_blur.get()
        k = k if k % 2 != 0 else k + 1

        sketch_img=pencil_sketch(self.original_img,k)
        self.processed_img = sketch_img
        
        # Convert OpenCV BGR to Tkinter PhotoImage
        img = Image.fromarray(self.processed_img)
        img_w, img_h = img.size
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        target_w = int(screen_w * 0.9)
        target_h = int(screen_h * 0.9)
        ratio = min((target_w - 40) / img_w, (target_h - 150) / img_h)
        new_size = (int(img_w * ratio), int(img_h * ratio))

        # resize keeping the aspect ratio same
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

def sketch_algo(x,blur_kernel):
    '''
    Apply pencil sketch algo to a variable x (gray for pencil_sketch and v for color_sketch)
    
    Args:
    x (numpy array) : Variable to apply pencil sketch algorithm to
    blur_kernel (int) : Gaussian kernel size for blurring(must be odd)

    Returns:
    sketch_img (numpy array of the final sketch)

    '''
    inverted_x = 255 - x
    blurred_inverted_x = cv2.GaussianBlur(inverted_x,(blur_kernel,blur_kernel),0)
    blurred_x = 255 - blurred_inverted_x
    safe_denom = blurred_x.astype(np.float32) + 1e-6
    sketch_img = cv2.divide(x.astype(np.float32),safe_denom,scale=256)
    sketch_img = np.clip(sketch_img, 0, 255).astype(np.uint8)
    return sketch_img

def pencil_sketch(original,blur_kernel=21):
    '''
    Converts original image to pencil sketch
    
    Args:
    original : The original image from the User
    blur_kernel : Gaussian kernel size for blurring(must be odd)

    Returns:
    pencil_sketch_img 

    '''
    gray = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
    pencil_sketch_img = sketch_algo(gray,blur_kernel)
    return pencil_sketch_img

def color_sketch(original,blur_kernel=21):
    '''
    Converts original image to color sketch
    
    Args:
    original : The original image from the User
    blur_kernel : Gaussian kernel size for blurring(must be odd)

    Returns:
    color_sketch_img 

    '''
    hsv=cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    sketch_v = sketch_algo(v,blur_kernel)

    #slight desaturation for realistic effect
    s=(s * 0.7).astype('uint8')

    final_hsv = cv2.merge([h,s,sketch_v])
    color_sketch_img=cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
    return color_sketch_img

def display_image(original,sketch,save_path):
    '''
    Displays the original and sketch side by side using matplotlib
    
    Args:
    original : The original image from the User
    sketch : Sketch made by program

    '''
    original_rgb=cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
    fig,axs=plt.subplots(1,2)
    axs[0].imshow(original_rgb)

    shape=sketch.shape
    if len(shape)==2:
        axs[1].imshow(sketch,cmap='gray')
    else:
        sketch_rgb=cv2.cvtColor(sketch,cv2.COLOR_BGR2RGB)
        axs[1].imshow(sketch_rgb)

    axs[0].set_title("original image")
    axs[1].set_title("sketch")

    for ax in axs:
        ax.axis("off") 

    if save_path is not None:
        if not save_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            save_path += ".jpg"  
        cv2.imwrite(save_path,sketch)
        print("image saved")
    else:
        print("save path not mentioned")
    
    plt.tight_layout()
    plt.show()

def video_capture(video_path,save_path,type_of_sketch,blur_kernel=21):
    '''
    Shows Preview of the sketch video and saves it in the path mentioned
    Shows the progress bar for the processing of the video 
    Kill Switch to stop processing : 'q' 
    NOTE: Pressing q stops processing but still saves the processed video to the mentioned save_path

    Args:
    video_path : Path where the inpuot video is stored
    save_path : path where the sketch video has to be stored
    type_of_sketch : pencil sketch or color sketch
    blur_kernel : Gaussian kernel size for blurring(must be odd)

    '''
    cap = cv2.VideoCapture(video_path) 
    if not save_path.lower().endswith('.mp4'):
            save_path += ".mp4" 
    if not cap.isOpened():
        print("video is not found in the given path")
        return

    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if type_of_sketch==1:
        x=False
    else:
        x=True
    writer=cv2.VideoWriter(save_path,fourcc,fps,(frame_width,frame_height),isColor=x)

    cv2.namedWindow('sketch video', cv2.WINDOW_NORMAL)
    screen_w = int(10 * cv2.getWindowImageRect('sketch video')[2])
    screen_h = int(30 * cv2.getWindowImageRect('sketch video')[3])
    cv2.resizeWindow('sketch video', screen_w, screen_h)

    #progress bar is shows using tqdm
    with tqdm(total=total_frames,desc="processing video",leave=False) as pbar:
        while True:
            ret,frame = cap.read()          
            if not ret:
                break

            #pencil sketch video
            if type_of_sketch==1:
                pencil_sketch_img = pencil_sketch(frame,blur_kernel)
                writer.write(pencil_sketch_img)
                cv2.imshow('sketch video', pencil_sketch_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    tqdm.write("Processing stopped by user")
                    break
            
            #color sketch video
            elif type_of_sketch == 2:
                color_sketch_img=color_sketch(frame,blur_kernel)
                writer.write(color_sketch_img)
                cv2.imshow('sketch video', color_sketch_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    tqdm.write("Processing stopped by user")
                    break
            pbar.update(1)      #updating the progress bar

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"video saved to: {save_path}")

def main():
    try:
        option=int(input("""pick a number:
                        1.pencil_sketch
                        2.color_sketch
                        3.video_processing
                        4.interactive_gui"""))
    except ValueError:
        print("Enter a number between 1-4")
        return
    
    if option == 4:
        root=tk.Tk()
        root.lift()

        root.attributes('-topmost', True)
        root.after(100, lambda: root.attributes('-topmost', False))
        root.focus_force()

        #Aligning the window to the center of the screen
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

        win_w = int(root.winfo_screenwidth())
        win_h = int(root.winfo_screenheight())
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2

        root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        gui=Sketch_GUI(root)
        root.mainloop()
        return
    
    save_path=input("enter save path")
    input_path=input("enter input_path(Dont forget to add .jpeg or .mp4)")
    try:
        blur_kernel=int(input("enter blur kernel size - it must be an odd number"))
    except ValueError:
        print("blur kernel must be an integer")
        return

    while blur_kernel % 2 == 0 or blur_kernel < 0 :
        print("blur_kernel must be an odd positive number")
        blur_kernel=int(input("Enter blur kernel size"))

    if option == 1:
        original = cv2.imread(input_path)
        if original is None:
            print("original image not found in given path")
            return
        pencil_sketch_img=pencil_sketch(original,blur_kernel)
        display_image(original,pencil_sketch_img,save_path)

    elif option == 2:
        original=cv2.imread(input_path)
        if original is None:
            print("original image not found in given path")
            return
        color_sketch_img=color_sketch(original,blur_kernel)
        display_image(original,color_sketch_img,save_path)

    elif option == 3:
        try:
            type_of_sketch=int(input("""pick a number:
                                1.pencil sketch
                                2.color_sketch"""))
        except ValueError:
            print("Enter 1 or 2")
            return
        if type_of_sketch in [1,2]:
            video_capture(input_path,save_path,type_of_sketch,blur_kernel)
        else:
            print("choose numbers that are given in the options")

if __name__ == '__main__':
    main()

                     
