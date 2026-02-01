SCOPE OF THIS PROJECT:
To design and implement a circle detection system that identifies, analyzes, and visualizes circular objects in images and videos using the Hough Circle Transform

This project supports:
1) Circle detection in images using Hough Circle Transform
2) Circle detection in videos with real time preview
3) Interactive GUI for tuning Hough parameters in real time
4) Size classification and color coding of detected circles
5) Statistics generation in a text file
6) Real time sketch processing for videos

The interactive GUI allows the user to:
1) Use 7 sliders to control the Hough Circle Transform parameters and Gaussian blur kernel size
2) Provide real time updates on the detected circles as sliders are moved
3) Option to load the image from the computer
4) Option to save the result in the computer

This program is designed to:
1) Work for both portrait and landscape images
2) Handle real time user interaction
3) Handle possible errors
4) Provide early termination of the video using a kill switch
5) Classify detected circles into small, medium, and large categories with color coding

Resizing and Alignment of the popups are included in the program

HOW TO RUN THE PROGRAM:
1) The user is given 3 choices :
        1.Interactive GUI
        2.Video Processing
        3.Hough Circles Analysis
   The user is expected to pick a number that matches the type of processing they wish to perform

2) INTERACTIVE GUI:
    A GUI window pops up from which the user is expected to press the 'Select Image' button and choose a BGR image from the computer
    The user can adjust all 7 parameters using the sliders:
        - dp : Inverse ratio of accumulator resolution
        - minDist : Minimum distance between circle centers
        - param1 : Upper threshold for Canny edge detector
        - param2 : Accumulator threshold - lower values detect more circles
        - minRadius : Minimum circle radius in pixels
        - maxRadius : Maximum circle radius in pixels
        - blur_kernel : Gaussian blur kernel size(must be odd)

    The annotated result will be displayed in the window keeping the aspect ratio same
    Circles are color coded based on size : green(small),blue(medium),red(large)
    Each circle is labeled with its ID, radius, and size category

    Now the user can choose to save the image by opting the 'Save Image' button

    Adjusting any slider provides real time circle detection updates

3) VIDEO PROCESSING:
    The user is expected to give the input and save paths and the blur kernel(must be odd) when the respective input statements are prompted

    Preview of the detected circles video will pop up and the progress bar of the video processing is shown in the terminal

    In case the user wants to stop the video processing - user can press 'q' to quit
    video processed up until the moment will be saved in the given path

4) HOUGH CIRCLES ANALYSIS:
    The user is expected to give the input path, save path for the annotated image, save path for the statistics text file, and the blur kernel(must be odd) when the respective input statements are prompted

    The original image and the annotated result will be displayed side by side

    A statistics text file will be exported containing:
        - Total number of circles detected
        - Minimum, maximum, and average radius
        - Circle details: ID, x coordinate, y coordinate, and radius

DEPENDENCIES:
The program uses the following libraries:
1) OpenCV(cv2)  -   Image and Video Processing
2) Numpy        -   Numerical Operations and array generation
3) Matplotlib   -   Image Visualizations
4) tqdm         -   Progress bar in Video Processing
5) Tkinter      -   GUI framework
6) PILLOW(PIL)  -   Image handling for GUI

OBSERVATIONS:
1) Lowering param2 detects more circles but increases false positives
4) The interactive GUI enables real time tuning of all 7 parameters helping users find the optimal settings for any image
5) minDist is critical for handling overlapping circles - lowering it allows closer circles to be detected separately

CHALLENGES FACED:
1) Tuning the parameters of HoughCircles to work for all the images
6) Color coding and labeling circles based on their sizes as both dynamic and fixed thresholds have their own problems

ERROR HANDLING IMPLEMENTED:
1) File not found and invalid format handling for images and videos
2) Parameter validation - minRadius must be less than maxRadius
3) Gaussian blur kernel being an odd positive number
4) Save path validation and automatic extension appending for output files
5) Kill switch for long video processing
6) Ensuring only the valid numbers in the choices given are chosen
7) dp, param1, and param2 must be positive values

BONUS CHALLENGES COMPLETED:
1) Bonus 2 - Interactive GUI
        Full Tkinter GUI with sliders, live preview, Load and Save buttons, and automatic resized image display

2) Bonus 3 - Video Processing
        Frame by frame circle detection on video with real time preview, tqdm progress bar, and a kill switch

3) Bonus 4 - Size Classification
        Detected circles are classified into small, medium, or large based on radius and color coded into green,blue and red respectively