SCOPE OF THIS PROJECT:
To design and implement a sketch generation system that converts BGR images and videos into pencil sketch and color sketch respresentations using computer vision

This project supports:
1) Pencil sketch generation for images
2) Color sketch generation for images
3) Real time sketch processing for videos
4) Interactive GUI for pencil sketch generation from images

The interactive GUI allows the user to:
1) Use a slider to choose the Gaussian blur kernel size
2) Provide real time updates on the sketch as the slider is moved
3) Option to load the image from the computer
4) Option to save the sketch in the computer

This program is designed to:
1) Work for both potrait and landscape images
2) Handle real time user interaction
3) Handle possible errors
4) Provide early termination of the video using a kill switch

Resizing and Alignment of the popups are included in the program

HOW TO RUN THE PROGRAM:
1) The user is given 4 choices :
        1.Pencil sketch
        2.Color sketch
        3.Video processing
        4.interactive GUI
The user is expected to pick a number that matches the type of sketching or processing they wish to perform

2) PENCIL SKETCH and COLOR SKETCH:
    The user is expected to give the input and save paths and the blur kernel(must be odd) when the respective input statements are prompted
    The original image and the sketch will be displayed side to side

3) VIDEO PROCESSING:
    The user is expected to give the input and save paths and the blur kernel(must be odd) when the respective input statements are prompted

    Now the user is given 2 choices:
    1.Pencil sketch
    2.Color sketch

    The user is expected to pick the number that matches their desired type of sketch generation for the video

    Preview of the sketch video will pop up and the progress bar of the video processing is shown in vs code terminal

    In case the user wants to kill the video processing - user can press 'q' to quit
    video processed up until the moment will be saved in the given path

4) Interactive GUI:
    A GUI window pops up from which the User is expected to press the 'Select Image' button and choose a BGR image from the computer and adjust the blur kernel using the slider

    The sketch will be displayed in the window keeping the aspect ratio same

    Now the user can choose to save the image by opting the 'Save Sketch' button

    Adjusting the slider provides real time sketch updates

DEPENDENCIES:
The program uses the following libraries:
1) OpenCV(cv2)   -   Image and Video Processing
2) Numpy         -   Numerical Operations and array generation
3) Matplotlib    -   Image Visualizations
4) tqdm          -   Progress bar in Video Processing
5) Tkinter       -   GUI framework
6) PILLOW(PIL)   -   Image handling for GUI

OBSERVATIONS:
1) Pencil sketches clearly highlight the contours of the image
2) Increasing blur kernel value results in smoother sketches but noise is increased
3) Desaturation improves realism in color sketches
4) The interative GUI enables real time visual comparison of the sketches helping users choose the optimal blur kernel size

CHALLENGES FACED:
1) Following the intuitive understanding of the sketch algorithm
2) Automatic resizing the images,videos and GUI window to fit the screen
3) Error Handling
4) Conversion of Color spaces to ensure compatability between different libraries
5) Progress bar output formatting (using tqdm.write() instead of print statements)

ERROR HANDLING IMPLEMENTED:
1) File not found and invalide format handling for images and videos
2) Safe handling of division operators ( Data type safety and division by zero)
3) Guassian blur kernel being odd positive number
4) Save path Validation
5) Kill switch for long video processing
6) Ensuring only the numbers in the choices given are chosen