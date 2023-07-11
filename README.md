# Real-Time Object Measurement

This project implements real-time object dimension measurements using an image or a webcam. It utilises fundamental computer vision techniques and the OpenCV library for the same. Since this is my first time doing a computer vision project, I have had to rely on a lot of online resources and tutorials, all of which are listed below in the [Resources](#resources) section.

## Prerequisites

- Python 3.x
- OpenCV
- Numpy

## Installation

1. Clone this repository or download the code files.

2. Install the required dependencies using pip:

   ```bash
   pip install opencv-python numpy
   ```

## Usage

1. Open the `main.py` file in a Python editor or IDE.

2. Set the `webcam` variable to `True` if you want to use the webcam for detection, or set it to `False` to use an image file.

3. If you're using an image file, provide the file path in the `path` variable.

4. Adjust the video capture parameters if necessary. You can modify the brightness, width, and height settings using the OpenCV `cap.set()` function.

5. Run the script.

   ```bash
   python main.py
   ```

6. If using the webcam, the script will start capturing video frames. If using an image file, it will process the image.

7. The script will detect the A4-size paper in the frame or image and overlay a bounding box around it.

8. If the A4-size paper is detected, the script will warp the image to extract the paper and display it in a new window titled "A4". It will also detect any objects within the paper and highlight them with green polygons.

9. For each detected object, the script will calculate and display its width and height in centimeters.

10. The original frame or image will be displayed in a window titled "Original".

11. Press `Ctrl + C` to exit the program.

## Customisations

- You can adjust the `SCALE` variable to change the size of the image. The default value is `3`, which I found to be suitable for an A4-sized sheet.

- You can modify the parameters in the `getContours()` and `warpImage()` functions to fine-tune the contour detection and image warping process.

- The `thresholdArea` parameter in the `getContours()` function controls the minimum area required for a contour to be considered. You can adjust this value to filter out small or large objects.

- The `filter` parameter in the `getContours()` function determines the contour approximation method. The default value is `0`. Since, we want to detect rectangles, while calling the function, we will pass `filter=4` into the parameter.
- I have used the basic `cv2.CHAIN_APPROX_SIMPLE` method to get the approx contour lines.

- The `cannyThreshold` parameter in the `getContours()` function sets the threshold values for the Canny edge detection algorithm. You can adjust these values to change the sensitivity of the edge detection.

- Feel free to modify the code to suit your specific requirements or integrate it into your own projects :\)

## Resources
- [OpenCV Documentation](https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html)
- [GeeksForGeeks](https://www.geeksforgeeks.org/python-opencv-canny-function/)
- [Tutorials Point](https://www.tutorialspoint.com/opencv/)

## License

This project is licensed under the [MIT License](LICENSE).




