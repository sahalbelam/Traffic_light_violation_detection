## Traffic Light Violation Detection System

This project implements a system to automatically detect traffic light violations at intersections.

 **Functionality**

The system performs the following tasks:

-   **Real-time video capture:** Continuously captures traffic flow from a camera.
-   **Vehicle detection:** Identifies and tracks vehicles within the video feed.
-   **Traffic light detection and recognition:** Locates the traffic light and classifies its color (red, yellow, green).
-   **Stop line detection:** Identifies the white line on the road that vehicles must stop behind.
-   **Violation detection:** Determines if a vehicle crosses the stop line while the light is red.

**Prerequisites:**

-   Python 3.x
-   OpenCV library ([https://opencv.org/](https://opencv.org/))
-   YOLOv5
