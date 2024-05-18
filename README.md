_Introduction_
This repo is the result of the development and implementation of a computer vision system designed to measure patient breathing rates. Using keypoint matching techniques, displacement tracking, and frequency domain analysis, we developed a method to monitor and analyze respiratory patterns. This system employs OpenCV’s goodFeaturesToTrack function to identify keypoints on a QR code placed on the patient's belly, which allows for precise tracking of respiratory movements.

_How to use run_

1. Run code/main.py with libraries installed.
2. Print a QR pattern on a paper or cloth then place them on the patient's belly
3. Aim the webcam to point to the QR code. Make sure that the pattern is in the green box on the screen.
4. press "W" to start measure.
5. After 60 seconds, the frequency graph and cleaned graph are shown.
6. The predicted frequency will be printed to console.
