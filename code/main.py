import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.stats import norm

BOX_HALF_WIDTH = 100
BOX_HALF_HEIGHT = 100
TIME = 60

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.5,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a directory to save frames
frame_dir = "frames"
os.makedirs(frame_dir, exist_ok=True)

# Take the first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("Error: Failed to capture initial frame.")
    cap.release()
    exit()

mid_y = old_frame.shape[0] // 2
mid_x = old_frame.shape[1] // 2
pt0 = (mid_x - BOX_HALF_WIDTH, mid_y - BOX_HALF_HEIGHT)
pt1 = (mid_x + BOX_HALF_WIDTH, mid_y + BOX_HALF_HEIGHT)
offset_x = mid_x - BOX_HALF_WIDTH
offset_y = mid_y - BOX_HALF_HEIGHT

# for aimming
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.rectangle(frame, pt0, pt1, (0, 255, 45), thickness=3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('w'):
        break

# Take the first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("Error: Failed to capture initial frame.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_gray = old_gray[mid_y - BOX_HALF_HEIGHT: mid_y + BOX_HALF_HEIGHT, mid_x - BOX_HALF_WIDTH: mid_x + BOX_HALF_WIDTH]
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Initialize the array to store average distances and timestamps
average_distances = []
timestamps = []

start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = frame_gray[mid_y - BOX_HALF_HEIGHT: mid_y + BOX_HALF_HEIGHT, mid_x - BOX_HALF_WIDTH: mid_x + BOX_HALF_WIDTH]

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Calculate distances between corresponding points
    distances = np.linalg.norm(good_new - good_old, axis=1)


    # Record the average distance and the timestamp
    avg_distance = np.mean(distances)
    timestamp = time.time() - start_time
    average_distances.append(avg_distance)
    timestamps.append(timestamp)

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a = int(a) + offset_x
        b = int(b) + offset_y
        c = int(c) + offset_x
        d = int(d) + offset_y
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)
    img = cv2.rectangle(img, pt0, pt1, (150, 20, 0), thickness=3)
    img = cv2.putText(img, str(round(TIME - timestamp, 2)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,100,0))

    # Save the frame with markers
    frame_filename = os.path.join(frame_dir, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, img)
    frame_count += 1

    cv2.imshow('frame', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27 or (time.time() - start_time) > TIME:  # ESC key to break or 10 seconds elapsed
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release the video capture
cap.release()
cv2.destroyAllWindows()

# Convert lists to numpy arrays
average_distances = np.array(average_distances)
timestamps = np.array(timestamps)

# Perform DFT on the average distances
dft = np.fft.fft(average_distances)
frequencies = np.fft.fftfreq(len(dft), d=(timestamps[1] - timestamps[0]))

# Plot the original frequency spectrum
plt.figure()
plt.title("Original Frequency Spectrum")
plt.plot(frequencies, np.abs(dft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

# Find the index of the maximum frequency component
max_freq_index = np.argmax(np.abs(dft))
max_freq = frequencies[max_freq_index]

# Create a normal distribution centered at the max frequency
std_dev = 1.0  # Standard deviation for the normal distribution
normal_dist = norm.pdf(frequencies, loc=max_freq, scale=std_dev)

# Apply the normal distribution as a filter to the DFT
dft_filtered = dft * normal_dist
print("max freq", max_freq*60)
print("mean freq", np.sum(dft_filtered) / (dft_filtered.size - 1) * 60)

# Plot the filtered frequency spectrum
plt.figure()
plt.title("Filtered Frequency Spectrum")
plt.plot(frequencies, np.abs(dft_filtered))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

# Perform inverse DFT to get the filtered time series
average_distances_filtered = np.fft.ifft(dft_filtered)

# Plot the original and filtered time series
plt.figure()
plt.plot(timestamps, average_distances, label='Original')
plt.plot(timestamps, average_distances_filtered, label='Filtered', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Average Distance')
plt.legend()
plt.title("Original and Filtered Time Series")
plt.show()

# Plot filtered time series
plt.figure()
plt.plot(timestamps, average_distances_filtered, label='Filtered', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Average Distance')
plt.legend()
plt.title("Filtered Time Series")
plt.show()

# Compile frames into a video
frame_size = (old_frame.shape[1], old_frame.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, frame_size)

for i in range(frame_count):
    frame_filename = os.path.join(frame_dir, f"frame_{i:04d}.png")
    frame = cv2.imread(frame_filename)
    out.write(frame)

out.release()
print("Video saved as output.avi")
