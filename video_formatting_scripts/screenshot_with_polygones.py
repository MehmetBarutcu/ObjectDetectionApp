import cv2
import os
import json
import numpy as np
import os
import glob

video_path = r'C:\Users\mbarut\Desktop\videos'
screenshoot_path = r'C:\Users\mbarut\Desktop\sc2'
# Function to take a screenshot of a video frame
def take_screenshot(video_name,polygones):
    color = (255, 0, 0)
    thickness = 2

    try:
        # Open the video file using cv2.VideoCapture
        video_file = f"{video_name}.mp4"  # Replace with the actual video file extension if needed
        cap = cv2.VideoCapture(os.path.join(video_path,video_file))
        
        if not cap.isOpened():
            raise Exception(f"Video '{video_file}' not found or unable to open.")

        # Capture a frame from the video
        ret, frame = cap.read()
        for p in polygones:
             frame = cv2.polylines(frame, [np.array(p,np.int32)],
                        True, color, thickness)
        if not ret:
            raise Exception(f"Failed to read frame from '{video_file}'.")

        # Save the frame as a screenshot image
        screenshot_file = f"{video_name}.png"
        cv2.imwrite(os.path.join(screenshoot_path,screenshot_file), frame)

        # Release the video capture object
        cap.release()

        print(f"Screenshot taken for {video_name}.")
    except Exception as e:
        print(f"Failed to take screenshot for {video_name}: {e}")

# Read config file
with open('config3.json','r') as f:
    config = json.load(f)

# Read the txt file and process each line
os.chdir(video_path)
files = glob.glob(f'*.mp4')
for file in files:
    file = file.replace('.mp4','')
    polygones = list(config[file].values())
    video_name = file
    take_screenshot(video_name,polygones)