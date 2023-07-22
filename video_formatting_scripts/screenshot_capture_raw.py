import cv2
import os
import glob
import logging

# Function to take a screenshot of a video frame
def take_screenshot(video_name,video_path, sc_path):
    #video_path = os.getcwd()     #r'C:\Users\mbarut\Desktop\videos'
    try:
        # Open the video file using cv2.VideoCapture
        video_file = f"{video_name}.mp4"  # Replace with the actual video file extension if needed
        cap = cv2.VideoCapture(os.path.join(video_path,video_file))
        
        if not cap.isOpened():
            raise Exception(f"Video '{video_file}' not found or unable to open.")

        # Capture a frame from the video
        ret, frame = cap.read()
        if not ret:
            raise Exception(f"Failed to read frame from '{video_file}'.")

        # Save the frame as a screenshot image
        screenshot_file = f"{video_name}.png"
        cv2.imwrite(os.path.join(sc_path,screenshot_file), frame)

        # Release the video capture object
        cap.release()

        print(f"Screenshot taken for {video_name}.")
    except Exception as e:
        print(f"Failed to take screenshot for {video_name}: {e}")

# Paths
video_paths = ['videos_p1','videos_p2','videos_p3']
main_path = r'C:\Users\mbarut\Desktop'
screen_path = r'C:\Users\mbarut\Desktop\screenshot_file'

log = logging.getLogger('screenshot_capture')
for path in video_paths:
    video_path = os.path.join(main_path,path)
    log.info(f'Current Path: {video_path}')
    os.chdir(video_path)
    video_names = glob.glob('*.mp4')
    for video_name in video_names:
        video_name = video_name.replace('.mp4','')
        take_screenshot(video_name,video_path,screen_path)
