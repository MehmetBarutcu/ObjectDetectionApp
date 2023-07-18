import cv2
import os

# Function to take a screenshot of a video frame
def take_screenshot(video_name):
    video_path = os.getcwd()     #r'C:\Users\mbarut\Desktop\videos'
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
        cv2.imwrite(screenshot_file, frame)

        # Release the video capture object
        cap.release()

        print(f"Screenshot taken for {video_name}.")
    except Exception as e:
        print(f"Failed to take screenshot for {video_name}: {e}")

# Read the txt file and process each line
with open("videos_list.txt", "r", encoding='utf8') as file:
    for line in file:
        video_name = line.strip()
        take_screenshot(video_name)
