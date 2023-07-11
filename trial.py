import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
image = cv2.imread('Trial.jpg')

#print(image.shape)
area = np.array([[201,831],[285,965],[1691,981],[1545,413]],np.int32)
color = (255, 0, 0)
thickness = 2

image = cv2.polylines(image, [area],
                      True, color, thickness)
# Displaying the image
while(1):
     
    cv2.imshow('image', image)
    if cv2.waitKey(20) & 0xFF == 27:
        break
         
cv2.destroyAllWindows()




'''
def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")

def save_frame(video_path, gap=5):
    name = video_path.split('\\')[-1]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = 0
    gap = int(fps*gap)

    #print(f'FPS of the video : {fps}')
    os.chdir(r'C:\Users\mbarut\Desktop\dataset')
    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break

        if idx == 0:
            
            cv2.imwrite(f"{name}_{idx}.png", frame)
        else:
            if idx % gap == 0:
                cv2.imwrite(f"{name}_{idx}.png", frame)
                #print(f"{save_path}/{name}_{idx}.png")

        idx += 1

if __name__ == "__main__":
    PATH = r'C:\Users\mbarut\Desktop\Eds kamera'
    SAVE_PATH = r'C:\Users\mbarut\Desktop'
    video_paths = glob(f'{PATH}/*')
    save_dir = 'dataset'

    save_path = os.path.join(SAVE_PATH, save_dir)
    create_dir(save_path)

    for path in tqdm(video_paths,total=len(video_paths)):
        save_frame(path, gap=5)
'''