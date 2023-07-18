import os
import glob

path = os.getcwd()
# to store files in a list
files = set()
 
# dirs=directories
for (root, dirs, file) in os.walk(path):
    for f in file:
        if '.mp4' in f:
            #print(f.split('_'))
            files.add(f.split('_')[0])

files = list(files)
print(files)
for file in files:
    pieces = glob.glob(f'{file}*.mp4')
    with open('list.txt','w') as f:
        enum = 0
        for i in pieces:
            os.rename(i,f'{enum}.mp4')
            f.write(f"file '{enum}.mp4'\n")
            enum +=1
    f.close()
    os.system(f'ffmpeg -safe 0 -f concat -i list.txt  -c copy -f mp4 "{file}.mp4"')
    os.remove('list.txt')
    for idx in range(enum):
        os.remove(f'{idx}.mp4')
