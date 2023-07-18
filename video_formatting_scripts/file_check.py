import glob
import os
import json

with open('config.json','r') as f:
    config=json.load(f)

files = glob.glob('*.mp4')
print(files)
for file in files:
    fname = file.replace('.mp4', '')
    if fname not in config:
        print('ERROR')
        print(fname)
print(f'Total Number of Videos: {len(files)}')
print(f'Total Number of Files in Config: {len(config)}')