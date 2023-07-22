import glob
import os

morning_path = r'C:\Users\mbarut\Desktop\sonuclar\ogle'
evening_path = r'C:\Users\mbarut\Desktop\sonuclar\aksam'

os.chdir(morning_path)
morning_files = glob.glob('*.json')
morning_files = [x.split(' ')[0] for x in morning_files]
print('Morning Files')
print(morning_files)
print('*'*50)

os.chdir(evening_path)
evening_files = glob.glob('*.json')
evening_files = [x.split(' ')[0] for x in evening_files]
print('Evening Files:')
print(evening_files)
print('*'*50)
m = set(morning_files)
e = set(evening_files)

print('Difference:')
print(m.difference(e))