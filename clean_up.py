import glob
import os
os.system('cls')
def Del():
 for f in files:
    os.remove(f)
current_directory = os.path.dirname(os.path.abspath(__file__))
files = glob.glob( os.path.join(current_directory, 'output/YOLO_darknet/*'))
Del()
print('Done')