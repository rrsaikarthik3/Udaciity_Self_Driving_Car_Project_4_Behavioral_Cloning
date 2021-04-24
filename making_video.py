import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import moviepy.video.io.ImageSequenceClip


image_folder ="run1" 
video_name = 'video_2.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape


act_im = []
for image in images:
    temp = cv2.imread(os.path.join(image_folder, image))    
    act_im.append(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(act_im, fps=48)
clip.write_videofile('my_video.mp4')