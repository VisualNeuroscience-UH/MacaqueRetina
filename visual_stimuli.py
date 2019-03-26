#python3
import os
import sys
import pdb

import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

cwd = os.getcwd()
work_path = 'C:\\Users\\vanni\\Laskenta\\Git_Repos\\MacaqueRetina_Git'
os.chdir(work_path)



'''
This module creates the visual stimuli. Stimuli include patches of sinusoidal gratings at different orientations
and spatial frequencies. The duration can be defined in seconds and size (radius), and center location (x,y) 
in degrees.

Input: stimulus definition
Output: video stimulus frames

Formats .avi .mov .mp4 ?


'''

class VideoBaseClass:
	def __init__(self):
		'''
		Initialize standard video stimulus
		'''
		
		self.image_width = 1280 # Image width
		self.image_height = 720 # Image height
		self.container = 'avi'
		self.codec = 'MP42'
		self.fps = 64 # Frames per second
		self.duration = 1 # seconds
		
		self.pattern = 'sine_grating' # Valid options sine_grating; square_grating; pink_noise; white_noise; natural_images; natural_video; phase_scrambled_video

		self.stimulus_form = 'circular_patch' # Valid options circular_patch, rectangular_patch, annulus
		self.stimulus_position = (0,0) # Stimulus center position in degrees inside the video. (0,0) is the center.
		self.stimulus_size = 1 # In degrees. Radius for circle and annulus, half-width for rectangle.
		
		# Limits, no need to go beyond these
		self.min_spatial_frequency = 0.0625 # cycles per degree
		self.max_spatial_frequency = 16 # cycles per degree
		self.min_temporal_frequency = 0.5 # cycles per second, Hz
		self.max_temporal_frequency = 32 # cycles per second, Hz. 
		
		
class StimulusPattern:
	'''
	Construct the stimulus images
	'''
	def sine_grating(self, width, height, fps, duration, orientation=0, spatial_frequency=1, temporal_frequency=1):
		pass
		
	def square_grating(self, width, height, fps, duration, orientation=0, spatial_frequency=1, temporal_frequency=1):
		pass
		
	def white_noise(self, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None):
		pass
		
	def pink_noise(self, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None):
		pass
		
	def natural_images(self, full_path_to_folder, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None, orientation=0):
		pass
		
	def natural_video(self, full_path, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None, orientation=0):
		pass
		
	def phase_scrambled_video(self, full_path, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None, orientation=0):
		pass

		
class StimulusForm:
	'''
	Mask the stimulus images
	'''

	def circular_patch(self, position, size):
		pass

	def rectangular_patch(self, position, size):
		pass

	def annulus(self, position, size_inner, size_outer):
		pass
		

class ConstructStimuli(VideoBaseClass):
	'''
	Create stimulus video and save
	'''

	def main(self, filename, **kwargs):
		'''
		Valid input keyword arguments include 
		
		image_width: (in pixels)
		image_height: (in pixels)
		container: (file format to export)
		codec: (compression format)
		fps: (frames per second)
		duration: (in seconds)
		pattern: 'sine_grating'; 'square_grating'; 'pink_noise'; 'white_noise'; 'natural_images'; 'natural_video'; 'hase_scrambled_video'
		stimulus_form: 'circular_patch'; 'rectangular_patch'; 'annulus'
		stimulus_position: in degrees, (0,0) is the center.
		stimulus_size = 1 # In degrees. Radius for circle and annulus, half-width for rectangle.
		
		------------------------
		Output: stimulus video file
		'''
		width = self.image_width	
		height = self.image_height
		FPS = self.fps
		seconds = self.duration

		fourcc = VideoWriter_fourcc(*self.codec)
		filename = './{0}.{1}'.format(filename, self.container)
		video = VideoWriter(filename, fourcc, float(FPS), (width, height)) # path, codec, fps, size

		for _ in range(FPS*seconds):
			frame = np.random.randint(0, 256, 
									  (height, width, 3), 
									  dtype=np.uint8)
			video.write(frame)

		video.release()

if __name__ == "__main__":

	my_video = ConstructStimuli()	# Instantiate
	filename = 'test1'
	my_video.main(filename) # Do the work.	Put here the needs in the keyword argumets