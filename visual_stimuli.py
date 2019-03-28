#python3
import os
import sys
import pdb

import numpy as np
import numpy.matlib as matlib
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
		
		self.image_width = 1280 # Image width in pixels
		self.image_height = 720 # Image height in pixels
		self.container = 'avi'
		self.codec = 'MP42'
		self.fps = 64.0 # Frames per second
		self.duration_seconds = 1.0 # seconds
		self.intensity = (0, 256) # video grey scale dynamic range
		self.intensity_pedestal = 0
		self.contrast = 1
		
		self.pattern = 'sine_grating' # Valid options sine_grating; square_grating; pink_noise; white_noise; natural_images; natural_video; phase_scrambled_video

		self.stimulus_form = 'circular_patch' # Valid options circular_patch, rectangular_patch, annulus
		self.stimulus_position = (0.0,0.0) # Stimulus center position in degrees inside the video. (0,0) is the center.
		self.stimulus_size = 1.0 # In degrees. Radius for circle and annulus, half-width for rectangle.
		
		# Init optional arguments
		self.spatial_frequency = None
		self.temporal_frequency = None
		self.spatial_band_pass = None
		self.temporal_band_pass = None
		self.orientation = 0.0 # No rotation or vertical
		
		# Limits, no need to go beyond these
		self.min_spatial_frequency = 0.0625 # cycles per degree
		self.max_spatial_frequency = 16.0 # cycles per degree
		self.min_temporal_frequency = 0.5 # cycles per second, Hz
		self.max_temporal_frequency = 32.0 # cycles per second, Hz. 
		
		self.background = 128 # Background grey value
		
		# Get resolution 
		self.pix_per_deg = self.max_spatial_frequency * 3 # min sampling at 1.5 x Nyquist frequency of the highest sf
		self.display_width_in_deg = self.image_width / self.pix_per_deg
	
	
class StimulusPattern:
	'''
	Construct the stimulus images
	'''
	def sine_grating(self, frames):
		# width, height, fps, duration, orientation=0, spatial_frequency=1, temporal_frequency=1

		spatial_frequency = self.spatial_frequency
		temporal_frequency = self.temporal_frequency
		fps=self.fps
		duration_seconds = self.duration_seconds
		orientation = self.orientation
		
		if not spatial_frequency:
			print('Spatial_frequency missing, setting to 1')
			spatial_frequency = 1
		if not temporal_frequency:
			print('Temporal_frequency missing, setting to 1')
			temporal_frequency = 1
					
		# Create sine wave
		one_cycle = 2 * np.pi
		cycles_per_degree = spatial_frequency
		image_width_in_degrees = self.display_width_in_deg
		image_width = self.image_width
		image_height = self.image_height
		
		# Draw temporospatial grating
		image_position_vector = np.linspace(0,one_cycle * cycles_per_degree * image_width_in_degrees, image_width)
		n_frames = frames.shape[2]
		image_temporospatial_3D_raw = np.tile(image_position_vector,(image_height,n_frames,1))
		image_temporospatial_3D_raw_rolled = np.moveaxis(image_temporospatial_3D_raw, 2, 1)
		total_temporal_shift = temporal_frequency * one_cycle * duration_seconds
		one_frame_temporal_shift = (temporal_frequency * one_cycle ) / fps
		temporal_shift_vector = np.arange(0, total_temporal_shift, one_frame_temporal_shift)
		image_temporospatial_3D = image_temporospatial_3D_raw_rolled + temporal_shift_vector # and then the magic happens?
		raw_frame = np.sin(image_temporospatial_3D)
		
		# Scale intensity to 8-bit grey scale. Calculating peak-to-peak here allows different luminances and contrasts
		raw_intensity_scale = np.ptp(raw_frame)
		intensity_min = np.min(self.intensity)
		intensity_max = np.max(self.intensity)
		full_intensity_scale = np.ptp((intensity_min,intensity_max))
		pedestal = self.intensity_pedestal # This is the bottom of final dynamic range
		contrast = self.contrast

		final_dynamic_range = (pedestal, intensity_max)
		final_scale = np.ptp(final_dynamic_range)
		
		#shift to zero
		raw_frame_at_zero = raw_frame - np.min(raw_frame)
		
		#scale to correct intensity scale
		raw_frame_at_zero_scaled = raw_frame_at_zero * (final_scale/raw_intensity_scale) * contrast
		
		#shift to pedestal
		frames = raw_frame_at_zero_scaled + pedestal

		# plt.imshow(frame[:,:,0], cmap='gray', vmin=intensity_min, vmax=intensity_max);plt.colorbar();plt.show()
		frames=frames.astype(np.uint8)
		return frames
		
	def square_grating(self, width, height, fps, duration, orientation=0, spatial_frequency=1, temporal_frequency=1):
		pass
		
	def white_noise(self, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None):
		pass
		
	def pink_noise(self, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None):
		pass
		
	def natural_images(self, full_path_to_folder, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None, orientation=0):
		pass
		
	def phase_scrambled_images(self, full_path_to_folder, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None, orientation=0):
		pass
		
	def natural_video(self, full_path, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None, orientation=0):
		pass
		
	def phase_scrambled_video(self, full_path, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None, orientation=0):
		pass

		
class StimulusForm:
	'''
	Mask the stimulus images
	'''

	def circular_patch(self, frames):
		# , position, size
		return frames

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
		Format: my_video_object.main(filename, keyword1=value1, keyword2=value2,...)
		
		Valid input keyword arguments include 
		
		image_width: in pixels
		image_height: in pixels
		container: file format to export
		codec: compression format
		fps: frames per second
		duration_seconds: stimulus duration
		pattern: 
			'sine_grating'; 'square_grating'; 'pink_noise'; 'white_noise'; 
			'natural_images'; 'phase_scrambled_images'; 'natural_video'; 'phase_scrambled_video'
		stimulus_form: 'circular_patch'; 'rectangular_patch'; 'annulus'
		stimulus_position: in degrees, (0,0) is the center.
		stimulus_size: In degrees. Radius for circle and annulus, half-width for rectangle.
		contrast: between 0 and 1
		pedestal: lowest stimulus intensity between 0, 256
		
		For sine_grating and square_grating, additional arguments are:
		spatial_frequency: in cycles per degree  
		temporal_frequency: in Hz
		orientation: in degrees
		
		For white_noise and pink_noise, additional arguments are:
		spatial_band_pass: (cycles per degree min, cycles per degree max)
		temporal_band_pass: (Hz min, Hz max)
		
		For natural_images, phase_scrambled_images, natural_video and phase_scrambled_video, additional arguments are:
		spatial_band_pass: (cycles per degree min, cycles per degree max)
		temporal_band_pass: (Hz min, Hz max)
		orientation: in degrees
		
		------------------------
		Output: stimulus video file
		'''
		
		# Set input arguments to video-object, updates the defaults from VideoBaseClass
		print("Setting the following attributes:\n")
		for kw in kwargs:
			print(kw, ":", kwargs[kw])
			assert kw in self.__dict__.keys(), "The keyword '{0}' was not recognized".format(kw)
		self.__dict__.update(kwargs)
		
		# Get basic video parameters
		width = self.image_width
		height = self.image_height
		fps = self.fps
		duration_seconds = self.duration_seconds
		
		# Init 3-D frames numpy array. Number of frames = frames per second * duration in seconds
		frames = np.ones((height, width, int(fps*duration_seconds)), dtype=np.uint8) * self.background
		
		# Call StimulusPattern class method to get patterns (numpy array)
		pattern = self.pattern
		frames=eval('StimulusPattern.{0}(self, frames)'.format(pattern)) # Direct call to class.method() requires the self argument

		TÄHÄN JÄIT: ROTATOI SINIGRATING, MASKAA, MUUT ÄRSYKKEET
		# # Call StimulusForm class method to mask frames
		# stimulus_form = self.stimulus_form
		# frames = eval('StimulusForm.{0}(self, frames)'.format(stimulus_form)) # Direct call to class.method() requires the self argument
		
		# Init openCV VideoWriter
		fourcc = VideoWriter_fourcc(*self.codec)
		filename = './{0}.{1}'.format(filename, self.container)	
		video = VideoWriter(filename, fourcc, float(fps), (width, height), isColor=False) # path, codec, fps, size. Note, the isColor the flag is currently supported on Windows only

		# Write frames to videofile frame-by-frame
		for index in np.arange(frames.shape[2]):
			video.write(frames[:,:,index])
		
		# for _ in range(int(fps*duration_seconds)):
		
			# # Test noise
			# frame = np.random.randint(0, 256, 
									  # (height, width), # (height, width, 3), # Color
									  # dtype=np.uint8)

		video.release()

		
if __name__ == "__main__":

	my_video = ConstructStimuli()	# Instantiate
	filename = 'test2'
	my_video.main(filename, pattern='sine_grating', duration_seconds=3, fps=30, spatial_frequency=2, temporal_frequency=20) # Do the work.	Put here the needs in the keyword argumets