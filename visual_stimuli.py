#python3
import os
import sys
import pdb

import numpy as np
import numpy.matlib as matlib
from scipy import ndimage
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
		The base class methods are applied to every stimulus
		'''
		options = {}
		options["image_width"] = 1280 # Image width in pixels
		options["image_height"] = 720 # Image height in pixels
		options["container"] = 'avi'
		options["codec"] = 'MP42'
		options["fps"] = 64.0 # Frames per second
		options["duration_seconds"] = 1.0 # seconds
		options["intensity"] = (0, 255) # video grey scale dynamic range. 
		options["pedestal"] = 0 # intensity pedestal
		options["contrast"] = 1
		
		options["pattern"] = 'sine_grating' # Valid options sine_grating; square_grating; pink_noise; white_noise; natural_images; natural_video; phase_scrambled_video

		options["stimulus_form"] = 'circular_patch' # Valid options circular_patch, rectangular_patch, annulus
		options["stimulus_position"] = (0.0,0.0) # Stimulus center position in degrees inside the video. (0,0) is the center.
		options["stimulus_size"] = 1.0 # In degrees. Radius for circle and annulus, half-width for rectangle.
		
		# Init optional arguments
		options["spatial_frequency"] = None
		options["temporal_frequency"] = None
		options["spatial_band_pass"] = None
		options["temporal_band_pass"] = None
		options["orientation"] = 0.0 # No rotation or vertical
		
		# Limits, no need to go beyond these
		options["min_spatial_frequency"] = 0.0625 # cycles per degree
		options["max_spatial_frequency"] = 16.0 # cycles per degree
		options["min_temporal_frequency"] = 0.5 # cycles per second, Hz
		options["max_temporal_frequency"] = 32.0 # cycles per second, Hz. 
		
		options["background"] = 128 # Background grey value
		
		# Get resolution 
		options["pix_per_deg"] = options["max_spatial_frequency"] * 3 # min sampling at 1.5 x Nyquist frequency of the highest sf
		options["display_width_in_deg"] = options["image_width"] / options["pix_per_deg"]
	
		self.options=options
	
	def _scale_intensity(self):
	
		'''Scale intensity to 8-bit grey scale. Calculating peak-to-peak here allows different 
		luminances and contrasts'''
		
		raw_intensity_scale = np.ptp(self.frames)
		intensity_min = np.min(self.options["intensity"])
		intensity_max = np.max(self.options["intensity"])
		full_intensity_scale = np.ptp((intensity_min,intensity_max))
		pedestal = self.options["pedestal"] # This is the bottom of final dynamic range
		contrast = self.options["contrast"]

		final_dynamic_range = (pedestal, intensity_max)
		final_scale = np.ptp(final_dynamic_range)
		
		# Shift to zero
		self.frames = self.frames - np.min(self.frames)
		
		# Scale to correct intensity scale
		self.frames = self.frames * (final_scale/raw_intensity_scale) * contrast
		
		# Shift to pedestal
		self.frames = self.frames + pedestal
		
		# Check that the values are between 0 and 255 to get correct conversion to uint8
		assert np.all(0 <= self.frames.flatten()) and np.all(self.frames.flatten() <= 255), "Cannot safely convert to uint8. Check intensity/dynamic range."
		# Return
		self.frames=self.frames.astype(np.uint8)

	def _prepare_grating(self):
		'''Create temporospatial grating
		'''
	
		spatial_frequency = self.options["spatial_frequency"]
		temporal_frequency = self.options["temporal_frequency"]
		fps=self.options["fps"]
		duration_seconds = self.options["duration_seconds"]
		orientation = self.options["orientation"]
		
		if not spatial_frequency:
			print('Spatial_frequency missing, setting to 1')
			spatial_frequency = 1
		if not temporal_frequency:
			print('Temporal_frequency missing, setting to 1')
			temporal_frequency = 1
					
		
		# Create sine wave
		one_cycle = 2 * np.pi
		cycles_per_degree = spatial_frequency
		image_width_in_degrees = self.options["display_width_in_deg"]
		image_width = self.options["image_width"]
		image_height = self.options["image_height"]
		
		# Calculate larger image size to allow rotations
		diameter = np.ceil(np.sqrt(image_height**2 + image_width**2)).astype(np.int)
		image_width_diameter = diameter
		image_height_diameter = diameter
		
		# Draw temporospatial grating
		image_position_vector = np.linspace(0,one_cycle * cycles_per_degree * image_width_in_degrees, image_width_diameter)
		n_frames = self.frames.shape[2]
		
		# Recycling large_frames and self.frames below, instead of descriptive variable names for the evolving video, saves a lot of memory
		# Create large 3D frames array covering the most distant corner when rotated
		large_frames = np.tile(image_position_vector,(image_height_diameter,n_frames,1))
		# Correct dimensions to image[0,1] and time[2]
		large_frames = np.moveaxis(large_frames, 2, 1)
		total_temporal_shift = temporal_frequency * one_cycle * duration_seconds
		one_frame_temporal_shift = (temporal_frequency * one_cycle ) / fps
		temporal_shift_vector = np.arange(0, total_temporal_shift, one_frame_temporal_shift)
		# Shift grating phase in time. Broadcasting temporal vector automatically to correct dimension.
		large_frames = large_frames + temporal_shift_vector  

		# Rotate to desired orientation
		large_frames = ndimage.rotate(large_frames, orientation, reshape=False) 
		
		# Cut back to original image dimensions
		marginal_height = (diameter - image_height) / 2
		marginal_width = (diameter - image_width) / 2
		marginal_height = np.floor(marginal_height).astype(np.int)
		marginal_width = np.floor(marginal_width).astype(np.int)
		self.frames = large_frames[marginal_height:-marginal_height,marginal_width:-marginal_width,:]
	
class StimulusPattern:
	'''
	Construct the stimulus images
	'''
	def sine_grating(self):

		# Create temporospatial grating
		self._prepare_grating()
		
		# Turn to sine values
		self.frames = np.sin(self.frames)
			
	def square_grating(self):

		# Create temporospatial grating
		self._prepare_grating()
		
		# Turn to sine values
		self.frames = np.sin(self.frames)

		# Turn to square grating values, threshold at zero.
		threshold = 0 # Change this between [-1 1] if you want uneven grating. Default is 0
		self.frames = (self.frames > threshold) * self.frames/self.frames * 2 - 1
		
	def white_noise(self, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None):
		pass
		
	def pink_noise(self, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None):
		# # import colorednoise as cn
		# # beta = 1 # the exponent
		# # samples = 2**18 # number of samples to generate
		# # y = cn.powerlaw_psd_gaussian(beta, samples)

		# # # optionally plot the Power Spectral Density with Matplotlib
		# # #from matplotlib import mlab
		# # #from matplotlib import pylab as plt
		# # #s, f = mlab.psd(y, NFFT=2**13)
		# # #plt.loglog(f,s)
		# # #plt.grid(True)
		# # #plt.show()
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
		
	def e_table(self, hight):
		'''For Khoa's driving's licence. This can easily become acuity table for our system. Just include the 
		Snellen letters, and scale the size'''

		sizes = np.array([0.1,0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
		scaling_size_to_angle = 5/60 # degree 1.0 size is 5/60 deg angle
		# optotype_height = 0.068 # 1.0, in meters
		# size = 0.1
		sizes_in_deg = (1/sizes) * scaling_size_to_angle
		sizes_in_rad = sizes_in_deg * np.pi/180
		distance = 6.0
		optotype_height = 2 * distance * np.tan(sizes_in_rad / 2) # in meters


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
			assert kw in self.options.keys(), "The keyword '{0}' was not recognized".format(kw)
		self.options.update(kwargs)
		
		# Get basic video parameters
		width = self.options["image_width"]
		height = self.options["image_height"]
		fps = self.options["fps"]
		duration_seconds = self.options["duration_seconds"]
		
		# Init 3-D frames numpy array. Number of frames = frames per second * duration in seconds
		self.frames = np.ones((height, width, int(fps*duration_seconds)), dtype=np.uint8) * self.options["background"]
		
		# Call StimulusPattern class method to get patterns (numpy array)
		eval('StimulusPattern.{0}(self)'.format(self.options["pattern"])) # Direct call to class.method() requires the self argument

		# TÄHÄN JÄIT: ROTATOI SINIGRATING, MASKAA, MUUT ÄRSYKKEET
		# # Call StimulusForm class method to mask frames
		# stimulus_form = self.options["stimulus_form
		# frames = eval('StimulusForm.{0}(self, frames)'.format(stimulus_form)) # Direct call to class.method() requires the self argument
		
		self._scale_intensity()
		
		# Init openCV VideoWriter
		fourcc = VideoWriter_fourcc(*self.options["codec"])
		filename = './{0}.{1}'.format(filename, self.options["container"])	
		video = VideoWriter(filename, fourcc, float(fps), (width, height), isColor=False) # path, codec, fps, size. Note, the isColor the flag is currently supported on Windows only

		# Write frames to videofile frame-by-frame
		for index in np.arange(self.frames.shape[2]):
			video.write(self.frames[:,:,index])
		
		# for _ in range(int(fps*duration_seconds)):
		
			# # Test noise
			# frame = np.random.randint(0, 256, 
									  # (height, width), # (height, width, 3), # Color
									  # dtype=np.uint8)

		video.release()

		
if __name__ == "__main__":

	my_video = ConstructStimuli()	# Instantiate
	filename = 'test2'
	my_video.main(filename, pattern='square_grating', duration_seconds=1, fps=30, spatial_frequency=2, 
		temporal_frequency=2, pedestal =0, orientation=45) # Do the work.	Put here the needs in the keyword argumets