#python3
import os
import sys
import pdb
import time

import numpy as np
import numpy.matlib as matlib
from scipy import ndimage
import colorednoise as cn
import h5py
from data_io_hdf5 import save_dict_to_hdf5, load_dict_from_hdf5, save_array_to_hdf5, load_array_from_hdf5

import matplotlib.pyplot as plt
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc


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
		options["container"] = 'mp4'
		options["codec"] = 'MP42'
		options["fps"] = 64.0 # Frames per second
		options["duration_seconds"] = 1.0 # seconds
		options["intensity"] = (0, 255) # video grey scale dynamic range. 
		options["pedestal"] = 0 # intensity pedestal
		options["contrast"] = 1
		
		options["pattern"] = 'sine_grating' # Valid options sine_grating; square_grating; colored_temporal_noise; white_noise; natural_images; natural_video; phase_scrambled_video

		options["stimulus_form"] = 'circular' # Valid options circular, rectangular, annulus
		options["stimulus_position"] = (0.0,0.0) # Stimulus center position in degrees inside the video. (0,0) is the center.
		options["stimulus_size"] = 1.0 # In degrees. Radius for circle and annulus, half-width for rectangle. 0 gives smallest distance from image borders, ie max radius
		
		# Init optional arguments
		options["spatial_frequency"] = None
		options["temporal_frequency"] = None
		options["spatial_band_pass"] = None
		options["temporal_band_pass"] = None
		options["orientation"] = 0.0 # No rotation or vertical
		options["size_inner"] = None
		options["size_outer"] = None
		
		# Limits, no need to go beyond these
		options["min_spatial_frequency"] = 0.0625 # cycles per degree
		options["max_spatial_frequency"] = 16.0 # cycles per degree
		options["min_temporal_frequency"] = 0.5 # cycles per second, Hz
		options["max_temporal_frequency"] = 32.0 # cycles per second, Hz. 
		
		options["background"] = 128 # Background grey value
		
		# Get resolution 
		options["pix_per_deg"] = options["max_spatial_frequency"] * 3 # min sampling at 1.5 x Nyquist frequency of the highest sf
		options["image_width_in_deg"] = options["image_width"] / options["pix_per_deg"]

		self.options=options

	def get_xrange_deg(self):
		stimulus_center_x = self.options['stimulus_position'][0]
		stimulus_width_deg = self.options['image_width'] / self.options['pix_per_deg']
		return [stimulus_center_x - (stimulus_width_deg/2), stimulus_center_x + (stimulus_width_deg/2)]

	def get_yrange_deg(self):
		stimulus_center_y = self.options['stimulus_position'][1]
		stimulus_height_deg = self.options['image_width'] / self.options['pix_per_deg']
		return [stimulus_center_y - (stimulus_height_deg / 2), stimulus_center_y + (stimulus_height_deg / 2)]

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

		# Round result to avoid unnecessary errors
		self.frames = np.round(self.frames, 1)

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
		image_width_in_degrees = self.options["image_width_in_deg"]
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
		marginal_height = np.round(marginal_height).astype(np.int)
		marginal_width = np.round(marginal_width).astype(np.int)
		self.frames = large_frames[marginal_height:-marginal_height,marginal_width:-marginal_width,:]
		#remove rounding error
		self.frames = self.frames[0:image_height,0:image_width,:]

	def _write_frames_to_videofile(self, filename):
		'''Write frames to videofile
		'''
		# Init openCV VideoWriter
		fourcc = VideoWriter_fourcc(*self.options["codec"])
		filename_out = './{0}.{1}'.format(filename, self.options["container"])	
		video = VideoWriter(filename_out, fourcc, float(self.options["fps"]), 
			(self.options["image_width"], self.options["image_height"]), isColor=False) # path, codec, fps, size. Note, the isColor the flag is currently supported on Windows only

		# Write frames to videofile frame-by-frame
		for index in np.arange(self.frames.shape[2]):
			video.write(self.frames[:,:,index])
		
		video.release()
	
	def _prepare_form(self, stimulus_size):
	
		center_deg = self.options["stimulus_position"] # in degrees
		radius_deg = stimulus_size # in degrees
		height = self.options["image_height"] # in pixels
		width = self.options["image_width"] # in pixels
		pix_per_deg = self.options["pix_per_deg"]
		
		# Turn position in degrees to position in mask, shift 0,0 to center of image
		center_pix = np.array([0,0])
		center_pix[0] = int(width/2 + pix_per_deg * center_deg[0]) # NOTE Width goes to x-coordinate
		center_pix[1] = int(height/2 + pix_per_deg * -center_deg[1]) # NOTE Height goes to y-coordinate. Inverted to get positive up

		if radius_deg == 0: # use the smallest distance between the center and image walls
			radius_pix = min(center_pix[0], center_pix[1], width - center_pix[0], height - center_pix[1])
		else:
			radius_pix = pix_per_deg * radius_deg

		Y, X = np.ogrid[:height, :width]

		return X, Y, center_pix, radius_pix
	
	def _prepare_circular_mask(self, stimulus_size):
	
		X, Y, center_pix, radius_pix = self._prepare_form(stimulus_size)
		dist_from_center = np.sqrt((X - center_pix[0])**2 + (Y - center_pix[1])**2)

		mask = dist_from_center <= radius_pix
		return mask

		
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
		
	def white_noise(self):

		self.frames = np.random.normal(loc=0.0, scale=1.0, size=self.frames.shape)
		
	def colored_temporal_noise(self):
	
		beta = 1 # the exponent. 1 = pink noise, 2 = brown noise, 0 = white noise?
		variance_limits = np.array([-3,3])
		samples = self.frames.shape[2] # number of time samples to generate
		frame_time_series_unit_variance = cn.powerlaw_psd_gaussian(beta, samples)
		
		# Cut variance to [-3,3]
		frame_time_series_unit_variance_clipped = np.clip(frame_time_series_unit_variance, variance_limits.min(), variance_limits.max())
		
		# Scale to [0 1]
		frame_time_series = (frame_time_series_unit_variance_clipped - variance_limits.min()) / variance_limits.ptp()
		
		# Cast time series to frames
		assert len(frame_time_series) not in self.frames.shape[:-1], "Oops. Two different dimensions match the time series length."
		self.frames = np.zeros(self.frames.shape) + frame_time_series
		
	def natural_images(self, full_path_to_folder, width, height, fps, duration, spatial_band_pass=None, temporal_band_pass=None, orientation=0):
		# filtering: http://www.djmannion.net/psych_programming/vision/sf_filt/sf_filt.html
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

	def circular(self):
		
		mask = self._prepare_circular_mask(self.options["stimulus_size"])
		# newaxis adds 3rd dim, multiplication broadcasts one 3rd dim to N 3rd dims in self.frames
		self.frames = self.frames * mask[...,np.newaxis] 
				
	def rectangular(self):

		X, Y, center_pix, radius_pix = self._prepare_form(self.options["stimulus_size"])
		
		# Prepare rectangular distance map in pixels
		x_distance_vector = np.abs(X - center_pix[0])
		X_distance_matrix = np.matlib.repmat(x_distance_vector,Y.shape[0],1)
		y_distance_vector = np.abs(Y - center_pix[1])
		Y_distance_matrix = np.matlib.repmat(y_distance_vector,1, X.shape[1])

		# rectangular_dist_from_center = np.abs(X - center_pix[0]) + np.abs(Y - center_pix[1])
		mask = np.logical_and((X_distance_matrix<=radius_pix),(Y_distance_matrix<=radius_pix))
			
		# newaxis adds 3rd dim, multiplication broadcasts one 3rd dim to N 3rd dims in self.frames
		self.frames = self.frames * mask[...,np.newaxis] 

	def annulus(self):
	
		size_inner = self.options["size_inner"]
		size_outer = self.options["size_outer"]
		if not size_inner:
			print('Size_inner missing, setting to 1')
			size_inner = 1
		if not size_outer:
			print('Size_outer missing, setting to 2')
			size_outer = 2

		mask_inner = self._prepare_circular_mask(size_inner)
		mask_outer = self._prepare_circular_mask(size_outer)

		mask = mask_outer ^ mask_inner
		self.frames = self.frames * mask[...,np.newaxis] 
			

class ConstructStimulus(VideoBaseClass):
	'''
	Create stimulus video and save
	'''

	def __init__(self, **kwargs):
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
			'sine_grating'; 'square_grating'; 'colored_temporal_noise'; 'white_noise'; 
			'natural_images'; 'phase_scrambled_images'; 'natural_video'; 'phase_scrambled_video'
		stimulus_form: 'circular'; 'rectangular'; 'annulus'
		stimulus_position: in degrees, (0,0) is the center.
		stimulus_size: In degrees. Radius for circle and annulus, half-width for rectangle.
		contrast: between 0 and 1
		pedestal: lowest stimulus intensity between 0, 256
		
		For sine_grating and square_grating, additional arguments are:
		spatial_frequency: in cycles per degree  
		temporal_frequency: in Hz
		orientation: in degrees
		
		For white_noise and colored_temporal_noise, additional arguments are:
		spatial_band_pass: (cycles per degree min, cycles per degree max)
		temporal_band_pass: (Hz min, Hz max)
		
		For natural_images, phase_scrambled_images, natural_video and phase_scrambled_video, additional arguments are:
		spatial_band_pass: (cycles per degree min, cycles per degree max)
		temporal_band_pass: (Hz min, Hz max)
		orientation: in degrees
		
		------------------------
		Output: stimulus video file
		'''

		super(ConstructStimulus, self).__init__()

		# Set input arguments to video-object, updates the defaults from VideoBaseClass
		print("Setting the following attributes:\n")
		for kw in kwargs:
			print(kw, ":", kwargs[kw])
			assert kw in self.options.keys(), f"The keyword '{kw}' was not recognized"
		self.options.update(kwargs)
		
		# Init 3-D frames numpy array. Number of frames = frames per second * duration in seconds
		self.frames = np.ones((self.options["image_height"], self.options["image_width"], 
								int(self.options["fps"]*self.options["duration_seconds"])), 
								dtype=np.uint8) * self.options["background"]
		
		# Call StimulusPattern class method to get patterns (numpy array)
		# self.frames updated according to the pattern
		eval(f'StimulusPattern.{self.options["pattern"]}(self)') # Direct call to class.method() requires the self argument

		# Call StimulusForm class method to mask frames
		# self.frames updated according to the form
		eval(f'StimulusForm.{self.options["stimulus_form"]}(self)') # Direct call to class.method() requires the self argument

		self._scale_intensity()

	def save_to_file(self, filename):
		self._write_frames_to_videofile(filename)
		
		# save video to hdf5 file
		filename_out = f"{filename}.hdf5"	
		save_array_to_hdf5(self.frames, filename_out)

		# save options as metadata in the same format
		filename_out_options = f"{filename}_options.hdf5"	
		save_dict_to_hdf5(self.options,filename_out_options)


if __name__ == "__main__":
	from pathlib import Path
	print(Path(__file__).parent)

	#filename = 'test3'
	#my_video = ConstructStimulus(filename, pattern='white_noise', stimulus_form='rectangular', duration_seconds=2,
	#							 fps=30, pedestal =0, orientation=0, stimulus_position=(0,0), stimulus_size=4)	# Instantiate

	#my_video.main(	filename, pattern='white_noise', stimulus_form='rectangular', duration_seconds=2,
	#				fps=30, pedestal =0, orientation=0, stimulus_position=(0,0), stimulus_size=4 ) # Do the work.	Put here the needs in the keyword argumets