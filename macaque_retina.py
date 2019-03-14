#python3
import os
import sys
import pdb

import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as ellipse
from tqdm import tqdm
import cv2

cwd = os.getcwd()
work_path = 'C:\\Users\\vanni\\OneDrive - University of Helsinki\\Work\\Simulaatiot\\Retinamalli\\retina_stats'
os.chdir(work_path)


class Mathematics:
	'''
	Constructor fit functions to read in data and provide continuous functions
	'''
	def gauss_plus_baseline(self, x,a,x0,sigma, baseline): # To fit GC density
		'''
		Function for Gaussian distribution with a baseline value. For optimization.
		'''
		return a*np.exp(-(x-x0)**2/(2*sigma**2)) + baseline

	def pol2cart(self, rho, phi):
		'''
		Get polar and return cartesian coordinates
		'''
		x = rho * np.cos(phi*np.pi/180) # assuming degrees
		ycoord = rho * np.sin(phi*np.pi/180)
		return(x, ycoord)

	def DoG2D_independent_surround(self, xy_tuple, amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes, xos, yos, semi_xs, semi_ys, orientation_surround, offset):
		'''
		DoG model with xo, yo, theta for surround independent from center.
		'''

		(x_fit, y_fit) = xy_tuple
		acen = (np.cos(orientation_center)**2)/(2*semi_xc**2) + (np.sin(orientation_center)**2)/(2*semi_yc**2)
		bcen = -(np.sin(2*orientation_center))/(4*semi_xc**2) + (np.sin(2*orientation_center))/(4*semi_yc**2)
		ccen = (np.sin(orientation_center)**2)/(2*semi_xc**2) + (np.cos(orientation_center)**2)/(2*semi_yc**2)

		asur = (np.cos(orientation_surround)**2)/(2*semi_xs**2) + (np.sin(orientation_surround)**2)/(2*semi_ys**2)
		bsur = -(np.sin(2*orientation_surround))/(4*semi_xs**2) + (np.sin(2*orientation_surround))/(4*semi_ys**2)
		csur = (np.sin(orientation_surround)**2)/(2*semi_xs**2) + (np.cos(orientation_surround)**2)/(2*semi_ys**2)

		## Difference of gaussians
		model_fit = offset + \
			amplitudec*np.exp( - (acen*((x_fit-xoc)**2) + 2*bcen*(x_fit-xoc)*(y_fit-yoc) + ccen*((y_fit-yoc)**2))) - \
			amplitudes*np.exp( - (asur*((x_fit-xos)**2) + 2*bsur*(x_fit-xos)*(y_fit-yos) + csur*((y_fit-yos)**2)))

		return model_fit.ravel()

	def DoG2D_fixed_surround(self, xy_tuple, amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes, sur_ratio, offset):
		'''
		DoG model with xo, yo, theta for surround coming from center.
		'''
		(x_fit, y_fit) = xy_tuple
		acen = (np.cos(orientation_center)**2)/(2*semi_xc**2) + (np.sin(orientation_center)**2)/(2*semi_yc**2)
		bcen = -(np.sin(2*orientation_center))/(4*semi_xc**2) + (np.sin(2*orientation_center))/(4*semi_yc**2)
		ccen = (np.sin(orientation_center)**2)/(2*semi_xc**2) + (np.cos(orientation_center)**2)/(2*semi_yc**2)

		asur = (np.cos(orientation_center)**2)/(2*sur_ratio*semi_xc**2) + (np.sin(orientation_center)**2)/(2*sur_ratio*semi_yc**2)
		bsur = -(np.sin(2*orientation_center))/(4*sur_ratio*semi_xc**2) + (np.sin(2*orientation_center))/(4*sur_ratio*semi_yc**2)
		csur = (np.sin(orientation_center)**2)/(2*sur_ratio*semi_xc**2) + (np.cos(orientation_center)**2)/(2*sur_ratio*semi_yc**2)

		## Difference of gaussians
		model_fit = offset + \
			amplitudec*np.exp( - (acen*((x_fit-xoc)**2) + 2*bcen*(x_fit-xoc)*(y_fit-yoc) + ccen*((y_fit-yoc)**2))) - \
			amplitudes*np.exp( - (asur*((x_fit-xoc)**2) + 2*bsur*(x_fit-xoc)*(y_fit-yoc) + csur*((y_fit-yoc)**2)))

		return model_fit.ravel()

	def sector2area(self, radius, angle): # Calculate sector area. Angle in deg, radius in mm
		pi = np.pi
		assert angle < 360, "Angle not possible, should be <360"

		# Calculating area of the sector
		sector_surface_area = (pi * (radius ** 2)) * (angle / 360)  # in mm2
		return sector_surface_area

	def circle_diameter2area(self, diameter):

		area_of_rf = np.pi * (diameter/2)**2
		
		return area_of_rf
		
	def area2circle_diameter(self, area_of_rf):
			
		diameter = np.sqrt(area_of_rf / np.pi) * 2
		
		return diameter
		
	def ellipse2area(self, sigma_x, sigma_y):
		
		area_of_ellipse = np.pi * sigma_x * sigma_y
		
		return area_of_ellipse
		
		
class GetLiteratureData:
	'''
	Read data from external mat files. Data-specific definitions are isolated here.
	'''
	def read_gc_density_data(self):
		'''
		Read re-digitized old literature data from mat files
		'''

		gc_density = sio.loadmat('Perry_1984_Neurosci_GCdensity_c.mat',variable_names=['Xdata','Ydata'])
		cell_eccentricity = np.squeeze(gc_density['Xdata'])
		cell_density = np.squeeze(gc_density['Ydata']) * 1e3 # Cells are in thousands, thus the 1e3
		return cell_eccentricity, cell_density

	def read_retina_glm_data(self, gc_type, responsetype):

		# Go to correct folder
		cwd2 = os.getcwd()
		work_path2 = 'C:\\Users\\vanni\\OneDrive - University of Helsinki\\Work\\Simulaatiot\\Retinamalli\\Retina_GLM\\apricot'
		#Define filename
		if gc_type=='parasol' and responsetype=='ON':
			filename = 'Parasol_ON_spatial.mat'
			bad_data_indices=[15, 67, 71, 86, 89]   # Manually selected for Chichilnisky apricot data
			# bad_data_indices=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,0] # Manually selected for Chichilnisky apricot data
		elif gc_type=='parasol' and responsetype=='OFF':
			filename = 'Parasol_OFF_spatial.mat'
			bad_data_indices=[6, 31, 73]
		elif gc_type=='midget' and responsetype=='ON':
			filename = 'Midget_ON_spatial.mat'
			bad_data_indices=[6, 13, 19, 23, 26, 28, 55,74, 93, 99, 160, 162, 203, 220]
		elif gc_type=='midget' and responsetype=='OFF':
			filename = 'Midget_OFF_spatial.mat'
			bad_data_indices=[4, 5, 13, 23, 39, 43, 50, 52, 55, 58, 71, 72, 86, 88, 94, 100, 104, 119, 137,
								154, 155, 169, 179, 194, 196, 224, 230, 234, 235, 239, 244, 250, 259, 278]
		else:
			print('Unkown celltype or responsetype, aborting')
			sys.exit()

		#Read data
		filepath=os.path.join(work_path2,filename)
		gc_spatial_data = sio.loadmat(filepath,variable_names=['c','stafit'])
		gc_spatial_data_array=gc_spatial_data['c']
		initial_center_values = gc_spatial_data['stafit']

		return gc_spatial_data_array, initial_center_values, bad_data_indices

	def read_dendritic_fields_vs_eccentricity_data(self):
		'''
		Read re-digitized old literature data from mat files
		'''
		if self.gc_type=='parasol':
			dendr_diam1 = sio.loadmat('Perry_1984_Neurosci_ParasolDendrDiam_c.mat',variable_names=['Xdata','Ydata'])
			dendr_diam2 = sio.loadmat('Watanabe_1989_JCompNeurol_GCDendrDiam_parasol_c.mat',variable_names=['Xdata','Ydata'])
		elif self.gc_type=='midget':
			dendr_diam1 = sio.loadmat('Perry_1984_Neurosci_MidgetDendrDiam_c.mat',variable_names=['Xdata','Ydata'])
			dendr_diam2 = sio.loadmat('Watanabe_1989_JCompNeurol_GCDendrDiam_midget_c.mat',variable_names=['Xdata','Ydata'])

		return dendr_diam1, dendr_diam2


class Visualize:
	'''
	Methods to visualize the retina
	'''
	def show_gc_positions_and_density(self, rho, phi, gc_density_func_params): 
		'''
		Show retina cell positions and receptive fields
		'''
					
		# to cartesian
		xcoord,ycoord = self.pol2cart(rho, phi)
		# xcoord,ycoord = FitFunctionsAndMath.pol2cart(matrix_eccentricity, matrix_orientation_surround)
		fig, ax = plt.subplots(nrows=2, ncols=1)
		ax[0].plot(xcoord.flatten(),ycoord.flatten(),'b.',label=self.gc_type)
		ax[0].axis('equal')
		ax[0].legend()
		ax[0].set_title('cartesian retina')
		ax[0].set_xlabel('Horizontal (mm)')
		ax[0].set_ylabel('Vertical (mm)')

		# quality control for density.
		nbins = 50
		# Fit for published data
		edge_ecc = np.linspace(np.min(rho), np.max(rho),nbins)
		my_gaussian_fit = self.gauss_plus_baseline(edge_ecc,*gc_density_func_params)
		my_gaussian_fit_current_GC = my_gaussian_fit * self.gc_proportion
		ax[1].plot(edge_ecc,my_gaussian_fit_current_GC,'r')
		
		# Density of model cells
		index = np.all([phi>np.min(self.theta),phi<np.max(self.theta),
						rho>np.min(self.eccentricity_in_mm), rho<np.max(self.eccentricity_in_mm)], axis=0) # Index only cells within original requested theta
		hist, bin_edges = np.histogram(rho[index],nbins)
		center_ecc = bin_edges[:-1] + ((bin_edges[1:] - bin_edges[:-1]) / 2)
		area_for_each_bin = self.sector2area(bin_edges[1:], np.ptp(self.theta)) \
							- self.sector2area(bin_edges[:-1], np.ptp(self.theta)) # in mm2. Vector length len(edge_ecc) - 1.
		# Cells/area
		model_cell_density = hist / area_for_each_bin # in cells/mm2
		ax[1].plot(center_ecc, model_cell_density, 'b.')
		
	def show_gc_receptive_fields(self, rho, phi, gc_rf_models, surround_fixed=0): 
		'''
		Show retina cell positions and receptive fields. Note that this is slow if you have a large patch.
		'''

		# to cartesian
		xcoord,ycoord = self.pol2cart(rho, phi)

		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(xcoord.flatten(),ycoord.flatten(),'b.',label=self.gc_type)

		if self.surround_fixed:
			# gc_rf_models parameters:'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center'
			# Ellipse parameters: Ellipse(xy, width, height, angle=0, **kwargs). Only possible one at the time, unfortunately.
			for index in np.arange(len(xcoord)):
				ellipse_center_x = xcoord[index]
				ellipse_center_y = ycoord[index]				
				semi_xc = gc_rf_models[index,0]
				semi_yc = gc_rf_models[index,1]
				angle_in_radians = gc_rf_models[index,5] # Orientation
				diameter_xc = semi_xc * 2
				diameter_yc = semi_yc * 2
				e1=ellipse((ellipse_center_x, ellipse_center_y),diameter_xc, diameter_yc, angle_in_radians*180/np.pi,
							edgecolor='b', linewidth=0.5, fill=False)
				ax.add_artist(e1)
			# e2=ellipse((popt[np.array([1,2])]),popt[7]*popt[3],popt[7]*popt[4],-popt[5]*180/np.pi,edgecolor='w', linewidth=2, fill=False, linestyle='--')

		ax.axis('equal')
		ax.legend()
		ax.set_title('cartesian retina')
		ax.set_xlabel('Horizontal (mm)')
		ax.set_ylabel('Vertical (mm)')

	def show_spatial_statistics(self,ydata, spatial_statistics_dict, model_fit_data=None):
		'''
		Show histograms of receptive field parameters
		'''

		distributions = [key for key in spatial_statistics_dict.keys()]
		n_distributions = len(spatial_statistics_dict)
		
		# plot the distributions and fits.
		fig, axes = plt.subplots(2, 3, figsize=(13,4))
		axes=axes.flatten()
		for index in np.arange(n_distributions):
			ax = axes[index]
			
			bin_values, foo, foo2 = ax.hist(ydata[:,index], bins=20, density=True);
			
			if model_fit_data!= None: # Assumes tuple of arrays, see below
				x_model_fit, y_model_fit = model_fit_data[0], model_fit_data[1]
				ax.plot(x_model_fit[:,index], y_model_fit[:,index], 'r-', linewidth=6, alpha=.6)
				
				spatial_statistics_dict[distributions[index]]
				shape = spatial_statistics_dict[distributions[index]]['shape']
				loc = spatial_statistics_dict[distributions[index]]['loc']
				scale = spatial_statistics_dict[distributions[index]]['scale']
				model_function = spatial_statistics_dict[distributions[index]]['distribution']

				if model_function == 'gamma':
					ax.annotate(s='shape = {0:.2f}\nloc = {1:.2f}\nscale = {2:.2f}'.format(shape, 
									loc, scale), xy=(.6,.4), xycoords='axes fraction')
					ax.set_title('{0} fit for {1}'.format(model_function, distributions[index]))
				elif model_function == 'beta':
					a_parameter, b_parameter = shape[0], shape[1]
					ax.annotate(s='a = {0:.2f}\nb = {1:.2f}\nloc = {2:.2f}\nscale = {3:.2f}'.format(a_parameter, b_parameter, 
									loc, scale), xy=(.6,.4), xycoords='axes fraction')
					ax.set_title('{0} fit for {1}'.format(model_function, distributions[index]))

				# Rescale y axis if model fit goes high. Shows histogram better
				if y_model_fit[:,index].max() > 1.5 * bin_values.max():
					ax.set_ylim([ax.get_ylim()[0],1.1 * bin_values.max()])

		# Check correlations
		# distributions = ['semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center']
		fig2, axes2 = plt.subplots(2, 3, figsize=(13,4))
		axes2=axes2.flatten()
		ref_index = 1
		for index in np.arange(n_distributions):
			ax2 = axes2[index]
			data_all_x = ydata[:,ref_index]
			data_all_y = ydata[:,index]

			r, p = stats.pearsonr(data_all_x, data_all_y);
			slope, intercept, r_value, p_value, std_err = stats.linregress(data_all_x, data_all_y)
			ax2.plot(data_all_x, data_all_y,'.');
			data_all_x.sort()
			ax2.plot(data_all_x,intercept + slope*data_all_x,"b-")
			ax2.annotate(s='\nr={0:.2g},\np={1:.2g}'.format(r,p), xy=(.8,.4),
							xycoords='axes fraction')
			ax2.set_title('Correlation between {0} and {1}'.format(distributions[ref_index],distributions[index]))
			
		# Save correlation figure
		# folder_name = 'Korrelaatiot\Midget_OFF'
		# save4save = os.getcwd()
		# os.chdir(folder_name)
		# plt.savefig('Corr_{0}.png'.format(distributions[ref_index]),format='png')
		# os.chdir(save4save)

		pass
	
	def show_dendritic_diameter_vs_eccentricity(self, gc_type, dataset_x, dataset_y, polynomials, dataset_name=''):

		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(dataset_x,dataset_y,'.')
		# pdb.set_trace()
		if dataset_name:
			if len(polynomials)==2: #check if only two parameters, ie intercept and slope
				intercept=polynomials[1]; slope=polynomials[0]
				ax.plot(dataset_x, intercept + slope*dataset_x,"k--")
				ax.annotate("{0} : \ny={1:.1f} + {2:.1f}x".format(dataset_name, intercept, slope), 
				xycoords='axes fraction', xy=(.5, .15), ha="left", color='k')
			elif len(polynomials)==3:
				intercept=polynomials[2]; slope=polynomials[1]; square=polynomials[0]
				ax.plot(dataset_x, intercept + slope*dataset_x + square*dataset_x**2,"k--")
				ax.annotate("{0}: \ny={1:.1f} + {2:.1f}x + {3:.1f}x^2".format(dataset_name, intercept, slope, square),
					xycoords='axes fraction', xy=(.5, .15), ha="left", color='k')
			elif len(polynomials)==4:
				intercept=polynomials[3]; slope=polynomials[2]; square=polynomials[1]; cube=polynomials[0]
				ax.plot(dataset_x, intercept + slope*dataset_x + square*dataset_x**2 + cube*dataset_x**3,"k--")
				ax.annotate("{0}: \ny={1:.1f} + {2:.1f}x + {3:.1f}x^2 + {4:.1f}x^3".format(dataset_name, intercept, slope, square, cube),
					xycoords='axes fraction', xy=(.5, .15), ha="left", color='k') 
		
		plt.title('Dendrite diam wrt ecc for {0} type, {1} dataset'.format(gc_type, dataset_name))

						
class ConstructReceptiveFields(GetLiteratureData, Visualize):
	'''
	Methods to build spatial receptive fields
	'''
	def fit_DoG2retina_data(self, visualize=False, surround_fixed=0): 
		'''
		2D DoG fit to Chichilnisky retina spike triggered average data. 
		The visualize parameter will show each DoG fit for search for bad cell fits and data.
		'''
		gc_type=self.gc_type
		responsetype=self.responsetype

		gc_spatial_data_array, initial_center_values, bad_data_indices = self.read_retina_glm_data(gc_type, responsetype)

		n_cells = int(gc_spatial_data_array.shape[2])
		pixel_array_shape_y = gc_spatial_data_array.shape[0] # Check indices: x horizontal, y vertical
		pixel_array_shape_x = gc_spatial_data_array.shape[1]

		#Make fit to all cells
		x_position_indices = np.linspace(1, pixel_array_shape_x, pixel_array_shape_x) # Note: input coming from matlab, thus indexing starts from 1
		y_position_indices = np.linspace(1, pixel_array_shape_y, pixel_array_shape_y)
		x_grid, y_grid = np.meshgrid(x_position_indices, y_position_indices)

		all_viable_cells = np.setdiff1d(np.arange(n_cells),bad_data_indices)
		# Empty numpy matrix to collect fitted RFs
		if surround_fixed:
			parameter_names = ['amplitudec', 'xoc', 'yoc', 'semi_xc', 'semi_yc', 'orientation_center', 'amplitudes', 'sur_ratio', 'offset']
			data_all_viable_cells = np.zeros(np.array([n_cells,len(parameter_names)]))
			surround_status = 'fixed'
		# if surround_fixed: # delta_semi_y
			# parameter_names = ['amplitudec', 'xoc', 'yoc', 'semi_xc', 'delta_semi_y', 'orientation_center', 'amplitudes', 'sur_ratio', 'offset']
			# data_all_viable_cells = np.zeros(np.array([n_cells,len(parameter_names)]))
			# surround_status = 'fixed'
		else:
			parameter_names = [ 'amplitudec', 'xoc', 'yoc', 'semi_xc', 'semi_yc', 'orientation_center', 'amplitudes', 'xos', 'yos', 'semi_xs',
								'semi_ys', 'orientation_surround', 'offset']
			data_all_viable_cells = np.zeros(np.array([n_cells,len(parameter_names)]))

			surround_status = 'independent'

		print(('Fitting DoG model, surround is {0}'.format(surround_status)))

		for cell_index in tqdm(all_viable_cells):
			# pbar(cell_index/n_cells)
			data_array = gc_spatial_data_array[:,:,cell_index]
			#Drop outlier cells

			# # Initial guess for center
			center_rotation_angle = float(initial_center_values[0,cell_index][4])
			if center_rotation_angle < 0: # For negative angles, turn positive
				center_rotation_angle = center_rotation_angle + 2*np.pi

			# Invert data arrays with negative sign for fitting and display. Fitting assumes that center peak is above mean
			if data_array.ravel()[np.argmax(np.abs(data_array))]  < 0:
				data_array = data_array * -1

			# Set initial guess for fitting
			if self.surround_fixed:
				# Build initial guess for (amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes, sur_ratio, offset)
				p0 = np.array([1, 7, 7, 3, 3,
							center_rotation_angle, 0.1, 3, 0])
				# boundaries=(np.array([.999, -np.inf, -np.inf, 0, 0, -2*np.pi, 0, 1, -np.inf]),
							# np.array([1, np.inf, np.inf, np.inf, np.inf, 2*np.pi, 1, np.inf, np.inf]))
				boundaries=(np.array([.999, -np.inf, -np.inf, 0, 0, 0, 0, 1, -np.inf]),
							np.array([1, np.inf, np.inf, np.inf, np.inf, 2*np.pi, 1, np.inf, np.inf]))
			# if surround_fixed: # delta_semi_y
				# # Build initial guess for (amplitudec, xoc, yoc, semi_xc, delta_semi_y, orientation_center, amplitudes, sur_ratio, offset)
				# p0 = np.array([1, 7, 7, 3, 0,
							# center_rotation_angle, 0.1, 3, 0])
				# boundaries=(np.array([.999, -np.inf, -np.inf, 0, 0, 0, 0, 1, -np.inf]),
							# np.array([1, np.inf, np.inf, np.inf, np.inf, 2*np.pi, 1, np.inf, np.inf]))
			else:
				# Build initial guess for (amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes, xos, yos, semi_xs, semi_ys, orientation_surround, offset)
				p0 = np.array([1, 7, 7, 3, 3,
							center_rotation_angle, 0.1, 7, 7,
							3*3, 3*3, center_rotation_angle, 0])
				boundaries=(np.array([.999, -np.inf, -np.inf, 0, 0, -2*np.pi, 0, -np.inf, -np.inf, 0, 0, -2*np.pi, -np.inf]),
							np.array([1, np.inf, np.inf, np.inf, np.inf, 2*np.pi, 1, np.inf, np.inf, np.inf, np.inf, 2*np.pi, np.inf]))

			try:
				if self.surround_fixed:
					popt, pcov = opt.curve_fit(self.DoG2D_fixed_surround, (x_grid, y_grid), data_array.ravel(), p0=p0, bounds=boundaries)
					data_all_viable_cells[cell_index,:]=popt
				else:
					popt, pcov = opt.curve_fit(self.DoG2D_independent_surround, (x_grid, y_grid), data_array.ravel(), p0=p0, bounds=boundaries)
					data_all_viable_cells[cell_index,:]=popt
			except:
				print(('Fitting failed for cell {0}'.format(str(cell_index))))
				data_all_viable_cells[cell_index,:]=np.nan
				bad_data_indices.append(cell_index)
				continue

			if visualize:
				#Visualize fits with data
				data_fitted = self.DoG2D_fixed_surround((x_grid, y_grid), *popt)
				fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
				plt.suptitle('celltype={0}, responsetype={1}, cell N:o {2}'.format(gc_type,responsetype,str(cell_index)), fontsize=10)
				cen=ax1.imshow(data_array, vmin=-0.1, vmax=0.4, cmap=plt.cm.gray, origin='bottom',
					extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()))
				fig.colorbar(cen, ax=ax1)

				# # Ellipses for DoG2D_fixed_surround

				if self.surround_fixed:
					e1=ellipse((popt[np.array([1,2])]),popt[3],popt[4],-popt[5]*180/np.pi,edgecolor='w', linewidth=2, fill=False)
					e2=ellipse((popt[np.array([1,2])]),popt[7]*popt[3],popt[7]*popt[4],-popt[5]*180/np.pi,edgecolor='w', linewidth=2, fill=False, linestyle='--')
					print(popt[0], popt[np.array([1,2])],popt[3],popt[4],-popt[5]*180/np.pi)
					print(popt[6], 'sur_ratio=', popt[7], 'offset=', popt[8])
				# if surround_fixed: # delta_semi_y
					# # Build initial guess for (amplitudec, xoc, yoc, semi_xc, delta_semi_y, orientation_center, amplitudes, sur_ratio, offset)
					# e1=ellipse((popt[np.array([1,2])]),popt[3],popt[3]+popt[4],-popt[5]*180/np.pi,edgecolor='w', linewidth=2, fill=False)
					# e2=ellipse((popt[np.array([1,2])]),popt[7]*popt[3],popt[7]*(popt[3]+popt[4]),-popt[5]*180/np.pi,edgecolor='w', linewidth=2, fill=False, linestyle='--')
					# print popt[0], popt[np.array([1,2])],'semi_xc=',popt[3], 'delta_semi_y=', popt[4],-popt[5]*180/np.pi
					# print popt[6], 'sur_ratio=', popt[7], 'offset=', popt[8]
				else:
					e1=ellipse((popt[np.array([1,2])]),popt[3],popt[4],-popt[5]*180/np.pi,edgecolor='w', linewidth=2, fill=False)
					e2=ellipse((popt[np.array([7,8])]),popt[9],popt[10],-popt[11]*180/np.pi,edgecolor='w', linewidth=2, fill=False, linestyle='--')
					print(popt[0], popt[np.array([1,2])],popt[3],popt[4],-popt[5]*180/np.pi)
					print(popt[6], popt[np.array([7,8])],popt[9],popt[10],-popt[11]*180/np.pi)

				print('\n')

				ax1.add_artist(e1)
				ax1.add_artist(e2)

				sur=ax2.imshow(data_fitted.reshape(pixel_array_shape_y,pixel_array_shape_x), vmin=-0.1, vmax=0.4, cmap=plt.cm.gray, origin='bottom')
				fig.colorbar(sur, ax=ax2)

				plt.show()

		#Calculate descriptive stats for params
		return parameter_names, data_all_viable_cells, bad_data_indices

	def fit_spatial_statistics(self, visualize=False): 
		'''
		Collect spatial statistics from Chichilnisky receptive field data
		'''
	
		# 2D DoG fit to Chichilnisky retina spike triggered average data. The visualize parameter will 
		# show each DoG fit for search for bad cell fits and data.
		parameter_names, data_all_viable_cells, bad_cell_indices = \
			self.fit_DoG2retina_data(visualize=False, surround_fixed=self.surround_fixed)

		all_viable_cells = np.delete(data_all_viable_cells, bad_cell_indices, 0)

		stat_df = pd.DataFrame(data=all_viable_cells,columns=parameter_names)

		# Save stats description to gc object
		self.rf_datafit_description_series=stat_df.describe()

		# Calculate xy_aspect_ratio
		xy_aspect_ratio_pd_series = stat_df['semi_yc'] / stat_df['semi_xc']
		xy_aspect_ratio_pd_series.rename('xy_aspect_ratio')
		stat_df['xy_aspect_ratio'] = xy_aspect_ratio_pd_series

		distributions = ['semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center']
		n_distributions = len(distributions)
		shape = np.zeros([n_distributions-1]) # orientation_center has two shape parameters, below alpha and beta
		loc = np.zeros([n_distributions]); scale = np.zeros([n_distributions])
		ydata=np.zeros([len(all_viable_cells),n_distributions])
		x_model_fit=np.zeros([100,n_distributions])
		y_model_fit=np.zeros([100,n_distributions])

		# Create dict for statistical parameters
		spatial_statistics_dict={}

		# Model 'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio' distributions with a gamma function. 
		for index, distribution in enumerate(distributions[:-1]):
			# fit the distributions, get the PDF distribution using the parameters
			ydata[:,index]=stat_df[distribution]
			shape[index], loc[index], scale[index] = stats.gamma.fit(ydata[:,index], loc=0)
			x_model_fit[:,index] = np.linspace(stats.gamma.ppf(0.001, shape[index], loc=loc[index], scale=scale[index]),
							stats.gamma.ppf(0.999,  shape[index], loc=loc[index], scale=scale[index]), 100)
			y_model_fit[:,index] = stats.gamma.pdf(x=x_model_fit[:,index], a=shape[index], loc=loc[index], scale=scale[index])
			
			# Collect parameters
			spatial_statistics_dict[distribution]={'shape':shape[index], 'loc':loc[index], 'scale':scale[index], 'distribution':'gamma'}

		# Model orientation distribution with beta function.  
		index += 1
		ydata[:,index]=stat_df[distributions[-1]]
		a_parameter, b_parameter, loc[index], scale[index] = stats.beta.fit(ydata[:,index], 0.6, 0.6, loc=0) #initial guess for a_parameter and b_parameter is 0.6
		x_model_fit[:,index] = np.linspace(stats.beta.ppf(0.001, a_parameter, b_parameter, loc=loc[index], scale=scale[index]),
						stats.beta.ppf(0.999,  a_parameter, b_parameter, loc=loc[index], scale=scale[index]), 100)
		y_model_fit[:,index] = stats.beta.pdf(x=x_model_fit[:,index], a=a_parameter, b=b_parameter, loc=loc[index], scale=scale[index])
		spatial_statistics_dict[distributions[-1]]={'shape':(a_parameter, b_parameter), 'loc':loc[index], 'scale':scale[index], 'distribution':'beta'}

		# Quality control images
		if visualize:
			self.show_spatial_statistics(ydata,spatial_statistics_dict, (x_model_fit, y_model_fit))
			
		# Return stats for RF creation
		return spatial_statistics_dict

	def fit_dendritic_diameter_vs_eccentricity(self, visualize=False):
		'''
		Dendritic field diameter with respect to eccentricity. Linear and quadratic fit. Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
		'''
		
		# Read dendritic field data and return linear fit with scipy.stats.linregress
		dendr_diam_parameters={}

		dendr_diam1, dendr_diam2 = self.read_dendritic_fields_vs_eccentricity_data()
		
		# Parasol fit
		gc_type = self.gc_type
	
		# Quality control. Datasets separately for visualization
		data_set_1_x = np.squeeze(dendr_diam1['Xdata'])
		data_set_1_y = np.squeeze(dendr_diam1['Ydata'])
		data_set_2_x = np.squeeze(dendr_diam2['Xdata'])
		data_set_2_y = np.squeeze(dendr_diam2['Ydata'])

		# Both datasets together
		data_all_x = np.concatenate((data_set_1_x, data_set_2_x))
		data_all_y = np.concatenate((data_set_1_y, data_set_2_y))
		
		# Limit eccentricities for central visual field studies to get better approximation at about 5 eg ecc (1mm)
		data_all_x_index = data_all_x<=self.visual_field_fit_limit
		data_all_x = data_all_x[data_all_x_index]
		data_all_y = data_all_y[data_all_x_index] # Don't forget to truncate values, too
		
		# Sort to ascending order
		data_all_x_index = np.argsort(data_all_x)
		data_all_x = data_all_x[data_all_x_index]
		data_all_y = data_all_y[data_all_x_index]

		# Get rf diameter vs eccentricity
		dendr_diam_model = self.dendr_diam_model # 'linear' # 'quadratic' # cubic
		dict_key = '{0}_{1}'.format(self.gc_type,dendr_diam_model)

		if dendr_diam_model == 'linear':
			polynomial_order = 1
			polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
			dendr_diam_parameters[dict_key] = {'intercept':polynomials[1], 'slope':polynomials[0]}
		elif dendr_diam_model == 'quadratic':
			polynomial_order = 2
			polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
			dendr_diam_parameters[dict_key] = {'intercept':polynomials[2], 'slope':polynomials[1], 
				'square':polynomials[0]}
		elif dendr_diam_model == 'cubic':
			polynomial_order = 3
			polynomials = np.polyfit(data_all_x, data_all_y, polynomial_order)
			dendr_diam_parameters[dict_key] = {'intercept':polynomials[3], 'slope':polynomials[2], 
				'square':polynomials[1], 'cube':polynomials[0]}
		
		if visualize:
			# self.show_dendritic_diameter_vs_eccentricity(gc_type, data_all_x, data_all_y, 
				# dataset_name='All data cubic fit', intercept=polynomials[3], slope=polynomials[2], square=polynomials[1], cube=polynomials[0])			
			self.show_dendritic_diameter_vs_eccentricity(self.gc_type, data_all_x, data_all_y, polynomials, 
				dataset_name='All data {0} fit'.format(dendr_diam_model))			

		return dendr_diam_parameters
		
		
class GanglionCells(Mathematics, ConstructReceptiveFields):
	'''
	Create ganglion cell object. Radii and theta in degrees. Use density to reduce N cells for faster processing.
	The key parameters are set to object instance here, called later as self.parameter_name
	'''

	def __init__(self, gc_type='parasol', responsetype='ON', eccentricity=[4,6], theta=[-5.0,5.0], model_density=1.0, randomize_position = 0.7):

		# Assertions
		assert isinstance(eccentricity,list) and len(eccentricity) == 2, 'Wrong type or length of eccentricity, aborting'
		assert isinstance(theta,list) and len(theta) == 2, 'Wrong type or length of theta, aborting'
		assert model_density <=1.0, 'Density should be <=1.0, aborting'

		# Proportion from all ganglion cells. Density of all ganglion cells is given later as a function of ecc from literature.
		proportion_of_parasol_gc_type = 0.1
		proportion_of_midget_gc_type = 0.8
		
		# Proportion of ON and OFF response types, assuming ON rf diameter = 1.2 x OFF rf diamter, Chichilnisky_2002_JNeurosci
		proportion_of_ON_response_type = 0.41
		proportion_of_OFF_response_type = 0.59

		# GC type specifications self.gc_proportion
		if all([gc_type == 'parasol', responsetype == 'ON']):
			self.gc_proportion = proportion_of_parasol_gc_type * proportion_of_ON_response_type * model_density
		elif all([gc_type == 'parasol', responsetype == 'OFF']):
			self.gc_proportion = proportion_of_parasol_gc_type * proportion_of_OFF_response_type * model_density
		elif all([gc_type == 'midget', responsetype == 'ON']):
			self.gc_proportion = proportion_of_midget_gc_type * proportion_of_ON_response_type * model_density
		elif all([gc_type == 'midget', responsetype == 'OFF']):
			self.gc_proportion = proportion_of_midget_gc_type * proportion_of_OFF_response_type * model_density
		else:
			print('Unkown GC type, aborting')
			sys.exit()
			
		self.gc_type = gc_type
		self.responsetype=responsetype

		self.deg_per_mm = 5  # Turn deg2mm retina. One mm retina is 5 deg visual field. 
		self.eccentricity = eccentricity
		self.eccentricity_in_mm = np.asarray([r / self.deg_per_mm for r in eccentricity]) # Turn list to numpy array
		self.theta = np.asarray(theta) # Turn list to numpy array
		self.randomize_position=randomize_position
		self.dendr_diam_model = 'quadratic' # 'linear' # 'quadratic' # cubic
		
		# If study concerns visual field within 4 mm (20 deg) of retinal eccentricity, the cubic fit for 
		# dendritic diameters fails close to fovea. Better limit it to more central part of the data
		if np.max(self.eccentricity_in_mm) <= 4:
			self.visual_field_fit_limit = 4
		else:
			self.visual_field_fit_limit = np.inf

		# If surround is fixed, the surround position, semi_x, semi_y (aspect_ratio) 
		# and orientation are are the same as center params. This appears to give better results.
		self.surround_fixed = 1 
		
	def fit_2GC_density_data(self):
		'''
		Fit continuous function for ganglion cell density.
		Return optimized paramters
		'''
		cell_eccentricity, cell_density = self.read_gc_density_data()

		# Gaussian + baseline fit initial values for fitting
		scale, mean, sigma, baseline0 = 1000, 0, 2, np.min(cell_density)
		popt,pcov = opt.curve_fit(self.gauss_plus_baseline,cell_eccentricity,cell_density,p0=[scale,mean,sigma,baseline0])

		return popt # gc_density_func_params

	def place_gc_units(self, gc_density_func_params, visualize=False):
		'''
		Place ganglion cells center positions to retina

		return matrix_eccentricity_randomized_all, matrix_orientation_surround_randomized_all
		'''

		# Place cells inside one polar sector with density according to mid-ecc
		eccentricity_in_mm_total = self.eccentricity_in_mm
		theta = self.theta
		randomize_position = self.randomize_position

		total_area_remove = self.sector2area(eccentricity_in_mm_total[0], np.ptp(theta))
		total_area_full = self.sector2area(eccentricity_in_mm_total[1], np.ptp(theta))
		self.total_surface_area = total_area_full - total_area_remove # Place total model area to gc object for further use

		# Loop for reasonable delta ecc to get correct density in one hand and good cell distribution from the algo on the other
		# Lets fit close to 0.1 mm intervals, which makes sense up to some 15 deg. Thereafter longer jumps would do fine.
		fit_interval = 0.1 # mm
		n_steps = int(np.round(np.ptp(eccentricity_in_mm_total)/fit_interval))
		eccentricity_steps = np.linspace(eccentricity_in_mm_total[0], eccentricity_in_mm_total[1], 1 + n_steps)

		# Initialize position arrays
		matrix_polar_angle_randomized_all = np.asarray([])
		matrix_eccentricity_randomized_all = np.asarray([])
		gc_eccentricity_group_index = np.asarray([])

		true_eccentricity_end = []
		for eccentricity_group_index, current_step in enumerate(np.arange(int(n_steps))):

			if true_eccentricity_end: # If the eccentricity has been adjusted below inside the loop
				eccentricity_in_mm = np.asarray([true_eccentricity_end, eccentricity_steps[current_step + 1]])
			else:
				eccentricity_in_mm = np.asarray([eccentricity_steps[current_step], eccentricity_steps[current_step + 1]])

			# fetch center ecc in mm
			center_ecc = np.mean(eccentricity_in_mm)

			# rotate theta to start from 0
			theta_rotated = theta - np.min(theta)
			angle = np.max(theta_rotated) # The angle is now == max theta

			# Calculate area
			assert eccentricity_in_mm[0] < eccentricity_in_mm[1], 'Radii in wrong order, give [min max], aborting'
			sector_area_remove = self.sector2area(eccentricity_in_mm[0], angle)
			sector_area_full = self.sector2area(eccentricity_in_mm[1], angle)
			sector_surface_area = sector_area_full - sector_area_remove # in mm2

			# N cells for given ecc
			my_gaussian_fit = self.gauss_plus_baseline(center_ecc,*gc_density_func_params)
			Ncells = sector_surface_area * my_gaussian_fit * self.gc_proportion

			# place cells in regular grid
			# Vector of cell positions in radial and polar directions. Angle in degrees.
			inner_arc_in_mm = (angle/360) * 2 * np.pi * eccentricity_in_mm[0]
			delta_eccentricity_in_mm = eccentricity_in_mm[1] - eccentricity_in_mm[0]
			n_segments_arc = np.sqrt(Ncells * (inner_arc_in_mm/delta_eccentricity_in_mm)) # note that the n_segments_arc and n_segments_eccentricity are floats
			n_segments_eccentricity =  (delta_eccentricity_in_mm/inner_arc_in_mm) * n_segments_arc
			int_n_segments_arc = int(round(n_segments_arc)) # cells must be integers
			int_n_segments_eccentricity = int(round(n_segments_eccentricity))

			# Recalc delta_eccentricity_in_mm given the n segments to avoid non-continuous cell densities
			true_n_cells = int_n_segments_arc * int_n_segments_eccentricity
			true_sector_area = true_n_cells / ( my_gaussian_fit * self.gc_proportion )
			true_delta_eccentricity_in_mm = (int_n_segments_eccentricity / int_n_segments_arc) * inner_arc_in_mm

			radius_segment_length = true_delta_eccentricity_in_mm / int_n_segments_eccentricity
			theta_segment_angle = angle / int_n_segments_arc

			# Set the true_eccentricity_end
			true_eccentricity_end = eccentricity_in_mm[0] + true_delta_eccentricity_in_mm

			vector_polar_angle = np.linspace(theta[0],theta[1],int_n_segments_arc)
			vector_eccentricity = np.linspace(eccentricity_in_mm[0],true_eccentricity_end - radius_segment_length,int_n_segments_eccentricity)
			# print vector_polar_angle
			# print '\n\n'
			# print vector_eccentricity
			# print '\n\n'

			# meshgrid and rotate every second to get good GC tiling
			matrix_polar_angle, matrix_eccentricity = np.meshgrid(vector_polar_angle, vector_eccentricity)
			matrix_polar_angle[::2] = matrix_polar_angle[::2] + (angle/(2 * n_segments_arc)) # rotate half the inter-cell angle

			# randomize for given proportion
			matrix_polar_angle_randomized = matrix_polar_angle + theta_segment_angle * randomize_position * (np.random.rand(matrix_polar_angle.shape[0],matrix_polar_angle.shape[1]) -0.5)
			matrix_eccentricity_randomized = matrix_eccentricity + radius_segment_length * randomize_position * (np.random.rand(matrix_eccentricity.shape[0],matrix_eccentricity.shape[1]) -0.5)

			matrix_polar_angle_randomized_all = np.append(matrix_polar_angle_randomized_all, matrix_polar_angle_randomized.flatten())
			matrix_eccentricity_randomized_all = np.append(matrix_eccentricity_randomized_all, matrix_eccentricity_randomized.flatten())

			assert true_n_cells == len(matrix_eccentricity_randomized.flatten()), "N cells dont match, check the code"			
			gc_eccentricity_group_index = np.append(gc_eccentricity_group_index, np.ones(true_n_cells) * eccentricity_group_index)
		
		# Save cell positions to current ganglion cell object
		self.gc_positions_eccentricity = matrix_eccentricity_randomized_all
		self.gc_positions_polar_angle = matrix_polar_angle_randomized_all
		self.gc_eccentricity_group_index = gc_eccentricity_group_index.astype(int) # Turn to integers, no need for floats
		
		# Visualize 2D retina with quality control for density
		# Pass the GC object to this guy, because the Visualize class is not inherited
		if visualize:
			self.show_gc_positions_and_density(matrix_eccentricity_randomized_all, matrix_polar_angle_randomized_all, gc_density_func_params)

		
		return matrix_eccentricity_randomized_all, matrix_polar_angle_randomized_all

	def get_random_samples_from_known_distribution(self,shape,loc,scale,n_cells,distribution):
		'''
		Create random samples from estimated model distribution
		'''
		if distribution == 'gamma':
			distribution_parameters = stats.gamma.rvs(a=shape, loc=loc, scale=scale, size=n_cells, random_state=1) # random_state is the seed
		elif distribution == 'beta':
			distribution_parameters = stats.beta.rvs(a=shape[0], b=shape[1], loc=loc, scale=scale, size=n_cells, random_state=1) # random_state is the seed
		
		return distribution_parameters

	def place_spatial_receptive_fields(self, spatial_statistics_dict, dendr_diam_vs_eccentricity_parameters_dict, visualize=False):
		'''
		Provide spatial receptive fields to model cells.
		Starting from 2D difference-of-gaussian parameters:
		'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center'
		'''
		
		# Get eccentricity data for all model cells
		gc_eccentricity = self.gc_positions_eccentricity

		# Get rf diameter vs eccentricity
		dendr_diam_model = self.dendr_diam_model # from __init__ method
		dict_key = '{0}_{1}'.format(self.gc_type,dendr_diam_model)
		diam_fit_params = dendr_diam_vs_eccentricity_parameters_dict[dict_key]

		if dendr_diam_model == 'linear':
			gc_diameters = diam_fit_params['intercept'] + diam_fit_params['slope']*gc_eccentricity # Units are micrometers
			polynomial_order = 1
		elif dendr_diam_model == 'quadratic':
			gc_diameters = diam_fit_params['intercept'] + diam_fit_params['slope']*gc_eccentricity \
											+ diam_fit_params['square']*gc_eccentricity**2
			polynomial_order = 2
		elif dendr_diam_model == 'cubic':
			gc_diameters = diam_fit_params['intercept'] + diam_fit_params['slope']*gc_eccentricity \
											+ diam_fit_params['square']*gc_eccentricity**2 \
											+ diam_fit_params['cube']*gc_eccentricity**3
			polynomial_order = 3

		# Set parameters for all cells
		n_cells = len(gc_eccentricity)
		n_parameters = len(spatial_statistics_dict.keys())
		gc_rf_models = np.zeros((n_cells,n_parameters))
		for index, key in enumerate(spatial_statistics_dict.keys()):
			
			shape = spatial_statistics_dict[key]['shape']
			loc = spatial_statistics_dict[key]['loc']
			scale = spatial_statistics_dict[key]['scale']
			distribution = spatial_statistics_dict[key]['distribution']
			gc_rf_models[:,index] = self.get_random_samples_from_known_distribution(shape,loc,scale,n_cells,distribution)

		# Quality control images
		if visualize:
			self.show_spatial_statistics(gc_rf_models, spatial_statistics_dict)
		
		# Calculate RF diameter scaling factor for all ganglion cells
		# Area of RF = Scaling_factor * Random_factor * Area of ellipse(semi_xc,semi_yc), solve Scaling_factor.
		scale_random_distribution=0.08 # Estimated by eye from Watanabe and Perry data. Normal distribution with scale_random_distribution 0.08 cover about 25% above and below the mean value
		random_normal_distribution = 1 + np.random.normal(scale=scale_random_distribution, size=n_cells)
		area_of_rf = self.circle_diameter2area(gc_diameters)

		'''
		The area_of_rf contains area for all model units. Its sum must fill the whole area (coverage factor = 1).
		'''
		scaling_for_coverage_1 = (self.total_surface_area *1e6 ) / np.sum(area_of_rf)   # in micrometers2

		area_of_ellipse = self.ellipse2area(gc_rf_models[:,0], gc_rf_models[:,1]) # Units are pixels
		area_scaling_factors = area_of_rf / area_of_ellipse 
		area_scaling_factors_coverage1 = area_scaling_factors * scaling_for_coverage_1
		
		# area' = scaling factor * area
		# area_of_ellipse' = scaling_factor * area_of_ellipse
		# pi*a'*b' = scaling_factor * pi*a*b
		# a and b are the semi-major and semi minor axis, like radius
		# a'*a'*constant = scaling_factor * a * a * constant
		# a'/a = sqrt(scaling_factor)
		
		# Apply scaling factors to semi_xc and semi_yc. Units are micrometers.
		semi_xc = np.sqrt(area_scaling_factors_coverage1) * gc_rf_models[:,0] * random_normal_distribution
		semi_yc = np.sqrt(area_scaling_factors_coverage1) * gc_rf_models[:,1] * random_normal_distribution

		# Scale from micrometers to millimeters and return to numpy matrix
		gc_rf_models[:,0],gc_rf_models[:,1] = semi_xc/1000, semi_yc/1000

		if visualize:
			# Quality control for diameter distribution. In micrometers.
			gc_diameters = self.area2circle_diameter(self.ellipse2area(semi_xc, semi_yc))

			polynomials = np.polyfit(gc_eccentricity, gc_diameters, polynomial_order)

			self.show_dendritic_diameter_vs_eccentricity(self.gc_type, gc_eccentricity, gc_diameters, polynomials, 
				dataset_name='All data {0} fit'.format(dendr_diam_model))			

			# gc_rf_models params: 'semi_xc', 'semi_yc', 'xy_aspect_ratio', 'amplitudes','sur_ratio', 'orientation_center'
			rho = self.gc_positions_eccentricity
			phi = self.gc_positions_polar_angle
			
			self.show_gc_receptive_fields(rho, phi, gc_rf_models, surround_fixed=self.surround_fixed) 

		return gc_rf_models
		
		
class SampleImage:
	'''
	This class gets one image at a time, and provides the cone response.
	After instantiation, the RGC group can get one frame at a time, and the system will give an impulse response.
	'''
			
	def __init__(self, mircometers_per_pixel=10, image_resolution=(100,100), temporal_resolution=1):
		'''
		Instantiate new stimulus.
		'''
		self.millimeters_per_pixel = mircometers_per_pixel / 1000 # Turn to millimeters
		self.temporal_resolution = temporal_resolution
		self.optical_aberration = 2/60 # unit is degree
		self.deg_per_mm = 5
		
	def get_image(self, image_file_name='testi.jpg'):
				
		# Load stimulus 
		image = cv2.imread(image_file_name,0) # The 0-flag calls for grayscale. Comes in as uint8 type

		# Normalize image intensity to 0-1, if RGB value
		if np.ptp(image) > 1:
			scaled_image = np.float32(image / 255)
		else:
			scaled_image = np.float32(image) # 16 bit to save space and memory
		
		return scaled_image
	
	def blur_image(self, image):
		'''
		Gaussian smoothing from Navarro 1993: 2 arcmin FWHM under 20deg eccentricity.
		'''
		
		# Turn the optical aberration of 2 arcmin FWHM to Gaussian function sigma
		sigma_in_degrees = self.optical_aberration / (2 * np.sqrt( 2 * np.log(2)))
		
		# Turn Gaussian function with defined sigma in degrees to pixel space
		sigma_in_mm = sigma_in_degrees / self.deg_per_mm
		sigma_in_pixels = sigma_in_mm / self.millimeters_per_pixel # This is small, 0.28 pixels for 10 microm/pixel resolution
		
		# Turn 
		kernel_size = (5,5) # Dimensions of the smoothing kernel in pixels, centered in the pixel to be smoothed
		image_after_optics = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_in_pixels) # sigmaY = sigmaX
		
		return image_after_optics
	
	def aberrated_image2cone_response(self, image):

		# Compressing nonlinearity. Parameters are manually scaled to give dynamic cone ouput. 
		# Equation, data from Baylor_1987_JPhysiol
		rm=25 # pA
		k=2.77e-4 # at 500 nm
		cone_sensitivity_min = 5e2
		cone_sensitivity_max = 1e4
		
		# Range
		response_range = np.ptp([cone_sensitivity_min,cone_sensitivity_max])

		# Scale
		image_at_response_scale = image * response_range # Image should be between 0 and 1
		cone_input = image_at_response_scale + cone_sensitivity_min

		# Cone nonlinearity
		cone_response = rm * (1 - np.exp(-k*cone_input))

		return cone_response
			
	
class Operator:
	'''
	Operate the generation and running of retina here
	'''

	def run_retina_construction(ganglion_cell_object, visualize=False): # class method
		'''
		Operate the construction process here
		'''

		# Run GC density fit to data, get func_params. Data from Perry_1984_Neurosci
		gc_density_func_params = ganglion_cell_object.fit_2GC_density_data()

		# Collect spatial statistics for receptive fields
		spatial_statistics_dict = ganglion_cell_object.fit_spatial_statistics(visualize=visualize)
		
		# Place ganglion cells to desired retina.
		gc_eccentricity, gc_polar_angle= ganglion_cell_object.place_gc_units(gc_density_func_params, visualize=visualize)

		# Get fit parameters for dendritic field diameter with respect to eccentricity. Linear and quadratic fit. 
		# Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
		dendr_diam_vs_eccentricity_parameters_dict = ganglion_cell_object.fit_dendritic_diameter_vs_eccentricity(visualize=visualize)

		# Construct receptive fields. Centers are saved in the object
		gc_rf_models = ganglion_cell_object.place_spatial_receptive_fields(spatial_statistics_dict, dendr_diam_vs_eccentricity_parameters_dict, visualize)
		
		# At this point the spatial receptive fieldS are constructed. The positions are in gc_eccentricity, gc_polar_angle, 
		# and the rf parameters in gc_rf_models
		TÄHÄN JÄIT: EKSENTRISYYSINDEXIT OVAT NYT OLEMASSA. JOS COVERAGE FACTOR PIDETÄÄN == 1, JA KUKIN DENDR ALUE NORMALISOIDAAN
		SEN MUKAISESTI NIIN GC_DENDRITIC_DIAMETER KIRJALLISUUDESTA EI ILMEISESTI PIDÄ ENÄÄ PAIKKAANSA. MIKSI?
		ovatko solutiheydet väärin vai ovatko diameter fitit väärin. Ehkä solutiheyksiin on eksynyt monia solutyyppejä.
		samoin ehkä diameter fitteihin. Päädytäänkö vaan coverage faktoriin 1 ja käyttämään dendr diam kirjallisuudesta vain vertailuna?
		
		plt.show()
		
	def run_stimulus_sampling(sample_image_object, visualize=0):

		one_frame = sample_image_object.get_image()
		# one_frame = np.logspace(-3, 0, num=1000)
		one_frame_after_optics = sample_image_object.blur_image(one_frame)
		cone_response = sample_image_object.aberrated_image2cone_response(one_frame_after_optics)
		
		if visualize:
			fig, ax = plt.subplots(nrows=2,ncols=3)
			axs = ax.ravel()
			axs[0].hist(one_frame.flatten(),20)
			axs[1].hist(one_frame_after_optics.flatten(),20)
			axs[2].hist(cone_response.flatten(),20)
			
			axs[3].imshow(one_frame, cmap='Greys')
			axs[4].imshow(one_frame_after_optics, cmap='Greys')
			axs[5].imshow(cone_response, cmap='Greys')
			
			# plt.xscale('log')
			plt.show()
	
		
if __name__ == "__main__":

	
	# # Define eccentricity and theta in degrees. Model_density is the relative density compared to true macaque values.
	# ganglion_cell_object = GanglionCells(gc_type='parasol', responsetype='ON', eccentricity=[3,7], theta=[-30.0,30.0], density=1.0, randomize_position = 0.6)

	# Operator.run_retina_construction(ganglion_cell_object, visualize=1)

	parasol_ON_object = GanglionCells(gc_type='parasol', responsetype='ON', eccentricity=[3,7], theta=[-30.0,30.0], model_density=1.0, randomize_position = 0.6)

	Operator.run_retina_construction(parasol_ON_object, visualize=0)

	# parasol_OFF_object = GanglionCells(gc_type='parasol', responsetype='OFF', eccentricity=[3,7], theta=[-30.0,30.0], model_density=1.0, randomize_position = 0.6)

	# Operator.run_retina_construction(parasol_OFF_object, visualize=1)

	# midget_ON_object = GanglionCells(gc_type='midget', responsetype='ON', eccentricity=[3,7], theta=[-30.0,30.0], model_density=1.0, randomize_position = 0.6)

	# Operator.run_retina_construction(midget_ON_object, visualize=0)

	# midget_OFF_object = GanglionCells(gc_type='midget', responsetype='OFF', eccentricity=[3,7], theta=[-30.0,30.0], model_density=1.0, randomize_position = 0.6)

	# Operator.run_retina_construction(midget_OFF_object, visualize=0)
	
	# sample_image_object = SampleImage()
	
	# Operator.run_stimulus_sampling(sample_image_object, visualize=1)
	
	
# TODO:
# 	TÄHÄN JÄIT: Jaa on ja off -tiheyksiin
# cone2CG:
	# # One photoreceptor type image convolution
	
	# # Temporal convolution missing.
# Visual stimuli series/video to cone response

#   -RF coverage is not right for midgets? -- QA estimate coverage factor? 
#   -xy_aspcects_ratio show to some extent bimodal distribution. It should be convertable to all y_sigma > x_sigma, but attempt to do this failed. Fit quality decreased
#	-consider implementing significant correlations between spatial parameters
#   -model and apply time behaviour
#   -construct LGN. Probably a filter rather than spiking neurons. The latter dont make sense because we are interested in cx, not sub-cx.s

os.chdir(cwd)

'''
This is code for building macaque retinal filters corresponding to midget and parasol cells' responses.
We keep modular code structure, to be able to add new features at later phase.

The cone photoreceptor sampling is approximated as achromatic (single) compressive cone response(Baylor_1987_JPhysiol).

Visual angle ( A) in degrees from previous studies (Croner and Kaplan, 1995) was approksimated with relation 5 deg/mm.
This works fine up to 20 deg ecc, but undesestimates the distance thereafter. If more peripheral representations are later
necessary, the millimeters should be calculates by inverting the inverting the relation 
A = 0.1 + 4.21E + 0.038E^2 (Drasdo and Fowler, 1974; Dacey and Petersen, 1992)

We have extracted statistics of macaque ganglion cell receptive fields from literature and build continuous functions.

The density of meny cell types is inversely proportional to dendritic field coverage, 
suggesting constant coverage factor (Perry_1984_Neurosci, Wassle_1991_PhysRev).
Midget coverage factor is 1  (Dacey_1993_JNeurosci for humans; Wassle_1991_PhysRev, Lee_2010_ProgRetEyeRes).
Parasol coverage factor is 3-4 close to fovea (Grunert_1993_VisRes); 2-7 according to Perry_1984_Neurosci.
These include ON- and OFF-center cells, and perhaps other cell types.
It is likely that coverage factor is 1 for midget and parasol ON- and OFF-center cells each, 
which is also in line with Doi_2012 JNeurosci, Field_2010_Nature

The spatiotemporal receptive fields for the four cell types (parasol & midget, ON & OFF) were modelled with double ellipsoid
difference-of-Gaussians model. The original spike triggered averaging RGC data in courtesy of Chichilnisky lab. The method is
described in Chichilnisky_2001_Network, Chichilnisky_2002_JNeurosci; Field_2010_Nature.

Chichilnisky_2002_JNeurosci states that L-ON (parasol) cells have on average 21% larger RFs than L-OFF cells. 
He also shows that OFF cells have more nonlinear response to input, which is not implemented currently (a no-brainer to imeplement 
if necessary).

NOTE: bad cell indices hard coded from Chichilnisky apricot data. For another data set, visualize fits, and change the bad cells.
NOTE: If eccentricity stays under 20 deg, dendritic diameter data fitted up to 25 deg only (better fit close to fovea)

-center-surround response ratio (in vivo, anesthetized, recorded from LGN; Croner_1995_VisRes) PC: ; MC: ;
-Michelson contrast definition for sinusoidal gratings (Croner_1995_VisRes).
-Optical quality probably poses no major limit to behaviorally measured spatial vision (Williams_1981_IOVS).
-spatial contrast sensitivity nonlinearity in the center subunits is omitted. This might reduce senstivity to natural scenes Turner_2018_eLife.

-quality control: compare to Watanabe_1989_JCompNeurol
	-dendritic diameter scatter is on average (lower,upper quartile) 21.3% of the median diameter in the local area

	Parasol dendritic field diameter: temporal retina 51.8 microm + ecc(mm) * 20.6 microm/mm,
	nasal retina; 115.5 microm + ecc(mm) * 6.97 microm/mm

'''