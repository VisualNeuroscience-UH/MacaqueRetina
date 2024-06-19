# Visualization
import matplotlib.pyplot as plt

# Builtin
from pathlib import Path
import time
import warnings
import math

# Comput Neurosci
import brian2.units as b2u

# sys.path.append(Path(__file__).resolve().parent.parent)
# Start measuring time
start_time = time.time()

# Local
from project.project_manager_module import ProjectManager

warnings.simplefilter("ignore")


"""
This is code for building macaque retinal filters corresponding to midget and parasol unit responses
for temporal hemiretina. We keep modular code structure, to be able to add new features at later phase. Name
'unit' is used as distinction from biological cell.

Visual angle (A) in degrees from previous studies (Croner and Kaplan, 1995) was approximated with relation 5 deg/mm. 
This works fine up to 20 deg ecc, but underestimates the distance thereafter. If more peripheral representations are 
necessary, the millimeters should be calculated by inverting the relation A = 0.1 + 4.21E + 0.038E^2 (Drasdo and Fowler, 1974;
Dacey and Petersen, 1992). Current implementation uses one deg = 220um (Perry et al 1985). One mm retina is ~4.55 deg visual field.

We have extracted statistics of macaque ganglion cell receptive fields from literature and build continuous functions.

The density of many cell types is inversely proportional to dendritic field coverage, suggesting constant coverage factor 
(Perry_1984_Neurosci, Wassle_1991_PhysRev). Midget coverage factor is 1  (Dacey_1993_JNeurosci for humans; Wassle_1991_PhysRev, 
Lee_2010_ProgRetEyeRes). It is likely that coverage factor is 1 for midget and parasol 
ON- and OFF-center cells each, which is also in line with Doi_2012 JNeurosci, Field_2010_Nature

The spatiotemporal receptive fields for the four unit types (parasol & midget, ON & OFF) were modelled with double ellipsoid 
difference-of-Gaussians model. The original spike triggered averaging RGC data in courtesy of Chichilnisky lab. The method is 
described in Chichilnisky_2001_Network, Chichilnisky_2002_JNeurosci Field_2010_Nature.

Chichilnisky_2002_JNeurosci states that L-ON (parasol) cells have on average 21% larger RFs than L-OFF cells. He also shows that 
OFF cells have more nonlinear response to input, which is not implemented currently (a no-brainer to implement if necessary).

NOTE: bad cell indices and metadata hard coded from Chichilnisky apricot data at fit_module ApricotData class. 
For another data set change metadata, visualize fits and change the bad cells.
NOTE: Visual stimulus video default options are hard coded at visual_stimulus_module.VideoBaseClass class.
NOTE: If eccentricity stays under 20 deg, dendritic diameter data fitted up to 25 deg only (better fit close to fovea)

-center-surround response ratio (in vivo, anesthetized, recorded from LGN; Croner_1995_VisRes) PC: ; MC: ;
-Michelson contrast definition for sinusoidal gratings (Croner_1995_VisRes).
-optical quality probably poses no major limit to behaviorally measured spatial vision (Williams_1981_IOVS).
-spatial contrast sensitivity nonlinearity in the center subunits is omitted. This might reduce sensitivity to natural scenes Turner_2018_eLife.

-quality control: compare to Watanabe_1989_JCompNeurol
    -dendritic diameter scatter is on average (lower,upper quartile) 21.3% of the median diameter in the local area

    Parasol dendritic field diameter: temporal retina 51.8 microm + ecc(mm) * 20.6 microm/mm, nasal retina; 115.5 microm + ecc(mm) * 6.97 microm/mm

spatial_model FIT : Fit ellipse to center and surround
spatial_model : VAE : Variational autoencoder. The model reconstructs the full receptive field and generates new samples from the latent space.
    
Contrast gain control (CGC) is implemented according to Victor_1987_JPhysiol using numerical integration in discretized temporal domain.
The unit parameters are drawn from Benardete_1999_VisNeurosci for parasol units and Benardete_1997_VisNeurosci_a for midget units.
We are sampling from Benardete Kaplan data assuming triangular distribution of the reported tables of statistics (original data points not shown).
For a review of physiological mechanisms, see Demb_2008_JPhysiol and Beaudoin_2007_JNeurosci.

Parasol ON firing rate for spatially uniform monochromatic light of wavelength 561 (530, 430) nm and intensity 4300
(4200, 2400) photons/microm2/s, incident on the photoreceptors: Across all preparations examined firing rates varied from about 5
to 20 Hz. From Shlens_2009_JNeurosci/Methods.

The firing rate gain (spikes/(second * unit contrast)), parameter "A" in the Victor model, comes from Benardete_1999_VisNeurosci for parasol units and Benardete_1997_VisNeurosci_a for midget units.
The max amplitude the the center spatial DoG model in Victor_1987_JPhysiol is 1.0. 

The cone photoreceptor sampling is approximated as achromatic (single) compressive cone response(Baylor_1987_JPhysiol).

Most of the functions fitted have no theoretical significance but provide empirical expressions for interpolation.
"""

"""
Technical notes:
Use keyword substring "file" in filenames, and "folder" or "path" in folder names to assert that they are turned into pathlib objects. 
Path structure is assumed to be model_root_path/project/experiment/output_folder

Abbreviations:
ana : analysis
cen : center
col : column
dd : dendritic diameter
exp : experimental
full : full absolute path 
gc : ganglion cell
gen : generated
lit : literature
mtx : matrix
param : parameter
sur : surround
viz : visualization

Custom suffixes:
_df : pandas dataframe
_mm : millimeter
_np : numpy array
_pix : pixel
_t : tensor
_um : micrometer
"""

"""
# TODO: Stodden_2016_Science: Software metadata should
include, at a minimum, the title, authors,
version, language, license, Uniform Resource
Identifier/DOI, software description (including
purpose, inputs, outputs, dependencies),
and execution requirements.
"""

"""
Main paths in different operating systems
"""
model_root_path = "/opt3/Laskenta/Models"
git_repo_root = Path(r"/opt2/Git_Repos/MacaqueRetina")
ray_root_path = None  # if None, ray_results are saved to model_root_path/project/experiment/output_folder/ray_results

"""
Project name
"""
project = "Retina"


"""
Current experiment. Use distinct folders fo distinct stimuli.
"""
experiment = "test_experiment"


"""
Input context
Existing retina model, any general files
"""
input_folder = "../in"  # input figs, videos, models


"""
Stimulus context
Stimulus images and videos
"""
stimulus_folder = "stim"  # stim_sine_grating_sf2p0_crf_14_7"  # "stim_luminance_onset"


"""
Data context for output. 
"""

output_folder = "testi_stim"  # "parasol_on_stim_sine_grating_sf2p0_crf_14_7"  # "parasol_on_luminance_onset"


"""
Remove random variations by setting the numpy random seed
"""
numpy_seed = 42  # random.randint(0, 1000000)  # 42

"""
Computing device
For small retinas cpu is faster. Use cpu if you do not have cuda.
"""
# After training with a device, the model must be loaded to same device. Seems to be a Pytorch quirk.
device = "cpu"  # "cpu" or "cuda"

"""
### Housekeeping ###. Do not comment out.
"""
model_root_path = Path(model_root_path)
path = Path.joinpath(model_root_path, Path(project), experiment)

# When training or tuning generative VAE model, multiple hyperparameters are set at the RetinaVAE class.
# For training, see __init__ method. For tuning, the __init__ contains search space and
# _set_ray_tuner contains the starting point.

# For "load_model" training_mode, the model is loaded from model_file_name at output_folder (primary)
# or input_folder. The correct model name (including time stamp) must be given in the model_file_name.

# VAE RF is generated in experimental data space originating from macaque peripheral retina.
# VAE RF sizes need to be scaled according to eccentricity.
# This scaling is based on dendritic field diameter (DoG model diameter) comparison between
# experimental data fit and literature data on dendritic field diameter vs eccentricity.
# When the spatial model is VAE, the DoG model is fitted twice. First, the experimental data is fitted
# to get the RF scaling. Second, the scaled VAE RF is fitted to get a description of the final RF.
# The experimental (first) DoG_model fit for VAE is automatically changed to to ellipse_fixed. This way
# the scaling is not dependent on the selected DoG_model. The final RF fit is what you call for in DoG_model.

# The model_file_name must must be of correct gc_type and response_type.

# Note: FIT model ellipse independent does not correlate the center and surround parameters. Thus they are independent, which
# is not the case in the VAE model, and not very physiological.

gc_type = "parasol"  # "parasol" or "midget"
response_type = "on"  # "on" or "off"

# These values are used for building a new retina
my_retina = {
    "gc_type": gc_type,
    "response_type": response_type,
    # "ecc_limits_deg": [36, 39.4],  # eccentricity in degrees
    # "pol_limits_deg": [-4, 4],  # polar angle in degrees
    "ecc_limits_deg": [4.5, 5.5],  # eccentricity in degrees
    "pol_limits_deg": [-3, 3],  # polar angle in degrees
    # "ecc_limits_deg": [4.0, 6.0],  # eccentricity in degrees
    # "pol_limits_deg": [-15.0, 15.0],  # polar angle in degrees
    "model_density": 1.0,  # 1.0 for 100% of the literature density of ganglion cells
    "dd_regr_model": "quadratic",  # linear, quadratic, cubic, loglog. For midget < 20 deg, use quadratic; for parasol use loglog
    "ecc_limit_for_dd_fit": 20,  # 20,  # degrees, math.inf for no limit
    "stimulus_center": 5.0 + 0j,  # degrees, this is stimulus_position (0, 0)
    "temporal_model": "subunit",  # fixed, dynamic, subunit
    "center_mask_threshold": 0.1,  # 0.1,  Limits rf center extent to values above this proportion of the peak values
    "spatial_model": "VAE",  # "FIT" or "VAE" for variational autoencoder
    "DoG_model": "ellipse_fixed",  # 'ellipse_independent', 'ellipse_fixed' or 'circular'
    "rf_coverage_adjusted_to_1": False,  # False or True. Applies to FIT only, scales sum(unit center areas) = retina area
    "training_mode": "load_model",  # "train_model" or "tune_model" or "load_model" for loading trained or tuned. Applies to VAE only
    "model_file_name": None,  # None for most recent or "model_[GC TYPE]_[RESPONSE TYPE]_[DEVICE]_[TIME_STAMP].pt" at input_folder. Applies to VAE "load_model" only
    "ray_tune_trial_id": None,  # Trial_id for tune, None for loading single run after "train_model". Applies to VAE "load_model" only
}


"""
Stimulus video will be saved on output_folder in mp4 format (viewing) and hdf5 format (reloading)
Valid stimulus_options include (overriding visual_stimulus_module.VideoBaseClass):

image_width: in pixels
image_height: in pixels
pix_per_deg: pixels per degree
fps: frames per second
duration_seconds: stimulus duration
baseline_start_seconds: midgray at the beginning
baseline_end_seconds: midgray at the end
pattern:
    'sine_grating'; 'square_grating'; 'colored_temporal_noise'; 'white_gaussian_noise';
    'natural_images'; 'natural_video'; 'temporal_sine_pattern'; 'temporal_square_pattern';
    'temporal_chirp_pattern'; contrast_chirp_pattern; 'spatially_uniform_binary_noise'
stimulus_form: 'circular'; 'rectangular'; 'annulus'

For stimulus_form annulus, additional arguments are:
size_inner: in degrees
size_outer: in degrees

stimulus_position: in degrees, (0,0) is the center.
stimulus_size: In degrees. Radius for circle and annulus, half-width for rectangle. 0 for full image.
background: "mean", "intensity_min", "intensity_max" or value. The "frame", around stimulus in time and space, incl baseline times
contrast: between 0 and 1
mean: mean stimulus intensity in cd/m2

If intensity (min, max) is defined, it overrides contrast and mean becomes baseline.

For sine_grating and square_grating, additional arguments are:
temporal_frequency: in Hz
spatial_frequency: in cycles per degree
orientation: in degrees

For all temporal and spatial gratings, additional argument is
phase_shift: between 0 and 2pi

For spatially_uniform_binary_noise, additional argument is
on_proportion: between 0 and 1, proportion of on-stimulus, default 0.5
direction: 'increment' or 'decrement'
stimulus_video_name: name of the stimulus video

With assuming rgb voltage = cd/m2, and average pupil diameter of 3 mm, the mean voltage of 128 in background
would mean ~ 905 Trolands. Td = lum * pi * (diam/2)^2, resulting in 128 cd/m2 = 128 * pi * (3/2)^2 ~ 905 Td.

VAE rf have different resolution from original RF data, if the estimated eccentricity is different from the original data.
VAE rf have different amplitude from original RF data, because the VAE model operates with values between 0 and 1. Later
the median is removed to get the zero level to approximately match the original data. 

Note that in experiments (below), tuple values are captured for varying each tuple value separately.
"""

my_stimulus_options = {
    # Shared btw stimulus and simulate_retina
    "image_width": 240,  # 752 for nature1.avi
    "image_height": 240,  # 432 for nature1.avi
    "pix_per_deg": 60,
    "dtype_name": "float64",  # low contrast needs "float16", for performance, use "uint8",
    "fps": 350,  # 300 for good cg integration
    "duration_seconds": 10,  # actual frames = floor(duration_seconds * fps)
    "baseline_start_seconds": 0.5,  # Total duration is duration + both baselines
    "baseline_end_seconds": 0.0,
    # "pattern": "sine_grating",  # One of the StimulusPatterns
    # "pattern": "natural_images",  # One of the StimulusPatterns
    # "pattern": "temporal_sine_pattern",  # One of the StimulusPatterns
    "pattern": "temporal_chirp_pattern",  # One of the StimulusPatterns
    # "pattern": "contrast_chirp_pattern",  # One of the StimulusPatterns
    "stimulus_form": "rectangular",
    "size_inner": 0.1,  # deg, applies to annulus only
    "size_outer": 1,  # deg, applies to annulus only
    "stimulus_position": (0.0, 0.0),  # relative to stimuls center in retina
    "stimulus_size": 1,  # 0.04,  # 2,  # deg, radius for circle, sidelen/2 for rectangle.
    "temporal_frequency": 10,  # 0.01,  # 4.0,  # 40,  # Hz
    "temporal_frequency_range": (0.5, 50),  # Hz, applies to temporal chirp only
    "spatial_frequency": 5.0,  # cpd
    "orientation": 0,  # degrees
    "phase_shift": 0,  # math.pi + 0.1,  # radians
    "stimulus_video_name": f"{stimulus_folder}.mp4",
    # Interdependent intensity variables:
    "contrast": 0.99,  # mean +- contrast * mean
    "mean": 128,  # Consider this as cd/m2
    # intensity (min, max) overrides contrast and mean unless the line is commented out
    # "intensity": (0, 100),
    "intensity": (0, 255),
    "background": 128,  # "mean", "intensity_min", "intensity_max" or value.
    "ND_filter": 0.0,  # 0.0, log10 neutral density filter factor, can be negative
}

# For external video and image input. See visual_stimulus_module.VideoBaseClass for more options.
my_stimulus_metadata = {
    "stimulus_file": "testi.jpg",  # nature1.avi, testi.jpg
    "pix_per_deg": 30,  # VanHateren_1998_ProcRSocLondB 2 arcmin per pixel
    "apply_cone_filter": False,
    "fps": 25,
}

# Each gc response file contain n_trials
n_files = 1

# Either n_trials or n_cells must be 1, and the other > 1
# Running multiple trials on multiple units is not supported
my_run_options = {
    "unit_index": None,  # list of ints or None for all units
    "n_trials": 1,  # For each of the response files
    "spike_generator_model": "poisson",  # poisson or refractory
    "save_data": True,
    "gc_response_filenames": [f"gc_response_{x:02}" for x in range(n_files)],
    "simulation_dt": 0.0001,  # in sec 0.001 = 1 ms
}


############################
###  Semi-constant variables
############################

apricot_metadata = {
    "data_microm_per_pix": 60,
    "data_spatialfilter_height": 13,
    "data_spatialfilter_width": 13,
    "data_fps": 30,
    "data_temporalfilter_samples": 15,
    "apricot_data_folder": git_repo_root.joinpath(r"retina/apricot_data"),
}

# Proportion from all ganglion cells. Density of all ganglion cells is given later as a function of ecc from literature.
proportion_of_parasol_gc_type = 0.08
proportion_of_midget_gc_type = 0.64

# Proportion of ON and OFF response type units, assuming ON rf diameter = 1.2 x OFF rf diameter, and
# coverage factor =1; Chichilnisky_2002_JNeurosci
proportion_of_ON_response_type = 0.40
proportion_of_OFF_response_type = 0.60

# Perry_1985_VisRes; 223 um/deg in the fovea, 169 um/deg at 90 deg ecc.
# With this relationship one mm retina is ~4.55 deg visual field.
# Constant (linear) approximation of quadratic formula from
# Goodchild_1996_JCompNeurol 0.038 * x**2 + 4.21 * x + 0.1
# gives 229 um/degree. The R2 (constant vs quadratic) is
# > 0.999 for ecc 0.1 - 4 mm, and > 0.97 for ecc 0.1 - 20 mm.
deg_per_mm = 1 / 0.229

optical_aberration = 2 / 60  # deg , 2 arcmin, Navarro 1993 JOSAA

# Parameters for cortical - visual coordinate transformation.
# a for macaques between 0.3 - 0.9, Schwartz 1994 citing Wilson et al 1990 "The perception of form"
# in Visual perception: The neurophysiological foundations, Academic Press
# k has been pretty open.
# However, if we relate 1/M = (a/k) + (1/k) * E and M = (1/0.077) + (1/(0.082 * E)), we get
# Andrew James, personal communication: k=1/.082, a=.077/.082
visual2cortical_params = {
    "a": 0.077 / 0.082,
    "k": 1 / 0.082,
}

# Compressing cone nonlinearity. Parameters are manually scaled to give dynamic cone ouput.
# Equation, data from Baylor_1987_JPhysiol
# Cone noise parameters from Angueyra_2013_NatNeurosci and Ala-laurila_2011_NatNeurosci
# Noise magnitude is a free parameter for now.
cone_general_params = {
    "rm": 25,  # pA
    "k": 2.77e-4,  # at 500 nm
    "sensitivity_min": 5e2,
    "sensitivity_max": 2e4,
    "cone2gc_midget": 9,  # um, 1 SD of Gaussian
    "cone2gc_parasol": 27,  # um 27
    "cone2gc_cutoff_SD": 1,  # 3 SD is 99.7% of Gaussian
    "cone2bipo_cutoff_SD": 1,
    "cone_noise_magnitude": 1.0,  # 0.2  # firing rate relative to Benardete mean values, 0 for no noise
    "cone_noise_wc": [14, 160],  # lorenzian freqs, Angueyra_2013_NatNeurosci Fig1
}

# # Parameters from Clark_2013_PLoSComputBiol model BHL
# # Light intensity photons/microm^2/second
# cone_signal_parameters = {
#     "unit": "mV",
#     "A_pupil": 9.0,  # * b2u.mm2,  # mm^2
#     "lambda_nm": 560,  # nm 555 monkey Clark models: DN 650
#     "input_gain": 1.0,  # unitless
#     "r_dark": -40 * b2u.mV,  # dark potential
#     "max_response": -24.4 * b2u.mV,  # "mV", measured for a strong flash
#     "alpha": -1.1 * b2u.mV * b2u.ms,
#     "beta": 0.0484 * b2u.ms,  # unitless 0.044 * 1.1
#     "gamma": 0.93,  # unitless
#     "tau_y": 38 * b2u.ms,
#     "n_y": 1.5,  # unitless
#     "tau_z": 20 * b2u.ms,
#     "n_z": 7.0,  # unitless
#     "tau_r": 39 * b2u.ms,
#     "filter_limit_time": 3.0 * b2u.second,
# }

# # Parameters from Clark_2013_PLoSComputBiol model B
# # Replicates Figs 5E
# # Light intensity photons/microm^2/second
# cone_signal_parameters = {
#     "unit": "mV",
#     "A_pupil": 9.3,  # * b2u.mm2,  # mm^2
#     "lambda_nm": 560,  # nm 555 monkey Clark models: DN 650
#     "input_gain": 1.0,  # unitless
#     "r_dark": -40 * b2u.mV,  # dark potential
#     "max_response": -26.2 * b2u.mV,  # "mV", measured for a strong flash
#     "alpha": -2.1 * b2u.mV * b2u.ms,
#     "beta": 0.1407 * b2u.ms,  #  0.067 * 2.1
#     "gamma": 0.57,  # unitless
#     "tau_y": 20 * b2u.ms,
#     "n_y": 3.0,  # unitless
#     "tau_z": 20 * b2u.ms,
#     "n_z": 7.0,  # unitless
#     "tau_r": 50 * b2u.ms,
#     "filter_limit_time": 3.0 * b2u.second,
# }

# # Parameters from Clark_2013_PLoSComputBiol model DN
# # Replicates Figs 5C
# # Light intensity photons/microm^2/second
# cone_signal_parameters = {
#     "unit": "mV",
#     "A_pupil": 9.0,  # * b2u.mm2,  # mm^2
#     "lambda_nm": 650,  # nm 555 monkey Clark models: DN 650
#     "input_gain": 1.0,  # unitless
#     "r_dark": -40 * b2u.mV,  # dark potential
#     "max_response": -61.3,  # "mV", measured for a strong flash
#     # Angueyra: unitless; Clark: mV * microm^2 * ms / photon
#     "alpha": -1.4 * b2u.mV * b2u.ms,
#     "beta": 0.1036 * b2u.ms,  # 0.074 * 1.4
#     "gamma": 0.22,  # unitless
#     "tau_y": 18 * b2u.ms,
#     "n_y": 3.7,  # unitless
#     "tau_z": 13 * b2u.ms,
#     "n_z": 7.8,  # unitless
#     "tau_r": 66 * b2u.ms,
#     "filter_limit_time": 3.0 * b2u.second,
# }

# Parameters from Angueyra_2022_JNeurosci, model according to Clark_2013_PLoSComputBiol
# Light intensity photons/microm^2/second
cone_signal_parameters = {
    "unit": "pA",
    "A_pupil": 9.0,  # * b2u.mm2,  # mm^2
    "lambda_nm": 555,  # nm 555 monkey Clark models: DN 650
    "input_gain": 1.0,  # unitless
    "r_dark": -136 * b2u.pA,  # dark current
    "max_response": 116.8 * b2u.pA,  # "pA", measured for a strong flash
    # Angueyra: unitless; Clark: mV * microm^2 * ms / photon
    "alpha": 19.4 * b2u.pA * b2u.ms,
    "beta": 0.36 * b2u.ms,  # unitless
    "gamma": 0.448,  # unitless
    "tau_y": 4.49 * b2u.ms,
    "n_y": 4.33,  # unitless
    "tau_z": 166 * b2u.ms,
    "n_z": 1.0,  # unitless
    "tau_r": 4.78 * b2u.ms,
    "filter_limit_time": 3.0 * b2u.second,
}

bipolar_general_params = {
    "bipo2gc_div": 6,  # Divide GC dendritic diameter to get bipolar/subunit SD
    "bipo2gc_cutoff_SD": 2,  # Multiplier for above value
    "cone2bipo_cen_sd": 10,  # um, Turner_2018_eLife
    "cone2bipo_sur_sd": 150,
    "bipo_sub_sur2cen": 1.0,  # Surround / Center amplitude ratio
}
# Recovery function from Berry_1998_JNeurosci, Uzzell_2004_JNeurophysiol
# abs and rel refractory estimated from Uzzell_2004_JNeurophysiol,
# Fig 7B, bottom row, inset. Parasol ON unit.
# Uzzell_2004_JNeurophysiol instataneous firing rates for binary noise go up to 320 Hz
refractory_params = {
    "abs_refractory": 1,
    "rel_refractory": 3,
    "p_exp": 4,
    "clip_start": 0,
    "clip_end": 100,
}

# If you see patterning, reduce unit_distance_threshold
# If you see clustering, increase increase diffusion_speed, unit_repulsion_stregth,
# or carefully unit_distance_threshold
# Algorithms are
# Voronoi-based Layout with Loyd's Relaxation
# "voronoi" (v) : better for big retinas, fast
# Force Based Layout Algorithm with Boundary Repulsion
# "force" (f) : better for small retinas, slow
# None : initial random placement. Nonvarying with fixed seed above. Good for testing and speed.
gc_placement_params = {
    "algorithm": None,  # "voronoi" or "force" or None
    "n_iterations": 30,  # v 20, f 5000
    "change_rate": 0.0005,  # f 0.001, v 0.5
    "unit_repulsion_stregth": 7,  # 10 f only
    "unit_distance_threshold": 0.1,  # f only, adjusted with ecc
    "diffusion_speed": 0.001,  # f only, adjusted with ecc
    "border_repulsion_stength": 0.2,  # f only
    "border_distance_threshold": 0.001,  # f only
    "show_placing_progress": True,  # True False
    "show_skip_steps": 1,  # v 1, f 100
}

cone_placement_params = {
    "algorithm": None,  # "voronoi" or "force" or None
    "n_iterations": 15,  # v 20, f 300
    "change_rate": 0.0005,  # f 0.0005, v 0.5
    "unit_repulsion_stregth": 2,  # 10 f only
    "unit_distance_threshold": 0.1,  # f only, adjusted with ecc
    "diffusion_speed": 0.001,  # f only, adjusted with ecc
    "border_repulsion_stength": 5,  # f only
    "border_distance_threshold": 0.0001,  # f only
    "show_placing_progress": True,  # True False
    "show_skip_steps": 1,  # v 1, f 100
}

bipolar_placement_params = {
    "algorithm": None,  # "voronoi" or "force" or None
    "n_iterations": 15,  # v 20, f 300
    "change_rate": 0.0005,  # f 0.0005, v 0.5
    "unit_repulsion_stregth": 2,  # 10 f only
    "unit_distance_threshold": 0.1,  # f only, adjusted with ecc
    "diffusion_speed": 0.005,  # f only, adjusted with ecc
    "border_repulsion_stength": 5,  # f only
    "border_distance_threshold": 0.0001,  # f only
    "show_placing_progress": True,  # True False
    "show_skip_steps": 1,  # v 1, f 100
}

# For VAE, this is enough to have good distribution between units.
rf_repulsion_params = {
    "n_iterations": 200,  # 200
    "change_rate": 0.01,
    "cooling_rate": 0.999,  # each iteration change_rate = change_rate * cooling_rate
    "border_repulsion_stength": 5,
    "show_repulsion_progress": False,  # True False
    "show_only_unit": None,  # None or int for unit idx
    "show_skip_steps": 5,
    "savefigname": None,
}

# For subunit model we assume constant bipolar to cone ratio as function of ecc.
# Boycott_1991_EurJNeurosci Table 1 has bipolar densities at 6.5 mm ecc.
# Inputs to ganglion cells:
# -OFF parasol: Diffuse Bipolars, DB2 and DB3 (Jacoby_2000_JCompNeurol, )
# -ON parasol:  Diffuse Bipolars, DB4 and DB5 (Marshak_2002_VisNeurosci, Boycott_1991_EurJNeurosci)
# -OFF midget: Flat Midget Bipolars, FMB (Wässle_1994_VisRes, Freeman_2015_eLife)
# -ON midget: Invaginating Midget Bipolars, IMB (Wässle_1994_VisRes)

bipolar2gc_dict = {
    "midget": {"on": ["IMB"], "off": ["FMB"]},
    "parasol": {"on": ["DB4", "DB5"], "off": ["DB2", "DB3"]},
}

my_retina_append = {
    "mosaic_file": gc_type + "_" + response_type + "_mosaic.csv",
    "spatial_rfs_file": gc_type + "_" + response_type + "_spatial_rfs.npz",
    "ret_file": gc_type + "_" + response_type + "_ret.npz",
    "proportion_of_parasol_gc_type": proportion_of_parasol_gc_type,
    "proportion_of_midget_gc_type": proportion_of_midget_gc_type,
    "proportion_of_ON_response_type": proportion_of_ON_response_type,
    "proportion_of_OFF_response_type": proportion_of_OFF_response_type,
    "deg_per_mm": deg_per_mm,
    "optical_aberration": optical_aberration,
    "cone_general_params": cone_general_params,
    "cone_signal_parameters": cone_signal_parameters,
    "bipolar_general_params": bipolar_general_params,
    "refractory_params": refractory_params,
    "gc_placement_params": gc_placement_params,
    "cone_placement_params": cone_placement_params,
    "bipolar_placement_params": bipolar_placement_params,
    "rf_repulsion_params": rf_repulsion_params,
    "visual2cortical_params": visual2cortical_params,
    "bipolar2gc_dict": bipolar2gc_dict,
}

my_retina.update(my_retina_append)

literature_data_folder = git_repo_root.joinpath(r"retina/literature_data")

# Define digitized literature data files for gc density and dendritic diameters.
# Dendritic diameter is the diameter of the circle that has the
# same area as the dendritic field polygon, as defined in Watanabe 1989.
# Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
# Define literature data files for linear temporal models.
# Data from Benardete_1999_VisNeurosci and Benardete_1997_VisNeurosci
# Define literature data files for cone density.
# Data from Packer_1989_JCompNeurol

# gc_density_fullpath = literature_data_folder / "Perry_1984_Neurosci_GCdensity_c.mat"
gc_density_fullpath = (
    literature_data_folder / "Perry_1984_Neurosci_GCdensity_Fig8_c.npz"
)
if my_retina["gc_type"] == "parasol":
    dendr_diam1_fullpath = (
        literature_data_folder / "Perry_1984_Neurosci_ParasolDendrDiam_Fig6A_c.npz"
    )
    dendr_diam2_fullpath = (
        literature_data_folder / "Watanabe_1989_JCompNeurol_ParasolDendrDiam_Fig7_c.npz"
    )
    dendr_diam3_fullpath = (
        literature_data_folder
        / "Goodchild_1996_JCompNeurol_Parasol_DendDiam_Fig2A_c.npz"
    )
    temporal_BK_model_fullpath = (
        literature_data_folder / "Benardete_1999_VisNeurosci_parasol.csv"
    )
    spatial_DoG_fullpath = (
        literature_data_folder
        / "Schottdorf_2021_JPhysiol_CenRadius_Fig4C_parasol_c.npz"
    )
elif my_retina["gc_type"] == "midget":
    dendr_diam1_fullpath = (
        literature_data_folder / "Perry_1984_Neurosci_MidgetDendrDiam_Fig6B_c.npz"
    )
    dendr_diam2_fullpath = (
        literature_data_folder / "Watanabe_1989_JCompNeurol_MidgetDendrDiam_Fig7_c.npz"
    )
    dendr_diam3_fullpath = (
        literature_data_folder
        / "Goodchild_1996_JCompNeurol_Midget_DendDiam_Fig2B_c.npz"
    )
    temporal_BK_model_fullpath = (
        literature_data_folder / "Benardete_1997_VisNeurosci_midget.csv"
    )
    spatial_DoG_fullpath = (
        literature_data_folder / "Schottdorf_2021_JPhysiol_CenRadius_Fig4C_midget_c.npz"
    )
dendr_diam_units = {
    "data1": ["mm", "um"],
    "data2": ["mm", "um"],
    "data3": ["deg", "um"],
}

cone_density1_fullpath = (
    literature_data_folder / "Packer_1989_JCompNeurol_ConeDensity_Fig6A_main_c.npz"
)
cone_density2_fullpath = (
    literature_data_folder / "Packer_1989_JCompNeurol_ConeDensity_Fig6A_insert_c.npz"
)
cone_noise_fullpath = literature_data_folder / "Angueyra_2013_NatNeurosci_Fig6E_c.npz"
cone_response_fullpath = (
    literature_data_folder / "Angueyra_2013_NatNeurosci_Fig6B_c.npz"
)

bipolar_table_fullpath = literature_data_folder / "Boycott_1991_EurJNeurosci_Table1.csv"

parasol_on_RI_values_fullpath = (
    literature_data_folder / "Turner_2018_eLife_Fig5C_ON_c.npz"
)
parasol_off_RI_values_fullpath = (
    literature_data_folder / "Turner_2018_eLife_Fig5C_OFF_c.npz"
)

temporal_pattern_fullpath = (
    literature_data_folder / "Angueyra_2022_JNeurosci_Fig2B_c.npz"
)

literature_data_files = {
    "gc_density_fullpath": gc_density_fullpath,
    "dendr_diam1_fullpath": dendr_diam1_fullpath,
    "dendr_diam2_fullpath": dendr_diam2_fullpath,
    "dendr_diam3_fullpath": dendr_diam3_fullpath,
    "dendr_diam_units": dendr_diam_units,
    "temporal_BK_model_fullpath": temporal_BK_model_fullpath,
    "spatial_DoG_fullpath": spatial_DoG_fullpath,
    "cone_density1_fullpath": cone_density1_fullpath,
    "cone_density2_fullpath": cone_density2_fullpath,
    "cone_noise_fullpath": cone_noise_fullpath,
    "cone_response_fullpath": cone_response_fullpath,
    "bipolar_table_fullpath": bipolar_table_fullpath,
    "parasol_on_RI_values_fullpath": parasol_on_RI_values_fullpath,
    "parasol_off_RI_values_fullpath": parasol_off_RI_values_fullpath,
    "temporal_pattern_fullpath": temporal_pattern_fullpath,
}


profile = False

#############################################################################################################
#############################################################################################################
###                                      End of  module-level script                                      ###
#############################################################################################################
#############################################################################################################


if __name__ == "__main__":
    if profile is True:
        import cProfile, pstats

        profiler = cProfile.Profile()
        profiler.enable()

    """
    Housekeeping. Do not comment out.

    All ProjectManager input parameters go to context. These are validated by the context object, and returned 
    to the class instance by set_context() method. They are available by class_instance.context.attribute. 
    """
    PM = ProjectManager(
        path=path,
        input_folder=input_folder,
        output_folder=output_folder,
        stimulus_folder=stimulus_folder,
        project=project,
        experiment=experiment,
        ray_root_path=ray_root_path,
        device=device,
        my_retina=my_retina,
        my_stimulus_metadata=my_stimulus_metadata,
        my_stimulus_options=my_stimulus_options,
        my_run_options=my_run_options,
        literature_data_files=literature_data_files,
        apricot_metadata=apricot_metadata,
        numpy_seed=numpy_seed,
    )

    #################################
    ###    Check cone response    ###
    #################################

    """
    For artificial stimuli, we want to measure the true transfer function of the retina.
    E.g. the Chichilnisky model receptive fields were measured from isolated retinas,
    where images were focused on photoreceptors. The natural_stimuli_cone_filter method 
    accounts for the blur by the eye and the nonlinear cone response for natural images 
    and videos.
    """

    ###########################################
    ##   Luminance and Photoisomerizations   ##
    ###########################################

    # I_cone = 4000  # photoisomerizations per second per cone

    # luminance = PM.cones.get_luminance_from_photoisomerizations(I_cone)
    # print(f"{luminance:.2f} cd/m2")

    # luminance = 128  # Luminance in cd/m2

    # I_cone = PM.cones.get_photoisomerizations_from_luminance(luminance)
    # print(f"{I_cone:.2f} photoisomerizations per second per cone")

    # ##########################################
    # #   Sample figure data from literature  ##
    # ##########################################

    # # # If possible, sample only temporal hemiretina
    # from project.project_utilities_module import DataSampler

    # filename = "Turner_2018_eLife_Fig5C_ON.jpg"
    # filename_full = git_repo_root.joinpath(r"retina/literature_data", filename)
    # # # Fig lowest and highest tick values in the image, use these as calibration points
    # min_X, max_X, min_Y, max_Y = (-5, 5, 0, 1)
    # ds = DataSampler(filename_full, min_X, max_X, min_Y, max_Y, logX=False, logY=False)
    # # ds.collect_and_save_points()
    # ds.quality_control(restore=True)

    #################################
    #################################
    ###        Build retina       ###
    #################################
    #################################

    """
    Build and test your retina here, one gc type at a time. 
    """

    PM.construct_retina.build()  # Main method for building the retina

    # The following visualizations are
    #  dependent on the ConstructRetina instance.
    # Thus, they are called after the retina is built.

    # The show_exp_build_process method visualizes the spatial and temporal filter responses, ganglion cell positions and density,
    # mosaic layout, spatial and temporal statistics, dendrite diameter versus eccentricity, and tonic drives
    # in the retina mosaic building process.

    # For FIT and VAE
    # PM.viz.show_cones_linked_to_bipolars(n_samples=4, savefigname=None)
    # PM.viz.show_bipolars_linked_to_gc(gc_list=[10, 17], savefigname=None)
    # PM.viz.show_bipolars_linked_to_gc(n_samples=4, savefigname=None)
    # PM.viz.show_cones_linked_to_gc(gc_list=[10, 17], savefigname=None)
    # PM.viz.show_cones_linked_to_gc(n_samples=4, savefigname=None)
    # PM.viz.show_DoG_img_grid(gc_list=[10, 17, 46], savefigname=None)
    # PM.viz.show_DoG_img_grid(n_samples=8)
    # PM.viz.show_cell_density_vs_ecc(unit_type="cone", savefigname=None)  # gc or cone
    # PM.viz.show_cell_density_vs_ecc(unit_type="gc", savefigname=None)  # gc or cone
    # PM.viz.show_cell_density_vs_ecc(unit_type="bipolar", savefigname=None)  # gc or cone
    # PM.viz.show_connection_histograms(savefigname=None)
    PM.viz.show_fan_in_out_distributions(savefigname=None)

    # PM.viz.show_DoG_model_fit(sample_list=[10], savefigname=None)
    # PM.viz.show_DoG_model_fit(n_samples=6, savefigname=None)
    # PM.viz.show_dendrite_diam_vs_ecc(log_x=False, log_y=False, savefigname=None)
    # PM.viz.show_retina_img(savefigname=None)
    # PM.viz.show_cone_noise_vs_freq(savefigname=None)
    # PM.viz.show_bipolar_nonlinearity(savefigname=None)

    # For FIT (DoG fits, temporal kernels and tonic drives)
    # PM.viz.show_exp_build_process(show_all_spatial_fits=False)
    # PM.viz.show_temporal_filter_response(n_curves=3, savefigname="temporal_filters.eps")
    # PM.viz.show_spatial_statistics(correlation_reference="ampl_s", savefigname=None)

    # For VAE
    # PM.viz.show_gen_exp_spatial_rf(ds_name="train_ds", n_samples=15, savefigname=None)
    # PM.viz.show_latent_tsne_space()
    # PM.viz.show_gen_spat_post_hist()
    # PM.viz.show_latent_space_and_samples()
    # PM.viz.show_rf_imgs(n_samples=10, savefigname="parasol_on_vae_gen_rf.eps")
    # PM.viz.show_rf_violinplot()  # Pixel values for each unit

    # # "train_loss", "val_loss", "mse", "ssim", "kid_mean", "kid_std"
    # this_dep_var = "val_loss"
    # ray_exp_name = None  # "TrainableVAE_2023-04-20_22-17-35"  # None for most recent
    # highlight_trial = None  # "2199e_00029"  # or None
    # PM.viz.show_ray_experiment(
    #     ray_exp_name, this_dep_var, highlight_trial=highlight_trial
    # )

    # For both FIT and VAE. Estimated luminance for validation data (Schottdorf_2021_JPhysiol,
    # van Hateren_2002_JNeurosci) is 222.2 td / (np.pi * (4 mm diam / 2)**2) = 17.7 cd/m2
    # PM.viz.validate_gc_rf_size(savefigname="rf_size_vs_Schottdorf_data.eps")

    ###################################
    ###################################
    ###         Single Trial        ###
    ###################################
    ###################################

    ########################
    ### Create stimulus ###
    ########################

    # Based on my_stimulus_options above
    # PM.stimulate.make_stimulus_video()

    ####################################
    ### Run multiple trials or units ###
    ####################################

    # Load stimulus to get working retina, necessary for running units
    # PM.simulate_retina.run_with_my_run_options()

    ##########################################
    ### Show single ganglion cell features ###
    ##########################################

    # PM.viz.show_spatiotemporal_filter(unit_index=39, savefigname=None)
    # PM.viz.show_temporal_kernel_frequency_response(unit_index=2, savefigname=None)
    # PM.viz.plot_midpoint_contrast(unit_index=2, savefigname=None)
    # PM.viz.plot_local_rms_contrast(unit_index=2, savefigname=None)
    # PM.viz.plot_local_michelson_contrast(unit_index=2, savefigname=None)
    # PM.viz.show_single_gc_view(unit_index=2, frame_number=300, savefigname=None)

    ################################################
    ###   Show multiple trials for single unit,  ###
    ###   or multiple units for single trial     ###
    ################################################

    # Based on my_run_options above
    # PM.viz.show_all_gc_responses(savefigname=None)
    # PM.viz.show_generator_potential_histogram(savefigname=None)
    # PM.viz.show_generator_potential_histogram(
    #     savefigname="Chirp_subunit_midOFF_256.eps"
    # )
    # PM.viz.show_cone_responses(time_range=[0.4, 1.1], savefigname=None)
    # PM.viz.show_cone_responses(time_range=None, savefigname=None)

    # PM.viz.show_stimulus_with_gcs(
    #     example_gc=0,  # [int,], my_run_options["unit_index"]
    #     frame_number=46,  # depends on fps, and video and baseline lengths
    #     show_rf_id=True,
    #     savefigname=None,
    # )

    # ##########################################
    # ###       Show impulse response        ###
    # ##########################################

    # # Contrast applies only for parasol units with dynamic model, use [1.0] for others
    # contrasts_for_impulse = [0.01, 1.0]
    # PM.simulate_retina.run_cells(
    #     unit_index=[16],  # list of ints
    #     get_impulse_response=True,  # Return with impulse response
    #     contrasts_for_impulse=contrasts_for_impulse,  # List of contrasts
    # )
    # savename = (
    #     f"{gc_type}_{response_type}_{my_retina['temporal_model']}_impulse" + ".eps"
    # )
    # # The PM.simulate_retina load_stimulus and run_cells must be active for impulse response viz
    # PM.viz.show_impulse_response(savefigname=None)

    ##########################################
    ###          Show unity data           ###
    ##########################################

    # # Get uniformity data and exit
    # PM.simulate_retina.run_cells(get_uniformity_data=True)
    # savename = f"{gc_type}_{response_type}_{my_retina['spatial_model']}_unity" + ".eps"
    # # The PM.simulate_retina load_stimulus and run_cells with option
    # # get_uniformity_data=True must be active for unity viz
    # PM.viz.show_unity(savefigname=None)

    #################################################################
    #################################################################
    ###   Experiment with multiple units, conditions and trials   ###
    #################################################################
    #################################################################

    # TODO Texture experiment from Schwartz_2012_NatNeurosci

    ################################
    ### Build and run Experiment ###
    ################################

    # TÄHÄN JÄIT:
    # - Montako yhteyttä tulee kuhunkin gc:uun? histogrammit voimakkuuksista ja määristä
    #   - tarkista tästä kirjallisuus, mikä on järkevä määrä kullekin bipo ja gc tyypille
    # - joutuuko pakottamaan yhteyksiä / tappamaan ganglionsoluja?
    # - Onko yhteyspainojen pakottamisella 1:een sivuvaikutuksia? Suuria painoja?
    # - Vakioi generaattoripotentiaali eri temporal mallien välillä käyttäen temporal chirp ärsykettä
    # - gen => fr transformaatio, Turner malli?
    # - SUBUNIT MALLIN VALIDOINTI vs Turner 2018

    # -BENARDETE INPUT [-1,1], VICTOR [0,1], CHICHI [-2,2], TURNER [-5, 5] & CDF NONLIN

    # Retina needs to be built for this to work.
    # my_stimulus_options above defines the stimulus. From that dictionary,
    # defined keys' values are dynamically changed in the experiment.
    # Note that tuple values from my_stimulus_options are captured for varying each tuple value separately.

    # exp_variables = ["background", "intensity"]  # key from my_stimulus_options
    exp_variables = ["temporal_frequency", "contrast"]  # from my_stimulus_options
    # # Define experiment parameters. List lengths must be equal.
    # # Examples: exp_variables = ["contrast"], min_max_values = [[0.015, 0.98]], n_steps = [30], logarithmic = [True]
    # experiment_dict = {
    #     "exp_variables": exp_variables,
    #     # two vals for each exp_variable, even is it is not changing
    #     # "min_max_values": [
    #     #     [1e-2, 1e5],
    #     #     ([0, 0], [1e-2, 1e8]),
    #     # ],  # background, intensity
    #     # "n_steps": [2, (1, 2)],
    #     # "n_steps": [8, (1, 10)], # background, intensity
    #     # "logarithmic": [True, (False, True)], # background, intensity
    #     "min_max_values": [[0.5, 30], [0.01, 0.64]],  # temporal frequency, contrast
    #     "n_steps": [14, 7],  # temporal frequency, contrast
    #     "logarithmic": [True, True],  # temporal frequency, contrast
    # }

    # # # # # N trials or N units must be 1, and the other > 1. This is set above in my_run_options.
    # PM.experiment.build_and_run(experiment_dict)

    # #########################
    # ## Analyze Experiment ###
    # #########################

    # my_analysis_options = {
    #     "exp_variables": exp_variables,
    #     "t_start_ana": 0.5,
    #     "t_end_ana": 30.5,
    # }

    # PM.ana.analyze_experiment(my_analysis_options)
    # # # PM.ana.unit_correlation(my_analysis_options, gc_type, response_type, gc_units=None)
    # # PM.ana.relative_gain(my_analysis_options)
    # # PM.ana.response_vs_background(my_analysis_options)

    ############################
    ### Visualize Experiment ###
    ############################

    # PM.viz.spike_raster_response(exp_variables, trial=0, savefigname=None)
    # PM.viz.show_relative_gain(exp_variables, savefigname=None)
    # PM.viz.show_response_vs_background_experiment(
    #     exp_variables, unit="cd/m2", savefigname=None
    # )

    # PM.viz.show_unit_correlation(
    #     exp_variables, time_window=[-0.2, 0.2], savefigname=None
    # )

    # PM.viz.F1F2_popul_response(exp_variables, xlog=False, savefigname=None)
    # PM.viz.F1F2_unit_response(exp_variables, xlog=False, savefigname=None)
    # PM.viz.ptp_response(exp_variables, x_of_interest=None, savefigname=None)
    # PM.viz.fr_response(exp_variables, xlog=True, savefigname=None)
    # PM.viz.tf_vs_fr_cg(
    #     exp_variables,
    #     n_contrasts=7,
    #     xlog=True,
    #     ylog=True,
    #     savefigname=None,
    # )

    #################################################################################
    ######              Categorical plot of parametric data              ############
    #################################################################################

    # """
    # Define what data is going to be visualized and how it is visualized

    # title : figures, multiple allowed, categorical independent variable
    # outer : panels (distinct subplots) multiple allowed, categorical independent variable
    # inner : inside one axis (subplot) multiple allowed, parametric dependent variable, statistics available
    # plot_type : seaborn plot types: "violin", "box", "strip", "swarm", "boxen", "point", "bar"
    # palette : seaborn color palette: "viridis", "plasma", "inferno", "magma", "Greys"
    # inner_stat_test : bool. If True, perform statistical test between inner variables.
    # -if N inner == 2, apply Wilcoxon signed-rank test, if N inner > 2, apply Friedman test.

    # inner_paths : bool (only inner available for setting paths). Provide comparison from arbitrary paths, e.g. controls. The 'inner' is ignored.
    # inner_path_names: list of names of paths to compare.
    # paths : provide list of tuples of full path parts to data folder.
    # E.g. [(path, 'Single_narrow_iterations_control', 'Bacon_gL_compiled_results'),]
    # The number of paths MUST be the same as the number of corresponding inner variables.
    # """

    # param_plot_dict = {
    #     "title": "parameters",
    #     "outer": "analyzes",
    #     "inner": "startpoints",
    #     "plot_type": "box",
    #     "sharey": False,
    #     "palette": "Greys",
    #     "inner_stat_test": False,
    #     # "inner_paths": False,
    #     # "inner_path_names": ["Comrad", "Bacon", "Random_EI", "Random_all"],
    #     # "paths": [
    #     #     (path, "Comrad_gL_compiled_results"),
    #     #     (path, "Bacon_gL_compiled_results"),
    #     #     (path, "Bacon_gL_compiled_results_EI"),
    #     #     (path, "Bacon_gL_compiled_results_ALL"),
    #     # ],
    # }
    # PM.viz.show_catplot(param_plot_dict)

    ###############################
    ###############################
    ###     Analog stimulus     ###
    ###############################
    ###############################

    # # For current injection, use this method.

    ##############################
    ### Create analog stimulus ###
    ##############################

    # N_tp = 20000
    # dt = 0.1  # ms

    # freq = 2
    # N_cycles = freq * (dt / 1000) * N_tp
    # filename_out = "test.mat"
    # analog_options = {
    #     "filename_out": filename_out,
    #     "N_units": 3,
    #     "coord_type": "real",
    #     "N_tp": N_tp,
    #     "input_type": "noise",  # 'quadratic_oscillation' or 'noise' or 'step_current'
    #     "N_cycles": [
    #         N_cycles,
    #         0,
    #         0,
    #     ],  # Scalar provides two units at quadrature, other units are zero. List of ints/floats provides separate freq to each. Ignored for noise.
    #     "dt": dt,  # IMPORTANT: assuming milliseconds
    #     "save_stimulus": True,
    # }

    # PM.analog_input.make_stimulus_video(analog_options=analog_options)
    # PM.viz.plot_analog_stimulus(PM.analog_input)

    """
    ### Housekeeping ###. Do not comment out.
    """
    # End measuring time and print the time in HH hours MM minutes SS seconds format
    end_time = time.time()
    print(
        "Total time taken: ",
        time.strftime(
            "%H hours %M minutes %S seconds", time.gmtime(end_time - start_time)
        ),
    )

    plt.show()

    if profile is True:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats(20)
