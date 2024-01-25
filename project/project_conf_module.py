# Visualization
import matplotlib.pyplot as plt

# Builtin
from pathlib import Path
import sys
import pdb
import time
import warnings
import random
import math

# sys.path.append(Path(__file__).resolve().parent.parent)
# Start measuring time
start_time = time.time()

# Local
from project.project_manager_module import ProjectManager

warnings.simplefilter("ignore")


"""
This is code for building macaque retinal filters corresponding to midget and parasol cell responses
for temporal hemiretina. We keep modular code structure, to be able to add new features at later phase.

Visual angle (A) in degrees from previous studies (Croner and Kaplan, 1995) was approximated with relation 5 deg/mm. 
This works fine up to 20 deg ecc, but underestimates the distance thereafter. If more peripheral representations are 
necessary, the millimeters should be calculated by inverting the relation A = 0.1 + 4.21E + 0.038E^2 (Drasdo and Fowler, 1974;
Dacey and Petersen, 1992). Current implementation uses one deg = 220um (Perry et al 1985). One mm retina is ~4.55 deg visual field.

We have extracted statistics of macaque ganglion cell receptive fields from literature and build continuous functions.

The density of many cell types is inversely proportional to dendritic field coverage, suggesting constant coverage factor 
(Perry_1984_Neurosci, Wassle_1991_PhysRev). Midget coverage factor is 1  (Dacey_1993_JNeurosci for humans; Wassle_1991_PhysRev, 
Lee_2010_ProgRetEyeRes). Parasol coverage factor is 3-4 close to fovea (Grunert_1993_VisRes); 2-7 according to Perry_1984_Neurosci. 
These include ON- and OFF-center cells, and perhaps other cell types. It is likely that coverage factor is 1 for midget and parasol 
ON- and OFF-center cells each, which is also in line with Doi_2012 JNeurosci, Field_2010_Nature

The spatiotemporal receptive fields for the four cell types (parasol & midget, ON & OFF) were modelled with double ellipsoid 
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
The unit parameters are drawn from Benardete_1999_VisNeurosci for parasol cells and Benardete_1997_VisNeurosci_a for midget cells.
We are sampling from Benardete Kaplan data assuming triangular distribution of the reported tables of statistics (original data points not shown).
For a review of physiological mechanisms, see Demb_2008_JPhysiol and Beaudoin_2007_JNeurosci.

The max firing rate, parameter "A" in the Victor model, comes from Benardete_1999_VisNeurosci for parasol cells and Benardete_1997_VisNeurosci_a for midget cells.
To get firing rate from generator potential, we fit a logistic function. The firing_rate = A / (1 + exp(-k*(x-x0))) - tonic_drive, where k is 
the steepness of the curve and x0 is the sigmoid's midpoint. 

The cone photoreceptor sampling is approximated as achromatic (single) compressive cone response(Baylor_1987_JPhysiol).
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
if sys.platform == "linux":
    model_root_path = "/opt3/Laskenta/Models"
    git_repo_root = Path(r"/opt2/Git_Repos/MacaqueRetina")
    ray_root_path = None  # if None, ray_results are saved to model_root_path/project/experiment/output_folder/ray_results
elif sys.platform == "win32":
    model_root_path = r"C:\Users\simov\Laskenta\Models"
    git_repo_root = Path(r"C:\Users\simov\Laskenta\GitRepos\MacaqueRetina")
    ray_root_path = r"C:\Data"


"""
Project name
"""
project = "Retina"


"""
Current experiment. Use distinct folders fo distinct stimuli.
"""
experiment = "dynamic_QA"


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
gc_type = "parasol"  # "parasol" or "midget"
response_type = "on"  # "on" or "off"

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

# These values are used for building a new retina
my_retina = {
    "gc_type": gc_type,
    "response_type": response_type,
    # "ecc_limits_deg": [36, 39.4],  # eccentricity in degrees
    # "pol_limits_deg": [-4, 4],  # polar angle in degrees
    "ecc_limits_deg": [4.5, 5.5],  # eccentricity in degrees
    "pol_limits_deg": [-3, 3],  # polar angle in degrees
    # "ecc_limits_deg": [4.8, 5.2],  # eccentricity in degrees
    # "pol_limits_deg": [-1.0, 1.0],  # polar angle in degrees
    "model_density": 1.0,  # 1.0 for 100% of the literature density of ganglion cells
    "dd_regr_model": "loglog",  # linear, quadratic, cubic, loglog. For midget < 20 deg, use quadratic; for parasol use loglog
    "ecc_limit_for_dd_fit": 20,  # 20,  # degrees, math.inf for no limit
    "stimulus_center": 5.0 + 0j,  # degrees, this is stimulus_position (0, 0)
    "temporal_model": "dynamic",  # fixed, dynamic
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
    'spatially_uniform_binary_noise'
stimulus_form: 'circular'; 'rectangular'; 'annulus'

For stimulus_form annulus, additional arguments are:
size_inner: in degrees
size_outer: in degrees

stimulus_position: in degrees, (0,0) is the center.
stimulus_size: In degrees. Radius for circle and annulus, half-width for rectangle. 0 for full image.
background: intensity between 0, 256
contrast: between 0 and 1
mean: mean stimulus intensity between 0, 256

Note if mean + ((contrast * max(intensity)) / 2) exceed 255 or if
        mean - ((contrast * max(intensity)) / 2) go below 0
        the stimulus generation fails

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
"""

my_stimulus_options = {
    # Shared btw stimulus and simulate_retina
    "image_width": 240,  # 752 for nature1.avi
    "image_height": 240,  # 432 for nature1.avi
    "pix_per_deg": 60,
    "fps": 300,  # 300 for good cg integration
    "duration_seconds": 1,  # actual frames = floor(duration_seconds * fps)
    "baseline_start_seconds": 0.5,  # Total duration is duration + both baselines
    "baseline_end_seconds": 0.5,
    "pattern": "temporal_square_pattern",  # One of the StimulusPatterns
    "stimulus_form": "rectangular",
    "size_inner": 0.1,  # deg, Applies to annulus only
    "size_outer": 1,  # deg, Applies to annulus only
    "stimulus_position": (0.0, 0.0),
    "stimulus_size": 2,  # 0.04,  # 2,  # deg, radius for circle, sidelen/2 for rectangle.
    "background": 128,
    "contrast": 0.95,  # Weber constrast
    "mean": 128,
    "temporal_frequency": 0.1,  # 0.01,  # 4.0,  # 40,  # Hz
    "spatial_frequency": 2.0,  # cpd
    "orientation": 0,  # degrees
    "phase_shift": 0,  # math.pi,  # radians
    "stimulus_video_name": f"{stimulus_folder}.mp4",
}

# For external video and image input. See visual_stimulus_module.VideoBaseClass for more options.
my_stimulus_metadata = {
    "stimulus_file": "testi2.jpg",  # nature1.avi, testi.jpg
    "pix_per_deg": 30,  # VanHateren_1998_ProcRSocLondB 2 arcmin per pixel
    "apply_cone_filter": False,
    "fps": 25,
}

# Each gc response file contain n_trials
n_files = 1

# Either n_trials or n_cells must be 1, and the other > 1
# Running multiple trials on multiple cells is not supported
my_run_options = {
    "cell_index": None,  # list of ints or None for all cells
    "n_trials": 1,  # For each of the response files
    # "cell_index": 2,  # list of ints or None for all cells
    # "n_trials": 10,  # For each of the response files
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

# Proportion of ON and OFF response type cells, assuming ON rf diameter = 1.2 x OFF rf diameter, and
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
    "cone_noise_magnitude": 1,  # Relative amplitude, 0 for no noise
}

# Recovery function from Berry_1998_JNeurosci, Uzzell_2004_JNeurophysiol
# abs and rel refractory estimated from Uzzell_2004_JNeurophysiol,
# Fig 7B, bottom row, inset. Parasol ON cell
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
    "n_iterations": 5000,  # v 20, f 5000
    "change_rate": 0.001,  # f 0.001, v 0.5
    "unit_repulsion_stregth": 5,  # 10 f only
    "unit_distance_threshold": 0.02,  # f only, adjusted with ecc
    "diffusion_speed": 0.0001,  # f only, adjusted with ecc
    "border_repulsion_stength": 10,  # f only
    "border_distance_threshold": 0.01,  # f only
    "show_placing_progress": False,  # True False
    "show_skip_steps": 100,  # v 1, f 100
}

cone_placement_params = {
    "algorithm": "force",  # "voronoi" or "force" or None
    "n_iterations": 300,  # v 20, f 300
    "change_rate": 0.0005,  # f 0.0005, v 0.5
    "unit_repulsion_stregth": 2,  # 10 f only
    "unit_distance_threshold": 0.1,  # f only, adjusted with ecc
    "diffusion_speed": 0.001,  # f only, adjusted with ecc
    "border_repulsion_stength": 5,  # f only
    "border_distance_threshold": 0.0001,  # f only
    "show_placing_progress": False,  # True False
    "show_skip_steps": 10,  # v 1, f 100
}

# For VAE, this is enough to have good distribution between units.
rf_repulsion_params = {
    "n_iterations": 200,
    "change_rate": 0.01,
    "cooling_rate": 0.999,  # each iteration change_rate = change_rate * cooling_rate
    "border_repulsion_stength": 5,
    "show_repulsion_progress": False,  # True False
    "show_only_unit": None,  # None or int for unit idx
    "show_skip_steps": 5,
    "savefigname": None,
}

my_retina_append = {
    "mosaic_file": gc_type + "_" + response_type + "_mosaic.csv",
    "spatial_rfs_file": gc_type + "_" + response_type + "_spatial_rfs.npz",
    "proportion_of_parasol_gc_type": proportion_of_parasol_gc_type,
    "proportion_of_midget_gc_type": proportion_of_midget_gc_type,
    "proportion_of_ON_response_type": proportion_of_ON_response_type,
    "proportion_of_OFF_response_type": proportion_of_OFF_response_type,
    "deg_per_mm": deg_per_mm,
    "optical_aberration": optical_aberration,
    "cone_general_params": cone_general_params,
    "refractory_params": refractory_params,
    "gc_placement_params": gc_placement_params,
    "cone_placement_params": cone_placement_params,
    "rf_repulsion_params": rf_repulsion_params,
    "visual2cortical_params": visual2cortical_params,
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
    # PM.cones.natural_stimuli_cone_filter()
    # PM.viz.show_cone_filter_response(
    #     PM.cones.image, PM.cones.image_after_optics, PM.cones.cone_response
    # )

    # TÄHÄN JÄIT: ETSI KIRJALLISUUDESTA TAUSTA AKTIIVISUUKSIA JA DYNAAMISIA MODULAATIOITA
    # SELVITÄ MIKSI FIXED JA VAE EROAVAT, JA MISTÄ TILEE FIXED SUSTAINED FIRING JOIHINKIN YKSIKÖIHIN
    # MIETI TAPPIKOHINAN LINKITYS UUDELLEEN:
    #  - PERIFERIASSA GRIDI LIIAN HARVA
    #  - KESKELLÄ GRIDI LIIAN TIHEÄ
    # FIT IRTI GRIDISTÄ EDELLEEN.

    ###########################################
    ##   Sample figure data from literature  ##
    ###########################################

    # # If possible, sample only temporal hemiretina
    # from project.project_utilities_module import DataSampler

    # filename = "Angueyra_2013_NatNeurosci_Fig6E.jpg"
    # filename_full = git_repo_root.joinpath(r"retina/literature_data", filename)
    # # Fig lowest and highest tick values in the image, use these as calibration points
    # min_X, max_X, min_Y, max_Y = (1, 600, 0.001, 1)
    # ds = DataSampler(filename_full, min_X, max_X, min_Y, max_Y, logX=True, logY=True)
    # ds.collect_and_save_points()
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

    # The following visualizations are dependent on the ConstructRetina instance.
    # Thus, they are called after the retina is built.

    # The show_exp_build_process method visualizes the spatial and temporal filter responses, ganglion cell positions and density,
    # mosaic layout, spatial and temporal statistics, dendrite diameter versus eccentricity, and tonic drives
    # in the retina mosaic building process.

    # For FIT and VAE
    # PM.viz.show_cones_linked_to_gc(gc_list=[10], savefigname="cones_linked_to_gc.svg")
    # PM.viz.show_cones_linked_to_gc(gc_list=[32, 58, 63, 67, 6], savefigname=None)
    # PM.viz.show_unit_density_vs_ecc(unit_type="cone", savefigname=None)  # gc or cone

    # PM.viz.show_DoG_model_fit(sample_list=[10], savefigname="DoG_model_fit.eps")
    # PM.viz.show_DoG_model_fit(n_samples=6, savefigname=None)
    # PM.viz.show_dendrite_diam_vs_ecc(log_x=False, log_y=False, savefigname=None)
    # PM.viz.show_coneq_noise_vs_freq(savefigname="cone_noise_vs_freq.svg")

    # For FIT (DoG fits, temporal kernels and tonic drives)
    # PM.viz.show_exp_build_process(show_all_spatial_fits=False)
    # PM.viz.show_temporal_filter_response(n_curves=3, savefigname="temporal_filters.eps")
    # PM.viz.show_spatial_statistics(correlation_reference="ampl_s", savefigname=None)

    # For VAE
    # PM.viz.show_gen_exp_spatial_rf(ds_name="train_ds", n_samples=15, savefigname=None)
    # PM.viz.show_latent_tsne_space()
    # PM.viz.show_gen_spat_post_hist()
    # PM.viz.show_latent_space_and_samples()
    # PM.viz.show_retina_img(savefigname=None)
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

    # # Based on my_stimulus_options above
    # PM.stimulate.make_stimulus_video()

    ####################################
    ### Run multiple trials or cells ###
    ####################################

    # Load stimulus to get working retina, necessary for running cells
    PM.simulate_retina.load_stimulus()
    PM.simulate_retina.run_with_my_run_options()

    ##########################################
    ### Show single ganglion cell features ###
    ##########################################

    # PM.viz.show_spatiotemporal_filter(cell_index=39, savefigname=None)
    # PM.viz.show_temporal_kernel_frequency_response(cell_index=2, savefigname=None)
    # PM.viz.plot_midpoint_contrast(cell_index=2, savefigname=None)
    # PM.viz.plot_local_rms_contrast(cell_index=2, savefigname=None)
    # PM.viz.plot_local_michelson_contrast(cell_index=2, savefigname=None)
    # PM.viz.show_single_gc_view(cell_index=2, frame_number=300, savefigname=None)

    ##########################################
    ###       Show impulse response        ###
    ##########################################

    # # Contrast applies only for parasol cells with dynamic model, use [1.0] for others
    # contrasts_for_impulse = [0.01, 1.0]
    # PM.simulate_retina.run_cells(
    #     cell_index=[16],  # list of ints
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

    ################################################
    ###   Show multiple trials for single cell,  ###
    ###   or multiple cells for single trial     ###
    ################################################

    # Based on my_run_options above
    PM.viz.show_all_gc_responses(savefigname="midget_on_sine.eps")

    # PM.viz.show_stimulus_with_gcs(
    #     example_gc=[9],  # [int,], my_run_options["cell_index"]
    #     frame_number=301,  # depends on fps, and video and baseline lengths
    #     show_rf_id=True,
    #     savefigname=None,
    # )

    #################################################################
    #################################################################
    ###   Experiment with multiple units, conditions and trials   ###
    #################################################################
    #################################################################

    # TODO Texture experiment from Schwartz_2012_NatNeurosci

    ################################
    ### Build and run Experiment ###
    ################################

    exp_variables = ["temporal_frequency"]  # from my_stimulus_options
    # exp_variables = ["temporal_frequency", "contrast"]  # from my_stimulus_options
    # # Define experiment parameters. List lengths must be equal.
    # # Examples: exp_variables = ["contrast"], min_max_values = [[0.015, 0.98]], n_steps = [30], logaritmic = [True]
    # experiment_dict = {
    #     "exp_variables": exp_variables,
    #     "min_max_values": [[0.01, 0.01]],  # two vals for each exp_variable # frequency
    #     "n_steps": [1],
    #     "logaritmic": [False],
    #     # "min_max_values": [[0.5, 46], [0.01, 0.64]],  # temporal frequency, contrast
    #     # "n_steps": [14, 7],  # temporal frequency, contrast
    #     # "logaritmic": [False, True],  # temporal frequency, contrast
    # }

    # PM.experiment.build_and_run(experiment_dict, n_trials=10)

    # # # #########################
    # # # ## Analyze Experiment ###
    # # # #########################

    # my_analysis_options = {
    #     "exp_variables": exp_variables,
    #     "t_start_ana": 1.0,
    #     "t_end_ana": 31.0,
    #     # "t_start_ana": 1,
    #     # "t_end_ana": 7,
    # }

    # # PM.ana.analyze_experiment(my_analysis_options)
    # PM.ana.unit_correlation(my_analysis_options, gc_type, response_type, gc_units=None)

    # ############################
    # ### Visualize Experiment ###
    # ############################

    # PM.viz.spike_raster_response(exp_variables, trial=0, savefigname=None)

    # PM.viz.show_unit_correlation(
    #     exp_variables, time_window=[-0.2, 0.2], savefigname=None
    # )

    # PM.viz.F1F2_popul_response(exp_variables, xlog=False, savefigname=None)
    # PM.viz.F1F2_unit_response(
    #     exp_variables, xlog=False, savefigname=output_folder + ".eps"
    # )
    # PM.viz.ptp_response(exp_variables, x_of_interest=None, savefigname=None)
    # PM.viz.fr_response(exp_variables, xlog=False, savefigname=None)
    # PM.viz.tf_vs_fr_cg(
    #     exp_variables,
    #     n_contrasts=3,
    #     xlog=True,
    #     ylog=False,
    #     savefigname="tf_vs_fr_cg_sine_grating.eps",
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
