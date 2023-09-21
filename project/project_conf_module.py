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

The cone photoreceptor sampling is approximated as achromatic (single) compressive cone response(Baylor_1987_JPhysiol).

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

model_type FIT : Fit ellipse to center and surround
model_type : VAE : Variational autoencoder. The model reconstructs the full receptive field and generates new samples from the latent space.
    
Contrast gain control (CGC) is implemented according to Victor_1987_JPhysiol using numerical integration in discretized temporal domain.
The unit parameters are drawn from Benardete_1999_VisNeurosci for parasol cells and Benardete_1997_VisNeurosci_a for midget cells.
We are sampling from Benardete Kaplan data assuming triangular distribution of the reported tables of statistics (original data points not shown).
For a review of physiological mechanisms, see Demb_2008_JPhysiol and Beaudoin_2007_JNeurosci.

The max firing rate, parameter "A" in the Victor model, comes from Benardete_1999_VisNeurosci for parasol cells and Benardete_1997_VisNeurosci_a for midget cells.
To get firing rate from generator potential, we fit a logistic function. The firing_rate = A / (1 + exp(-k*(x-x0))) - tonic_drive, where k is 
the steepness of the curve and x0 is the sigmoid's midpoint. 
"""

"""
Technical notes:
Use keyword substring "file" in filenames, and "folder" or "path" in folder names to assert that they are turned into pathlib objects. 
Path structure is assumed to be model_root_path/project/experiment/output_folder

Abbreviations:
ana : analysis
col : column
full : full absolute path 
param : parameter
viz : visualization
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
experiment = "contrast_response_function"  # "test"


"""
Input context
Existing retina model, any general files
"""
input_folder = "../in"  # input figs, videos


"""
Stimulus context
Stimulus images and videos
"""
stimulus_folder = "stim_temporal_c0p8"  # input figs, videos


"""
Data context for output. 
"""

output_folder = "temp_freq_testi"  # "crf_midget_on_vae_tf10_c6"


"""
Remove random variations by setting the numpy random seed
"""
numpy_seed = 42  # random.randint(0, 1000000)  # 42

"""
### Housekeeping ###. Do not comment out.
"""
model_root_path = Path(model_root_path)
path = Path.joinpath(model_root_path, Path(project), experiment)

# When training or tuning generative models, multiple hyperparameters are set at the RetinaVAE class.
# For training, see __init__ method. For tuning, the __init__ contains search space and
# _set_ray_tuner contains the starting point.
gc_type = "midget"
response_type = "on"

my_retina = {
    "gc_type": gc_type,
    "response_type": response_type,
    "ecc_limits": [4.7, 5.3],  # degrees
    "sector_limits": [-1.0, 1.0],  # polar angle in degrees
    "model_density": 1.0,  # 1.0 for 100% of the literature density of ganglion cells
    "dd_regr_model": "cubic",  # linear, quadratic, cubic.
    "randomize_position": 0.1,  # between 0 and 1
    "stimulus_center": 5.0 + 0j,  # degrees, this is stimulus_position (0, 0)
    "temporal_model": "dynamic",  # fixed, dynamic # Gain control for parasol cells only
    "model_type": "VAE",  # "FIT" or "VAE" for variational autoencoder.
    "rf_coverage_adjusted_to_1": False,  # False or True. Applies both to FIT and VAE models
    "training_mode": "load_model",  # "train_model" or "tune_model" or "load_model" for loading trained or tuned. Applies to VAE only.
    "ray_tune_trial_id": None,  # Trial_id for tune, None for loading single run after "train_model". Applies to VAE "load_model" only.
}

# For external video and image input. See visual_stimulus_module.VideoBaseClass for more options.
my_stimulus_metadata = {
    "stimulus_file": "nature1.avi",  # nature1.avi, testi.jpg
    "pix_per_deg": 60,
    "apply_cone_filter": False,
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
    'natural_images'; 'phase_scrambled_images'; 'natural_video'; 'phase_scrambled_video';
    'temporal_sine_pattern'; 'temporal_square_pattern'; 'spatially_uniform_binary_noise'
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


"""

my_stimulus_options = {
    # Shared btw stimulus and working_retina
    "image_width": 180,  # 752 for nature1.avi
    "image_height": 180,  # 432 for nature1.avi
    "pix_per_deg": 120,
    "fps": 300,  # 300 for good cg integration
    "duration_seconds": 7.0,  # actual frames = floor(duration_seconds * fps)
    "baseline_start_seconds": 0.5,  # Total duration is duration + both baselines
    "baseline_end_seconds": 0.5,
    "pattern": "sine_grating",  # Natural video is not supported yet. One of the StimulusPatterns
    "stimulus_form": "circular",
    "size_inner": 0.02,  # Applies to annulus only
    "size_outer": 2,  # Applies to annulus only
    "stimulus_position": (0, 0),
    "stimulus_size": 2,  # 4.6 deg in Lee_1990_JOSA
    "background": 128,
    "contrast": 0.8,  # Weber constrast
    "mean": 128,
    "temporal_frequency": 0.5,  # 40,  # Hz
    "spatial_frequency": 1.0,
    "orientation": 0,  # degrees
    "phase_shift": 0,  # math.pi,  # radians
    "stimulus_video_name": "sin_drift_120Hz.mp4",
}

# Each gc response file contain n_trials
n_files = 1

# Either n_trials or n_cells must be 1, and the other > 1
my_run_options = {
    "cell_index": None,  # list of ints or None for all cells
    "n_trials": 1,  # For each of the response files
    "spike_generator_model": "poisson",  # poisson or refractory
    "save_data": True,
    "gc_response_filenames": [f"gc_response_{x:02}" for x in range(n_files)],
    "simulation_dt": 0.0001,  # in sec 0.001 = 1 ms
}


############################
###  Semi-constant variables
############################

# TODO: Refactor apricot_data_module.py to use these. Requires new composition of the module.
apricot_metadata = {
    "data_microm_per_pix": 60,
    "data_spatialfilter_height": 13,
    "data_spatialfilter_width": 13,
    "data_fps": 30,
    "data_temporalfilter_samples": 15,
}

# Proportion from all ganglion cells. Density of all ganglion cells is given later as a function of ecc from literature.
proportion_of_parasol_gc_type = 0.08
proportion_of_midget_gc_type = 0.64

# Proportion of ON and OFF response type cells, assuming ON rf diameter = 1.2 x OFF rf diameter, and
# coverage factor =1; Chichilnisky_2002_JNeurosci
proportion_of_ON_response_type = 0.40
proportion_of_OFF_response_type = 0.60

# Perry_1985_VisRes; 0.223 um/deg in the fovea, 169 um/deg at 90 deg ecc
# One mm retina is ~4.55 deg visual field.
deg_per_mm = 1 / 0.223

# Compressing cone nonlinearity. Parameters are manually scaled to give dynamic cone ouput.
# Equation, data from Baylor_1987_JPhysiol
cone_params = {
    "rm": 25,  # pA
    "k": 2.77e-4,  # at 500 nm
    "sensitivity_min": 5e2,
    "sensitivity_max": 1e4,
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

my_retina_append = {
    "mosaic_file": gc_type + "_" + response_type + "_mosaic.csv",
    "spatial_rfs_file": gc_type + "_" + response_type + "_spatial_rfs.npy",
    "proportion_of_parasol_gc_type": proportion_of_parasol_gc_type,
    "proportion_of_midget_gc_type": proportion_of_midget_gc_type,
    "proportion_of_ON_response_type": proportion_of_ON_response_type,
    "proportion_of_OFF_response_type": proportion_of_OFF_response_type,
    "deg_per_mm": deg_per_mm,
    "optical_aberration": 2 / 60,  # unit is degree
    "cone_params": cone_params,
    "refractory_params": refractory_params,
}

my_retina.update(my_retina_append)

apricot_data_folder = git_repo_root.joinpath(r"retina/apricot_data")
literature_data_folder = git_repo_root.joinpath(r"retina/literature_data")

# Define digitized literature data files for gc density and dendritic diameters.
# Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci
# Define literature data files for linear temporal models.
# Data from Benardete_1999_VisNeurosci and Benardete_1997_VisNeurosci

# gc_density_file = literature_data_folder / "Perry_1984_Neurosci_GCdensity_c.mat"
gc_density_file = literature_data_folder / "Perry_1984_Neurosci_GCdensity_Fig8_c.npz"
if my_retina["gc_type"] == "parasol":
    dendr_diam1_file = (
        # literature_data_folder / "Perry_1984_Neurosci_ParasolDendrDiam_c.mat"
        literature_data_folder
        / "Perry_1984_Neurosci_ParasolDendrDiam_Fig6A_c.npz"
    )
    dendr_diam2_file = (
        # literature_data_folder / "Watanabe_1989_JCompNeurol_GCDendrDiam_parasol_c.mat"
        literature_data_folder
        / "Watanabe_1989_JCompNeurol_ParasolDendrDiam_Fig7_c.npz"
    )
    temporal_BK_model_file = (
        literature_data_folder / "Benardete_1999_VisNeurosci_parasol.csv"
    )
elif my_retina["gc_type"] == "midget":
    dendr_diam1_file = (
        # literature_data_folder / "Perry_1984_Neurosci_MidgetDendrDiam_c.mat"
        literature_data_folder
        / "Perry_1984_Neurosci_MidgetDendrDiam_Fig6B_c.npz"
    )
    dendr_diam2_file = (
        # literature_data_folder / "Watanabe_1989_JCompNeurol_GCDendrDiam_midget_c.mat"
        literature_data_folder
        / "Watanabe_1989_JCompNeurol_MidgetDendrDiam_Fig7_c.npz"
    )
    temporal_BK_model_file = (
        literature_data_folder / "Benardete_1997_VisNeurosci_midget.csv"
    )


profile = False

if __name__ == "__main__":
    if profile is True:
        import cProfile, pstats

        profiler = cProfile.Profile()
        profiler.enable()

    """
    Housekeeping. Do not comment out.

    All ProjectManager input parameters go to context.
    Init methods ask for these parameters by _properties_list class attribute.
    """
    PM = ProjectManager(
        path=path,
        input_folder=input_folder,
        output_folder=output_folder,
        stimulus_folder=stimulus_folder,
        project=project,
        experiment=experiment,
        ray_root_path=ray_root_path,
        my_retina=my_retina,
        my_stimulus_metadata=my_stimulus_metadata,
        my_stimulus_options=my_stimulus_options,
        my_run_options=my_run_options,
        apricot_data_folder=apricot_data_folder,
        literature_data_folder=literature_data_folder,
        dendr_diam1_file=dendr_diam1_file,
        dendr_diam2_file=dendr_diam2_file,
        gc_density_file=gc_density_file,
        temporal_BK_model_file=temporal_BK_model_file,
        apricot_metadata=apricot_metadata,
        numpy_seed=numpy_seed,
    )

    #################################
    ### Check cone response ###
    #################################

    """
    The Chichilnisky model receptive fields were measured from isolated retinas.
    Images were focused on photoreceptors. To account for the blur by the eye,
    we have the cone sample image method.
    """
    # PM.cones.image2cone_response()
    # PM.viz.show_cone_response(PM.cones.image, PM.cones.image_after_optics, PM.cones.cone_response)

    # TODO take raw hdf5 image through cone response to working retina

    #################################
    ### Sample image data ###
    #################################

    # # Note: sample only temporal hemiretina
    # from project.project_utilities_module import DataSampler
    # filename = "Perry_1984_Neurosci_GCdensity_Fig8.jpg"
    # filename_full = git_repo_root.joinpath(r"retina/literature_data", filename)
    # min_X, max_X, min_Y, max_Y = 0, 17, 0, 35 # Fig lowest and highest tick values, use these as calibration points
    # ds = DataSampler(filename_full, min_X, max_X, min_Y, max_Y)
    # ds.collect_and_save_points()
    # ds.quality_control()

    #################################
    #################################
    ### Build retina ###
    #################################
    #################################

    """
    Build and test your retina here, one gc type at a time. Temporal hemiretina of macaques.
    """

    # # Main retina construction method. This method calls all other methods in the retina construction process.
    # PM.construct_retina.build()

    # The following visualizations are dependent on the ConstructRetina instance.
    # This is why they are called via the construct_retina attribute. The instance
    # object is attached to the call for viz.

    # This function visualizes the spatial and temporal filter responses, ganglion cell positions and density,
    # mosaic layout, spatial and temporal statistics, dendrite diameter versus eccentricity, and tonic drives
    # in the retina mosaic building process.
    # PM.viz.show_exp_build_process(show_all_spatial_fits=False)

    # PM.viz.show_gen_exp_spatial_fit(n_samples=20)
    # PM.viz.show_gen_exp_spatial_rf(ds_name="test_ds", n_samples=10)
    # PM.viz.show_latent_tsne_space()
    # PM.viz.show_gen_spat_post_hist()
    # PM.viz.show_latent_space_and_samples()
    # PM.viz.show_retina_img()
    # PM.viz.show_rf_imgs(n_samples=10)
    # PM.viz.show_rf_violinplot()

    # # "train_loss", "val_loss", "mse", "ssim", "kid_mean", "kid_std"
    # this_dep_var = "val_loss"
    # ray_exp_name = None  # "TrainableVAE_2023-04-20_22-17-35"  # None for most recent
    # highlight_trial = None  # "2199e_00029"  # or None
    # PM.viz.show_ray_experiment(
    #     ray_exp_name, this_dep_var, highlight_trial=highlight_trial
    # )

    ###################################
    ###################################
    ###         Single Trial        ###
    ###################################
    ###################################

    ########################
    ### Create stimulus ###
    ########################

    # See my_stimulus_options for valid stimulus_options
    # PM.stimulate.make_stimulus_video()

    ###########################################
    ### Load stimulus to get working retina ###
    ###########################################

    # PM.working_retina.load_stimulus()

    ##########################################
    ### Show single ganglion cell response ###
    ##########################################

    # example_gc = 0  # int or 'None'
    # PM.working_retina.convolve_stimulus(example_gc)

    # # # PM.viz.show_spatiotemporal_filter(PM.working_retina)
    # PM.viz.show_convolved_stimulus(PM.working_retina)

    ########################################
    # Show impulse response and exit
    ########################################

    # contrast_for_impulses = [1.0]  # [1.0] for midget units
    # PM.working_retina.run_cells(
    #     cell_index=my_run_options["cell_index"],  # int
    #     get_impulse_response=True,  # Return with impulse response
    #     contrast_for_impulses=contrast_for_impulses,  # List of contrasts
    # )

    # PM.viz.show_impulse_response(PM.working_retina, savefigname="testi.png")

    ####################################
    ### Run multiple trials or cells ###
    ####################################

    # PM.working_retina.run_with_my_run_options()

    # PM.viz.show_gc_responses(PM.working_retina, savefigname="sin_drift_40Hz.png")

    # PM.viz.show_stimulus_with_gcs(
    #     PM.working_retina,
    #     example_gc=my_run_options["cell_index"],
    #     frame_number=450,
    #     show_rf_id=True,
    # )
    # TODO:
    # VISUALISOI VAE RF ÄRSYKKEEN KANSSA, TARKISTA ETTEI OLE JO YLLÄ KUN RF LUODAAN
    # VISUALISOI SOLUJEN VASTEET YKSITELLEN HARMAINA KÄPPYRÖINÄ KUN AJETAAN KOKONAINEN KOE
    # REFAKTOROI WORKING RETINA OBJEKTI VIZ IIN PROJECT MANAGERISSA JA POISTA FUNKTIOSIGNATUUREISTA

    # PM.viz.show_single_gc_view(
    #     PM.working_retina, cell_index=example_gc, frame_number=21
    # )

    # PM.viz.plot_tf_amplitude_response(PM.working_retina, my_run_options["cell_index"])

    # PM.viz.plot_midpoint_contrast(PM.working_retina, example_gc)
    # plt.show(block=False)

    # PM.viz.plot_local_rms_contrast(PM.working_retina, example_gc)
    # plt.show(block=False)

    # PM.viz.plot_local_michelson_contrast(PM.working_retina, example_gc)
    # plt.show(block=False)

    #####################
    ### Run all cells ###
    #####################

    # PM.viz.show_gc_responses(PM.working_retina)

    # PM.working_retina.save_spikes_csv(filename='testi_spikes.csv') # => METADATA
    # PM.working_retina.save_structure_csv(filename='testi_structure.csv') # => METADATA

    ###############################################
    ###############################################
    ###     Experiment with multiple trials     ###
    ###############################################
    ###############################################

    exp_variables = ["temporal_frequency"]  # from my_stimulus_options
    # exp_variables = ["temporal_frequency", "contrast"]  # from my_stimulus_options
    # # Define experiment parameters. List lengths must be equal.
    # # Examples: exp_variables = ["contrast"], min_max_values = [[0.015, 0.98]], n_steps = [30], logaritmic = [True]
    experiment_dict = {
        "exp_variables": exp_variables,
        "min_max_values": [[0.5, 32]],  # two vals for each exp_variable # frequency
        "n_steps": [10],
        "logaritmic": [True],
        # "min_max_values": [[0.5, 32], [0.02, 0.8]],  # temporal frequency, contrast
        # "n_steps": [10, 6],  # temporal frequency, contrast
        # "logaritmic": [True, True],  # temporal frequency, contrast
    }

    PM.experiment.build_and_run(experiment_dict, n_trials=5)

    ###############################
    ## Analyze Experiment ###
    ###############################

    my_analysis_options = {
        "exp_variables": exp_variables,
        "t_start_ana": 1,
        "t_end_ana": 7,
    }

    PM.ana.analyze_response(my_analysis_options)

    ################################
    ### Visualize Experiment ###
    ################################

    # PM.viz.F1F2_popul_response(exp_variables, xlog=False, savefigname=None)
    PM.viz.F1F2_unit_response(exp_variables, xlog=True, savefigname=output_folder)
    # PM.viz.fr_response(exp_variables, xlog=True, savefigname=None)
    # PM.viz.spike_raster_response(exp_variables, savefigname=None)
    # PM.viz.tf_vs_fr_cg(
    #     exp_variables, n_contrasts=None, xlog=False, ylog=False, savefigname=None
    # )

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
