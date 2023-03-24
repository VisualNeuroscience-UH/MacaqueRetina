# Visualization
import matplotlib.pyplot as plt

# Builtin
from pathlib import Path
import sys
import pdb
import time

# sys.path.append(Path(__file__).resolve().parent.parent)
# Start measuring time
start_time = time.time()

# Local
from project.project_manager_module import ProjectManager

# from project_manager_module import ProjectManager

"""
This is code for building macaque retinal filters corresponding to midget and parasol cell responses. We keep modular code structure, to be able to add new features at later phase.

The cone photoreceptor sampling is approximated as achromatic (single) compressive cone response(Baylor_1987_JPhysiol).

Visual angle (A) in degrees from previous studies (Croner and Kaplan, 1995) was approximated with relation 5 deg/mm. This works fine up to 20 deg ecc, but underestimates the distance thereafter. If more peripheral representations are later necessary, the millimeters should be calculated by inverting the relation 
A = 0.1 + 4.21E + 0.038E^2 (Drasdo and Fowler, 1974; Dacey and Petersen, 1992). Current implementation uses one deg = 220um (Perry et al 1985). One mm retina is ~4.55 deg visual field.

We have extracted statistics of macaque ganglion cell receptive fields from literature and build continuous functions.

The density of many cell types is inversely proportional to dendritic field coverage, suggesting constant coverage factor (Perry_1984_Neurosci, Wassle_1991_PhysRev). Midget coverage factor is 1  (Dacey_1993_JNeurosci for humans; Wassle_1991_PhysRev, Lee_2010_ProgRetEyeRes). Parasol coverage factor is 3-4 close to fovea (Grunert_1993_VisRes); 2-7 according to Perry_1984_Neurosci. These include ON- and OFF-center cells, and perhaps other cell types. It is likely that coverage factor is 1 for midget and parasol ON- and OFF-center cells each, which is also in line with Doi_2012 JNeurosci, Field_2010_Nature

The spatiotemporal receptive fields for the four cell types (parasol & midget, ON & OFF) were modelled with double ellipsoid difference-of-Gaussians model. The original spike triggered averaging RGC data in courtesy of Chichilnisky lab. The method is described in Chichilnisky_2001_Network, Chichilnisky_2002_JNeurosci Field_2010_Nature.

Chichilnisky_2002_JNeurosci states that L-ON (parasol) cells have on average 21% larger RFs than L-OFF cells. He also shows that OFF cells have more nonlinear response to input, which is not implemented currently (a no-brainer to implement if necessary).

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

"""

"""
Use keyword substring "file" in filenames, and "folder" in folder names to assert that they are turned into pathlib objects. Path structure is assumed to be root_path/project/experiment/output_folder

Abbreviations:
ana : analysis
col : column
full : full absolute path 
param : parameter
"""

"""
Main paths in different operating systems
"""
if sys.platform == "linux":
    root_path = "/opt3/Laskenta/Models"  # pikkuveli
    git_repo_root = Path(r"/opt2/Git_Repos/MacaqueRetina")
    # git_repo_root = r"/opt2/Git_Repos/MacaqueRetina"
    # root_path = "/opt2/Models"  # isosisko
elif sys.platform == "win32":
    root_path = r"C:\Users\simov\Laskenta\Models"
    git_repo_root = Path(r"C:\Users\simov\Laskenta\GitRepos\MacaqueRetina")


"""
Project name
"""
project = "Retina"


"""
Current experiment
"""
experiment = "test"  # "test"


"""
Input context
Stimulus images and videos
"""
input_folder = "../in"  # input figs, videos


"""
Data context for output. 
"""

output_folder = "out"

"""
### Housekeeping ###. Do not comment out.
"""
root_path = Path(root_path)
path = Path.joinpath(root_path, Path(project), experiment)


my_retina = {
    "gc_type": "parasol",
    "response_type": "on",
    # "ecc_limits": [4.8, 5.2],
    # "sector_limits": [-0.4, 0.4],
    "ecc_limits": [4, 5],
    "sector_limits": [-1, 1],
    "model_density": 1.0,
    "randomize_position": 0.05,
    "stimulus_center": 4.45 + 0j,
    "model_type": "VAE",  # "FIT", "VAE" for variational autoencoder, or GAN for generative adversarial network.
    "training_mode": "load_model",  # "train_model" or "tune_model" or "load_model" Applies to VAE or GAN only.
}


my_stimulus_metadata = {
    "stimulus_file": "nature1.avi",  # nature1.avi, testi.jpg
    "stimulus_video_name": "testi.mp4",  # REFACTOR
    "pix_per_deg": 60,
    "apply_cone_filter": False,
}

"""
Valid stimulus_options include (overriding visual_stimulus_module.VideoBaseClass):

image_width: in pixels
image_height: in pixels
container: file format to export
codec: compression format
fps: frames per second
duration_seconds: stimulus duration
baseline_start_seconds: midgray at the beginning
baseline_end_seconds: midgray at the end
pattern:
    'sine_grating'; 'square_grating'; 'colored_temporal_noise'; 'white_gaussian_noise';
    'natural_images'; 'phase_scrambled_images'; 'natural_video'; 'phase_scrambled_video';
    'temporal_sine_pattern'; 'temporal_square_pattern'; 'spatially_uniform_binary_noise'
stimulus_form: 'circular'; 'rectangular'; 'annulus'
stimulus_position: in degrees, (0,0) is the center.
stimulus_size: In degrees. Radius for circle and annulus, half-width for rectangle.
contrast: between 0 and 1
mean: mean stimulus intensity between 0, 256

Note if mean + ((contrast * max(intensity)) / 2) exceed 255 or if
        mean - ((contrast * max(intensity)) / 2) go below 0
        the stimulus generation fails

For sine_grating and square_grating, additional arguments are:
spatial_frequency: in cycles per degree
temporal_frequency: in Hz
orientation: in degrees

For all temporal and spatial gratings, additional argument is
phase_shift: between 0 and 2pi

For spatially_uniform_binary_noise, additional argument is
on_proportion: between 0 and 1, proportion of on-stimulus, default 0.5
direction: 'increment' or 'decrement'
stimulus_video_name: name of the stimulus video
"""

my_stimulus_options = {
    # Shared btw stimulus and working_retina
    "image_width": 240,  # 752 for nature1.avi
    "image_height": 240,  # 432 for nature1.avi
    "pix_per_deg": 60,
    "fps": 30,
    "pattern": "sine_grating",  # Natural video is not supported yet. One of the StimulusPatterns
    # stimulus only
    "stimulus_form": "rectangular",
    "temporal_frequency": 2,
    "spatial_frequency": 2.0,
    "stimulus_position": (0, 0),  # center_deg
    "duration_seconds": 4.0,
    "stimulus_size": 1.0,
    "contrast": 0.99,
    "baseline_start_seconds": 0.5,
    "baseline_end_seconds": 0.5,
    "background": 128,
    "mean": 128,
    "phase_shift": 0,
}

# Each gc response file contain n_trials
n_files = 1

my_run_options = {
    "cell_index": 2,  # int or None for all cells
    "n_trials": 10,  # For each of the response files
    "spike_generator_model": "poisson",  # poisson or refractory
    "save_data": True,
    "gc_response_filenames": [f"gc_response_{x:02}" for x in range(n_files)],
}


"""
Semi-constant variables
"""

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
rm_param = 25  # pA
k_param = 2.77e-4  # at 500 nm
cone_sensitivity_min = 5e2
cone_sensitivity_max = 1e4

my_retina_append = {
    "proportion_of_parasol_gc_type": proportion_of_parasol_gc_type,
    "proportion_of_midget_gc_type": proportion_of_midget_gc_type,
    "proportion_of_ON_response_type": proportion_of_ON_response_type,
    "proportion_of_OFF_response_type": proportion_of_OFF_response_type,
    "mosaic_file_name": "parasol_on_single.csv",
    "deg_per_mm": deg_per_mm,
    "optical_aberration": 2 / 60,  # unit is degree
    "cone_sensitivity_min": cone_sensitivity_min,
    "cone_sensitivity_max": cone_sensitivity_max,
    "rm": rm_param,
    "k": k_param,
}

my_retina.update(my_retina_append)

apricot_data_folder = git_repo_root.joinpath(r"retina/apricot_data")
literature_data_folder = git_repo_root.joinpath(r"retina/literature_data")

# Define digitized literature data files for gc density and dendritic diameters.
# Data from Watanabe_1989_JCompNeurol and Perry_1984_Neurosci

gc_density_file = literature_data_folder / "Perry_1984_Neurosci_GCdensity_c.mat"
if my_retina["gc_type"] == "parasol":
    dendr_diam1_file = (
        literature_data_folder / "Perry_1984_Neurosci_ParasolDendrDiam_c.mat"
    )
    dendr_diam2_file = (
        literature_data_folder / "Watanabe_1989_JCompNeurol_GCDendrDiam_parasol_c.mat"
    )
elif my_retina["gc_type"] == "midget":
    dendr_diam1_file = (
        literature_data_folder / "Perry_1984_Neurosci_MidgetDendrDiam_c.mat"
    )
    dendr_diam2_file = (
        literature_data_folder / "Watanabe_1989_JCompNeurol_GCDendrDiam_midget_c.mat"
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
        project=project,
        experiment=experiment,
        my_retina=my_retina,
        my_stimulus_metadata=my_stimulus_metadata,
        my_stimulus_options=my_stimulus_options,
        my_run_options=my_run_options,
        apricot_data_folder=apricot_data_folder,
        literature_data_folder=literature_data_folder,
        dendr_diam1_file=dendr_diam1_file,
        dendr_diam2_file=dendr_diam2_file,
        gc_density_file=gc_density_file,
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
    ### Build retina ###
    #################################

    """
    Build and test your retina here, one gc type at a time. Temporal hemiretina of macaques.
    """

    # options are defined in my_retina_options

    PM.construct_retina.build()

    # PM.construct_retina.save_mosaic()

    PM.construct_retina.show_build_process(show_all_spatial_fits=False)

    #################################
    ### Create stimulus ###
    #################################

    # options are defined in my_stimulus_options
    # stimulus video will be saved on output_folder in mp4 format (viewing and hdf5 format (for reloading)
    # PM.stimulate.make_stimulus_video()

    ##############################
    ### Create analog stimulus ###
    ##############################

    # # Analog stimulus comprises of continuous waveforms of types 'quadratic_oscillation', 'noise' or 'step_current'. You get few input  channels (N_units) of temporal signals. These signals do not pass through the retina, instead they are saved as .mat files.

    # N_tp = 20000
    # dt = 0.1 # ms

    # # for freq in range(1,101):
    # #     N_cycles = freq * (dt/1000) * N_tp
    # #     print(f"Creating stim with {freq=}, holding {N_cycles=}")

    # #     filename_out =  f'freq_{freq:02}.mat'

    # freq = 2
    # N_cycles = freq * (dt/1000) * N_tp
    # filename_out =  'test.mat'
    # analog_options = {
    # "filename_out" : filename_out,
    # "N_units" :3,
    # "coord_type" :"real",
    # "N_tp" : N_tp,
    # "input_type" : 'quadratic_oscillation', # 'quadratic_oscillation' or 'noise' or 'step_current'
    # "N_cycles" : [N_cycles, 0, 0], # Scalar provides two units at quadrature, other units are zero. List of ints/floats provides separate freq to each. Ignored for noise.
    # "dt" : dt, # IMPORTANT: assuming milliseconds
    # "save_stimulus" : True
    # }

    # PM.analog_input.make_stimulus_video(analog_options=analog_options)
    # PM.viz.plot_analog_stimulus(PM.analog_input)

    #################################
    ### Load stimulus to get working retina ###
    #################################

    # # # If you want to load with object, it is possible by:
    # PM.working_retina.load_stimulus(PM.stimulate)
    PM.working_retina.load_stimulus()

    # movie = vs.NaturalMovie(r'C:\Users\Simo\Laskenta\Stimuli\videoita\naturevids\nature1.avi', fps=100, pix_per_deg=60)# => METADATA
    # ret.load_stimulus(movie)# => METADATA

    #################################
    ### Show single ganglion cell response ###
    #################################

    # example_gc = 2  # int or 'None'
    # PM.working_retina.convolve_stimulus(example_gc)

    # PM.viz.show_spatiotemporal_filter(PM.working_retina)
    # PM.viz.show_convolved_stimulus(PM.working_retina)

    #################################
    ### Run multiple trials for single cell ###
    #################################

    # PM.working_retina.run_with_my_run_options()

    # PM.viz.show_gc_responses(PM.working_retina)

    # PM.viz.show_stimulus_with_gcs(
    #     PM.working_retina, example_gc=my_run_options["cell_index"], frame_number=51
    # )

    # PM.viz.show_single_gc_view(PM.working_retina, cell_index=example_gc, frame_number=21)

    # PM.viz.plot_tf_amplitude_response(PM.working_retina, example_gc)

    # PM.viz.plot_midpoint_contrast(PM.working_retina, example_gc)
    # plt.show(block=False)

    # PM.viz.plot_local_rms_contrast(PM.working_retina, example_gc)
    # plt.show(block=False)

    # PM.viz.plot_local_michelson_contrast(PM.working_retina, example_gc)
    # plt.show(block=False)

    #################################
    ### Run all cells ###
    #################################

    # PM.working_retina.run_all_cells(spike_generator_model="poisson", save_data=False)
    # PM.viz.show_gc_responses(PM.working_retina)

    # PM.working_retina.save_spikes_csv(filename='testi_spikes.csv') # => METADATA
    # PM.working_retina.save_structure_csv(filename='testi_structure.csv') # => METADATA

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
