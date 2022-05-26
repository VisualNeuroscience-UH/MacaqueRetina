# Numeric
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Builtin
from pathlib import Path
import sys
from math import nan
from itertools import islice
import pdb

# This computer git repos
from project.project_manager_module import ProjectManager

"""
This is code for building macaque retinal filters corresponding to midget and parasol cell responses. We keep modular code structure, to be able to add new features at later phase.

The cone photoreceptor sampling is approximated as achromatic (single) compressive cone response(Baylor_1987_JPhysiol).

Visual angle (A) in degrees from previous studies (Croner and Kaplan, 1995) was approximated with relation 5 deg/mm. This works fine up to 20 deg ecc, but underestimates the distance thereafter. If more peripheral representations are later necessary, the millimeters should be calculates by inverting the relation 
A = 0.1 + 4.21E + 0.038E^2 (Drasdo and Fowler, 1974; Dacey and Petersen, 1992). Current implementation uses one deg = 220um (Perry et al 1985). One mm retina is ~4.55 deg visual field.

We have extracted statistics of macaque ganglion cell receptive fields from literature and build continuous functions.

The density of many cell types is inversely proportional to dendritic field coverage, suggesting constant coverage factor (Perry_1984_Neurosci, Wassle_1991_PhysRev). Midget coverage factor is 1  (Dacey_1993_JNeurosci for humans; Wassle_1991_PhysRev, Lee_2010_ProgRetEyeRes). Parasol coverage factor is 3-4 close to fovea (Grunert_1993_VisRes); 2-7 according to Perry_1984_Neurosci. These include ON- and OFF-center cells, and perhaps other cell types. It is likely that coverage factor is 1 for midget and parasol ON- and OFF-center cells each, which is also in line with Doi_2012 JNeurosci, Field_2010_Nature

The spatiotemporal receptive fields for the four cell types (parasol & midget, ON & OFF) were modelled with double ellipsoid difference-of-Gaussians model. The original spike triggered averaging RGC data in courtesy of Chichilnisky lab. The method is described in Chichilnisky_2001_Network, Chichilnisky_2002_JNeurosci Field_2010_Nature.

Chichilnisky_2002_JNeurosci states that L-ON (parasol) cells have on average 21% larger RFs than L-OFF cells. He also shows that OFF cells have more nonlinear response to input, which is not implemented currently (a no-brainer to implement if necessary).

NOTE: bad cell indices hard coded from Chichilnisky apricot data. For another data set, visualize fits, and change the bad cells.
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
ci : current injection
col : column
coll : collated, collected
conn : connections
full : full absolute path
mid : midpoint
param : parameter
"""

"""
Main paths in different operating systems
"""
if sys.platform == "linux":
    root_path = "/opt3/Laskenta/Models"  # pikkuveli
    # root_path = "/opt2/Laskenta_ssd/Models"  # isosisko
elif sys.platform == "win32":
    root_path = r"C:\Users\Simo\Laskenta\Models"


"""
Project name
"""
project = "Retina"

"""
Current experiment
"""
experiment = "test"  # "test"

"""
### Housekeeping ###. Do not comment out.
"""
path = Path.joinpath(Path(root_path), Path(project), experiment)


"""
Input context
"""
input_folder = "../in"

"""
Data context for single files and arrays. These midpoint and parameter strings are used only in this module.
"""

output_folder = "out"

my_retina = {
    "mosaic_file": "parasol_on_single.csv",
}

my_stimuli = {
    "stimulus_file": "testi.jpg",
    "stimulus_type": "image",  # "image", "video" or "grating"
    "gc_response_file": "my_gc_response",  # check extension
}


stimulus_video_name = "tmp"


profile = False

if __name__ == "__main__":

    if profile is True:
        import cProfile, pstats

        profiler = cProfile.Profile()
        profiler.enable()

    """
    ### Housekeeping ###. Do not comment out.
    """
    PM = ProjectManager(
        path=path,
        input_folder=input_folder,
        output_folder=output_folder,
        project=project,
        experiment=experiment,
        retina=my_retina,
    )

    #################################
    ### Get image ###
    #################################

    """
    """
    # PM.cones.sample_image(image_file_name="testi.jpg")
    # PM.viz.show_cone_response(PM.cones.image, PM.cones.image_after_optics, PM.cones.cone_response)

    #################################
    ### Build retina ###
    #################################

    """
    Build and test your retina here, one gc type at a time. Temporal hemiretina of macaques.
    """
    PM.mosaic_constructor.initialize(
        gc_type="parasol",
        response_type="on",
        ecc_limits=[4.8, 5.2],
        sector_limits=[-0.4, 0.4],
        model_density=1.0,
        randomize_position=0.05,
    )

    PM.mosaic_constructor.build(show_build_process=False)
    PM.mosaic_constructor.save_mosaic("parasol_on_single.csv")

    testmosaic = pd.read_csv("parasol_on_single.csv", index_col=0)

    PM.functional_mosaic.initialize(
        testmosaic,
        "parasol",
        "on",
        stimulus_center=5 + 0j,
        stimulus_width_pix=240,
        stimulus_height_pix=240,
    )

    PM.stimulate.make_stimulus_video(
        pattern="temporal_square_pattern",
        stimulus_form="circular",
        temporal_frequency=0.1,
        spatial_frequency=1.0,
        stimulus_position=(-0.06, 0.03),
        duration_seconds=0.4,
        image_width=240,
        image_height=240,
        stimulus_size=0.1,
        contrast=0.99,
        baseline_start_seconds=0.5,
        baseline_end_seconds=0.5,
        background=128,
        mean=128,
        phase_shift=0,
        stimulus_video_name=stimulus_video_name,  # If empty, does not save the video
    )

    # ret.load_stimulus(grating)
    # ret.load_stimulus(stim)
    PM.functional_mosaic.load_stimulus(PM.stimulate)

    # # movie = vs.NaturalMovie('/home/henhok/nature4_orig35_fps100.avi', fps=100, pix_per_deg=60)
    # movie = vs.NaturalMovie(r'C:\Users\Simo\Laskenta\Stimuli\videoita\naturevids\nature1.avi', fps=100, pix_per_deg=60)
    # ret.load_stimulus(movie)

    # ret.plot_midpoint_contrast(0)
    # plt.show()
    # ret.plot_local_rms_contrast(0)
    # plt.show()
    # ret.plot_local_michelson_contrast(0)
    # plt.show()

    example_gc = 2  # int or 'None'
    # ret.convolve_stimulus(example_gc, show_convolved_stimulus=True)
    # plt.show()

    filenames = [f"Response_foo_{x}" for x in np.arange(1)]

    for filename in filenames:

        PM.functional_mosaic.run_cells(
            cell_index=example_gc,
            n_trials=5,
            save_data=False,
            spike_generator_model="poisson",
            return_monitor=False,
            filename=filename,
            show_gc_response=True,
        )
    plt.show(block=False)

    # # ret.run_all_cells(show_gc_response=True, spike_generator_model='refractory', reload_last=False)
    # # plt.show(block = False)
    # # ret.save_spikes_csv()

    PM.functional_mosaic.viz.show_stimulus_with_gcs(example_gc=example_gc, frame_number=51)
    # ret.show_single_gc_view(cell_index=example_gc, frame_number=21)
    # plt.show(block = False)

    # plt.show()

    #################################
    ### Project files & Utilities ###
    #################################

    ##########################################
    ###### Analysis & Viz, single files ######
    ##########################################

    #######################################################
    ###### Viz, single array runs, multiple analyzes ######
    #######################################################

    ############################################################
    ###### Analyze & Viz, array runs, multiple iterations ######
    ############################################################

    """
    Show xy plot allows any parametric data plotted against each other.
    Uses seaborn regplot or lineplot. Seaborn options easy to include into code (viz_module).
    All analyzes MUST be included into to_mpa_dict
    Same data at the x and y axis simultaneously can be used with regplot.
    If compiled_type is accuracy, and only mean datatype is available, 
    uses the mean. 

    midpoints: 'Comrad', 'HiFi', 'Bacon'
    parameters: 'C', 'gL', 'VT', 'EL', 'delay'
    analyzes: 
    'Coherence', 'Granger Causality', 'Transfer Entropy', 'Simulation Error', 'Excitatory Firing Rate', 'Inhibitory Firing Rate', 'Euclidean Distance'

    kind: regplot, binned_lineplot 
        regplot is scatterplot, where only single midpoint and parameter should be plotted at a time. draw_regression available.
        binned_lineplot bins x-data, then compiles parameters/midpoints and finally shows distinct midpoints/parameters (according to "hue") with distinct hues. Error shading 
        indicates 95% confidence interval, obtained by bootstrapping the data 1000 times (seaborn default)
    """
    # xy_plot_dict = {
    #     "x_ana": [
    #         "Coherence",
    #         # "Transfer Entropy",
    #     ],  # multiple allowed => subplot rows, unless ave
    #     "x_mid": [
    #         'Comrad',
    #         # 'Bacon',
    #         # 'HiFi',
    #     ],  # single allowed, multiple (same mids for y) if type binned_lineplot
    #     "x_para": ['C'],  # single allowed, multiple (same params for y) if type binned_lineplot
    #     "x_ave": False,  # Weighted average over NGs. Works only for kind = regplot

    #     "y_ana": [
    #         # "Coherence",
    #         "Granger Causality",
    #         # "GC as TE",
    #         # "Transfer Entropy",
    #         # 'Euclidean Distance', # Note: Euclidean Distance cannot have accuracy
    #         # 'Simulation Error'
    #     ],  # multiple allowed => subplot columns, unless ave
    #     "y_mid": [
    #         'Comrad',
    #         # 'Bacon',
    #         # 'HiFi',
    #         ],  # single allowed, multiple (same mids for x) if type binned_lineplot
    #     "y_para": ['C'],  # single allowed, multiple (same params for x) if type binned_lineplot
    #     "y_ave": False,  # Weighted average over NGs. Works only for kind = regplot

    #     "kind": "regplot",  # binned_lineplot, regplot
    #     "n_bins": 10,  # ignored for regplot
    #     "hue": "Midpoint",  # Midpoint or Parameter. If Midpoint is selected, each line is one midpoint and parameters will be combined. And vice versa. Ignored for regplot
    #     "compiled_results": True,  # x and y data from folder XX'_compiled_results'
    #     "compiled_type": "mean",  # mean, accuracy; falls back to mean if accuracy not found
    #     "draw_regression": True, # only for regplot
    #     "order": 1,  # Regression polynomial fit order, only for regplot
    #     "draw_diagonal": False,  # only for regplot

    #     "xlog": False,
    #     "ylog": False,
    #     "sharey": False,
    # }

    # PM.viz.show_xy_plot(xy_plot_dict)

    """
    Show input-to-output classification confusion matrix
    midpoints Comrad, Bacon, HiFi; parameter 'C', 'gL', 'VT', 'EL', 'delay'
    """
    # PM.viz.show_IxO_conf_mtx(midpoint='Comrad', parameter='VT', ana_list=['Coherence', 'GrCaus', 'TransferEntropy', 'NormError'],
    #     ana_suffix_list=['sum', 'Information', 'TransfEntropy', 'SimErr'], par_value_string_list=['-44.0', '-46.0'],
    #     best_is_list=['max', 'max', 'max', 'min'])

    ##################################################################################
    ###### Analyze & Viz, array runs, multiple iterations, multiple paths ############
    ##################################################################################

    """
    Categorical plot of parametric data.
    Definitions for parametric plotting of multiple conditions/categories.
    First, define what data is going to be visualized in to_mpa_dict.
    Second, define how it is visualized in param_plot_dict.

    Limitations: 
        You cannot have analyzes as title AND inner_sub = True.
        For violinplot and inner_sub = True, N bin edges MUST be two (split view)

    outer : panel (distinct subplots) # analyzes, midpoints, parameters, controls
    inner : inside one axis (subplot) # midpoints, parameters, controls
    inner_sub : bool, further subdivision by value, such as mean firing rate
    inner_sub_ana : name of ana. This MUST be included into to_mpa_dict "analyzes"
    plot_type : parametric plot type # box

    compiled_results : data at compiled_results folder, mean over iterations

    inner_paths : bool (only inner available for setting paths). Provide comparison from arbitrary paths, e.g. controls. The 'inner' is ignored.
    inner_path_names: list of names of paths to compare.
    paths : provide list of tuples of full path parts to data folder. 
    E.g. [(path, 'Single_narrow_iterations_control', 'Bacon_gL_compiled_results'),] 
    The number of paths MUST be the same as the number of corresponding inner variables. 
    save_description: bool, if True, saves pd.describe() to csv files for each title into path/Description/
    """

    # param_plot_dict = {
    #     "title": "parameters",  # multiple allowed => each in separate figure
    #     "outer": "analyzes",  # multiple allowed => plt subplot panels
    #     "inner": "midpoints",  # multiple allowed => direct comparison
    #     "inner_sub": False,  # A singular analysis => subdivisions
    #     "inner_sub_ana": "Excitatory Firing Rate",  #  The singular analysis
    #     "bin_edges": [[0.001, 150], [150, 300]],
    #     "plot_type": "box",  # "violin" (2), "box", "strip", "swarm", "boxen", "point", "bar"
    #     "compiled_results": True, # True, False
    #     "sharey": False,
    #     "palette": "Greys",
    #     "inner_paths": False,
    #     # "inner_path_names": ["Comrad", "Bacon", "Bacon_EI", "Bacon_ALL"],
    #     "paths": [
    #         # (Path(root_path), Path(project), 'Single_narrow_iterations', 'Comrad_gL_compiled_results'),
    #         # (Path(root_path), Path(project), 'Single_narrow_iterations_control_EI', 'Bacon_gL_compiled_results'),
    #         ],
    #     "inner_stat_test": True,
    #     "save_description": False,
    #     "save_name": "description_simulated",
    #     "display_optimal_values": True, # If True, draws optimal values in the plot
    #     "optimal_value_foldername": optimal_value_foldername,
    #     "optimal_description_name": optimal_description_name
    # }
    # PM.viz.show_catplot(param_plot_dict)

    """
    ### Housekeeping ###. Do not comment out.
    """
    plt.show()

    if profile is True:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats(20)
