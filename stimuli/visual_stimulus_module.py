# Numerical
import numpy as np
import numpy.matlib as npm

# import numpy.matlib as matlib
from scipy import ndimage
import colorednoise as cn

# Data IO
import cv2

# Viz
import matplotlib.pyplot as plt

# Builtin
from pathlib import Path
import pdb
# import time

plt.rcParams["image.cmap"] = "gray"

"""
This module creates the visual stimuli. Stimuli include patches of sinusoidal gratings at different orientations
and spatial frequencies. The duration can be defined in seconds and size (radius), and center location (x,y) 
in degrees.

Input: stimulus definition
Output: video stimulus frames

Formats .avi .mov .mp4 ?


"""


class VideoBaseClass(object):
    def __init__(self):
        """
        Initialize standard video stimulus
        The base class methods are applied to every stimulus
        """
        options = {}
        options["image_width"] = 1280  # Image width in pixels
        options["image_height"] = 720  # Image height in pixels
        options["container"] = "mp4"
        # options["codec"] = 'MP42'
        options["codec"] = "mp4v"
        options["fps"] = 100.0  # 64.0  # Frames per second
        options["duration_seconds"] = 1.0  # seconds
        options["intensity"] = (0, 255)  # video grey scale dynamic range.
        options["mean"] = 128  # intensity mean
        options["contrast"] = 1
        options[
            "raw_intensity"
        ] = None  # Dynamic range before scaling, set by each stimulus pattern method

        # Valid options sine_grating; square_grating; colored_temporal_noise; white_gaussian_noise; natural_images; natural_video; phase_scrambled_video
        options["pattern"] = "sine_grating"
        options[
            "phase_shift"
        ] = 0  # 0 - 2pi, to have grating or temporal oscillation phase shifted
        options[
            "stimulus_form"
        ] = "circular"  # Valid options circular, rectangular, annulus
        options["stimulus_position"] = (
            0.0,
            0.0,
        )  # Stimulus center position in degrees inside the video. (0,0) is the center.

        # In degrees. Radius for circle and annulus, half-width for rectangle. 0 gives smallest distance from image borders, ie max radius
        options["stimulus_size"] = 0.0

        # Init optional arguments
        options["spatial_frequency"] = None
        options["temporal_frequency"] = None
        options["spatial_band_pass"] = None
        options["temporal_band_pass"] = None
        options["orientation"] = 0.0  # No rotation or vertical
        options["size_inner"] = None
        options["size_outer"] = None
        options[
            "on_proportion"
        ] = 0.5  # between 0 and 1, proportion of stimulus-on time
        options["direction"] = "increment"  # or 'decrement'

        # Limits, no need to go beyond these
        options["min_spatial_frequency"] = 0.0625  # cycles per degree
        options["max_spatial_frequency"] = 16.0  # cycles per degree
        options["min_temporal_frequency"] = 0.5  # cycles per second, Hz
        options["max_temporal_frequency"] = 32.0  # cycles per second, Hz.

        options["background"] = 255  # Background grey value

        # Get resolution
        options["pix_per_deg"] = 60


        options["baseline_start_seconds"] = 0
        options["baseline_end_seconds"] = 0
        options["stimulus_video_name"] = None

        self.options = options

    def _scale_intensity(self):

        """Scale intensity to 8-bit grey scale. Calculating peak-to-peak here allows different
        luminances and contrasts"""

        intensity_max = np.max(self.options["intensity"])
        mean = self.options["mean"]  # This is the mean of final dynamic range
        contrast = self.options["contrast"]
        raw_min_value = np.min(self.options["raw_intensity"])
        raw_peak_to_peak = np.ptp(self.options["raw_intensity"])

        frames = self.frames

        # Simo's new version, Lmin, Lmax from mean and Michelson contrast, check against 0 and intensity_max
        # Michelson contrast = (Lmax -Lmin) / (Lmax + Lmin); mean = (Lmax + Lmin) / 2 =>
        Lmax = mean * (contrast + 1)
        Lmin = 2 * mean - Lmax
        peak_to_peak = Lmax - Lmin

        # Scale values
        # Shift to 0
        frames = frames - raw_min_value
        # Scale to 1
        frames = frames / raw_peak_to_peak

        # Final scale
        frames = frames * peak_to_peak

        # Final offset
        frames = frames + Lmin

        # Round result to avoid unnecessary errors
        frames = np.round(frames, 1)

        # Check that the values are between 0 and 255 to get correct conversion to uint8
        assert np.all(0 <= frames.flatten()) and np.all(
            frames.flatten() <= 255
        ), f"Cannot safely convert range {np.min(frames.flatten())}- {np.max(frames.flatten())}to uint8. Check intensity/dynamic range."
        # Return
        self.frames = frames.astype(np.uint8)

    def _prepare_grating(self):
        """Create temporospatial grating"""

        spatial_frequency = self.options["spatial_frequency"]
        temporal_frequency = self.options["temporal_frequency"]
        fps = self.options["fps"]
        duration_seconds = self.options["duration_seconds"]
        orientation = self.options["orientation"]

        if not spatial_frequency:
            print("Spatial_frequency missing, setting to 1")
            spatial_frequency = 1
        if not temporal_frequency:
            print("Temporal_frequency missing, setting to 1")
            temporal_frequency = 1

        # Create sine wave
        one_cycle = 2 * np.pi
        cycles_per_degree = spatial_frequency
        # image_width_in_degrees = self.options["image_width_in_deg"]
        image_width = self.options["image_width"]
        image_height = self.options["image_height"]
        image_width_in_degrees = image_width / self.options["pix_per_deg"]

        # Calculate larger image size to allow rotations
        diameter = np.ceil(np.sqrt(image_height**2 + image_width**2)).astype(np.int)
        image_width_diameter = diameter
        image_height_diameter = diameter
        image_width_diameter_in_degrees = (
            image_width_diameter / self.options["pix_per_deg"]
        )

        # Draw temporospatial grating
        # NB! one_cycle * cycles_per_degree needs to be multiplied with the scaled width to have
        # the desired number of cpd in output image
        image_position_vector = np.linspace(
            0,
            one_cycle * cycles_per_degree * image_width_diameter_in_degrees,
            image_width_diameter,
        )
        n_frames = self.frames.shape[2]

        # Recycling large_frames and self.frames below, instead of descriptive variable names for the evolving video, saves a lot of memory
        # Create large 3D frames array covering the most distant corner when rotated
        large_frames = np.tile(
            image_position_vector, (image_height_diameter, n_frames, 1)
        )
        # Correct dimensions to image[0,1] and time[2]
        large_frames = np.moveaxis(large_frames, 2, 1)
        total_temporal_shift = temporal_frequency * one_cycle * duration_seconds
        one_frame_temporal_shift = (temporal_frequency * one_cycle) / fps
        temporal_shift_vector = np.arange(
            0, total_temporal_shift, one_frame_temporal_shift
        )
        # Shift grating phase in time. Broadcasting temporal vector automatically to correct dimension.
        large_frames = large_frames + temporal_shift_vector

        # Rotate to desired orientation
        large_frames = ndimage.rotate(large_frames, orientation, reshape=False)

        # Cut back to original image dimensions
        marginal_height = (diameter - image_height) / 2
        marginal_width = (diameter - image_width) / 2
        marginal_height = np.round(marginal_height).astype(np.int)
        marginal_width = np.round(marginal_width).astype(np.int)
        self.frames = large_frames[
            marginal_height:-marginal_height, marginal_width:-marginal_width, :
        ]
        # remove rounding error
        self.frames = self.frames[0:image_height, 0:image_width, :]

    def _prepare_form(self, stimulus_size):

        center_deg = self.options["stimulus_position"]  # in degrees
        radius_deg = stimulus_size  # in degrees
        height = self.options["image_height"]  # in pixels
        width = self.options["image_width"]  # in pixels
        pix_per_deg = self.options["pix_per_deg"]

        # Turn position in degrees to position in mask, shift 0,0 to center of image
        center_pix = np.array([0, 0])
        center_pix[0] = int(
            width / 2 + pix_per_deg * center_deg[0]
        )  # NOTE Width goes to x-coordinate
        center_pix[1] = int(
            height / 2 + pix_per_deg * -center_deg[1]
        )  # NOTE Height goes to y-coordinate. Inverted to get positive up

        if (
            radius_deg == 0
        ):  # use the smallest distance between the center and image walls
            radius_pix = min(
                center_pix[0],
                center_pix[1],
                width - center_pix[0],
                height - center_pix[1],
            )
        else:
            radius_pix = pix_per_deg * radius_deg

        Y, X = np.ogrid[:height, :width]

        return X, Y, center_pix, radius_pix

    def _prepare_circular_mask(self, stimulus_size):

        X, Y, center_pix, radius_pix = self._prepare_form(stimulus_size)
        dist_from_center = np.sqrt((X - center_pix[0]) ** 2 + (Y - center_pix[1]) ** 2)

        mask = dist_from_center <= radius_pix
        return mask

    def _combine_background(self, mask):
        # OLD: self.frames = self.frames * mask[..., np.newaxis]
        self.frames_background = np.ones(self.frames.shape) * self.options["background"]
        self.frames_background[mask] = self.frames[mask]
        self.frames = self.frames_background

    def _prepare_temporal_sine_pattern(self):
        """Prepare temporal sine pattern"""

        temporal_frequency = self.options["temporal_frequency"]
        fps = self.options["fps"]
        duration_seconds = self.options["duration_seconds"]
        phase_shift = self.options["phase_shift"]

        if not temporal_frequency:
            print("Temporal_frequency missing, setting to 1")
            temporal_frequency = 1

        # Create sine wave
        one_cycle = 2 * np.pi
        cycles_per_second = temporal_frequency

        n_frames = self.frames.shape[2]
        image_width = self.options["image_width"]
        image_height = self.options["image_height"]

        # time_vector in radians, temporal modulation via np.sin()
        time_vec_end = 2 * np.pi * temporal_frequency * duration_seconds
        time_vec = np.linspace(
            0 + phase_shift, time_vec_end + phase_shift, int(fps * duration_seconds)
        )
        temporal_modulation = np.sin(time_vec)

        # Set the frames to sin values
        frames = np.ones(self.frames.shape) * temporal_modulation

        # Set raw_intensity to [-1 1]
        self.options["raw_intensity"] = (-1, 1)

        assert temporal_modulation.shape[0] == n_frames, "Unequal N frames, aborting..."
        assert (
            image_width != n_frames
        ), "Errors in 3D broadcasting, change image width/height NOT to match n frames "
        assert (
            image_height != n_frames
        ), "Errors in 3D broadcasting, change image width/height NOT to match n frames "

        self.frames = frames

    def _raw_intensity_from_data(self):

        self.options["raw_intensity"] = (np.min(self.frames), np.max(self.frames))

    def _create_frames(self, epoch__in_seconds):
        # Create frames for the requested duration in sec
        frames = (
            np.ones(
                (
                    self.options["image_height"],
                    self.options["image_width"],
                    int(self.options["fps"] * epoch__in_seconds),
                ),
                dtype=np.uint8,
            )
            * self.options["background"]
        )

        return frames

    def _set_zero_masked_pixels_to_bg(self):
        # Set masked pixels to background value.
        mask_value = 1
        self.frames[self.frames == mask_value] = self.options["background"]

    def _extract_frames(self, cap, n_frames):
        """
        Extracts and resizes the frames from a cv2.VideoCapture object

        :param cap:
        :return:
        """
        w = self.options["image_width"]
        h = self.options["image_height"]
        # Load each frame as array of gray values between 0-255
        for frame_ix in range(n_frames):
            _, frame = cap.read()

            frame_out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frames[:, :, frame_ix] = cv2.resize(frame_out, (w, h), interpolation=cv2.INTER_AREA)


class StimulusPattern:
    """
    Construct the stimulus image pattern.
    This class is for isolating logic. Self comes from
    the calling function as argument
    """

    def sine_grating(self):
        # Create temporospatial grating
        self._prepare_grating()

        # Turn to sine values
        self.frames = np.sin(self.frames + self.options["phase_shift"])

        # Set raw_intensity to [-1 1]
        self.options["raw_intensity"] = (-1, 1)

    def square_grating(self):
        # Create temporospatial grating
        self._prepare_grating()

        # Turn to sine values
        self.frames = np.sin(self.frames + self.options["phase_shift"])

        # Set raw_intensity to [-1 1]
        self.options["raw_intensity"] = (-1, 1)

        # Turn to square grating values, threshold at zero.
        threshold = (
            0  # Change this between [-1 1] if you want uneven grating. Default is 0
        )
        self.frames = (self.frames > threshold) * self.frames / self.frames * 2 - 1

    def white_gaussian_noise(self):
        self.frames = np.random.normal(loc=0.0, scale=1.0, size=self.frames.shape)

        self._raw_intensity_from_data()

    def temporal_sine_pattern(self):
        """Create temporal sine pattern"""

        self._prepare_temporal_sine_pattern()

    def temporal_square_pattern(self):
        """Create temporal sine pattern"""

        self._prepare_temporal_sine_pattern()
        # Turn to square grating values, threshold at zero.
        threshold = (
            0  # Change this between [-1 1] if you want uneven grating. Default is 0
        )

        self.frames[self.frames >= threshold] = 1
        self.frames[self.frames < threshold] = -1

    def colored_temporal_noise(self):
        beta = 1  # the exponent. 1 = pink noise, 2 = brown noise, 0 = white noise?
        variance_limits = np.array([-3, 3])
        samples = self.frames.shape[2]  # number of time samples to generate
        frame_time_series_unit_variance = cn.powerlaw_psd_gaussian(beta, samples)

        # Cut variance to [-3,3]
        frame_time_series_unit_variance_clipped = np.clip(
            frame_time_series_unit_variance,
            variance_limits.min(),
            variance_limits.max(),
        )
        # Scale to [0 1]
        frame_time_series = (
            frame_time_series_unit_variance_clipped - variance_limits.min()
        ) / variance_limits.ptp()

        self.options["raw_intensity"] = (0, 1)

        # Cast time series to frames
        assert (
            len(frame_time_series) not in self.frames.shape[:-1]
        ), "Oops. Two different dimensions match the time series length."
        self.frames = np.zeros(self.frames.shape) + frame_time_series

    def spatially_uniform_binary_noise(self):
        on_proportion = self.options["on_proportion"]
        samples = self.frames.shape[2]
        direction = self.options["direction"]

        def _rand_bin_array(samples, on_proportion):
            N = samples
            K = np.int(N * on_proportion)
            arr = np.zeros(N)
            arr[:K] = 1
            np.random.shuffle(arr)
            return arr

        frame_time_series = _rand_bin_array(samples, on_proportion)

        if direction == "decrement":
            frame_time_series = frame_time_series * -1  # flip
        elif direction == "increment":
            frame_time_series = (
                frame_time_series * 1
            )  # Isn't it beautiful? For consistency
        else:
            raise NotImplementedError(
                'Unknown option for "direction", should be "increment" or "decrement"'
            )

        # Note the stim has dyn range 1 and thus contrast will be halved by the
        # dyn range -1 1, thus the doubling
        self.options["contrast"] = self.options["contrast"] * 2
        self.options["raw_intensity"] = (-1, 1)

        self.frames = np.zeros(self.frames.shape) + frame_time_series

    def natural_images(self):

        if self.context.my_stimulus_metadata["apply_cone_filter"] is True:
            self.cones.image2cone_response()
            self.image = self.cones.cone_response
        else:
            image_file_name = self.context.my_stimulus_metadata["stimulus_file"]
            self.image = self.data_io.get_data(image_file_name)

        # resize image by specifying custom width and height
        resized_image = cv2.resize(self.image, self.frames.shape[:2])

        # Prep for temporal pattern. Writes on top of self.frames,
        self._prepare_temporal_sine_pattern()


        # Change temporal sine pattern to temporal square pattern.
        threshold = (
            0  # Change this between [-1 1] if you want uneven grating. Default is 0
        )
        self.frames[self.frames >= threshold] = 1
        self.frames[self.frames < threshold] = 0

        # add new axis to b to use numpy broadcasting
        resized_image = resized_image[:,:,np.newaxis]

        self.frames =  self.frames * resized_image

        # filtering: http://www.djmannion.net/psych_programming/vision/sf_filt/sf_filt.html
        self._raw_intensity_from_data()

    def phase_scrambled_images(self):
        raise NotImplementedError("Phase scrambled images are not implemented yet.")
        self._raw_intensity_from_data()

    def natural_video(self):

        
        # self.cones.image2cone_response()
        # self.image = self.cones.cone_response
        video_file_name = self.context.my_stimulus_metadata["stimulus_file"]
        video_cap = self.data_io.get_data(video_file_name)


        self.fps = self.options["fps"]
        self.pix_per_deg = self.options["pix_per_deg"]

        # Cut to desired length at desired fps
        n_frames = self.frames.shape[2]
        self.frames = np.ones(
            (self.options["image_height"], self.options["image_width"], n_frames)
        )

        video_n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_cap.get(cv2.CAP_PROP_FPS)

        self._extract_frames(video_cap, n_frames)

        print(
            "Original movie dimensions %d x %d px, %d frames at %d fps."
            % (video_width, video_height, video_n_frames, fps)
        )
        print(
            "Resized movie dimensions %d x %d px, %d frames at %d fps."
            % (self.options["image_width"], self.options["image_height"], n_frames, self.fps)
        )

        if self.context.my_stimulus_metadata["apply_cone_filter"] is True:
            pass
        # TÄHÄN JÄIT. MUUTA  PHOTORESEPTOR KÄYTTÄMÄÄN MYÖS VIDEOKUVAA.
        # TILAA VIDEOFILTTERÖINTI TÄSSÄ

        self._raw_intensity_from_data()
        video_cap.release()

    def phase_scrambled_video(self):
        raise NotImplementedError("Phase scrambled video is not implemented yet.")
        self._raw_intensity_from_data()


class StimulusForm:
    """
    Mask the stimulus images. This class is for isolating logic. 
    Self comes from the calling function as argument.
    """

    def circular(self):

        mask = self._prepare_circular_mask(self.options["stimulus_size"])

        # self.frames = self.frames * mask[..., np.newaxis]
        self._combine_background(mask)

    def rectangular(self):

        X, Y, center_pix, radius_pix = self._prepare_form(self.options["stimulus_size"])

        # Prepare rectangular distance map in pixels
        x_distance_vector = np.abs(X - center_pix[0])
        X_distance_matrix = npm.repmat(x_distance_vector, Y.shape[0], 1)
        y_distance_vector = np.abs(Y - center_pix[1])
        Y_distance_matrix = npm.repmat(y_distance_vector, 1, X.shape[1])

        # rectangular_dist_from_center = np.abs(X - center_pix[0]) + np.abs(Y - center_pix[1])
        mask = np.logical_and(
            (X_distance_matrix <= radius_pix), (Y_distance_matrix <= radius_pix)
        )

        # newaxis adds 3rd dim, multiplication broadcasts one 3rd dim to N 3rd dims in self.frames
        # self.frames = self.frames * mask[..., np.newaxis]
        self._combine_background(mask)

    def annulus(self):

        size_inner = self.options["size_inner"]
        size_outer = self.options["size_outer"]
        if not size_inner:
            print("Size_inner missing, setting to 1")
            size_inner = 1
        if not size_outer:
            print("Size_outer missing, setting to 2")
            size_outer = 2

        mask_inner = self._prepare_circular_mask(size_inner)
        mask_outer = self._prepare_circular_mask(size_outer)

        mask = mask_outer ^ mask_inner
        # self.frames = self.frames * mask[..., np.newaxis]
        self._combine_background(mask)

    def stencil(self):
        raise NotImplementedError


class ConstructStimulus(VideoBaseClass):
    """
    Create stimulus video and save
    """

    _properties_list = [
        "path",
        "output_folder",
        "input_folder",
        "my_stimulus_options",
        "my_stimulus_metadata",
    ]

    def __init__(self, context, data_io, cones):
        super().__init__()

        self._context = context.set_context(self._properties_list)
        self._data_io = data_io
        self._cones = cones


    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def cones(self):
        return self._cones

    def make_stimulus_video(self):
        """
        Valid stimulus_options include

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

        TODO Below not implemented yet.
        For natural_images, phase_scrambled_images, natural_video and phase_scrambled_video,
        additional arguments are:
        spatial_band_pass: (cycles per degree min, cycles per degree max)
        temporal_band_pass: (Hz min, Hz max)
        orientation: in degrees

        ------------------------
        Output: saves the stimulus video file to output path if stimulus_video_name is not empty or None
        """

        # Set input arguments to video-object, updates the defaults from VideoBaseClass
        print("Making a stimulus with the following properties:")
        
        my_stimulus_options = self.context.my_stimulus_options

        for this_option in my_stimulus_options:
            print(this_option, ":", my_stimulus_options[this_option])
            assert this_option in self.options.keys(), f"The option '{this_option}' was not recognized"

        self.options.update(my_stimulus_options)


        self.frames = self._create_frames(
            self.options["duration_seconds"]
        )  # background for stimulus

        # Check that phase shift is in radians
        assert (
            0 <= self.options["phase_shift"] <= 2 * np.pi
        ), "Phase shift should be between 0 and 2 pi"

        # Call StimulusPattern class method to get patterns (numpy array)
        # self.frames updated according to the pattern
        eval(
            f'StimulusPattern.{self.options["pattern"]}(self)'
        )  # Direct call to class.method() requires the self as argument

        # Now only the stimulus is scaled. The baseline and bg comes from options
        self._scale_intensity()

        # For natural images, set zero-masked pixels to background value
        if self.options["pattern"] == "natural_images":
            self._set_zero_masked_pixels_to_bg()

        # Call StimulusForm class method to mask frames
        # self.frames updated according to the form
        eval(
            f'StimulusForm.{self.options["stimulus_form"]}(self)'
        )  # Direct call to class.method() requires the self argument

        # Apply cone filtering to grating frames here
        # Not implemented yet
        # Natural images are filtered at the StimulusPattern method,
        # because they are typically not evolving over time

        self.frames_baseline_start = self._create_frames(
            self.options["baseline_start_seconds"]
        )  # background for baseline before stimulus
        self.frames_baseline_end = self._create_frames(
            self.options["baseline_end_seconds"]
        )  # background for baseline after stimulus

        # Concatenate baselines and stimulus, recycle to self.frames
        self.frames = np.concatenate(
            (self.frames_baseline_start, self.frames, self.frames_baseline_end), axis=2
        )
        self.frames = self.frames.astype(np.uint8)

        self.video = self.frames.transpose(2, 0, 1)
        self.fps = self.options["fps"]
        self.pix_per_deg = self.options["pix_per_deg"]

        self.video_n_frames = len(self.video)
        self.video_width = self.video[0].shape[1]
        self.video_height = self.video[1].shape[0]
        self.video_width_deg = self.video_width / self.pix_per_deg
        self.video_height_deg = self.video_height / self.pix_per_deg

        stimulus_video = self

        # Save video
        stimulus_video_name = Path(self.context.my_stimulus_metadata["stimulus_video_name"])
        self.data_io.save_stimulus_to_videofile(stimulus_video_name, stimulus_video)

    def get_2d_video(self):
        stim_video_2d = np.reshape(
            self.video, (self.video_n_frames, self.video_height * self.video_width)
        ).T  # pixels as rows, time as cols
        return stim_video_2d

    def set_test_image(self):
        raise NotImplementedError


class AnalogInput():
    '''
    Creates analog input in CxSystem compatible video mat file format.

    frameduration assumes milliseconds
    '''
    _properties_list = [
        "path",
        "output_folder",
        "input_folder",
        "my_stimulus_options",
        "my_stimulus_metadata",
    ]

    def __init__(self, context, data_io, viz, **kwargs):
        super().__init__()

        self._context = context.set_context(self._properties_list)
        self._data_io = data_io
        self._viz = viz
        
        for attr, value in kwargs.items():
            setattr(self, attr, value)


    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def viz(self):
        return self._viz

    def make_stimulus_video(self, analog_options=None): 

        assert analog_options is not None, "analog_options not set, aborting... "

        N_units = analog_options["N_units"]
        N_tp = analog_options["N_tp"]
        filename_out = analog_options["filename_out"]
        input_type = analog_options["input_type"]
        coord_type = analog_options["coord_type"]
        N_cycles = analog_options["N_cycles"]
        frameduration = analog_options["dt"]

        # get Input
        if input_type == 'noise':
            Input = self.create_noise_input(Nx = N_units, N_tp = N_tp)
        elif input_type == 'quadratic_oscillation':
            if N_units != 2:
                print(f'NOTE: You requested {input_type} input type, setting excessive units to 0 value')
            Input = self.create_quadratic_oscillation_input(Nx = N_units, N_tp = N_tp, N_cycles = N_cycles)
        elif input_type == 'step_current':
            Input = self.create_step_input(Nx = N_units, N_tp = N_tp)
        # if current_injection is True:
        #     Input = self.create_current_injection(Input)
        #     filename_out = filename_out[:-4] + '_ci.mat'

        # get coordinates
        if coord_type == 'dummy':
            w_coord, z_coord = self.get_dummy_coordinates(Nx = N_units)
        elif coord_type == 'real':
            w_coord, z_coord = self.get_real_coordinates(Nx = N_units)

        assert 'w_coord' in locals(), 'coord_type not set correctly, check __init__, aborting'
        w_coord = np.expand_dims(w_coord, 1)
        z_coord = np.expand_dims(z_coord, 1)

        # For potential plotting from conf
        self.Input = Input
        self.frameduration = frameduration

        if analog_options["save_stimulus"] is True:
            self.data_io.save_analog_stimulus(
                filename_out = filename_out, 
                Input = Input, 
                z_coord = z_coord, 
                w_coord = w_coord,
                frameduration = frameduration,
            )

    def _gaussian_filter(self):

        sigma = 30 # was abs(30)
        w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp( -1 * np.power(np.arange(1000) - 500,2) / (2 * np.power(sigma,2)))
        w=w/np.sum(w)
        return w

    def _normalize(self, Input):
        # Scale to interval [0, 1]
        Input = Input - min(np.ravel(Input))
        Input = Input / max(np.ravel(Input))
        return Input

    def create_noise_input(self, Nx = 0, N_tp = None):   

        assert Nx != 0, 'N units not set, aborting...'
        assert N_tp is not None, 'N timepoints not set, aborting...'
        Input=(np.random.multivariate_normal(np.zeros([Nx]), np.eye(Nx), N_tp)).T

        # Get gaussian filter, apply
        w = self._gaussian_filter()
        A = 15 # Deneve project was 2000, from their Learning.py file
        for d in np.arange(Nx):
            Input[d,:] = A * np.convolve(Input[d,:],w,'same')

        return Input


    def create_quadratic_oscillation_input(self, Nx = 0, N_tp = None, N_cycles = 0): 
        '''
        Creates analog oscillatory input

        :param Nx: int, number of units
        :param N_tp: int, number of time points
        :param N_cycles: int, float or list of ints or floats, number of oscillatory cycles. Scalar value creates a quadratic pair. List enables assigning distinct frequencies to distinct channels. Every 1,3,5... unit will be sine and 2,4,6... will be cosine transformed. 
        '''  

        assert Nx != 0, 'N units not set, aborting...'
        assert N_cycles != 0, 'N cycles not set, aborting...'
        assert N_tp is not None, 'N timepoints not set, aborting...'

        tp_vector = np.arange(N_tp)
        A = 5 # Deneve project was 2000, from their Learning.py file

        if isinstance(N_cycles, int) or isinstance(N_cycles, float):
            # frequency, this gives N_cycles over all time points
            freq = N_cycles * 2 * np.pi * 1/N_tp 
            sine_wave = np.sin(freq * tp_vector)
            cosine_wave = np.cos(freq * tp_vector)
            Input = A * np.array([sine_wave, cosine_wave])
            if Nx > 2:
                unit_zero_input = np.zeros(sine_wave.shape)
                stack_to_add = np.tile(unit_zero_input, (Nx - 2, 1))
                zero_padded_input_stack = np.vstack((Input, stack_to_add))
                Input = zero_padded_input_stack
        elif isinstance(N_cycles, list):
            for index, this_Nx in enumerate(range(Nx)):
                if index > len(N_cycles) - 1:
                    freq = 0
                else:
                    freq = N_cycles[this_Nx] * 2 * np.pi * 1/N_tp

                if index % 2 == 0:
                    oscillations = np.sin(freq * tp_vector)
                else:
                    oscillations = np.cos(freq * tp_vector)
                    if freq == 0:
                        oscillations = oscillations * 0
                if 'Input' not in locals():
                    Input = A * np.array([oscillations])
                else:
                    Input = np.vstack((Input, A * np.array([oscillations])))

        return Input

    def create_step_input(self, Nx = 0, N_tp = None):   

        assert Nx != 0, 'N units not set, aborting...'
        assert N_tp is not None, 'N timepoints not set, aborting...'

        # Create your input here. Zeros and ones at this point.
        # Create matrix of zeros with shape of Input
        Input = (np.concatenate((np.zeros((N_tp//3,), dtype=int), np.ones((N_tp//3), dtype=int), np.zeros((N_tp//3), dtype=int)), axis=None))
        Input = (np.concatenate((Input, np.zeros((N_tp-np.size((Input),0),), dtype=int)), axis=None))
        Input = (np.tile(Input.T, (Nx,1)))

        A = 5 # Amplification, Units = ?
        Input = A * Input

        minI = np.min(Input)
        maxI = np.max(Input)
        print(f'minI = {minI}')
        print(f'maxI = {maxI}')
        return Input

    def get_dummy_coordinates(self, Nx = 0):
        # Create dummy coordinates for CxSystem format video input.
        # NOTE: You are safer with local mode on in CxSystem to use these

        assert Nx != 0, 'N units not set, aborting...'

        # N units btw 4 and 6 deg ecc
        z_coord = np.linspace(4.8, 5.2, Nx)
        z_coord = z_coord + 0j # Add second dimension

        # Copied from macaque retina, to keep w and z coords consistent
        a = .077 / .082 # ~ 0.94
        k = 1 / .082 # ~ 12.2
        w_coord = k * np.log(z_coord + a)

        return w_coord, z_coord 

    def get_real_coordinates(self, Nx = 0):
        # For realistic coordinates, we use Macaque retina module

        assert Nx != 0, 'N units not set, aborting...'

        # Initialize WorkingRetina
        self.wr_initialize()
        w_coord, z_coord = self.get_w_z_coords()

        # Get random sample sized N_units, assert for too small sample

        Nmosaic_units = w_coord.size
        assert Nx <= Nmosaic_units, 'Too few units in mosaic, increase ecc and / or sector limits in get_real_coordinates method'
        idx = np.random.choice(Nmosaic_units, size=Nx, replace=False)
        w_coord, z_coord = w_coord[idx], z_coord[idx]

        return w_coord, z_coord 

