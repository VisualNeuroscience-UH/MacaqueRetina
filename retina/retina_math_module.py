# Numerical
import numpy as np
from scipy.stats import norm
from scipy import ndimage
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


# Viz
import matplotlib.pyplot as plt

# BUiltin
import pdb


class RetinaMath:
    """
    Constructor fit functions to read in data and provide continuous functions
    """

    # Need object instance of this class at ProjectManager
    def __init__(self) -> None:
        pass

    # RetinaConstruction methods
    def double_exponential_func(self, x, a, b, c, d):
        return a * np.exp(b * x) + c * np.exp(d * x)

    def triple_exponential_func(self, x, a, b, c, d, e, f):
        return np.maximum(a * np.exp(b * x) + c * np.exp(d * x) + e * np.exp(f * x), 0)

    def gauss_plus_baseline_func(self, x, a, x0, sigma, baseline):  # To fit GC density
        """
        Function for Gaussian distribution with a baseline value. For optimization.
        """
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + baseline

    def generalized_gauss_func(self, x, a, x0, alpha, beta):
        """
        Generalized Gaussian distribution function with variable kurtosis.
        """
        coeff = beta / (2 * alpha * gamma(1 / beta))
        return a * coeff * np.exp(-np.abs((x - x0) / alpha) ** beta)

    def sector2area_mm2(self, radius, angle):
        """
        Calculate sector area.

        Parameters
        ----------
        radius : float
            The radius of the sector in mm.
        angle : float
            The angle of the sector in degrees.

        Returns
        -------
        sector_surface_area : float
            The area of the sector in mm2.
        """
        assert angle < 360, "Angle not possible, should be <360"

        # Calculating area of the sector
        sector_surface_area = (np.pi * (radius**2)) * (angle / 360)  # in mm2
        return sector_surface_area

    def area2circle_diameter(self, area_of_rf):
        diameter = np.sqrt(area_of_rf / np.pi) * 2

        return diameter

    def ellipse2diam(self, semi_xc, semi_yc):
        """
        Compute the spherical diameter of an ellipse given its semi-major and semi-minor axes.

        Parameters
        ----------
        semi_xc : array-like
            The lengths of the semi-major axes of the ellipses.
        semi_yc : array-like
            The lengths of the semi-minor axes of the ellipses.

        Returns
        -------
        diameters : numpy array
            The spherical diameters of the ellipses.

        Notes
        -----
        The spherical diameter is calculated as the diameter of a circle with the same area as the ellipse.
        """
        # Calculate the area of each ellipse
        areas = np.pi * semi_xc * semi_yc

        # Calculate the diameter of a circle with the same area
        diameters = 2 * np.sqrt(areas / np.pi)

        return diameters

    # RetinaConstruction & SimulateRetina methods
    def pol2cart_df(self, df):
        """
        Convert retinal positions (eccentricity, polar angle) to visual space positions in degrees (x, y).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing retinal positions with columns 'pos_ecc_mm' and 'pos_polar_deg'.

        Returns
        -------
        numpy.ndarray
            Numpy array of visual space positions in degrees, with shape (n, 2), where n is the number of rows in the DataFrame.
            Each row represents the Cartesian coordinates (x, y) in visual space.
        """
        rspace_pos_mm = np.array(
            [
                self.pol2cart(gc.pos_ecc_mm, gc.pos_polar_deg, deg=True)
                for index, gc in df.iterrows()
            ]
        )

        return rspace_pos_mm

    def create_ellipse_mask(self, xo, yo, semi_x, semi_y, ori, s):
        """
        Create an ellipse mask.

        This function creates an ellipse mask with the specified parameters.

        Parameters
        ----------
        xo : float
            The x-coordinate of the ellipse center.
        yo : float
            The y-coordinate of the ellipse center.
        semi_x : float
            The semi-major axis of the ellipse.
        semi_y : float
            The semi-minor axis of the ellipse.
        ori : float
            The orientation of the ellipse in radians.
        s : int
            The side length of the mask.

        Returns
        -------
        mask : ndarray
            The ellipse mask.
        """

        # Create a grid of x and y coordinates
        x = np.arange(0, s)
        y = np.arange(0, s)
        # x, y = np.meshgrid(x, y)
        Y, X = np.meshgrid(
            np.arange(0, s),
            np.arange(0, s),
            indexing="ij",
        )  # y, x
        # Rotate the coordinates
        x_rot = (X - xo) * np.cos(ori) + (Y - yo) * np.sin(ori)
        y_rot = -(X - xo) * np.sin(ori) + (Y - yo) * np.cos(ori)

        # Create the mask
        mask = ((x_rot / semi_x) ** 2 + (y_rot / semi_y) ** 2 <= 1).astype(np.uint8)

        return mask

    def get_center_masks(self, img_stack, mask_threshold):
        """
        Extracts the contours around the maximum of each receptive field in an image stack. The contour for a field is
        defined as the set of pixels with a value of at least 10% of the maximum pixel value in the field. Only the
        connected region of the contour that contains the maximum value is included.

        Parameters
        ----------
        img_stack : numpy.ndarray
            3D numpy array representing a stack of images. The shape of the array should be (N, H, W).
        mask_threshold : float between 0 and 1
            The threshold for the contour mask.

        Returns
        -------
        numpy.ndarray
            3D numpy array of boolean masks (N, H, W). In each mask, True indicates
            a pixel is part of the contour, and False indicates it is not.
        """

        assert (
            mask_threshold >= 0 and mask_threshold <= 1
        ), "mask_threshold must be between 0 and 1, aborting..."

        masks = []
        for img in img_stack:
            max_val = np.max(img)
            mask = img >= max_val * mask_threshold

            # Label the distinct regions in the mask
            labeled_mask, num_labels = ndimage.label(mask)

            # Find the label of the region that contains the maximum value
            max_label = labeled_mask[np.unravel_index(np.argmax(img), img.shape)]

            # Keep only the region in the mask that contains the maximum value
            mask = labeled_mask == max_label

            masks.append(mask)

        return np.array(masks)

    # SimulateRetina methods
    def pol2cart(self, radius, phi, deg=True):
        """
        Converts polar coordinates to Cartesian coordinates

        Parameters
        ----------
        radius : float
            The radius value in real distance such as mm.
        phi : float
            The polar angle value.
        deg : bool, optional
            Whether the polar angle is given in degrees or radians.
            If True, the angle is given in degrees; if False, the angle is given in radians.
            Default is True.

        Returns
        -------
        tuple
            A tuple containing the Cartesian coordinates (x, y) in same units as the radius.
        """
        # Check that radius and phi are floats or numpy arrays
        assert type(radius) in [float, np.float64, np.ndarray], "Radius must be a float"
        assert type(phi) in [float, np.float64, np.ndarray], "Phi must be a float"

        if deg is True:
            theta = phi * np.pi / 180
        else:
            theta = phi

        x = radius * np.cos(theta)  # radians fed here
        y = radius * np.sin(theta)

        return (x, y)

    def cart2pol(self, x, y, deg=True):
        """
        Converts Cartesian coordinates to polar coordinates.

        Parameters
        ----------
        x : float
            The x-coordinate in real distance such as mm.
        y : float
            The y-coordinate in real distance such as mm.
        deg : bool, optional
            Whether to return the polar angle in degrees or radians.
            If True, the angle is returned in degrees; if False, the angle is returned in radians.
            Default is True.

        Returns
        -------
        tuple
            A tuple containing the polar coordinates (radius, phi).
        """
        # Check that x and y are floats or numpy arrays
        assert type(x) in [float, np.float64, np.ndarray], "x must be a float"
        assert type(y) in [float, np.float64, np.ndarray], "y must be a float"

        radius = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        if deg:
            phi = theta * 180 / np.pi
        else:
            phi = theta

        return (radius, phi)

    # General function fitting methods
    def hyperbolic_function(self, x, y_max, x_half):
        # Define the generalized hyperbolic function
        return y_max / (1 + x / x_half)

    def log_hyperbolic_function(self, x_log, log_y_max, x_half_log):
        # Define the hyperbolic function in log space
        return log_y_max - np.log(1 + np.exp(x_log - x_half_log))

    def victor_model_frequency_domain(self, f, NL, TL, HS, TS, A0, M0, D):
        """
        The model by Victor 1987 JPhysiol
        """
        # Linearized low-pass filter in frequency domain
        x_hat = (1 + 1j * f * TL) ** (-NL)

        # Adaptive high-pass filter (linearized representation)
        y_hat = (1 - HS / (1 + 1j * f * TS)) * x_hat

        # Impulse generation stage
        r_hat = A0 * np.exp(-1j * f * D) * y_hat + M0
        # Power spectrum is the square of the magnitude of the impulse generation stage output
        power_spectrum = np.abs(r_hat) ** 2

        return power_spectrum

    def lorenzian_function(self, f, a, wc):
        """
        Define a Lorentzian function for curve fitting.

        Parameters
        ----------
        f : numpy.ndarray
            Frequency axis.
        a : float
            Amplitude of the Lorentzian function.
        wc : float
            Cutoff frequency of the Lorentzian function.

        Returns
        -------
        numpy.ndarray
            The Lorentzian function evaluated at each frequency in `f`.
        """
        return a / (1 + ((f / wc) ** 2))

    def interpolation_function(self, x, y, kind="linear"):
        """Interpolate empirical data to get a continuous function for the cone response."""

        # Finite fill values for extrapolation are necessary for the model
        # when the frequency is outside the range of the empirical data.
        # In linear space the option "extrapolate" will push high-fr response negative
        # causing nan values in the log power spectrum.
        fill_value = (y[0], 0)

        interp1d_function = interp1d(
            x,
            y,
            kind=kind,
            fill_value=fill_value,
            bounds_error=False,
        )

        return interp1d_function

    def lin_interp_and_double_lorenzian(
        self, f, a0, L1_params, L2_params, cone_interp_response
    ):
        """
        Calculate the power spectrum in linear space as a combination of
        interpolated cone response and two lorenzian functions.
        """
        L1 = self.lorenzian_function(f, *L1_params)
        L2 = self.lorenzian_function(f, *L2_params)
        fitted_interpolated_data = a0 * cone_interp_response(f)

        return L1 + L2 + fitted_interpolated_data

    def fit_log_interp_and_double_lorenzian(self, log_f, *log_params):
        """
        Wrapper function for fitting interpolated cone response and
        two lorenzian functions.

        Incoming and returned data are in log space to equalize fitting
        errors across the power scale.

        Parameters
        ----------
        log_f : numpy.ndarray
            Logarithm of frequency axis.
        log_params : list
            List of parameters to be fitted in log space.

        Returns
        -------
        log of power_spectrum : numpy.ndarray
        """

        # Parameters are scalars and frequency (x_axis) is an array.
        f = np.exp(log_f)

        # Combine the fitted amplitudes with fixed corner frequencies
        L1_params = np.array([np.exp(log_params[1]), self.cone_noise_wc[0]])
        L2_params = np.array([np.exp(log_params[2]), self.cone_noise_wc[1]])
        a0 = np.exp(log_params[0])

        # Calculate the power spectrum in linear space
        power_spectrum = self.lin_interp_and_double_lorenzian(
            f, a0, L1_params, L2_params, self.cone_interp_response
        )

        log_power_spectrum = np.log(power_spectrum)

        return log_power_spectrum

    # Fit method
    def lowpass(self, t, n, p, tau):
        """
        Returns a lowpass filter kernel with a given time constant and order.

        Parameters
        ----------
        - t (numpy.ndarray): Time points at which to evaluate the kernel.
        - n (float): Order of the filter.
        - p (float): Normalization factor for the kernel.
        - tau (float): Time constant of the filter.

        Returns
        -------
        - y (numpy.ndarray): Lowpass filter kernel evaluated at each time point in `t`.
        """

        y = p * (t / tau) ** (n) * np.exp(-n * (t / tau - 1))
        return y

    def get_triangular_parameters(self, minimum, maximum, median, mean, sd, sem):
        """
        Estimate the parameters for a triangular distribution based on the provided
        statistics: minimum, maximum, median, mean, and standard deviation.

        Parameters
        ----------
        minimum : float
            The smallest value of the data.
        maximum : float
            The largest value of the data.
        median : float
            The median of the data.
        mean : float
            The mean of the data.
        sd : float
            The standard deviation of the data.

        Returns
        -------
        c : float
            The shape parameter of the triangular distribution, representing the mode.
        loc : float
            The location parameter, equivalent to the minimum.
        scale : float
            The scale parameter, equivalent to the difference between the maximum and minimum.

        Raises
        ------
        ValueError:
            If the provided mean and standard deviation don't closely match the expected
            values for the triangular distribution.

        Notes
        -----
        The returned parameters can be used with scipy's triang function to represent
        a triangular distribution and perform further sampling or analysis.
        """
        # The location is simply the minimum.
        loc = minimum

        # The scale is the difference between maximum and minimum.
        scale = maximum - minimum

        # Estimating c (shape parameter) based on the position of median within the range.
        c = (median - minimum) / scale

        # Validate the given mean and SD against expected values for triangular distribution
        expected_mean = (minimum + maximum + median) / 3
        expected_sd = np.sqrt(
            (
                minimum**2
                + maximum**2
                + median**2
                - minimum * maximum
                - minimum * median
                - maximum * median
            )
            / 18
        )

        tolerance = 3 * sem
        if not (
            np.abs(expected_mean - mean) < tolerance
            and np.abs(expected_sd - sd) < tolerance
        ):
            raise ValueError(
                f"The provided mean ({mean}) and SD ({sd}) don't match the expected values for a triangular distribution with the given min, max, and median. Expected mean: {expected_mean}, Expected SD: {expected_sd}"
            )

        return c, loc, scale

    # Fit & RetinaVAE method

    def flip_negative_spatial_rf(self, spatial_rf_unflipped):
        """
        Flips negative values of a spatial RF to positive values.

        Parameters
        ----------
        spatial_rf_unflipped: numpy.ndarray of shape (N, H, W)
            Spatial receptive field.

        Returns
        -------
        spatial_rf: numpy.ndarray of shape (N, H, W)
            Spatial receptive field with negative values flipped to positive values.
        """

        # Number of pixels to define maximum value of RF
        max_pixels = 5

        # Copy spatial_rf_unflipped to spatial_rf
        spatial_rf = np.copy(spatial_rf_unflipped)

        # Find max_pixels number of pixels with absolute maximum value
        # and their indices
        for i in range(spatial_rf.shape[0]):
            # max_pixels_values = np.sort(np.abs(spatial_rf[i].ravel()))[-max_pixels:]
            max_pixels_indices = np.argsort(np.abs(spatial_rf[i].ravel()))[-max_pixels:]

            # Calculate mean value of the original max_pixels_values
            mean_max_pixels_values = np.mean(spatial_rf[i].ravel()[max_pixels_indices])

            # If mean value of the original max_pixels_values is negative,
            # flip the RF
            if mean_max_pixels_values < 0:
                spatial_rf[i] = spatial_rf[i] * -1

        return spatial_rf

    # Fit & SimulateRetina method
    def DoG2D_fixed_surround(
        self,
        xy_tuple,
        ampl_c,
        xoc,
        yoc,
        semi_xc,
        semi_yc,
        orient_cen_rad,
        ampl_s,
        relat_sur_diam,
        offset,
    ):
        """
        DoG model with xo, yo, theta for surround coming from center.
        Note that semi_xc and semi_yc correspond to radii while matplotlib Ellipse assumes diameters.
        """

        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orient_cen_rad) ** 2) / (2 * semi_xc**2) + (
            np.sin(orient_cen_rad) ** 2
        ) / (2 * semi_yc**2)
        bcen = -(np.sin(2 * orient_cen_rad)) / (4 * semi_xc**2) + (
            np.sin(2 * orient_cen_rad)
        ) / (4 * semi_yc**2)
        ccen = (np.sin(orient_cen_rad) ** 2) / (2 * semi_xc**2) + (
            np.cos(orient_cen_rad) ** 2
        ) / (2 * semi_yc**2)

        asur = (np.cos(orient_cen_rad) ** 2) / (2 * (relat_sur_diam * semi_xc) ** 2) + (
            np.sin(orient_cen_rad) ** 2
        ) / (2 * (relat_sur_diam * semi_yc) ** 2)
        bsur = -(np.sin(2 * orient_cen_rad)) / (4 * (relat_sur_diam * semi_xc) ** 2) + (
            np.sin(2 * orient_cen_rad)
        ) / (4 * (relat_sur_diam * semi_yc) ** 2)
        csur = (np.sin(orient_cen_rad) ** 2) / (2 * (relat_sur_diam * semi_xc) ** 2) + (
            np.cos(orient_cen_rad) ** 2
        ) / (2 * (relat_sur_diam * semi_yc) ** 2)

        ## Difference of gaussians
        model_fit = (
            offset
            + ampl_c
            * np.exp(
                -(
                    acen * ((x_fit - xoc) ** 2)
                    + 2 * bcen * (x_fit - xoc) * (y_fit - yoc)
                    + ccen * ((y_fit - yoc) ** 2)
                )
            )
            - ampl_s
            * np.exp(
                -(
                    asur * ((x_fit - xoc) ** 2)
                    + 2 * bsur * (x_fit - xoc) * (y_fit - yoc)
                    + csur * ((y_fit - yoc) ** 2)
                )
            )
        )

        return model_fit.ravel()

    def DoG2D_independent_surround(
        self,
        xy_tuple,
        ampl_c,
        xoc,
        yoc,
        semi_xc,
        semi_yc,
        orient_cen_rad,
        ampl_s,
        xos,
        yos,
        semi_xs,
        semi_ys,
        orient_sur_rad,
        offset,
    ):
        """
        DoG model with xo, yo, theta for surround independent from center.
        """

        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orient_cen_rad) ** 2) / (2 * semi_xc**2) + (
            np.sin(orient_cen_rad) ** 2
        ) / (2 * semi_yc**2)
        bcen = -(np.sin(2 * orient_cen_rad)) / (4 * semi_xc**2) + (
            np.sin(2 * orient_cen_rad)
        ) / (4 * semi_yc**2)
        ccen = (np.sin(orient_cen_rad) ** 2) / (2 * semi_xc**2) + (
            np.cos(orient_cen_rad) ** 2
        ) / (2 * semi_yc**2)

        asur = (np.cos(orient_sur_rad) ** 2) / (2 * semi_xs**2) + (
            np.sin(orient_sur_rad) ** 2
        ) / (2 * semi_ys**2)
        bsur = -(np.sin(2 * orient_sur_rad)) / (4 * semi_xs**2) + (
            np.sin(2 * orient_sur_rad)
        ) / (4 * semi_ys**2)
        csur = (np.sin(orient_sur_rad) ** 2) / (2 * semi_xs**2) + (
            np.cos(orient_sur_rad) ** 2
        ) / (2 * semi_ys**2)

        ## Difference of gaussians
        model_fit = (
            offset
            + ampl_c
            * np.exp(
                -(
                    acen * ((x_fit - xoc) ** 2)
                    + 2 * bcen * (x_fit - xoc) * (y_fit - yoc)
                    + ccen * ((y_fit - yoc) ** 2)
                )
            )
            - ampl_s
            * np.exp(
                -(
                    asur * ((x_fit - xos) ** 2)
                    + 2 * bsur * (x_fit - xos) * (y_fit - yos)
                    + csur * ((y_fit - yos) ** 2)
                )
            )
        )

        return model_fit.ravel()

    def DoG2D_circular(self, xy_tuple, ampl_c, x0, y0, rad_c, ampl_s, rad_s, offset):
        """
        DoG model with the center and surround as concentric circles and a shared center (x0, y0).
        """

        (x_fit, y_fit) = xy_tuple

        # Distance squared from the center for the given (x_fit, y_fit) points
        distance_sq = (x_fit - x0) ** 2 + (y_fit - y0) ** 2

        # Gaussian for the center
        center_gaussian = ampl_c * np.exp(-distance_sq / (2 * rad_c**2))

        # Gaussian for the surround
        surround_gaussian = ampl_s * np.exp(-distance_sq / (2 * rad_s**2))

        # Difference of gaussians
        model_fit = offset + center_gaussian - surround_gaussian

        return model_fit.ravel()

    def diff_of_lowpass_filters(self, t, n, p1, p2, tau1, tau2):
        """
        Returns the difference between two lowpass filters with different time constants and orders.
        From Chichilnisky & Kalmar JNeurosci 2002

        Parameters
        ----------
        - t (numpy.ndarray): Time points at which to evaluate the filters.
        - n (float): Order of the filters.
        - p1 (float): Normalization factor for the first filter.
        - p2 (float): Normalization factor for the second filter.
        - tau1 (float): Time constant of the first filter.
        - tau2 (float): Time constant of the second filter.

        Returns
        -------
        - y (numpy.ndarray): Difference between the two lowpass filters evaluated at each time point in `t`.
        """

        #
        y = self.lowpass(t, n, p1, tau1) - self.lowpass(t, n, p2, tau2)
        return y

    def photon_flux_density_to_luminance(self, F, lambda_nm=555):
        """
        Convert photon flux density to luminance using human photopic vision V(lambda).

        Parameters
        ----------
        F : float
            Photon flux density at the cornea in photons/mm²/s.
        lambda_nm : float, optional
            Wavelength of the monochromatic light in nanometers, default is 555 nm (peak of human photopic sensitivity).

        Returns
        -------
        L : float
            Luminance in cd/m².

        Notes
        -----
        The conversion uses the formula:

        L = F * (hc/lambda) * kappa * V(lambda)

        where:
        - h is Planck's constant (6.626 x 10^-34 J·s),
        - c is the speed of light (3.00 x 10^8 m/s),
        - lambda is the wavelength of light in meters,
        - kappa is the luminous efficacy of monochromatic radiation (683 lm/W at 555 nm),
        - V(lambda) is the photopic luminous efficiency function value at lambda,
        assumed to be 1 at 555 nm for peak human photopic sensitivity.
        """
        # Constants
        h = 6.626e-34  # Planck's constant in J·s
        c = 3.00e8  # Speed of light in m/s
        lambda_m = lambda_nm * 1e-9  # Convert wavelength from nm to m
        kappa = 683  # Luminous efficacy of monochromatic radiation in lm/W at 555 nm

        # Energy of a photon at wavelength lambda in joules
        E_photon = (h * c) / lambda_m

        # Convert photon flux density F to luminance L in cd/m²
        F_m2 = F * 1e6  # Convert photon flux density from mm² to m²
        L = F_m2 * E_photon * kappa

        return L

    def calculate_F_cornea(self, I_cone, a_c_end_on, A_pupil, A_retina, tau_media=1.0):
        """
        Calculate the photon flux density at the cornea (F_cornea) for a given rate of photoisomerization in cones.

        Parameters
        ----------
        I_cone : float
            The rate of photoisomerizations per cone per second (R* cone^-1 s^-1).
        a_c_end_on : float
            The end-on collecting area for the cones (in mm^2).
        A_pupil : float
            The area of the pupil (in mm^2).
        A_retina : float
            The area of the retina (in mm^2).
        tau_media : float, optional
            The transmittance of the ocular media at wavelength λ, default is 1.0 (unitless).

        Returns
        -------
        F_cornea : float
            The photon flux density at the cornea (in photons/mm²/s).

        Notes
        -----
        The function assumes that the transmittance of the ocular media (tau_media) is 1.0, indicating no loss of light due to absorption or scattering within the eye's media.
        """

        # Calculate the photon flux density at the cornea (F_cornea)
        F_cornea = I_cone / (a_c_end_on * (A_pupil / A_retina) * tau_media)

        return F_cornea
