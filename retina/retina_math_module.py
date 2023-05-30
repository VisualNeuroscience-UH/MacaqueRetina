# Numerical
import numpy as np
from scipy.stats import norm

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
    def gauss_plus_baseline(self, x, a, x0, sigma, baseline):  # To fit GC density
        """
        Function for Gaussian distribution with a baseline value. For optimization.
        """
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + baseline

    def sector2area(
        self, radius, angle
    ):  # Calculate sector area. Angle in deg, radius in mm
        pi = np.pi
        assert angle < 360, "Angle not possible, should be <360"

        # Calculating area of the sector
        sector_surface_area = (pi * (radius**2)) * (angle / 360)  # in mm2
        return sector_surface_area

    def area2circle_diameter(self, area_of_rf):
        diameter = np.sqrt(area_of_rf / np.pi) * 2

        return diameter

    def ellipse2area(self, sigma_x, sigma_y):
        area_of_ellipse = np.pi * sigma_x * sigma_y

        return area_of_ellipse

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

    # RetinaConstruction & WorkingRetina methods
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

    # WorkingRetina methods
    def pol2cart(self, radius, phi, deg=True):
        """
        Converts polar coordinates to Cartesian coordinates

        Parameters
        ----------
        radius : float
            The radius value in polar coordinates.
        phi : float
            The polar angle value.
        deg : bool, optional
            Whether the polar angle is given in degrees or radians.
            If True, the angle is given in degrees; if False, the angle is given in radians.
            Default is True.

        Returns
        -------
        tuple
            A tuple containing the Cartesian coordinates (x, y).
        """

        if deg is True:
            theta = phi * np.pi / 180
        else:
            theta = phi

        x = radius * np.cos(theta)  # radians fed here
        y = radius * np.sin(theta)
        return (x, y)

    # Fit method
    def DoG2D_independent_surround(
        self,
        xy_tuple,
        ampl_c,
        xoc,
        yoc,
        semi_xc,
        semi_yc,
        orient_cen,
        ampl_s,
        xos,
        yos,
        semi_xs,
        semi_ys,
        orientation_surround,
        offset,
    ):
        """
        DoG model with xo, yo, theta for surround independent from center.
        """

        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orient_cen) ** 2) / (2 * semi_xc**2) + (
            np.sin(orient_cen) ** 2
        ) / (2 * semi_yc**2)
        bcen = -(np.sin(2 * orient_cen)) / (4 * semi_xc**2) + (
            np.sin(2 * orient_cen)
        ) / (4 * semi_yc**2)
        ccen = (np.sin(orient_cen) ** 2) / (2 * semi_xc**2) + (
            np.cos(orient_cen) ** 2
        ) / (2 * semi_yc**2)

        asur = (np.cos(orientation_surround) ** 2) / (2 * semi_xs**2) + (
            np.sin(orientation_surround) ** 2
        ) / (2 * semi_ys**2)
        bsur = -(np.sin(2 * orientation_surround)) / (4 * semi_xs**2) + (
            np.sin(2 * orientation_surround)
        ) / (4 * semi_ys**2)
        csur = (np.sin(orientation_surround) ** 2) / (2 * semi_xs**2) + (
            np.cos(orientation_surround) ** 2
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

    # Fit & WorkingRetina method
    def DoG2D_fixed_surround(
        self,
        xy_tuple,
        ampl_c,
        xoc,
        yoc,
        semi_xc,
        semi_yc,
        orient_cen,
        ampl_s,
        relat_sur_diam,
        offset,
    ):
        """
        DoG model with xo, yo, theta for surround coming from center.
        Note that semi_xc and semi_yc correspond to radii while matplotlib Ellipse assumes diameters.
        """

        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orient_cen) ** 2) / (2 * semi_xc**2) + (
            np.sin(orient_cen) ** 2
        ) / (2 * semi_yc**2)
        bcen = -(np.sin(2 * orient_cen)) / (4 * semi_xc**2) + (
            np.sin(2 * orient_cen)
        ) / (4 * semi_yc**2)
        ccen = (np.sin(orient_cen) ** 2) / (2 * semi_xc**2) + (
            np.cos(orient_cen) ** 2
        ) / (2 * semi_yc**2)

        asur = (np.cos(orient_cen) ** 2) / (2 * (relat_sur_diam * semi_xc) ** 2) + (
            np.sin(orient_cen) ** 2
        ) / (2 * (relat_sur_diam * semi_yc) ** 2)
        bsur = -(np.sin(2 * orient_cen)) / (4 * (relat_sur_diam * semi_xc) ** 2) + (
            np.sin(2 * orient_cen)
        ) / (4 * (relat_sur_diam * semi_yc) ** 2)
        csur = (np.sin(orient_cen) ** 2) / (2 * (relat_sur_diam * semi_xc) ** 2) + (
            np.cos(orient_cen) ** 2
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

    # Not in use
    def generator2firing(self, generator=0, show_generator_vs_fr=True):
        """
        Generator potential to firing rate by cumulative normal distribution
        From Chichilnisky_2002_JNeurosci:
        The response nonlinearity N was well approximated using the lower
        portion of a sigmoidal function: n(x) = aG(bx + c), where
        x is the generator signal, n(x) is the firing rate, G(x) is the
        cumulative normal (indefinite integral of standard normal distribution),
        and a, b, and c are free parameters.
        """
        max_firing_rate = 160  # max firing rate, plateau, demo 1
        slope = 1  # slope, demo 1
        half_height = 1  # at what generator signal is half-height, demo 0
        firing_freq = max_firing_rate * norm.cdf(
            generator, loc=half_height, scale=slope
        )
        if show_generator_vs_fr == True:
            generator = np.linspace(-3, 3, num=200)
            firing_freq = max_firing_rate * norm.cdf(
                generator, loc=half_height, scale=slope
            )
            plt.plot(generator, firing_freq)
            plt.show()
        # return firing_freq
