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

    # MosaicConstruction methods
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

    # WorkingRetina methods
    def pol2cart(self, radius, phi, deg=True):
        """
        Converts polar coordinates to Cartesian coordinates

        :param radius: float
        :param phi: float, polar angle
        :param deg: True/False, whether polar angle given in degrees or radians (default True)
        :return: (x,y) tuple
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
        amplitudec,
        xoc,
        yoc,
        semi_xc,
        semi_yc,
        orientation_center,
        amplitudes,
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
        acen = (np.cos(orientation_center) ** 2) / (2 * semi_xc**2) + (
            np.sin(orientation_center) ** 2
        ) / (2 * semi_yc**2)
        bcen = -(np.sin(2 * orientation_center)) / (4 * semi_xc**2) + (
            np.sin(2 * orientation_center)
        ) / (4 * semi_yc**2)
        ccen = (np.sin(orientation_center) ** 2) / (2 * semi_xc**2) + (
            np.cos(orientation_center) ** 2
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
            + amplitudec
            * np.exp(
                -(
                    acen * ((x_fit - xoc) ** 2)
                    + 2 * bcen * (x_fit - xoc) * (y_fit - yoc)
                    + ccen * ((y_fit - yoc) ** 2)
                )
            )
            - amplitudes
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

    # Fit & WorkingRetina method
    def DoG2D_fixed_surround(
        self,
        xy_tuple,
        amplitudec,
        xoc,
        yoc,
        semi_xc,
        semi_yc,
        orientation_center,
        amplitudes,
        sur_ratio,
        offset,
    ):
        """
        DoG model with xo, yo, theta for surround coming from center.
        Note that semi_xc and semi_yc correspond to radii while matplotlib Ellipse assumes diameters.
        """

        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orientation_center) ** 2) / (2 * semi_xc**2) + (
            np.sin(orientation_center) ** 2
        ) / (2 * semi_yc**2)
        bcen = -(np.sin(2 * orientation_center)) / (4 * semi_xc**2) + (
            np.sin(2 * orientation_center)
        ) / (4 * semi_yc**2)
        ccen = (np.sin(orientation_center) ** 2) / (2 * semi_xc**2) + (
            np.cos(orientation_center) ** 2
        ) / (2 * semi_yc**2)

        asur = (np.cos(orientation_center) ** 2) / (2 * (sur_ratio * semi_xc) ** 2) + (
            np.sin(orientation_center) ** 2
        ) / (2 * (sur_ratio * semi_yc) ** 2)
        bsur = -(np.sin(2 * orientation_center)) / (4 * (sur_ratio * semi_xc) ** 2) + (
            np.sin(2 * orientation_center)
        ) / (4 * (sur_ratio * semi_yc) ** 2)
        csur = (np.sin(orientation_center) ** 2) / (2 * (sur_ratio * semi_xc) ** 2) + (
            np.cos(orientation_center) ** 2
        ) / (2 * (sur_ratio * semi_yc) ** 2)

        ## Difference of gaussians
        model_fit = (
            offset
            + amplitudec
            * np.exp(
                -(
                    acen * ((x_fit - xoc) ** 2)
                    + 2 * bcen * (x_fit - xoc) * (y_fit - yoc)
                    + ccen * ((y_fit - yoc) ** 2)
                )
            )
            - amplitudes
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
