# Numerical
import numpy as np
from scipy.stats import norm

# Viz
import matplotlib.pyplot as plt


class Mathematics:
    '''
    Constructor fit functions to read in data and provide continuous functions
    '''

    def gauss_plus_baseline(self, x, a, x0, sigma, baseline):  # To fit GC density
        '''
        Function for Gaussian distribution with a baseline value. For optimization.
        '''
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + baseline

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

    def DoG2D_independent_surround(self, xy_tuple, amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center,
                                   amplitudes, xos, yos, semi_xs, semi_ys, orientation_surround, offset):
        '''
        DoG model with xo, yo, theta for surround independent from center.
        '''

        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orientation_center) ** 2) / (2 * semi_xc ** 2) + (np.sin(orientation_center) ** 2) / (
                    2 * semi_yc ** 2)
        bcen = -(np.sin(2 * orientation_center)) / (4 * semi_xc ** 2) + (np.sin(2 * orientation_center)) / (
                    4 * semi_yc ** 2)
        ccen = (np.sin(orientation_center) ** 2) / (2 * semi_xc ** 2) + (np.cos(orientation_center) ** 2) / (
                    2 * semi_yc ** 2)

        asur = (np.cos(orientation_surround) ** 2) / (2 * semi_xs ** 2) + (np.sin(orientation_surround) ** 2) / (
                    2 * semi_ys ** 2)
        bsur = -(np.sin(2 * orientation_surround)) / (4 * semi_xs ** 2) + (np.sin(2 * orientation_surround)) / (
                    4 * semi_ys ** 2)
        csur = (np.sin(orientation_surround) ** 2) / (2 * semi_xs ** 2) + (np.cos(orientation_surround) ** 2) / (
                    2 * semi_ys ** 2)

        ## Difference of gaussians
        model_fit = offset + \
                    amplitudec * np.exp(
            - (acen * ((x_fit - xoc) ** 2) + 2 * bcen * (x_fit - xoc) * (y_fit - yoc) + ccen * ((y_fit - yoc) ** 2))) - \
                    amplitudes * np.exp(
            - (asur * ((x_fit - xos) ** 2) + 2 * bsur * (x_fit - xos) * (y_fit - yos) + csur * ((y_fit - yos) ** 2)))

        return model_fit.ravel()

    def DoG2D_fixed_surround(self, xy_tuple, amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes,
                             sur_ratio, offset):
        '''
        DoG model with xo, yo, theta for surround coming from center.
        Note that semi_xc and semi_yc correspond to radii while matplotlib Ellipse assumes diameters.
        '''

        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orientation_center) ** 2) / (2 * semi_xc ** 2) + (np.sin(orientation_center) ** 2) / (
                    2 * semi_yc ** 2)
        bcen = -(np.sin(2 * orientation_center)) / (4 * semi_xc ** 2) + (np.sin(2 * orientation_center)) / (
                    4 * semi_yc ** 2)
        ccen = (np.sin(orientation_center) ** 2) / (2 * semi_xc ** 2) + (np.cos(orientation_center) ** 2) / (
                    2 * semi_yc ** 2)

        asur = (np.cos(orientation_center) ** 2) / (2 * (sur_ratio * semi_xc) ** 2) + (
                    np.sin(orientation_center) ** 2) / (2 * (sur_ratio * semi_yc) ** 2)
        bsur = -(np.sin(2 * orientation_center)) / (4 * (sur_ratio * semi_xc) ** 2) + (np.sin(2 * orientation_center)) / (
                    4 * (sur_ratio * semi_yc) ** 2)
        csur = (np.sin(orientation_center) ** 2) / (2 * (sur_ratio * semi_xc) ** 2) + (
                    np.cos(orientation_center) ** 2) / (2 * (sur_ratio * semi_yc) ** 2)

        ## Difference of gaussians
        model_fit = offset + \
                    amplitudec * np.exp(
            - (acen * ((x_fit - xoc) ** 2) + 2 * bcen * (x_fit - xoc) * (y_fit - yoc) + ccen * ((y_fit - yoc) ** 2))) - \
                    amplitudes * np.exp(
            - (asur * ((x_fit - xoc) ** 2) + 2 * bsur * (x_fit - xoc) * (y_fit - yoc) + csur * ((y_fit - yoc) ** 2)))

        return model_fit.ravel()

    def DoG2D_fixed_double_surround(self, xy_tuple, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes):
        """
        DoG model with the angle of orientation and center positions identical and diameter of the surround
        twice that of the center.
        """

        raise NotImplementedError

    def sector2area(self, radius, angle):  # Calculate sector area. Angle in deg, radius in mm
        pi = np.pi
        assert angle < 360, "Angle not possible, should be <360"

        # Calculating area of the sector
        sector_surface_area = (pi * (radius ** 2)) * (angle / 360)  # in mm2
        return sector_surface_area

    def circle_diameter2area(self, diameter):
        area_of_rf = np.pi * (diameter / 2) ** 2

        return area_of_rf

    def area2circle_diameter(self, area_of_rf):
        diameter = np.sqrt(area_of_rf / np.pi) * 2

        return diameter

    def ellipse2area(self, sigma_x, sigma_y):
        area_of_ellipse = np.pi * sigma_x * sigma_y

        return area_of_ellipse

    def lowpass(self, t, n, p, tau):
        y = p * (t / tau) ** (n) * np.exp(-n * (t / tau - 1))
        return y

    def diff_of_lowpass_filters(self, t, n, p1, p2, tau1, tau2):
        # From Chichilnisky & Kalmar JNeurosci 2002
        y = self.lowpass(t, n, p1, tau1) - self.lowpass(t, n, p2, tau2)
        return y

    def generator2firing(self, generator=0, viz_module=True):
        '''
        Generator potential to firing rate by cumulative normal distribution
        From Chichilnisky_2002_JNeurosci: 
        The response nonlinearity N was well approximated using the lower 
        portion of a sigmoidal function: n(x) = aG(bx + c), where
        x is the generator signal, n(x) is the firing rate, G(x) is the 
        cumulative normal (indefinite integral of standard normal distribution), 
        and a, b, and c are free parameters.
        '''
        max_firing_rate=160 # max firing rate, plateau, demo 1
        slope=1 # slope, demo 1
        half_height=1 # at what generator signal is half-height, demo 0
        firing_freq = max_firing_rate * norm.cdf(generator, loc=half_height, scale=slope)
        if viz_module==True:
            generator=np.linspace(-3,3,num=200)
            firing_freq = max_firing_rate * norm.cdf(generator, loc=half_height, scale=slope)
            plt.plot(generator,firing_freq);plt.show()
        # return firing_freq

