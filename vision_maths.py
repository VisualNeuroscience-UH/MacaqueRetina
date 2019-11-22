import numpy as np


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


    # TODO - Replace with call to previous fn
    def DoG2D_fixed_surround(self, xy_tuple, amplitudec, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes,
                             sur_ratio, offset):
        '''
        DoG model with xo, yo, theta for surround coming from center.
        '''
        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orientation_center) ** 2) / (2 * semi_xc ** 2) + (np.sin(orientation_center) ** 2) / (
                    2 * semi_yc ** 2)
        bcen = -(np.sin(2 * orientation_center)) / (4 * semi_xc ** 2) + (np.sin(2 * orientation_center)) / (
                    4 * semi_yc ** 2)
        ccen = (np.sin(orientation_center) ** 2) / (2 * semi_xc ** 2) + (np.cos(orientation_center) ** 2) / (
                    2 * semi_yc ** 2)

        asur = (np.cos(orientation_center) ** 2) / (2 * sur_ratio * semi_xc ** 2) + (
                    np.sin(orientation_center) ** 2) / (2 * sur_ratio * semi_yc ** 2)
        bsur = -(np.sin(2 * orientation_center)) / (4 * sur_ratio * semi_xc ** 2) + (np.sin(2 * orientation_center)) / (
                    4 * sur_ratio * semi_yc ** 2)
        csur = (np.sin(orientation_center) ** 2) / (2 * sur_ratio * semi_xc ** 2) + (
                    np.cos(orientation_center) ** 2) / (2 * sur_ratio * semi_yc ** 2)

        ## Difference of gaussians
        model_fit = offset + \
                    amplitudec * np.exp(
            - (acen * ((x_fit - xoc) ** 2) + 2 * bcen * (x_fit - xoc) * (y_fit - yoc) + ccen * ((y_fit - yoc) ** 2))) - \
                    amplitudes * np.exp(
            - (asur * ((x_fit - xoc) ** 2) + 2 * bsur * (x_fit - xoc) * (y_fit - yoc) + csur * ((y_fit - yoc) ** 2)))

        return model_fit.ravel()

    # TODO - Replace with call to previous fn
    def DoG2D_fixed_double_surround(self, xy_tuple, xoc, yoc, semi_xc, semi_yc, orientation_center, amplitudes):
        """
        DoG model with the angle of orientation and center positions identical and diameter of the surround
        twice that of the center. This is the model used in the data paper.
        """
        sur_ratio = 2
        offset = 0
        amplitudec = 1

        (x_fit, y_fit) = xy_tuple
        acen = (np.cos(orientation_center) ** 2) / (2 * semi_xc ** 2) + (np.sin(orientation_center) ** 2) / (
                2 * semi_yc ** 2)
        bcen = -(np.sin(2 * orientation_center)) / (4 * semi_xc ** 2) + (np.sin(2 * orientation_center)) / (
                4 * semi_yc ** 2)
        ccen = (np.sin(orientation_center) ** 2) / (2 * semi_xc ** 2) + (np.cos(orientation_center) ** 2) / (
                2 * semi_yc ** 2)

        asur = (np.cos(orientation_center) ** 2) / (2 * sur_ratio * semi_xc ** 2) + (
                np.sin(orientation_center) ** 2) / (2 * sur_ratio * semi_yc ** 2)
        bsur = -(np.sin(2 * orientation_center)) / (4 * sur_ratio * semi_xc ** 2) + (np.sin(2 * orientation_center)) / (
                4 * sur_ratio * semi_yc ** 2)
        csur = (np.sin(orientation_center) ** 2) / (2 * sur_ratio * semi_xc ** 2) + (
                np.cos(orientation_center) ** 2) / (2 * sur_ratio * semi_yc ** 2)

        ## Difference of gaussians
        model_fit = offset + \
                    amplitudec * np.exp(
            - (acen * ((x_fit - xoc) ** 2) + 2 * bcen * (x_fit - xoc) * (y_fit - yoc) + ccen * ((y_fit - yoc) ** 2))) - \
                    amplitudes * np.exp(
            - (asur * ((x_fit - xoc) ** 2) + 2 * bsur * (x_fit - xoc) * (y_fit - yoc) + csur * ((y_fit - yoc) ** 2)))

        return model_fit.ravel()



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

    def cosinebump(self, j, t):
        a = 3.0  # Hand-tuned
        c = 0.01  # Hand-tuned
        phi_j = j * np.pi/2  # Spacing as in Pillow et al. 2008 Nature

        # First, scale time to logtime
        t_logtime = a*np.log(t + c) - phi_j

        # Then create the bump
        if -np.pi <= t_logtime <= np.pi:
            return (np.cos(t_logtime) + 1) / 2
        else:
            return 0

