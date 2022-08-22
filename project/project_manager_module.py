# This computer git repos
from project.project_base_module import ProjectBase
from project.project_utilities_module import ProjectUtilities
from context.context_module import Context
from data_io.data_io_module import DataIO
from analysis.analysis_module import Analysis
from viz.viz_module import Viz
from  retina.macaque_retina_module import ConstructRetina, WorkingRetina, PhotoReceptor
from retina.retina_math_module import RetinaMath
from stimuli.visual_stimulus_module import ConstructStimulus, AnalogInput

# Builtin
import pdb
# import time
# import shlex
# import subprocess
# from types import ModuleType
# from copy import deepcopy


# Analysis
# import pandas as pd


"""
Module on retina management

We use dependency injection to make the code more modular and easier to test.
It means that during construction here at the manager level, we can inject
an object instance to constructor of a "client". Thus the constructed "client" is holding the injected
object instance.

Simo Vanni 2022
"""


class ProjectManager(ProjectBase, ProjectUtilities):



    def __init__(self, **all_properties):
        """
        Main project manager.
        In init we construct other classes and inject necessary dependencies. This class is allowed to house project-dependent data and methods.
        """

        context = Context(all_properties)

        # Get correct context attributes. Empty properties return all existing project attributes to context. That is what we want for the project manager
        self.context = context.set_context()

        data_io = DataIO(context)
        self.data_io = data_io
        
        cones = PhotoReceptor(context, data_io)
        self.cones = cones

        stimulate = ConstructStimulus(context, data_io, cones) 
        self.stimulate = stimulate

        # natural_image = NaturalImage(context, data_io, cones) 
        # self.natural_image = natural_image

        ana = Analysis(
            # Interfaces
            context,
            data_io,
            # Dictionaries
            # Methods, which are needed also elsewhere
            round_to_n_significant=self.round_to_n_significant,
            # pp_df_full=self.pp_df_full,
        )

        self.ana = ana
        
        retina_math = RetinaMath()
        
        viz = Viz(
            # Interfaces
            context,
            data_io,
            ana,
            # Dictionaries
            # Methods, which are needed also elsewhere
            round_to_n_significant = self.round_to_n_significant,
            DoG2D_fixed_surround = retina_math.DoG2D_fixed_surround,
            DoG2D_independent_surround = retina_math.DoG2D_independent_surround,
            pol2cart = retina_math.pol2cart,
            gauss_plus_baseline = retina_math.gauss_plus_baseline,
            sector2area = retina_math.sector2area,
        )

        self.viz = viz

        self.construct_retina = ConstructRetina(context, data_io, viz)
        self.working_retina = WorkingRetina(context, data_io, viz)

        analog_input = AnalogInput(context, data_io, viz, wr_initialize = self.working_retina.initialize, get_w_z_coords = self.working_retina.get_w_z_coords)
        self.analog_input = analog_input
        


    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        if isinstance(value, Context):
            self._context = value
        else:
            raise AttributeError(
                "Trying to set improper context. Context must be a context object."
            )

    @property
    def data_io(self):
        return self._data_io

    @data_io.setter
    def data_io(self, value):
        if isinstance(value, DataIO):
            self._data_io = value
        else:
            raise AttributeError(
                "Trying to set improper data_io. Data_io must be a DataIO object."
            )

    @property
    def construct_retina(self):
        return self._construct_retina

    @construct_retina.setter
    def construct_retina(self, value):
        if isinstance(value, ConstructRetina):
            self._construct_retina = value
        else:
            raise AttributeError(
                "Trying to set improper construct_retina. construct_retina must be a ConstructRetina instance."
            )

    @property
    def working_retina(self):
        return self._working_retina

    @working_retina.setter
    def working_retina(self, value):
        if isinstance(value, WorkingRetina):
            self._working_retina = value
        else:
            raise AttributeError(
                "Trying to set improper working_retina. working_retina must be a WorkingRetina instance."
            )

    @property
    def stimulate(self):
        return self._stimulate

    @stimulate.setter
    def stimulate(self, value):
        if isinstance(value, ConstructStimulus):
            self._stimulate = value
        else:
            raise AttributeError(
                "Trying to set improper stimulate. stimulate must be a ConstructStimulus instance."
            )

    @property
    def analog_input(self):
        return self._analog_input

    @analog_input.setter
    def analog_input(self, value):
        if isinstance(value, AnalogInput):
            self._analog_input = value
        else:
            raise AttributeError(
                "Trying to set improper analog_input. analog_input must be a AnalogInput instance."
            )

# if __name__=='__main__':
