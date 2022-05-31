# This computer git repos
from project.project_base_module import ProjectBase
from project.project_utilities_module import ProjectUtilities
from context.context_module import Context
from data_io.data_io_module import DataIO
from analysis.analysis_module import Analysis
from viz.viz_module import Viz
from  construct.macaque_retina_module import ConstructRetina, WorkingRetina
from stimuli.visual_stimuli_module import ConstructStimulus, PhotoReceptor

# Builtin
import pdb
import time
import shlex
import subprocess
from types import ModuleType
from copy import deepcopy


# Analysis
import pandas as pd


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

        # ProjectManager is facade to Context.
        context = Context(all_properties)

        # Get correct context attributes. Empty properties return all existing project attributes to context. That is what we want for the project manager
        self.context = context.set_context()

        data_io = DataIO(context)
        self.data_io = data_io

        # # Monkey-patching macaque retina construct module
        # self.construct = construct

        stimulate = ConstructStimulus(context, data_io) 
        self.stimulate = stimulate

        self.cones = PhotoReceptor(context, data_io)

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
        
        viz = Viz(
            # Interfaces
            context,
            data_io,
            ana,
            # Dictionaries
            # Methods, which are needed also elsewhere
            round_to_n_significant=self.round_to_n_significant,
        )

        self.viz = viz

        # Constructor for macaque retina mosaic. For client object injection, 
        # we need deep copy of viz to avoid masking of the instances from distinct classes
        self.mosaic_constructor = ConstructRetina(context, data_io, deepcopy(viz))
        self.functional_mosaic = WorkingRetina(context, data_io, deepcopy(viz))



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
    def mosaic_constructor(self):
        return self._mosaic_constructor

    @mosaic_constructor.setter
    def mosaic_constructor(self, value):
        if isinstance(value, ConstructRetina):
            self._mosaic_constructor = value
        else:
            raise AttributeError(
                "Trying to set improper mosaic_constructor. mosaic_constructor must be a ConstructRetina instance."
            )

    @property
    def functional_mosaic(self):
        return self._functional_mosaic

    @functional_mosaic.setter
    def functional_mosaic(self, value):
        if isinstance(value, WorkingRetina):
            self._functional_mosaic = value
        else:
            raise AttributeError(
                "Trying to set improper functional_mosaic. functional_mosaic must be a WorkingRetina instance."
            )

# if __name__=='__main__':
