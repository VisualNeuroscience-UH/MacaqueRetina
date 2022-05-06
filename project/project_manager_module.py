# This computer git repos
from project.project_base_module import ProjectBase
from project.project_utilities_module import ProjectUtilities
from context.context_module import Context
from data_io.data_io_module import DataIO
from analysis.analysis_module import Analysis
from viz.viz_module import Viz
import construct.macaque_retina_module as construct
from stimuli.visual_stimuli_module import ConstructStimulus

# Builtin
import pdb
import time
import shlex
import subprocess
from types import ModuleType


# Analysis
import pandas as pd


"""
Module on project-specific data analysis.
This configures analysis. 
Specific methods are called at the bottom after the if __name__=='__main__':

Simo Vanni 2021
"""


class ProjectManager(ProjectBase, ProjectUtilities):



    def __init__(self, **all_properties):
        """
        Main project manager.
        In init we construct other classes and inject necessary dependencies. This class is allowed to house project-dependent data and methods.
        """

        # ProjectManager is facade to Context.
        context = Context(all_properties)

        # Get corrent context attributes. Empty properties return all existing project attributes to context. That is what we want for the project manager
        self.context = context.set_context()

        data_io = DataIO(context)
        self.data_io = data_io

        # Monkey-patching macaque retina construct module
        self.construct = construct

        stimulate = ConstructStimulus(context, data_io) 
        self.stimulate = stimulate

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
        
        self.viz = Viz(
            # Interfaces
            context,
            data_io,
            ana,
            # Dictionaries
            # Methods, which are needed also elsewhere
            round_to_n_significant=self.round_to_n_significant,
        )

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
    def construct(self):
        return self._construct

    @construct.setter
    def construct(self, value):
        if isinstance(value, ModuleType):
            self._construct = value
        else:
            raise AttributeError(
                "Trying to set improper construct. Construct must be a macaque_retina_module."
            )

# if __name__=='__main__':
