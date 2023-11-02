# Numerical
import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.stats as stats
import pandas as pd
from sklearn.manifold import TSNE

# Machine learning
import torch
import torchvision.transforms.functional as TF
from torchsummary import summary

# import cv2

# Comput Neurosci
import brian2.units as b2u

# Viz
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

# from tqdm import tqdm

# Local
from retina.vae_module import AugmentedDataset

# Builtin
import os
from pathlib import Path
import pdb
import copy
from functools import reduce
import math


class Viz:
    """
    Methods to viz_module the retina

    Some methods import object instance as call parameter (ConstructRetina, WorkingRetina, etc).
    """

    cmap = "gist_earth"  # viridis or cividis would be best for color-blind

    def __init__(self, context, data_io, project_data, ana, **kwargs) -> None:
        self._context = context.set_context(self)
        self._data_io = data_io
        self._project_data = project_data
        self._ana = ana

        for attr, value in kwargs.items():
            setattr(self, attr, value)

        # Some settings related to plotting
        self.cmap_stim = "gray"
        self.cmap_spatial_filter = "bwr"

    @property
    def context(self):
        return self._context

    @property
    def data_io(self):
        return self._data_io

    @property
    def project_data(self):
        return self._project_data

    @property
    def ana(self):
        return self._ana

    def data_is_valid(self, data, accept_empty=False):
        try:
            data = data / data.get_best_unit()
        except:
            pass

        if accept_empty == True:
            is_valid = isinstance(data, np.ndarray)
        else:
            is_valid = isinstance(data, np.ndarray) and data.size > 0

        return is_valid

    def _figsave(self, figurename="", myformat="png", subfolderpath="", suffix=""):
        """
        Save the current figure to the working directory or a specified subfolder path.

        This method saves the current figure with various customization options for the
        filename, format, and location. By default, figures are saved as 'MyFigure.png'.
        The figure's font settings are configured such that fonts are preserved as they are,
        and not converted into paths.

        Parameters
        ----------
        figurename : str, optional
            The name of the figure file. If it's specified with an extension, the figure
            is saved with that name. If it's a relative path, the figure is saved to that path.
            If not provided, the figure is saved as 'MyFigure.png'. Defaults to "".
        myformat : str, optional
            The format of the figure (e.g., 'png', 'jpg', 'svg', etc.).
            If provided with a leading ".", the "." is removed. Defaults to 'png'.
        subfolderpath : str, optional
            The subfolder within the working directory to which the figure is saved.
            If figurename is a path, this value will be overridden by the parent directory
            of figurename. Defaults to "".
        suffix : str, optional
            A suffix that is added to the end of the filename, just before the file extension.
            Defaults to "".

        Returns
        -------
        None

        Notes
        -----
        - The fonts in the figure are configured to be saved as fonts, not as paths.
        - If the specified subfolder doesn't exist, it is created.
        - If both `figurename` and `subfolderpath` are paths, `figurename` takes precedence,
        and `subfolderpath` is overridden.
        """

        plt.rcParams["svg.fonttype"] = "none"  # Fonts as fonts and not as paths
        plt.rcParams["ps.fonttype"] = "type3"  # Fonts as fonts and not as paths

        # Confirm pathlib type
        figurename = Path(figurename)
        subfolderpath = Path(subfolderpath)

        # Check if figurename is a path. If yes, parse the figurename and subfolderpath
        if str(figurename.parent) != ".":
            subfolderpath = figurename.parent
            figurename = Path(figurename.name)

        if myformat[0] == ".":
            myformat = myformat[1:]

        filename, file_extension = figurename.stem, figurename.suffix

        filename = filename + suffix

        if not file_extension:
            file_extension = "." + myformat

        if not figurename:
            figurename = "MyFigure" + file_extension
        else:
            figurename = filename + file_extension

        path = self.context.path
        figurename_fullpath = Path.joinpath(path, subfolderpath, figurename)
        full_subfolderpath = Path.joinpath(path, subfolderpath)
        if not Path.is_dir(full_subfolderpath):
            Path.mkdir(full_subfolderpath)
        print(f"Saving figure to {figurename_fullpath}")
        plt.savefig(
            figurename_fullpath,
            dpi=None,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=file_extension[1:],
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )

    # Fit visualization
    def show_temporal_filter_response(self, n_curves=None, savefigname=None):
        """
        Show temporal filter response for each cell.
        """
        exp_temp_filt = self.project_data.fit["exp_temp_filt"]
        xdata = exp_temp_filt["xdata"]
        xdata_finer = exp_temp_filt["xdata_finer"]
        title = exp_temp_filt["title"]

        # get cell_ixs
        cell_ixs_list = [ci for ci in exp_temp_filt.keys() if ci.startswith("cell_ix_")]

        if n_curves is not None:
            cell_ixs_list = np.random.choice(cell_ixs_list, n_curves, replace=False)

        for this_cell_ix in cell_ixs_list:
            ydata = exp_temp_filt[f"{this_cell_ix}"]["ydata"]
            y_fit = exp_temp_filt[f"{this_cell_ix}"]["y_fit"]
            plt.scatter(xdata, ydata)
            plt.plot(
                xdata_finer,
                y_fit,
                c="grey",
            )

        N_cells = len(cell_ixs_list)
        plt.title(f"{title} ({N_cells} cells)")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_spatial_filter_response(
        self, spat_filt, n_samples=1, sample_list=None, title="", savefigname=None
    ):
        """
        Display the spatial filter response of the selected cells, along with the corresponding DoG models.

        Parameters
        ----------
        spat_filt : dict
            Dictionary containing spatial filter data.
            - 'data_all_viable_cells': numpy.ndarray
            - 'x_grid', 'y_grid': numpy.ndarray
            - 'DoG_model': str
            - 'num_pix_x', 'num_pix_y': int
            - Other keys starting with "cell_ix_" for individual cell data.
        n_samples : int, optional
            Number of cells to sample. The default is 1. np.inf will display all cells.
        sample_list : list of int, optional
            Indices of specific cells to display. Overrides n_samples if provided.
        title : str, optional
            Title for the plot. Default is an empty string.
        savefigname : str, optional
            If provided, saves the plot to a file with this name.

        """

        data_all_viable_cells = spat_filt["data_all_viable_cells"]
        x_grid = spat_filt["x_grid"]
        y_grid = spat_filt["y_grid"]
        DoG_model = spat_filt["DoG_model"]
        pixel_array_shape_x = spat_filt["num_pix_x"]
        pixel_array_shape_y = spat_filt["num_pix_y"]

        # get cell_ixs
        cell_ixs_list = [ci for ci in spat_filt.keys() if ci.startswith("cell_ix_")]
        if sample_list is not None:
            cell_ixs_list = [cell_ixs_list[i] for i in sample_list]
            n_samples = len(cell_ixs_list)
        elif n_samples < len(cell_ixs_list):
            cell_ixs_list = np.random.choice(cell_ixs_list, n_samples, replace=False)
        elif n_samples == np.inf:
            n_samples = len(cell_ixs_list)

        # Create a single figure for all the samples
        fig, axes = plt.subplots(figsize=(8, 2 * n_samples), nrows=n_samples, ncols=2)
        if n_samples == 1:  # Ensure axes is a 2D array for consistency
            axes = np.array([axes])

        imshow_cmap = "viridis"
        ellipse_edgecolor = "white"
        colorscale_min_max = [None, None]  # [-0.2, 0.8]

        for idx, this_cell_ix in enumerate(cell_ixs_list):
            this_cell_ix_numerical = int(this_cell_ix.split("_")[-1])

            # Add cell index text to the left side of each row
            axes[idx, 0].text(
                x_grid.min() - 5,  # Adjust the x-coordinate as needed
                (y_grid.min() + y_grid.max()) / 2,  # Vertical centering
                f"Cell Index: {this_cell_ix_numerical}",
                verticalalignment="center",
                horizontalalignment="right",
            )

            # Get DoG model fit parameters to popt
            popt = data_all_viable_cells[this_cell_ix_numerical, :]
            spatial_data_array = spat_filt[this_cell_ix]["spatial_data_array"]
            suptitle = spat_filt[this_cell_ix]["suptitle"]
            suptitle = f"{title}, {suptitle})"

            cen = axes[idx, 0].imshow(
                spatial_data_array,
                cmap=imshow_cmap,
                origin="lower",
                extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
                vmin=colorscale_min_max[0],
                vmax=colorscale_min_max[1],
            )
            fig.colorbar(cen, ax=axes[idx, 0])

            # Ellipses for DoG2D_fixed_surround. Circular params are mapped to ellipse_fixed params
            if DoG_model == "ellipse_fixed":
                data_fitted = self.DoG2D_fixed_surround((x_grid, y_grid), *popt)
                e1 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[3],
                    popt[4],
                    -popt[5] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                )
                e2 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[7] * popt[3],
                    popt[7] * popt[4],
                    -popt[5] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                    linestyle="--",
                )

            elif DoG_model == "ellipse_independent":
                data_fitted = self.DoG2D_independent_surround((x_grid, y_grid), *popt)
                e1 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[3],
                    popt[4],
                    -popt[5] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                )
                e2 = Ellipse(
                    (popt[np.array([7, 8])]),
                    popt[9],
                    popt[10],
                    -popt[11] * 180 / np.pi,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                    linestyle="--",
                )
            elif DoG_model == "circular":
                data_fitted = self.DoG2D_circular((x_grid, y_grid), *popt)
                e1 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[3],
                    popt[3],
                    0.0,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                )
                e2 = Ellipse(
                    (popt[np.array([1, 2])]),
                    popt[5],
                    popt[5],
                    0.0,
                    edgecolor=ellipse_edgecolor,
                    linewidth=2,
                    fill=False,
                    linestyle="--",
                )

            axes[idx, 0].add_artist(e1)
            axes[idx, 0].add_artist(e2)

            sur = axes[idx, 1].imshow(
                data_fitted.reshape(pixel_array_shape_y, pixel_array_shape_x),
                cmap=imshow_cmap,
                origin="lower",
                extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
                vmin=colorscale_min_max[0],
                vmax=colorscale_min_max[1],
            )
            fig.colorbar(sur, ax=axes[idx, 1])

        plt.tight_layout()
        plt.suptitle(title, fontsize=10)
        plt.subplots_adjust(top=0.95)

        if savefigname:
            self._figsave(figurename=title + "_" + savefigname)

    # ConstructRetina visualization

    def _get_imgs(
        self,
        df,
        nsamples,
        exp_spat_filt,
        this_trial_idx,
    ):
        log_dir = df["logdir"][this_trial_idx]

        # Get folder name starting "checkpoint"
        checkpoint_folder_name = [f for f in os.listdir(log_dir) if "checkpoint" in f][
            0
        ]
        checkpoint_path = Path(log_dir) / checkpoint_folder_name / "model.pth"

        # Load the model
        model = torch.load(checkpoint_path)

        if hasattr(model, "test_data"):
            test_data = model.test_data[:nsamples, :, :, :]
        else:
            # Make a list of dict keys starting "cell_ix_" from exp_spat_filt dictionary
            keys = exp_spat_filt.keys()
            cell_ix_names_list = [key for key in keys if "cell_ix_" in key]
            # Make a numpy array of numbers following "cell_ix_"
            cell_ix_array = np.array(
                [int(key.split("cell_ix_")[1]) for key in cell_ix_names_list]
            )

            # Take first nsamples from cell_ix_array. They have to be constant,
            # because we are showing multiple sets of images on top of each other.
            samples = cell_ix_array[:nsamples]

            test_data = np.zeros(
                [
                    nsamples,
                    1,
                    exp_spat_filt["num_pix_y"],
                    exp_spat_filt["num_pix_x"],
                ]
            )
            for idx, this_sample in enumerate(samples):
                test_data[idx, 0, :, :] = exp_spat_filt[f"cell_ix_{this_sample}"][
                    "spatial_data_array"
                ]

        # Hack to reuse the AugmentedDataset._feature_scaling method. Scales to [0,1]
        test_data = AugmentedDataset._feature_scaling("", test_data)

        test_data = torch.from_numpy(test_data).float()
        img_size = model.decoder.unflatten.unflattened_size
        test_data = TF.resize(test_data, img_size[-2:], antialias=True)

        self.device = self.context.device
        samples = range(0, nsamples)

        model.eval()
        model.to(self.device)

        img = test_data.to(self.device)

        with torch.no_grad():
            rec_img = model(img)

        img = img.cpu().squeeze().numpy()
        rec_img = rec_img.cpu().squeeze().numpy()

        return img, rec_img, samples

    def _subplot_dependent_boxplots(self, axd, kw, df, dep_vars, config_vars_changed):
        """Boxplot dependent variables for one ray tune experiment"""

        # config_vars_changed list contain the varied columns in dataframe df
        # From config_vars_changed, 'config/model_id' contain the replications of the same model
        # Other config_vars_changed contain the models of interest
        # Make an seaborn boxplot for each model of interest
        config_vars_changed.remove("config/model_id")

        # If there are more than one config_vars_changed,
        # make a new dataframe column with the values of the config_vars_changed as strings
        if len(config_vars_changed) > 1:
            df["config_vars"] = (
                df[config_vars_changed].astype(str).agg(",".join, axis=1)
            )
            # config_vars_changed = ["config_vars"]
            config_vars_for_label = [
                col.removeprefix("config/") for col in config_vars_changed
            ]
            # Combine the string listed in config_vars_changed to one string
            config_vars_label = ",".join(config_vars_for_label)

        else:
            df["config_vars"] = df[config_vars_changed[0]]
            config_vars_label = [
                col.removeprefix("config/") for col in config_vars_changed
            ][0]

        # Make one subplot for each dependent variable
        # Plot labels only after the last subplot
        for idx, dep_var in enumerate(dep_vars):
            ax = axd[f"{kw}{idx}"]

            # Create the boxplot with seaborn
            ax_sns = sns.boxplot(
                x="config_vars", y=dep_var, data=df, ax=ax, whis=[0, 100]
            )
            # If any of the df["config_vars"] has length > 4, make x-label rotated 90 degrees
            if any(df["config_vars"].astype(str).str.len() > 4):
                ax_sns.set_xticklabels(ax.get_xticklabels(), rotation=90)

            # Set the title of the subplot to be the dependent variable name
            ax.set_title(dep_var)

            # Set y-axis label
            ax.set_ylabel("")

            # Set x-axis label
            if idx == 0:  # only the first subplot gets an x-axis label
                ax.set_xlabel(config_vars_label)
            else:
                ax.set_xlabel("")

    def _get_best_trials(self, df, dep_var, best_is, num_best_trials):
        """
        Get the indices of the best trials for a dependent variable.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the results of the hyperparameter search.
        dep_var : str
            Name of the dependent variable.
        best_is : str
            Whether the best trials are the ones with the highest or lowest values.
        num_best_trials : int
            Number of best trials to return. Overrides frac_best.

        Returns
        -------
        best_trials : list
            List of indices of the best trials.
        """

        # get array of values for this dependent variable
        dep_var_vals = df[dep_var].values

        # get the indices of the num_best_trials
        if best_is == "min":
            best_trials = np.argsort(dep_var_vals)[:num_best_trials]
        elif best_is == "max":
            best_trials = np.argsort(dep_var_vals)[-num_best_trials:]

        return best_trials, dep_var_vals

    def _subplot_dependent_variables(self, axd, kw, result_grid, dep_vars, best_trials):
        """Plot dependent variables as a function of epochs."""

        df = result_grid.get_dataframe()
        # Find all columns with string "config/"
        config_cols = [x for x in df.columns if "config/" in x]

        # From the config_cols, identify columns where there is more than one unique value
        # These are the columns which were varied in the search space
        varied_cols = []
        for col in config_cols:
            if len(df[col].unique()) > 1:
                varied_cols.append(col)

        # Drop the "config/" part from the column names
        varied_cols = [x.replace("config/", "") for x in varied_cols]

        # # remove "model_id" from the varied columns
        # varied_cols.remove("model_id")

        num_colors = len(best_trials)
        colors = plt.cm.get_cmap("tab20", num_colors).colors

        total_n_epochs = 0
        # Make one subplot for each dependent variable
        for idx, dep_var in enumerate(dep_vars):
            # Create a new plot for each label
            color_idx = 0
            ax = axd[f"{kw}{idx}"]

            for i, result in enumerate(result_grid):
                if i not in best_trials:
                    continue

                if idx == 0:
                    label = f"{dep_vars[color_idx]}: " + ",".join(
                        f"{x}={result.config[x]}" for x in varied_cols
                    )
                    legend = True
                    first_ax = ax

                else:
                    label = None
                    legend = False

                result.metrics_dataframe.plot(
                    "training_iteration",
                    dep_var,
                    ax=ax,
                    label=label,
                    color=colors[color_idx],
                    legend=legend,
                )

                if len(result.metrics_dataframe) > total_n_epochs:
                    total_n_epochs = len(result.metrics_dataframe)

                # At the end (+1) of the x-axis, add mean and SD of last 50 epochs as dot and vertical line, respectively
                last_50 = result.metrics_dataframe.tail(50)
                mean = last_50[dep_var].mean()
                std = last_50[dep_var].std()
                n_epochs = result.metrics_dataframe.tail(1)["training_iteration"]
                ax.plot(
                    n_epochs + n_epochs // 5,
                    mean,
                    "o",
                    color=colors[color_idx],
                )
                ax.plot(
                    [n_epochs + n_epochs // 5] * 2,
                    [mean - std, mean + std],
                    "-",
                    color=colors[color_idx],
                )

                color_idx += 1

            if idx == 0:
                ax.set_ylabel("Metrics")

            # Add legend and bring it to the front
            leg = first_ax.legend(
                loc="center left", bbox_to_anchor=((idx + 2.0), 0.5, 1.0, 0.2)
            )
            first_ax.set_zorder(1)

            # change the line width for the legend
            for line in leg.get_lines():
                line.set_linewidth(3.0)

            # Set x and y axis tick font size 8
            ax.tick_params(axis="both", which="major", labelsize=8)

            # Change legend font size to 8
            for text in leg.get_texts():
                text.set_fontsize(8)

            ax.grid(True)

            # set x axis labels off
            ax.set_xlabel("")
            # set x ticks off
            ax.set_xticks([])

        first_ax.set_title(
            f"Evolution for best trials (ad {total_n_epochs} epochs)\nDot and vertical line indicate mean and SD of last 50 epochs",
            loc="left",
        )

    def _subplot_img_recoimg(self, axd, kw, subidx, img, samples, title):
        """
        Plot sample images
        """
        for pos_idx, sample_idx in enumerate(samples):
            if subidx is None:
                ax = axd[f"{kw}{pos_idx}"]
            else:
                ax = axd[f"{kw}{subidx}{pos_idx}"]
            ax.imshow(img[sample_idx], cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if pos_idx == 0:
                # ax.set_title(title, fontsize=8, fontdict={'verticalalignment': 'baseline', 'horizontalalignment': 'left'})
                # Print title to the left of the first image. The coordinates are in axes coordinates
                ax.text(
                    -1.0,
                    0.5,
                    title,
                    fontsize=8,
                    fontdict={
                        "verticalalignment": "baseline",
                        "horizontalalignment": "left",
                    },
                    transform=ax.transAxes,
                )

    def _show_tune_depvar_evolution(self, result_grid, dep_vars, highlight_trial=None):
        """Plot results from ray tune"""

        df = result_grid.get_dataframe()
        # Find all columns with string "config/"
        config_cols = [x for x in df.columns if "config/" in x]

        # From the config_cols, identify columns where there is more than one unique value
        # These are the columns which were varied in the search space
        varied_cols = []
        for col in config_cols:
            if len(df[col].unique()) > 1:
                varied_cols.append(col)

        # Drop the "config/" part from the column names
        varied_cols = [x.replace("config/", "") for x in varied_cols]

        num_colors = len(result_grid.get_dataframe())
        if highlight_trial is None:
            colors = plt.cm.get_cmap("tab20", num_colors).colors
            highlight_idx = None
        else:
            [highlight_idx] = [
                idx
                for idx, r in enumerate(result_grid)
                if highlight_trial in r.metrics["trial_id"]
            ]
            # set all other colors low contrast gray, and the highlight color to red
            colors = np.array(
                ["gray" if idx != highlight_idx else "red" for idx in range(num_colors)]
            )

        # Make one subplot for each dependent variable
        nrows = 2
        ncols = len(dep_vars) // 2
        plt.figure(figsize=(ncols * 5, nrows * 5))

        for idx, dep_var in enumerate(dep_vars):
            # Create a new plot for each label
            color_idx = 0
            ax = plt.subplot(nrows, ncols, idx + 1)
            label = None

            for result in result_grid:
                # Too cluttered for a legend
                # if idx == 0 and highlight_idx is None:
                #     label = ",".join(f"{x}={result.config[x]}" for x in varied_cols)
                #     legend = True
                # else:
                #     legend = False

                ax_plot = result.metrics_dataframe.plot(
                    "training_iteration",
                    dep_var,
                    ax=ax,
                    label=label,
                    color=colors[color_idx],
                    legend=False,
                )

                # At the end (+1) of the x-axis, add mean and SD of last 50 epochs as dot and vertical line, respectively
                last_50 = result.metrics_dataframe.tail(50)
                mean = last_50[dep_var].mean()
                std = last_50[dep_var].std()
                n_epochs = result.metrics_dataframe.tail(1)["training_iteration"]
                ax.plot(
                    n_epochs + n_epochs // 5,
                    mean,
                    "o",
                    color=colors[color_idx],
                )
                ax.plot(
                    [n_epochs + n_epochs // 5] * 2,
                    [mean - std, mean + std],
                    "-",
                    color=colors[color_idx],
                )

                color_idx += 1
            ax.set_title(f"{dep_var}")
            ax.set_ylabel(dep_var)
            ax.grid(True)

    def show_gc_positions(self):
        """
        Show retina cell positions and receptive fields

        ConstructRetina call.
        """

        ecc_mm = self.construct_retina.gc_df["pos_ecc_mm"].to_numpy()
        pol_deg = self.construct_retina.gc_df["pos_polar_deg"].to_numpy()
        gc_density_func_params = self.construct_retina.gc_density_func_params

        # to cartesian
        xcoord, ycoord = self.pol2cart(ecc_mm, pol_deg)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(
            xcoord.flatten(),
            ycoord.flatten(),
            "b.",
            label=self.construct_retina.gc_type,
        )
        ax.axis("equal")
        ax.legend()
        ax.set_title("Cartesian retina")
        ax.set_xlabel("Eccentricity (mm)")
        ax.set_ylabel("Elevation (mm)")

    def boundary_polygon(self, ecc_lim_mm, polar_lim_deg, n_points=100):
        """
        Create a boundary polygon based on given eccentricity and polar angle limits.

        Parameters
        ----------
        ecc_lim_mm : np.ndarray
            An array representing the eccentricity limits in millimeters for
            left and right boundaries (shape: [2]).
        polar_lim_deg : np.ndarray
            An array representing the polar angle limits in degrees for
            bottom and top boundaries (shape: [2]).
        n_points : int
            Number of points to generate along each arc.

        Returns
        -------
        boundary_polygon : np.ndarray
            Array of Cartesian coordinates forming the vertices of the boundary polygon.
        """

        # Generate points for bottom and top polar angle limits
        bottom_x, bottom_y = self.pol2cart(
            np.full(n_points, ecc_lim_mm[0]),
            np.linspace(polar_lim_deg[0], polar_lim_deg[1], n_points),
        )
        top_x, top_y = self.pol2cart(
            np.full(n_points, ecc_lim_mm[1]),
            np.linspace(polar_lim_deg[0], polar_lim_deg[1], n_points),
        )

        # Generate points along the arcs for min and max eccentricities
        theta_range = np.linspace(polar_lim_deg[0], polar_lim_deg[1], n_points)
        min_ecc_x, min_ecc_y = self.pol2cart(
            np.full_like(theta_range, ecc_lim_mm[0]), theta_range
        )
        max_ecc_x, max_ecc_y = self.pol2cart(
            np.full_like(theta_range, ecc_lim_mm[1]), theta_range
        )

        # Combine them to form the vertices of the bounding polygon
        boundary_polygon = []

        # Add points from bottom arc
        for bx, by in zip(min_ecc_x, min_ecc_y):
            boundary_polygon.append((bx, by))

        # Add points from top arc (in reverse order)
        for tx, ty in reversed(list(zip(max_ecc_x, max_ecc_y))):
            boundary_polygon.append((tx, ty))

        return np.array(boundary_polygon)

    def visualize_mosaic(self, savefigname=None):
        """
        Visualize the mosaic of ganglion cells in retinal mm coordinates.

        This function plots the ganglion cells as ellipses on a Cartesian plane and adds 
        a boundary polygon representing sector limits. 

        Parameters
        ----------
        savefigname : str, optional
            The name of the file to save the figure. If None, the figure is not saved.
        """
        gc_df = self.project_data.construct_retina["gc_df"]
        ecc_mm = gc_df["pos_ecc_mm"].to_numpy()
        pol_deg = gc_df["pos_polar_deg"].to_numpy()

        ecc_lim_deg = self.context.my_retina["ecc_limits_deg"]
        ecc_lim_mm = np.array(ecc_lim_deg) / self.context.my_retina["deg_per_mm"]
        pol_lim_deg = self.context.my_retina["pol_limits_deg"]
        boundary_polygon = self.boundary_polygon(ecc_lim_mm, pol_lim_deg)
        
        # Obtain mm values
        if self.context.my_retina["DoG_model"] == "circular":
            semi_xc = gc_df["rad_c_mm"]
            semi_yc = gc_df["rad_c_mm"]
            angle_in_deg = np.zeros(len(self.construct_retina.gc_df))
        elif self.context.my_retina["DoG_model"] in [
            "ellipse_independent",
            "ellipse_fixed",
        ]:
            semi_xc = gc_df["semi_xc_mm"]
            semi_yc = gc_df["semi_yc_mm"]
            angle_in_deg = gc_df["orient_cen_rad"] * 180 / np.pi

        # to cartesian
        xcoord, ycoord = self.pol2cart(ecc_mm, pol_deg)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        polygon = Polygon(
            boundary_polygon, closed=True, fill=None, edgecolor="r"
        )
        ax.add_patch(polygon)

        ax.plot(
            xcoord.flatten(),
            ycoord.flatten(),
            "b.",
            label=self.construct_retina.gc_type,
        )
        # Ellipse parameters: Ellipse(xy, width, height, angle=0, **kwargs). Only possible one at the time, unfortunately.
        for index in np.arange(len(xcoord)):
            ellipse_center_x = xcoord[index]
            ellipse_center_y = ycoord[index]
            diameter_xc = semi_xc[index] * 2
            diameter_yc = semi_yc[index] * 2
            e1 = Ellipse(
                (ellipse_center_x, ellipse_center_y),
                diameter_xc,
                diameter_yc,
                angle_in_deg[index],
                edgecolor="b",
                linewidth=0.5,
                fill=False,
            )
            ax.add_artist(e1)

        ax.axis("equal")
        ax.legend()
        ax.set_title("Cartesian retina")
        ax.set_xlabel("Eccentricity (mm)")
        ax.set_ylabel("Elevation (mm)")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_spatial_statistics(self, savefigname=None):
        """
        Show histograms of receptive field parameters

        ConstructRetina call.
        """
        ydata = self.project_data.fit["exp_spat_stat"]["ydata"]
        spatial_statistics_dict = self.project_data.fit["exp_spat_stat"][
            "spatial_statistics_dict"
        ]
        spatial_statistics_dict.pop("ampl_c", None)
        model_fit_data = self.project_data.fit["exp_spat_stat"]["model_fit_data"]

        distributions = [key for key in spatial_statistics_dict.keys()]
        n_distributions = len(spatial_statistics_dict)

        # plot the distributions and fits.
        n_ax_cols = 3
        n_ax_rows = math.ceil(n_distributions / n_ax_cols)
        fig, axes = plt.subplots(n_ax_rows, n_ax_cols, figsize=(13, 4))
        axes = axes.flatten()
        for index in np.arange(n_distributions):
            ax = axes[index]

            bin_values, foo, foo2 = ax.hist(ydata[:, index], bins=20, density=True)

            if model_fit_data != None:  # Assumes tuple of arrays, see below
                x_model_fit, y_model_fit = model_fit_data[0], model_fit_data[1]
                ax.plot(
                    x_model_fit[:, index],
                    y_model_fit[:, index],
                    "r-",
                    linewidth=6,
                    alpha=0.6,
                )

                spatial_statistics_dict[distributions[index]]
                shape = spatial_statistics_dict[distributions[index]]["shape"]
                loc = spatial_statistics_dict[distributions[index]]["loc"]
                scale = spatial_statistics_dict[distributions[index]]["scale"]
                model_function = spatial_statistics_dict[distributions[index]][
                    "distribution"
                ]

                ax.annotate(
                    "shape = {0:.2f}\nloc = {1:.2f}\nscale = {2:.2f}".format(
                        shape, loc, scale
                    ),
                    xy=(0.6, 0.4),
                    xycoords="axes fraction",
                )
                ax.set_title(
                    "{0} fit for {1}".format(model_function, distributions[index])
                )

                # Rescale y axis if model fit goes high. Shows histogram better
                if y_model_fit[:, index].max() > 1.5 * bin_values.max():
                    ax.set_ylim([ax.get_ylim()[0], 1.1 * bin_values.max()])

        if savefigname:
            self._figsave(figurename=savefigname)

        # Check correlations
        fig2, axes2 = plt.subplots(n_ax_rows, n_ax_cols, figsize=(13, 4))
        axes2 = axes2.flatten()
        ref_index = 1
        for index in np.arange(n_distributions):
            ax2 = axes2[index]
            data_all_x = ydata[:, ref_index]
            data_all_y = ydata[:, index]

            r, p = stats.pearsonr(data_all_x, data_all_y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                data_all_x, data_all_y
            )
            ax2.plot(data_all_x, data_all_y, ".")
            data_all_x.sort()
            ax2.plot(data_all_x, intercept + slope * data_all_x, "b-")
            ax2.annotate(
                "\nr={0:.2g},\np={1:.2g}".format(r, p),
                xy=(0.8, 0.4),
                xycoords="axes fraction",
            )
            ax2.set_title(
                "Correlation between {0} and {1}".format(
                    distributions[ref_index], distributions[index]
                )
            )

    def show_dendrite_diam_vs_ecc(self, savefigname=None):
        """
        Plot dendritic diameter as a function of retinal eccentricity with linear, quadratic, or cubic fitting.
        """
        dd_vs_ecc = self.project_data.construct_retina["dd_vs_ecc"]
        data_all_x = dd_vs_ecc["data_all_x"]
        data_all_y = dd_vs_ecc["data_all_y"]
        dd_DoG_x = dd_vs_ecc["dd_DoG_x"]
        dd_DoG_y = dd_vs_ecc["dd_DoG_y"]
        fit_parameters = dd_vs_ecc["fit_parameters"]
        dd_model_caption = dd_vs_ecc["dd_model_caption"]
        title = dd_vs_ecc["title"]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(data_all_x, data_all_y, "b.", label="Data")
        ax.plot(dd_DoG_x, dd_DoG_y, "r.", label="DoG fit")

        ax.set_xlabel("Retinal eccentricity (mm)")
        ax.set_ylabel("Dendritic diameter (um)")
        ax.legend()

        if dd_model_caption:
            if self.context.my_retina["dd_regr_model"] == "linear":
                intercept = fit_parameters[1]
                slope = fit_parameters[0]
                ax.plot(data_all_x, intercept + slope * data_all_x, "k--")
                ax.annotate(
                    "{0} : \ny={1:.1f} + {2:.1f}x".format(
                        dd_model_caption, intercept, slope
                    ),
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )
            elif self.context.my_retina["dd_regr_model"] == "quadratic":
                intercept = fit_parameters[2]
                slope = fit_parameters[1]
                square = fit_parameters[0]
                ax.plot(
                    data_all_x,
                    intercept + slope * data_all_x + square * data_all_x**2,
                    "k--",
                )
                ax.annotate(
                    "{0}: \ny={1:.1f} + {2:.1f}x + {3:.1f}x^2".format(
                        dd_model_caption, intercept, slope, square
                    ),
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )
            elif self.context.my_retina["dd_regr_model"] == "cubic":
                intercept = fit_parameters[3]
                slope = fit_parameters[2]
                square = fit_parameters[1]
                cube = fit_parameters[0]
                ax.plot(
                    data_all_x,
                    intercept
                    + slope * data_all_x
                    + square * data_all_x**2
                    + cube * data_all_x**3,
                    "k--",
                )
                ax.annotate(
                    "{0}: \ny={1:.1f} + {2:.1f}x + {3:.1f}x^2 + {4:.1f}x^3".format(
                        dd_model_caption, intercept, slope, square, cube
                    ),
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )
            elif self.context.my_retina["dd_regr_model"] == "exponential":
                constant = fit_parameters[0]
                lamda = fit_parameters[1]
                ax.plot(data_all_x, constant + np.exp(data_all_x / lamda), "k--")
                ax.annotate(
                    "{0}: \ny={1:.1f} + exp(x/{2:.1f})".format(
                        dd_model_caption, constant, lamda
                    ),
                    xycoords="axes fraction",
                    xy=(0.5, 0.15),
                    ha="left",
                    color="k",
                )

        plt.title(title)

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_temp_stat(self):
        """
        Show the temporal statistics of the retina units.
        """

        exp_temp_stat = self.project_data.fit["exp_temp_stat"]

        temporal_filter_parameters = exp_temp_stat["temporal_filter_parameters"]
        distrib_params = exp_temp_stat["distrib_params"]
        suptitle = exp_temp_stat["suptitle"]
        all_data_fits_df = exp_temp_stat["all_data_fits_df"]
        good_idx_experimental = exp_temp_stat["good_idx_experimental"]

        plt.subplots(2, 3)
        plt.suptitle(suptitle)
        for i, param_name in enumerate(temporal_filter_parameters):
            plt.subplot(2, 3, i + 1)
            ax = plt.gca()
            shape, loc, scale = distrib_params[i, :]
            param_array = np.array(
                all_data_fits_df.iloc[good_idx_experimental][param_name]
            )

            x_min, x_max = stats.gamma.ppf(
                [0.001, 0.999], a=shape, loc=loc, scale=scale
            )
            xs = np.linspace(x_min, x_max, 100)
            pdf = stats.gamma.pdf(xs, a=shape, loc=loc, scale=scale)
            ax.plot(xs, pdf)
            ax.hist(param_array, density=True)
            ax.set_title(param_name)

    def show_tonic_drives(self):
        """
        ConstructRetina call.
        """

        exp_tonic_dr = self.project_data.fit["exp_tonic_dr"]

        xs = exp_tonic_dr["xs"]
        pdf = exp_tonic_dr["pdf"]
        tonicdrive_array = exp_tonic_dr["tonicdrive_array"]
        title = exp_tonic_dr["title"]

        plt.plot(xs, pdf)
        plt.hist(tonicdrive_array, density=True)
        plt.title(title)
        plt.xlabel("Tonic drive (a.u.)")

    def show_exp_build_process(self, show_all_spatial_fits=False):
        """
        Visualize the stages of retina mosaic building process.
        """

        # If show_all_spatial_fits is true, show the spatial fits
        if show_all_spatial_fits is True:
            spat_filt = self.project_data.fit["exp_spat_filt"]
            self.show_spatial_filter_response(
                spat_filt,
                n_samples=np.inf,
                title="Experimental",
                pause_to_show=True,
            )
            return

        self.show_temporal_filter_response()
        self.visualize_mosaic()
        self.show_spatial_statistics()
        self.show_dendrite_diam_vs_ecc()
        self.show_temp_stat()
        self.show_tonic_drives()

    def show_DoG_model_fit(self, n_samples=2, sample_list=None, savefigname=None):
        """
        Show the experimental and generated spatial receptive fields. The type of
        spatial model in use (VAE or other) determines what exactly is displayed.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to show. Default is 2.
        sample_list : list, optional
            List of specific samples to display. Overrides n_samples if provided.
        savefigname : str, optional
            Name of the file to save the figure. If None, the figure won't be saved.

        Notes
        -----
        - When the spatial model is VAE, the experimental title changes to indicate the
        use of 'ellipse_fixed'.
        """
        if self.construct_retina.spatial_model == "VAE":
            spat_filt = self.project_data.fit["gen_spat_filt"]
            self.show_spatial_filter_response(
                spat_filt,
                n_samples=n_samples,
                sample_list=sample_list,
                title="Generated",
                savefigname=savefigname,
            )
            # VAE RF is originally generated in experimental data space at peripheral retina.
            # When eccentricity changes also the VAE RF sizes need to be scaled.
            # The VAE RF sizes are scaled according to dendritic field diameter comparison between
            # experimental data fit and literature data.
            # Thus the experimental (first) fit for VAE is fixed to ellipse_fixed not to have the
            # VAE RF resolution depend on DoG model.
            exp_title = "Experimental shows ellipse_fixed when spatial_model: VAE"
        else:
            exp_title = "Experimental"

        spat_filt = self.project_data.fit["exp_spat_filt"]
        self.show_spatial_filter_response(
            spat_filt,
            n_samples=n_samples,
            sample_list=sample_list,
            title=exp_title,
            savefigname=savefigname,
        )

    def show_latent_space_and_samples(self):
        """
        Plot the latent samples on top of the estimated kde, one sublot for each successive two dimensions of latent_dim
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        gen_latent_space = self.project_data.construct_retina["gen_latent_space"]

        latent_samples = gen_latent_space["samples"]
        latent_data = gen_latent_space["data"]
        latent_dim = gen_latent_space["dim"]

        # Make a grid of subplots
        n_cols = 4
        n_rows = int(np.ceil(latent_dim / n_cols))
        if n_rows == 1:
            n_cols = latent_dim
        elif n_rows > 4:
            n_rows = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2 * n_rows))
        fig_suptitle_text = "Latent space and samples"
        axes = axes.flatten()

        # Plot the latent samples on top of the estimated kde
        for ax_idx, i in enumerate(range(0, latent_dim, 2)):
            if ax_idx > 15:
                fig_suptitle_text = (
                    "Latent space and samples (plotting only the first 32 dimensions)"
                )
                break

            # Get only two dimensions at a time
            values = latent_data[:, [i, i + 1]].T
            # Evaluate the kde using only the same two dimensions
            # Both uniform and normal distr during learning is sampled
            # using gaussian kde estimate. The kde estimate is basically smooth histogram,
            # so it is not a problem that the data is not normal.
            kernel = stats.gaussian_kde(values)

            # Construct X and Y grids using the same two dimensions
            x = np.linspace(latent_data[:, i].min(), latent_data[:, i].max(), 100)
            y = np.linspace(
                latent_data[:, i + 1].min(), latent_data[:, i + 1].max(), 100
            )
            X, Y = np.meshgrid(x, y)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions).T, X.shape)

            # Plot the estimated kde and samples on top of it
            axes[ax_idx].contour(X, Y, Z, levels=10)
            axes[ax_idx].scatter(latent_samples[:, i], latent_samples[:, i + 1])
            # axes[ax_idx].scatter(latent_data[:, i], latent_data[:, i + 1])

            # Make marginal plots of the contours as contours and samples as histograms.
            # Place the marginal plots on the right and top of the main plot
            ax_marg_x = inset_axes(
                axes[ax_idx],
                width="100%",  # width  of parent_bbox width
                height="30%",  # height : 1 inch
                loc="upper right",
                # bbox_to_anchor=(1.05, 1.05),
                bbox_to_anchor=(0, 0.95, 1, 0.3),
                bbox_transform=axes[ax_idx].transAxes,
                borderpad=0,
            )
            ax_marg_y = inset_axes(
                axes[ax_idx],
                width="30%",  # width of parent_bbox width
                height="100%",  # height : 1 inch
                loc="lower left",
                # bbox_to_anchor=(-0.05, -0.05),
                bbox_to_anchor=(1, 0, 0.4, 1),
                bbox_transform=axes[ax_idx].transAxes,
                borderpad=0,
            )

            # Plot the marginal plots
            nx, bins, _ = ax_marg_x.hist(latent_samples[:, i], bins=20, density=True)
            ny, bins, _ = ax_marg_y.hist(
                latent_samples[:, i + 1],
                bins=20,
                density=True,
                orientation="horizontal",
            )

            # Plot the one-dimensional marginal shapes of the kde
            x_margin_contour = nx.max() * Z.mean(axis=0) / Z.mean(axis=0).max()
            y_margin_contour = ny.max() * Z.mean(axis=1) / Z.mean(axis=1).max()
            ax_marg_x.plot(x, x_margin_contour, color="r")
            ax_marg_y.plot(y_margin_contour, y, color="r")

            # Remove the ticks from the marginal plots
            ax_marg_x.tick_params(
                axis="both", which="both", bottom=False, top=False, labelbottom=False
            )
            ax_marg_y.tick_params(
                axis="both", which="both", left=False, right=False, labelleft=False
            )

            # Set the title of the main plot
            axes[ax_idx].set_title(f"Latent dims {i}, {i+1}")

        # plt.tight_layout()
        fig.suptitle(fig_suptitle_text)

    def show_gen_spat_post_hist(self):
        """
        Show the original experimental spatial receptive fields and
        the generated spatial receptive fields before and after postprocessing.
        """

        # Get the keys for the cell_ix arrays
        cell_key_list = [
            key
            for key in self.project_data.fit["exp_spat_filt"].keys()
            if "cell_ix" in key
        ]
        img_shape = self.project_data.fit["exp_spat_filt"]["cell_ix_0"][
            "spatial_data_array"
        ].shape
        # The shape of the array is N cells, y_pixels, x_pixels
        img_exp = np.zeros([len(cell_key_list), img_shape[0], img_shape[1]])
        for i, cell_key in enumerate(cell_key_list):
            img_exp[i, :, :] = self.project_data.fit["exp_spat_filt"][cell_key][
                "spatial_data_array"
            ]

        img_pre = self.project_data.construct_retina["gen_spat_img"]["img_raw"]
        img_post = self.project_data.construct_retina["gen_spat_img"]["img_processed"]

        plt.subplot(1, 3, 1)
        plt.hist(img_exp.flatten(), bins=100)
        # plot median value as a vertical line
        plt.axvline(np.median(img_exp), color="r")
        plt.title(f"Experimental, median: {np.median(img_exp):.2f}")

        plt.subplot(1, 3, 2)
        plt.hist(img_pre.flatten(), bins=100)
        plt.axvline(np.median(img_pre), color="r")
        plt.title(f"Generated raw, median: {np.median(img_pre):.2f}")

        plt.subplot(1, 3, 3)
        plt.hist(img_post.flatten(), bins=100)
        plt.axvline(np.median(img_post), color="r")
        plt.title(f"Generated processed, median: {np.median(img_post):.2f}")

    def show_gen_exp_spatial_rf(
        self, ds_name="test_ds", n_samples=10, savefigname=None
    ):
        """
        Plot the outputs of the autoencoder.
        """
        assert (
            self.construct_retina.spatial_model == "VAE"
        ), "Only model type VAE is supported for show_gen_exp_spatial_rf()"
        if ds_name == "train_ds":
            ds = self.construct_retina.retina_vae.train_loader.dataset
        elif ds_name == "valid_ds":
            ds = self.construct_retina.retina_vae.val_loader.dataset
        else:
            ds = self.construct_retina.retina_vae.test_loader.dataset

        plt.figure(figsize=(16, 4.5))

        vae = self.construct_retina.retina_vae.vae
        vae.eval()
        len_ds = len(ds)
        samples = np.random.choice(len_ds, n_samples, replace=False)

        for pos_idx, sample_idx in enumerate(samples):
            ax = plt.subplot(2, len(samples), pos_idx + 1)
            img = (
                ds[sample_idx][0]
                .unsqueeze(0)
                .to(self.construct_retina.retina_vae.device)
            )
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.text(
                0.05,
                0.85,
                self.construct_retina.retina_vae.apricot_data.data_labels2names_dict[
                    ds[sample_idx][1].item()
                ],
                fontsize=10,
                color="red",
                transform=ax.transAxes,
            )
            if pos_idx == 0:
                ax.set_title("Original images")

            ax = plt.subplot(2, len(samples), len(samples) + pos_idx + 1)
            with torch.no_grad():
                rec_img = vae(img)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if pos_idx == 0:
                ax.set_title("Reconstructed images")

        # Set the whole figure title as ds_name
        plt.suptitle(ds_name)

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_latent_tsne_space(self):
        train_df = self.construct_retina.retina_vae.get_encoded_samples(
            dataset=self.construct_retina.retina_vae.train_loader.dataset
        )
        valid_df = self.construct_retina.retina_vae.get_encoded_samples(
            dataset=self.construct_retina.retina_vae.val_loader.dataset
        )
        test_df = self.construct_retina.retina_vae.get_encoded_samples(
            dataset=self.construct_retina.retina_vae.test_loader.dataset
        )

        # Add a column to each df with the dataset name
        train_df["dataset"] = "train"
        valid_df["dataset"] = "valid"
        test_df["dataset"] = "test"

        # Concatenate the dfs
        encoded_samples = pd.concat([train_df, valid_df, test_df])

        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30)

        if encoded_samples.shape[0] < tsne.perplexity:
            tsne.perplexity = encoded_samples.shape[0] - 1

        tsne_results = tsne.fit_transform(
            encoded_samples.drop(["label", "dataset"], axis=1)
        )

        ax0 = sns.relplot(
            # data=tsne_results,
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            hue=encoded_samples.dataset.astype(str),
        )
        ax0.set(xlabel="tsne-2d-one", ylabel="tsne-2d-two")
        plt.title("TSNE plot of encoded samples")

    def show_ray_experiment(self, ray_exp, this_dep_var, highlight_trial=None):
        """
        Show the results of a ray experiment. If ray_exp is None, then
        the most recent experiment is shown.

        Parameters
        ----------
        ray_exp : str
            The name of the ray experiment
        this_dep_var : str
            The dependent variable to use for selecting the best trials
        highlight_trial : int
            The trial to highlight in the evolution plot
        """

        info_columns = ["trial_id", "iteration"]
        dep_vars = ["train_loss", "val_loss", "mse", "ssim", "kid_mean", "kid_std"]
        dep_vars_best = ["min", "min", "min", "max", "min", "min"]
        config_prefix = "config/"

        if ray_exp is None:
            most_recent = True
        else:
            most_recent = False

        result_grid = self.data_io.load_ray_results_grid(
            most_recent=most_recent, ray_exp=ray_exp
        )
        df = result_grid.get_dataframe()

        # Get configuration variables
        config_vars_all = [c for c in df.columns if config_prefix in c]

        # Drop columns that are constant in the experiment
        constant_cols = []
        for col in config_vars_all:
            if len(df[col].unique()) == 1:
                constant_cols.append(col)
        config_vars_changed = [
            col for col in config_vars_all if col not in constant_cols
        ]
        config_vars = [col.removeprefix(config_prefix) for col in config_vars_changed]

        # Remove all rows containing nan values in the dependent variables
        df = df.dropna(subset=dep_vars)

        # Collect basic data from the experiment
        n_trials = len(df)
        n_errors = result_grid.num_errors

        # Columns to describe the experiment
        exp_info_columns = info_columns + config_vars_changed + dep_vars
        print(df[exp_info_columns].describe())

        # Find the row indeces of the n best trials
        best_trials_across_dep_vars = []
        for dep_var, dep_var_best in zip(dep_vars, dep_vars_best):
            if dep_var_best == "min":
                best_trials_across_dep_vars.append(df[dep_var].idxmin())
            elif dep_var_best == "max":
                best_trials_across_dep_vars.append(df[dep_var].idxmax())
            if this_dep_var in dep_var:
                this_dep_var_best = dep_var_best

        df_filtered = df[exp_info_columns].loc[best_trials_across_dep_vars]
        # Print the exp_info_columns for the best trials
        print(f"Best trials: in order of {dep_vars=}")
        print(df_filtered)

        self._show_tune_depvar_evolution(
            result_grid, dep_vars, highlight_trial=highlight_trial
        )

        nrows = 9
        ncols = len(dep_vars)
        nsamples = 10

        layout = [
            ["dh0", "dh1", "dh2", "dh3", "dh4", "dh5", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            # ["dv0", "dv1", "dv2", "dv3", "dv4", "dv5", ".", ".", ".", "."],
            ["im0", "im1", "im2", "im3", "im4", "im5", "im6", "im7", "im8", "im9"],
            ["re0" + str(i) for i in range(10)],
            ["re1" + str(i) for i in range(10)],
            ["re2" + str(i) for i in range(10)],
            ["re3" + str(i) for i in range(10)],
            ["re4" + str(i) for i in range(10)],
        ]
        fig, axd = plt.subplot_mosaic(layout, figsize=(ncols * 2, nrows))

        # Fraction of best = 1/4
        frac_best = 0.25
        num_best_trials = int(len(df) * frac_best)

        self._subplot_dependent_boxplots(axd, "dh", df, dep_vars, config_vars_changed)

        exp_spat_filt = self.project_data.fit["exp_spat_filt"]

        num_best_trials = 5  # Also N reco img to show
        best_trials, dep_var_vals = self._get_best_trials(
            df, this_dep_var, this_dep_var_best, num_best_trials
        )

        img, rec_img, samples = self._get_imgs(
            df, nsamples, exp_spat_filt, best_trials[0]
        )

        title = f"Original \nimages"
        self._subplot_img_recoimg(axd, "im", None, img, samples, title)

        title = f"Reco for \n{this_dep_var} = \n{dep_var_vals[best_trials[0]]:.3f}, \nidx = {best_trials[0]}"
        self._subplot_img_recoimg(axd, "re", 0, rec_img, samples, title)

        for idx, this_trial in enumerate(best_trials[1:]):
            img, rec_img, samples = self._get_imgs(
                df, nsamples, exp_spat_filt, this_trial
            )

            title = f"Reco for \n{this_dep_var} = \n{dep_var_vals[this_trial]:.3f}, \nidx = {this_trial}"
            # Position idx in layout: enumerate starts at 0, so add 1.
            self._subplot_img_recoimg(axd, "re", idx + 1, rec_img, samples, title)

    def show_gc_placement_progress(
        self,
        original_positions,
        positions=None,
        init=False,
        iteration=0,
        intersected_polygons=None,
        boundary_polygon=None,
        **fig_args,
    ):
        if init is True:
            ecc_lim_mm = self.construct_retina.ecc_lim_mm
            polar_lim_deg = self.construct_retina.polar_lim_deg

            # Init plotting
            # Convert self.polar_lim_deg to Cartesian coordinates
            pol2cart = self.construct_retina.pol2cart

            bottom_x, bottom_y = pol2cart(
                np.array([ecc_lim_mm[0], ecc_lim_mm[1]]),
                np.array([polar_lim_deg[0], polar_lim_deg[0]]),
            )
            top_x, top_y = pol2cart(
                np.array([ecc_lim_mm[0], ecc_lim_mm[1]]),
                np.array([polar_lim_deg[1], polar_lim_deg[1]]),
            )

            # Concatenate to get the corner points
            corners_x = np.concatenate([bottom_x, top_x])
            corners_y = np.concatenate([bottom_y, top_y])

            # Initialize the plot before the loop
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.scatter(corners_x, corners_y, color="black", marker="x", zorder=2)
            ax2.scatter(corners_x, corners_y, color="black", marker="x", zorder=2)

            ax1.set_aspect("equal")
            ax2.set_aspect("equal")
            scatter1 = ax1.scatter([], [], color="blue", marker="o")
            scatter2 = ax2.scatter([], [], color="red", marker="o")

            # Obtain corners based on original_positions
            min_x = np.min(original_positions[:, 0]) - 0.1
            max_x = np.max(original_positions[:, 0]) + 0.1
            min_y = np.min(original_positions[:, 1]) - 0.1
            max_y = np.max(original_positions[:, 1]) + 0.1

            # Set axis limits based on min and max values of original_positions
            ax1.set_xlim(min_x, max_x)
            ax1.set_ylim(min_y, max_y)
            ax2.set_xlim(min_x, max_x)
            ax2.set_ylim(min_y, max_y)

            # set horizontal (x) and vertical (y) units as mm for both plots
            ax1.set_xlabel("horizontal (mm)")
            ax1.set_ylabel("vertical (mm)")
            ax2.set_xlabel("horizontal (mm)")
            ax2.set_ylabel("vertical (mm)")

            plt.ion()  # Turn on interactive mode
            plt.show()

            return {
                "fig": fig,
                "ax1": ax1,
                "ax2": ax2,
                "scatter1": scatter1,
                "scatter2": scatter2,
                "intersected_voronoi_polygons": [],
            }

        else:
            fig = fig_args["fig"]
            ax1 = fig_args["ax1"]
            ax2 = fig_args["ax2"]
            scatter1 = fig_args["scatter1"]
            scatter2 = fig_args["scatter2"]

            scatter1.set_offsets(original_positions)
            ax1.set_title(f"orig pos")

            scatter2.set_offsets(positions)
            ax2.set_title(f"new pos iteration {iteration}")

            # Draw boundary polygon with no fill
            if boundary_polygon is not None:
                polygon = Polygon(
                    boundary_polygon, closed=True, fill=None, edgecolor="r"
                )
                ax2.add_patch(polygon)

            if intersected_polygons is not None:
                if fig_args["intersected_voronoi_polygons"] is not None:
                    # Remove old polygons
                    for poly in fig_args["intersected_voronoi_polygons"]:
                        poly.remove()
                    fig_args["intersected_voronoi_polygons"].clear()

                # Plot intersected Voronoi polygons
                for polygon in intersected_polygons:
                    poly = ax2.fill(*zip(*polygon), alpha=0.4, edgecolor="black")
                    fig_args["intersected_voronoi_polygons"].extend(poly)
            # Update the plot
            fig.canvas.flush_events()

    # WorkingRetina visualization
    def show_stimulus_with_gcs(
        self,
        frame_number=0,
        ax=None,
        example_gc=5,
        show_rf_id=False,
        savefigname=None,
    ):
        """
        Plots the 1SD ellipses of the RGC mosaic. This method is a WorkingRetina call.

        Parameters
        ----------
        retina : object
            The retina object that contains all the relevant information about the stimulus video and ganglion cells.
        frame_number : int, optional
            The index of the frame from the stimulus video to be displayed. Default is 0.
        ax : matplotlib.axes.Axes, optional
            The axes object to draw the plot on. If None, the current axes is used. Default is None.
        example_gc : int, optional
            The index of the ganglion cell to be highlighted. Default is 5.
        show_rf_id : bool, optional
            If True, the index of each ganglion cell will be printed at the center of its ellipse. Default is False..
        """
        stim_to_show = self.project_data.working_retina["stim_to_show"]

        stimulus_video = stim_to_show["stimulus_video"]
        gc_df_stimpix = stim_to_show["gc_df_stimpix"]
        stimulus_height_pix = stim_to_show["stimulus_height_pix"]
        pix_per_deg = stim_to_show["pix_per_deg"]
        deg_per_mm = stim_to_show["deg_per_mm"]
        stimulus_center = stim_to_show["stimulus_center"]

        DoG_model = self.context.my_retina["DoG_model"]

        fig = plt.figure()
        ax = ax or plt.gca()
        ax.imshow(stimulus_video.frames[:, :, frame_number], vmin=0, vmax=255)
        ax = plt.gca()

        gc_rot_deg = gc_df_stimpix["orient_cen_rad"] * (-1) * 180 / np.pi

        for index, gc in gc_df_stimpix.iterrows():
            if index == example_gc:
                facecolor = "yellow"
            else:
                facecolor = "None"

            if DoG_model in ["ellipse_independent", "ellipse_fixed"]:
                circ = Ellipse(
                    (gc.q_pix, gc.r_pix),
                    width=2 * gc.semi_xc,
                    height=2 * gc.semi_yc,
                    angle=gc_rot_deg[index],  # Rotation in degrees anti-clockwise.
                    edgecolor="blue",
                    facecolor=facecolor,
                )
            elif DoG_model == "circular":
                circ = Ellipse(
                    (gc.q_pix, gc.r_pix),
                    width=2 * gc.rad_c,
                    height=2 * gc.rad_c,
                    angle=gc_rot_deg[index],  # Rotation in degrees anti-clockwise.
                    edgecolor="blue",
                    facecolor=facecolor,
                )

            ax.add_patch(circ)

            # If show_rf_id is True, annotate each ellipse with the index
            if show_rf_id:
                ax.annotate(
                    str(index),
                    (gc.q_pix, gc.r_pix),
                    color="black",
                    weight="bold",
                    fontsize=8,
                    ha="center",
                    va="center",
                )

        locs, labels = plt.yticks()

        locs = locs[locs < stimulus_height_pix]
        locs = locs[locs > 0]

        left_y_labels = locs.astype(int)
        plt.yticks(ticks=locs)
        ax.set_ylabel("pix")

        xlocs = locs - np.mean(locs)
        down_x_labels = np.round(xlocs / pix_per_deg, decimals=2) + np.real(
            stimulus_center
        )
        plt.xticks(ticks=locs, labels=down_x_labels)
        ax.set_xlabel("deg")

        ax2 = ax.twinx()
        ax2.tick_params(axis="y")
        right_y_labels = np.round((locs / pix_per_deg) / deg_per_mm, decimals=2)
        plt.yticks(ticks=locs, labels=right_y_labels)
        ax2.set_ylabel("mm")

        fig.tight_layout()

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_single_gc_view(
        self, cell_index, frame_number=0, ax=None, savefigname=None
    ):
        """
        Overlays the receptive field center of the specified retinal ganglion cell (RGC) on top of
        a given stimulus frame. whoch is cropped around the RGC.

        Parameters
        ----------
        cell_index : int
            Index of the RGC for which the view is to be shown.
        frame_number : int, optional
            Frame number of the stimulus to display. Default is 0.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If not provided, uses the current axis.
        """

        stim_to_show = self.project_data.working_retina["stim_to_show"]
        stimulus_video = stim_to_show["stimulus_video"]
        gc_df_stimpix = stim_to_show["gc_df_stimpix"]
        qmin_all, qmax_all, rmin_all, rmax_all = stim_to_show["qr_min_max"]
        qmin = qmin_all[cell_index]
        qmax = qmax_all[cell_index]
        rmin = rmin_all[cell_index]
        rmax = rmax_all[cell_index]

        if ax is None:
            fig, ax = plt.subplots()

        gc = gc_df_stimpix.iloc[cell_index]

        # Show stimulus frame cropped to RGC surroundings & overlay 1SD center RF on top of that
        ax.imshow(
            stimulus_video.frames[:, :, frame_number],
            cmap=self.cmap_stim,
            vmin=0,
            vmax=255,
        )
        ax.set_xlim([qmin, qmax])
        ax.set_ylim([rmax, rmin])

        # When in pixel coordinates, positive value in Ellipse angle is clockwise. Thus minus here.
        # Note that Ellipse angle is in degrees.
        # Width and height in Ellipse are diameters, thus x2.
        circ = Ellipse(
            (gc.q_pix, gc.r_pix),
            width=2 * gc.semi_xc,
            height=2 * gc.semi_yc,
            angle=gc.orient_cen_rad * (-1),  # Rotation in degrees anti-clockwise.
            edgecolor="white",
            facecolor="yellow",
        )
        ax.add_patch(circ)
        plt.xticks([])
        plt.yticks([])

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_temporal_kernel_frequency_response(
        self, cell_index=0, ax=None, savefigname=None
    ):
        """
        Plot the frequency response of the temporal kernel for a specified or all retinal ganglion cells (RGCs).

        Parameters
        ----------
        cell_index : int, optional
            Index of the RGC for which the frequency response is to be shown.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If not provided, uses the current axis.
        """
        spat_temp_filter_to_show = self.project_data.working_retina[
            "spat_temp_filter_to_show"
        ]
        temporal_filters = spat_temp_filter_to_show["temporal_filters"]
        data_filter_duration = spat_temp_filter_to_show["data_filter_duration"]

        tf = temporal_filters[cell_index, :]

        if ax is None:
            fig, ax = plt.subplots()

        ft_tf = np.fft.fft(tf)
        timestep = data_filter_duration / len(tf) / 1000  # in seconds
        freqs = np.fft.fftfreq(tf.size, d=timestep)
        ampl_s = np.abs(ft_tf)

        ax.set_xscale("log")
        ax.set_xlim([0.1, 100])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain")
        ax.plot(freqs, ampl_s, ".")

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def plot_midpoint_contrast(self, cell_index=0, ax=None, savefigname=None):
        """
        Plot the contrast at the midpoint pixel of the stimulus cropped to a specified RGC's surroundings.

        Parameters
        ----------
        cell_index : int, optional
            Index of the RGC for which to plot the contrast. Default is 0.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If not provided, uses the current axis.

        """
        stim_to_show = self.project_data.working_retina["stim_to_show"]
        spatial_filter_sidelen = stim_to_show["spatial_filter_sidelen"]
        stimulus_video = stim_to_show["stimulus_video"]
        stimulus_cropped_all = stim_to_show["stimulus_cropped"]
        stimulus_cropped = stimulus_cropped_all[cell_index]

        midpoint_ix = (spatial_filter_sidelen - 1) // 2
        signal = stimulus_cropped[midpoint_ix, midpoint_ix, :]

        video_dt = (1 / stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(tvec, signal)
        ax.set_ylim([-1, 1])

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def plot_local_rms_contrast(self, cell_index=0, ax=None, savefigname=None):
        """
        Plot the local RMS contrast for the stimulus cropped to a specified RGC's surroundings.

        Parameters
        ----------
        cell_index : int, optional
            Index of the RGC for which to plot the local RMS contrast. Default is 0.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If not provided, uses the current axis.
        """
        stim_to_show = self.project_data.working_retina["stim_to_show"]
        stimulus_cropped_all = stim_to_show["stimulus_cropped"]
        stimulus_cropped = stimulus_cropped_all[cell_index]
        stimulus_video = stim_to_show["stimulus_video"]
        spatial_filter_sidelen = stim_to_show["spatial_filter_sidelen"]
        # Invert from Weber contrast
        stimulus_cropped = 127.5 * (stimulus_cropped + 1.0)

        n_frames = stimulus_video.video_n_frames
        sidelen = spatial_filter_sidelen
        signal = np.zeros(n_frames)

        for t in range(n_frames):
            frame_mean = np.mean(stimulus_cropped[:, :, t])
            squared_sum = np.sum((stimulus_cropped[:, :, t] - frame_mean) ** 2)
            signal[t] = np.sqrt(1 / (frame_mean**2 * sidelen**2) * squared_sum)

        video_dt = (1 / stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(tvec, signal)
        ax.set_ylim([0, 1])

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def plot_local_michelson_contrast(self, cell_index=0, ax=None, savefigname=None):
        """
        Plot the local Michelson contrast for the stimulus cropped to a specified RGC's surroundings.

        Parameters
        ----------
        cell_index : int, optional
            Index of the RGC for which to plot the local Michelson contrast. Default is 0.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If not provided, uses the current axis.
        """
        stim_to_show = self.project_data.working_retina["stim_to_show"]
        stimulus_cropped_all = stim_to_show["stimulus_cropped"]
        stimulus_cropped = stimulus_cropped_all[cell_index]
        stimulus_video = stim_to_show["stimulus_video"]

        # Invert from Weber contrast
        stimulus_cropped = 127.5 * (stimulus_cropped + 1.0)

        n_frames = stimulus_video.video_n_frames
        signal = np.zeros(n_frames)

        # unsigned int will overflow when frame_max + frame_min = 256
        stimulus_cropped = stimulus_cropped.astype(np.uint16)
        for t in range(n_frames):
            frame_min = np.min(stimulus_cropped[:, :, t])
            frame_max = np.max(stimulus_cropped[:, :, t])
            signal[t] = (frame_max - frame_min) / (frame_max + frame_min)

        video_dt = (1 / stimulus_video.fps) * b2u.second
        tvec = np.arange(0, len(signal)) * video_dt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(tvec, signal)
        ax.set_ylim([0, 1])

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_all_gc_responses(self, savefigname=None):
        """
        Visualize ganglion cell (gc) responses based on the data in the WorkingRetina object.

        Parameters
        ----------
        savefigname : str, optional
            The name of the file where the figure will be saved. If None, the figure is not saved.

        Attributes Accessed
        --------------------
        project_data.working_retina : dict
            Dictionary attached to ProjectData class instance containing the gc responses
            and other information to show.
        """
        gc_responses_to_show = self.project_data.working_retina["gc_responses_to_show"]
        n_trials = gc_responses_to_show["n_trials"]
        n_cells = gc_responses_to_show["n_cells"]
        all_spiketrains = gc_responses_to_show["all_spiketrains"]
        duration = gc_responses_to_show["duration"]
        generator_potential = gc_responses_to_show["generator_potential"]
        video_dt = gc_responses_to_show["video_dt"]
        tvec_new = gc_responses_to_show["tvec_new"]

        # Prepare data for manual visualization
        if n_trials > 1 and n_cells == 1:
            for_eventplot = all_spiketrains  # list of different leght arrays
            for_histogram = np.concatenate(all_spiketrains)
            for_generatorplot = generator_potential.flatten()
            n_samples = n_trials
            sample_name = "Trials"
        elif n_trials == 1 and n_cells > 1:
            for_eventplot = all_spiketrains
            for_histogram = np.concatenate(all_spiketrains)
            for_generatorplot = np.nanmean(generator_potential, axis=0)
            n_samples = n_cells
            sample_name = "Cell #"
        else:
            raise ValueError(
                """You attempted to visualize gc activity, but either n_trials or n_cells must be 1, 
                and the other > 1"""
            )

        # Create subplots
        fig, ax = plt.subplots(2, 1, sharex=True)

        # Event plot on first subplot
        ax[0].eventplot(for_eventplot)
        ax[0].set_xlim([0, duration / b2u.second])
        ax[0].set_ylabel(sample_name)

        # Generator potential and average firing rate on second subplot
        tvec = np.arange(0, generator_potential.shape[-1], 1) * video_dt
        ax[1].plot(tvec, for_generatorplot, label="Generator")
        ax[1].set_xlim([0, duration / b2u.second])

        # Given bin_width in ms, convert it to the correct unit
        bin_width = 10 * b2u.ms

        # Find the nearest integer number of simulation_dt units for hist_dt
        simulation_dt = self.context.my_run_options["simulation_dt"] * b2u.second
        hist_dt = np.round(bin_width / simulation_dt) * simulation_dt

        # Update bin_edges based on the new hist_dt
        num_bins = int(np.ceil(duration / hist_dt))
        bin_edges = np.linspace(0, duration / b2u.second, num_bins + 1)

        # Compute histogram with the new hist_dt
        hist, _ = np.histogram(for_histogram, bins=bin_edges)

        # Update average firing rate calculation based on the new hist_dt
        avg_fr = hist / n_samples / (hist_dt / b2u.second)

        # # Smoothing remains the same
        # xsmooth = np.arange(-3, 3 + 1)
        # smoothing = stats.norm.pdf(xsmooth, scale=1)
        # smoothed_avg_fr = np.convolve(smoothing, avg_fr, mode="same")

        # ax[1].plot(bin_edges[:-1], smoothed_avg_fr, label="Measured")
        ax[1].plot(bin_edges[:-1], avg_fr, label="Measured")

        ax[1].set_ylabel("Firing rate (Hz)")
        ax[1].set_xlabel("Time (s)")
        ax[1].legend()

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_spatiotemporal_filter(self, cell_index=0, savefigname=None):
        """
        Display the spatiotemporal filter for a given cell in the retina.

        This method retrieves the specified cell's spatial and temporal filters
        from the 'working_retina' attribute of the 'project_data' object.

        Parameters
        ----------
        cell_index : int, optional
            Index of the cell for which the spatiotemporal filter is to be shown.
        savefigname : str or None, optional
            If a string is provided, the figure will be saved with this filename.
        """
        spat_temp_filter_to_show = self.project_data.working_retina[
            "spat_temp_filter_to_show"
        ]
        spatial_filters = spat_temp_filter_to_show["spatial_filters"]
        temporal_filters = spat_temp_filter_to_show["temporal_filters"]
        gc_type = spat_temp_filter_to_show["gc_type"]
        response_type = spat_temp_filter_to_show["response_type"]
        temporal_filter_len = spat_temp_filter_to_show["temporal_filter_len"]
        spatial_filter_sidelen = spat_temp_filter_to_show["spatial_filter_sidelen"]

        temporal_filter = temporal_filters[cell_index, :]
        spatial_filter = spatial_filters[cell_index, :]
        spatial_filter = spatial_filter.reshape(
            (spatial_filter_sidelen, spatial_filter_sidelen)
        )

        vmax = np.max(np.abs(spatial_filter))
        vmin = -vmax

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        plt.suptitle(gc_type + " " + response_type + " / cell ix " + str(cell_index))
        plt.subplot(121)
        im = ax[0].imshow(
            spatial_filter, cmap=self.cmap_spatial_filter, vmin=vmin, vmax=vmax
        )
        ax[0].grid(True)
        plt.colorbar(im, ax=ax[0])

        plt.subplot(122)
        if self.context.my_retina["temporal_model"] == "dynamic":
            # Print text to middle of ax[1]: "No fixed temporal filter for dynamic temporal model"
            ax[1].text(
                0.5,
                0.5,
                "No fixed temporal filter for dynamic temporal model",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
            )            
            
        else:
            ax[1].plot(range(temporal_filter_len), np.flip(temporal_filter))

        plt.tight_layout()

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    def show_impulse_response(self, savefigname=None):
        viz_dict = self.project_data.working_retina["impulse_to_show"]

        tvec = viz_dict["tvec"]  # in seconds
        svec = viz_dict["svec"]

        contrasts = viz_dict["contrasts"]  # contrasts_for_impulse
        yvecs = viz_dict["impulse_responses"]  # yvecs
        start_delay = viz_dict["start_delay"]  # in milliseconds

        tvec = tvec * 1000  # convert to milliseconds
        tvec = tvec - start_delay  # shift to start at 0

        cell_index = viz_dict["Unit idx"]
        ylims = np.array([np.min(yvecs), np.max(yvecs)])

        plt.figure()

        for u_idx, this_unit in enumerate(cell_index):
            for c_idx, this_contrast in enumerate(contrasts):
                if len(contrasts) > 1:
                    label = f"Unit {this_unit}, contrast {this_contrast}"
                else:
                    label = f"Unit {this_unit}"
                plt.plot(
                    tvec[:-1],
                    yvecs[u_idx, c_idx, :-1],
                    label=label,
                )

        # Set vertical dashed line at max (svec) time point, i.e. at the impulse time
        plt.axvline(x=tvec[np.argmax(np.abs(svec))], color="k", linestyle="--")
        plt.legend()
        plt.ylim(ylims[0] * 1.1, ylims[1] * 1.1)

        gc_type = viz_dict["gc_type"]
        response_type = viz_dict["response_type"]
        temporal_model = viz_dict["temporal_model"]

        plt.title(
            f"{gc_type} {response_type} ({temporal_model} model) impulse response(s)"
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Normalized response")
        # Put grid on
        plt.grid(True)

        if savefigname is not None:
            self._figsave(figurename=savefigname)

    # PhotoReceptor visualization
    def show_cone_response(self, image, image_after_optics, cone_response):
        """
        PhotoReceptor call.
        """
        fig, ax = plt.subplots(nrows=2, ncols=3)
        axs = ax.ravel()
        axs[0].hist(image.flatten(), 20)
        axs[1].hist(image_after_optics.flatten(), 20)
        axs[2].hist(cone_response.flatten(), 20)

        axs[3].imshow(image, cmap="Greys")
        axs[4].imshow(image_after_optics, cmap="Greys")
        axs[5].imshow(cone_response, cmap="Greys")

    def plot_analog_stimulus(self, analog_input):
        data = analog_input.Input

        plt.figure()
        plt.plot(data.T)

    def show_retina_img(self, savefigname=None):
        """
        Show the image of whole retina with all the receptive fields summed up.
        """

        gen_ret = self.project_data.construct_retina["gen_ret"]

        img_ret = gen_ret["img_ret"]
        img_ret_masked = gen_ret["img_ret_masked"]
        img_ret_adjusted = gen_ret["img_ret_adjusted"]

        plt.figure()
        plt.subplot(221)
        plt.imshow(img_ret, cmap="gray")
        plt.colorbar()
        plt.title("Original coverage")
        plt.subplot(222)

        cmap = plt.cm.get_cmap("viridis")
        custom_cmap = mcolors.ListedColormap(
            cmap(np.linspace(0, 1, int(np.max(img_ret_masked)) + 1))
        )
        plt.imshow(img_ret_masked, cmap=custom_cmap)
        plt.colorbar()
        plt.title("Summed masks")

        plt.subplot(223)
        plt.imshow(img_ret_adjusted, cmap="gray")
        plt.colorbar()
        plt.title("Adjusted coverage")

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_rf_imgs(self, n_samples=10, savefigname=None):
        """
        Show the individual RFs of the VAE retina

        img_rf: (n_cells, n_pix, n_pix)
        img_rf_mask: (n_cells, n_pix, n_pix)
        img_rfs_adjusted: (n_cells, n_pix, n_pix)
        """

        gen_rfs = self.project_data.construct_retina["gen_rfs"]

        img_rf = gen_rfs["img_rf"]
        img_rf_mask = gen_rfs["img_rf_mask"]
        img_rfs_adjusted = gen_rfs["img_rfs_adjusted"]

        fig, axs = plt.subplots(3, n_samples, figsize=(n_samples, 3))
        samples = np.random.choice(img_rf.shape[0], n_samples, replace=False)
        for i, sample in enumerate(samples):
            axs[0, i].imshow(img_rf[sample], cmap="gray")
            axs[0, i].axis("off")
            axs[0, i].set_title("Cell " + str(sample))

            axs[1, i].imshow(img_rf_mask[sample], cmap="gray")
            axs[1, i].axis("off")

            axs[2, i].imshow(img_rfs_adjusted[sample], cmap="gray")
            axs[2, i].axis("off")

        # On the left side of the first axis of each row, set text labels.
        axs[0, 0].set_ylabel("RF")
        axs[0, 0].axis("on")
        axs[1, 0].set_ylabel("Mask")
        axs[1, 0].axis("on")
        axs[2, 0].set_ylabel("Adjusted RF")
        axs[2, 0].axis("on")

        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])

        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])

        axs[2, 0].set_xticks([])
        axs[2, 0].set_yticks([])

        # # Adjust the layout so labels are visible
        # fig.subplots_adjust(left=0.15)
        if savefigname:
            self._figsave(figurename=savefigname)

    def show_rf_violinplot(self):
        """
        Show the individual RFs of the VAE retina

        img_rf: (n_cells, n_pix, n_pix)
        img_rfs_adjusted: (n_cells, n_pix, n_pix)
        """

        gen_rfs = self.project_data.construct_retina["gen_rfs"]

        img_rf = gen_rfs["img_rf"]
        img_rfs_adjusted = gen_rfs["img_rfs_adjusted"]

        fig, axs = plt.subplots(
            2, 1, figsize=(10, 10)
        )  # I assume you want a bigger figure size.

        # reshape and transpose arrays so that we have one row per cell
        df_rf = pd.DataFrame(img_rf.reshape(img_rf.shape[0], -1).T)
        df_pruned = pd.DataFrame(
            img_rfs_adjusted.reshape(img_rfs_adjusted.shape[0], -1).T
        )

        # Show seaborn boxplot with RF values, one box for each cell
        sns.violinplot(data=df_rf, ax=axs[0])
        axs[0].set_title("RF values")
        # Put grid on
        axs[0].grid(True)
        # ...and RF_adjusted values
        sns.violinplot(data=df_pruned, ax=axs[1])
        axs[1].set_title("RF adjusted values")
        axs[1].grid(True)

    # Results visualization
    def _string_on_plot(
        self, ax, variable_name=None, variable_value=None, variable_unit=None
    ):
        plot_str = f"{variable_name} = {variable_value:6.2f} {variable_unit}"
        ax.text(
            0.05,
            0.95,
            plot_str,
            fontsize=8,
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="Square,pad=0.2", fc="white", ec="white", lw=1),
        )

    def fr_response(self, exp_variables, xlog=False, ylog=False, savefigname=None):
        """
        Plot the mean firing rate response curve.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)
        assert (
            len(exp_variables) == 1
        ), "Only one variable can be plotted at a time, aborting..."

        experiment_df = pd.read_csv(
            data_folder / f"exp_metadata_{cond_names_string}.csv", index_col=0
        )
        data_df = pd.read_csv(
            data_folder / f"{cond_names_string}_population_means.csv", index_col=0
        )
        data_df_units = pd.read_csv(
            data_folder / f"{cond_names_string}_unit_means.csv", index_col=0
        )

        response_levels_s = experiment_df.loc["contrast", :]
        mean_response_levels_s = data_df.mean()
        response_levels_s = pd.to_numeric(response_levels_s)
        response_levels_s = response_levels_s.round(decimals=2)

        response_function_df = pd.DataFrame(
            {cond_names_string: response_levels_s, "response": mean_response_levels_s}
        )

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        sns.lineplot(
            data=response_function_df,
            x=cond_names_string,
            y="response",
            marker="o",
            color="black",
            ax=ax[0],
        )

        # Title
        ax[0].set_title(f"{cond_names_string} response function (population mean)")

        if xlog:
            ax[0].set_xscale("log")
        if ylog:
            ax[0].set_yscale("log")

        sns.boxplot(data=data_df_units, color="white", linewidth=2, whis=100, ax=ax[1])
        sns.swarmplot(data=data_df_units, color="black", size=3, ax=ax[1])

        # Title
        ax[1].set_title(f"{cond_names_string} response function (individual units)")

        if savefigname:
            self._figsave(figurename=savefigname)

    def F1F2_popul_response(
        self, exp_variables, xlog=False, ylog=False, savefigname=None
    ):
        """
        Plot oF1 and  F2 frequency response curves for all conditions.
        Population response, i.e. mean across units.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)
        n_variables = len(exp_variables)

        experiment_df = pd.read_csv(
            data_folder / f"exp_metadata_{cond_names_string}.csv", index_col=0
        )
        F_popul_df = pd.read_csv(
            data_folder / f"{cond_names_string}_F1F2_population_means.csv", index_col=0
        )

        F_popul_long_df = pd.melt(
            F_popul_df,
            id_vars=["trial", "F_peak"],
            value_vars=F_popul_df.columns[:-2],
            var_name=f"{cond_names_string}_names",
            value_name="amplitudes",
        )

        # Make new columns with conditions' levels
        for cond in exp_variables:
            levels_s = experiment_df.loc[cond, :]
            levels_s = pd.to_numeric(levels_s)
            levels_s = levels_s.round(decimals=2)
            F_popul_long_df[cond] = F_popul_long_df[f"{cond_names_string}_names"].map(
                levels_s
            )

        fig, ax = plt.subplots(1, n_variables, figsize=(8, 4))

        if n_variables == 1:
            sns.lineplot(
                data=F_popul_long_df,
                x=exp_variables[0],
                y="amplitudes",
                hue="F_peak",
                palette="tab10",
                ax=ax,
            )
            ax.set_title("Population amplitude spectra for " + exp_variables[0])
            if xlog:
                ax.set_xscale("log")
            if ylog:
                ax.set_yscale("log")

        else:
            for i, cond in enumerate(exp_variables):
                sns.lineplot(
                    data=F_popul_long_df,
                    x=cond,
                    y="amplitudes",
                    hue="F_peak",
                    palette="tab10",
                    ax=ax[i],
                )

                # Title
                ax[i].set_title("Population amplitude spectra for " + cond)
                if xlog:
                    ax[i].set_xscale("log")
                if ylog:
                    ax[i].set_yscale("log")

        if savefigname:
            self._figsave(figurename=savefigname)

    def F1F2_unit_response(
        self, exp_variables, xlog=False, ylog=False, savefigname=None
    ):
        """
        Plot F1 and  F2 frequency response curves for all conditions.
        Unit response, i.e. mean across trials.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)
        n_variables = len(exp_variables)

        experiment_df = pd.read_csv(
            data_folder / f"exp_metadata_{cond_names_string}.csv", index_col=0
        )

        F_unit_ampl_df = pd.read_csv(
            data_folder / f"{cond_names_string}_F1F2_unit_ampl_means.csv", index_col=0
        )
        F_unit_long_df = pd.melt(
            F_unit_ampl_df,
            id_vars=["unit", "F_peak"],
            value_vars=F_unit_ampl_df.columns[:-2],
            var_name=f"{cond_names_string}_names",
            value_name="amplitudes",
        )

        # Make new columns with conditions' levels
        for cond in exp_variables:
            levels_s = experiment_df.loc[cond, :]
            levels_s = pd.to_numeric(levels_s)
            levels_s = levels_s.round(decimals=2)
            F_unit_long_df[cond] = F_unit_long_df[f"{cond_names_string}_names"].map(
                levels_s
            )

        fig, ax = plt.subplots(1, n_variables, figsize=(8, 4))

        if n_variables == 1:
            sns.lineplot(
                data=F_unit_long_df,
                x=exp_variables[0],
                y="amplitudes",
                hue="F_peak",
                palette="tab10",
                ax=ax,
            )
            ax.set_title("Unit amplitude spectra for " + exp_variables[0])
            if xlog:
                ax.set_xscale("log")
            if ylog:
                ax.set_yscale("log")

        else:
            for i, cond in enumerate(exp_variables):
                sns.lineplot(
                    data=F_unit_long_df,
                    x=cond,
                    y="amplitudes",
                    hue="F_peak",
                    palette="tab10",
                    ax=ax[i],
                )

                # Title
                ax[i].set_title("Unit amplitude spectra for " + cond)
                if xlog:
                    ax[i].set_xscale("log")
                if ylog:
                    ax[i].set_yscale("log")

        if savefigname:
            self._figsave(figurename=savefigname)

    def ptp_response(self, exp_variables, x_of_interest=None, savefigname=None):
        """
        Plot the peak-to-peak firing rate magnitudes across conditions.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)
        assert (
            len(exp_variables) == 1
        ), "Only one variable can be plotted at a time, aborting..."

        experiment_df = pd.read_csv(
            data_folder / f"exp_metadata_{cond_names_string}.csv", index_col=0
        )
        data_df = pd.read_csv(
            data_folder / f"{cond_names_string}_PTP_population_means.csv", index_col=0
        )

        if x_of_interest is None:
            data_df_selected = data_df
        else:
            data_df_selected = data_df.loc[:, x_of_interest]

        # Turn series into array of values
        x_values_df = experiment_df.loc[exp_variables, :]
        x_values_series = x_values_df.iloc[0, :]
        x_values_series = pd.to_numeric(x_values_series)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        plt.subplot(121)
        plt.plot(
            x_values_series.values,
            data_df.mean().values,
            color="black",
        )

        sns.boxplot(
            data=data_df_selected,
            color="black",
            ax=ax[1],
        )

        # Title
        ax[0].set_title(f"{cond_names_string} ptp (population mean)")
        ax[1].set_title(f"{cond_names_string} ptp at two peaks and a through")

        if savefigname:
            self._figsave(figurename=savefigname)

    def spike_raster_response(self, exp_variables, trial=0, savefigname=None):
        """
        Show spikes from a results file.

        Parameters
        ----------
        results_filename : str or None
            This name is searched from current directory, input_folder and output_folder (defined in config file). If None, the latest results file is used.
        savefigname : str or None
            If not empty, the figure is saved to this filename.

        Returns
        -------
        None
        """
        cond_names_string = "_".join(exp_variables)
        filename = f"exp_metadata_{cond_names_string}.csv"
        experiment_df = self.data_io.get_data(filename=filename)
        cond_names = experiment_df.columns.values
        n_trials_vec = pd.to_numeric(experiment_df.loc["n_trials", :].values)
        assert trial < np.min(n_trials_vec), "Trial id too high, aborting..."

        # Visualize

        fig, ax = plt.subplots(len(experiment_df.columns), 1, figsize=(8, 4))

        # Loop conditions
        for idx, cond_name in enumerate(cond_names):
            filename = f"Response_{cond_name}.gz"
            data_dict = self.data_io.get_data(filename)

            cond_s = experiment_df[cond_name]
            duration_seconds = pd.to_numeric(cond_s.loc["duration_seconds"])
            baseline_start_seconds = pd.to_numeric(cond_s.loc["baseline_start_seconds"])
            baseline_end_seconds = pd.to_numeric(cond_s.loc["baseline_end_seconds"])
            duration_tot = (
                duration_seconds + baseline_start_seconds + baseline_end_seconds
            )

            units, times = self.ana._get_spikes_by_interval(
                data_dict, trial, 0, duration_tot
            )

            ax[idx].plot(
                times,
                units,
                ".",
            )

            this_contrast = pd.to_numeric(experiment_df.loc["contrast", cond_name])
            ax[idx].set_title(
                f"{cond_name}, contrast {this_contrast:.2f}",
                fontsize=10,
            )

            MeanFR = self.ana._analyze_meanfr(data_dict, trial, 0, duration_tot)
            self._string_on_plot(
                ax[idx],
                variable_name="Mean FR",
                variable_value=MeanFR,
                variable_unit="Hz",
            )

        if savefigname:
            self._figsave(figurename=savefigname)

    def tf_vs_fr_cg(
        self, exp_variables, n_contrasts=None, xlog=False, ylog=False, savefigname=None
    ):
        """
        Plot F1 frequency response curves for 2D frequency-contrast experiment.
        Unit response, i.e. mean across trials.
        Subplot 1: temporal frequency vs firing rate at n_contrasts
        Subplot 2: temporal frequency vs contrast gain (cg) at n_contrasts. Contrast gain is defined as the F1 response divided by contrast.

        Parameters
        ----------
        exp_variables : list of str
            List of experiment variables to be plotted.
        n_contrasts : int
            Number of contrasts to be plotted. If None, all contrasts are plotted.
        xlog : bool
            If True, x-axis is logarithmic.
        ylog : bool
            If True, y-axis is logarithmic.
        """

        data_folder = self.context.output_folder
        cond_names_string = "_".join(exp_variables)

        # Experiment metadata
        experiment_df = pd.read_csv(
            data_folder / f"exp_metadata_{cond_names_string}.csv", index_col=0
        )

        # Results
        F_unit_ampl_df = pd.read_csv(
            data_folder / f"{cond_names_string}_F1F2_unit_ampl_means.csv", index_col=0
        )

        F_unit_long_df = pd.melt(
            F_unit_ampl_df,
            id_vars=["unit", "F_peak"],
            value_vars=F_unit_ampl_df.columns[:-2],
            var_name=f"{cond_names_string}_names",
            value_name="amplitudes",
        )

        # Make new columns with conditions' levels
        for cond in exp_variables:
            levels_s = experiment_df.loc[cond, :]
            levels_s = pd.to_numeric(levels_s)
            levels_s = levels_s.round(decimals=2)
            F_unit_long_df[cond] = F_unit_long_df[f"{cond_names_string}_names"].map(
                levels_s
            )

        # Make new columns cg and phase.
        F_unit_long_df["cg"] = F_unit_long_df["amplitudes"] / F_unit_long_df["contrast"]

        F_unit_long_df = F_unit_long_df[F_unit_long_df["F_peak"] == "F1"].reset_index(
            drop=True
        )

        # Select only the desired number of contrasts at about even intervals, including the lowest and the highest contrast
        if n_contrasts:
            contrasts = F_unit_long_df["contrast"].unique()
            contrasts.sort()
            contrasts = contrasts[:: int(len(contrasts) / n_contrasts)]
            contrasts = np.append(contrasts, F_unit_long_df["contrast"].max())
            F_unit_long_df = F_unit_long_df.loc[
                F_unit_long_df["contrast"].isin(contrasts)
            ]

        fig, ax = plt.subplots(2, 1, figsize=(8, 12))

        # Make the three subplots using seaborn lineplot
        sns.lineplot(
            data=F_unit_long_df,
            x="temporal_frequency",
            y="amplitudes",
            hue="contrast",
            palette="tab10",
            ax=ax[0],
        )
        ax[0].set_title("Firing rate vs temporal frequency")
        if xlog:
            ax[0].set_xscale("log")
        if ylog:
            ax[0].set_yscale("log")

        sns.lineplot(
            data=F_unit_long_df,
            x="temporal_frequency",
            y="cg",
            hue="contrast",
            palette="tab10",
            ax=ax[1],
        )
        ax[1].set_title("Contrast gain vs temporal frequency")
        if xlog:
            ax[1].set_xscale("log")
        if ylog:
            ax[1].set_yscale("log")

        if savefigname:
            self._figsave(figurename=savefigname)

    # Validation viz
    def _build_param_plot(self, coll_ana_df_in, param_plot_dict, to_spa_dict_in):
        """
        Prepare for parametric plotting of multiple conditions.

        Parameters
        ----------
        coll_ana_df_in : pandas.DataFrame
            Mapping from to_spa_dict in conf file to dataframes which include parameter and analysis details.
        param_plot_dict : dict
            Dictionary guiding the parametric plot. See :func:`show_catplot` for details.
        to_spa_dict_in : dict
            Dictionary containing the startpoints, parameters and analyzes which are active in conf file.

        Returns
        -------
        data_list : list
            A nested list of data for each combination of outer and inner conditions.
        data_name_list : list
            A nested list of names for each combination of outer and inner conditions.
        outer_name_list : list
            A list of names for the outer conditions.
        """

        to_spa_dict = to_spa_dict_in
        coll_ana_df = coll_ana_df_in

        [title] = to_spa_dict[param_plot_dict["title"]]
        outer_list = to_spa_dict[param_plot_dict["outer"]]
        inner_list = to_spa_dict[param_plot_dict["inner"]]

        # if paths to data provided, take inner names from distinct list
        if param_plot_dict["inner_paths"] is True:
            inner_list = param_plot_dict["inner_path_names"]

        mid_idx = list(param_plot_dict.values()).index("startpoints")
        par_idx = list(param_plot_dict.values()).index("parameters")
        ana_idx = list(param_plot_dict.values()).index("analyzes")

        key_list = list(param_plot_dict.keys())

        # Create dict whose key is folder hierarchy and value is plot hierarchy
        hdict = {
            "mid": key_list[mid_idx],
            "par": key_list[par_idx],
            "ana": key_list[ana_idx],
        }

        data_list = []  # nested list, N items = N outer x N inner
        data_name_list = []  # nested list, N items = N outer x N inner
        outer_name_list = []  # list , N items = N outer

        for outer in outer_list:
            inner_data_list = []  # list , N items = N inner
            inner_name_list = []  # list , N items = N inner

            for in_idx, inner in enumerate(inner_list):
                # Nutcracker. eval to "outer", "inner" and "title"
                mid = eval(f"{hdict['mid']}")
                par = eval(f"{hdict['par']}")
                this_folder = f"{mid}_{par}"
                this_ana = eval(f"{hdict['ana']}")

                this_ana_col = coll_ana_df.loc[this_ana]["csv_col_name"]
                if param_plot_dict["compiled_results"] is True:
                    this_folder = f"{this_folder}_compiled_results"
                    this_ana_col = f"{this_ana_col}_mean"

                if param_plot_dict["inner_paths"] is True:
                    csv_path_tuple = param_plot_dict["paths"][in_idx]
                    csv_path = reduce(
                        lambda acc, y: Path(acc).joinpath(y), csv_path_tuple
                    )
                else:
                    csv_path = None

                # get data
                (
                    data0_df,
                    data_df_compiled,
                    independent_var_col_list,
                    dependent_var_col_list,
                    time_stamp,
                ) = self.data_io.get_csv_as_df(
                    folder_name=this_folder, csv_path=csv_path, include_only=None
                )

                df = data_df_compiled[this_ana_col]
                inner_data_list.append(df)
                inner_name_list.append(inner)

            data_list.append(inner_data_list)
            data_name_list.append(inner_name_list)
            outer_name_list.append(outer)

        return (
            data_list,
            data_name_list,
            outer_name_list,
        )

    def validate_gc_rf_size(self, savefigname=None):
        gen_rfs = self.project_data.construct_retina["gen_rfs"]

        if self.context.my_retina["spatial_model"] == "VAE":
            gen_rfs = gen_rfs
            img_rf = gen_rfs["img_rf"]

            new_um_per_pix = self.construct_retina.updated_vae_um_per_pix

            # Get ellipse FIT and VAE FIT values
            gc_df = self.construct_retina.gc_df_original
            gc_vae_df = self.construct_retina.gc_vae_df

            fit = self.construct_retina.Fit(
                self.context.apricot_metadata,
                self.construct_retina.gc_type,
                self.construct_retina.response_type,
                spatial_data=img_rf,
                fit_type="concentric_rings",
                new_um_per_pix=new_um_per_pix,
            )

            all_data_fits_df = fit.all_data_fits_df
            gen_spat_filt = fit.gen_spat_filt
            good_idx_rings = fit.good_idx_rings
        else:
            raise ValueError(
                "Only VAE spatial_model is supported for validate_gc_rf_size, it shows FIT values, too."
            )

        # cr for concentric rings, i.e. the symmetric DoG model
        # Scales pix to mm for semi_xc i.e. central radius for cd fits
        gc_vae_cr_df = self.construct_retina._update_gc_vae_df(
            all_data_fits_df, new_um_per_pix
        )

        # Center radius and eccentricity for cr
        deg_per_mm = self.construct_retina.deg_per_mm
        cen_mm_cr = gc_vae_cr_df["semi_xc"].values
        cen_deg_cr = cen_mm_cr * deg_per_mm
        cen_min_arc_cr = cen_deg_cr * 60
        ecc_mm_cr = self.construct_retina.gc_vae_df["pos_ecc_mm"].values
        ecc_deg_cr = ecc_mm_cr * deg_per_mm

        # Center radius and eccentricity for ellipse fit
        cen_mm_fit = np.sqrt(gc_df["semi_xc"].values * gc_df["semi_yc"].values)
        cen_deg_fit = cen_mm_fit * deg_per_mm
        cen_min_arc_fit = cen_deg_fit * 60
        ecc_mm_fit = gc_df["pos_ecc_mm"].values
        ecc_deg_fit = ecc_mm_fit * deg_per_mm

        # Center radius and eccentricity for vae ellipse fit
        cen_mm_vae = np.sqrt(gc_vae_df["semi_xc"].values * gc_vae_df["semi_yc"].values)
        cen_deg_vae = cen_mm_vae * deg_per_mm
        cen_min_arc_vae = cen_deg_vae * 60
        ecc_mm_vae = gc_vae_df["pos_ecc_mm"].values
        ecc_deg_vae = ecc_mm_vae * deg_per_mm

        # Read in corresponding data from literature
        spatial_DoG_fullpath = self.context.literature_data_files[
            "spatial_DoG_fullpath"
        ]
        spatial_DoG_data = self.data_io.get_data(spatial_DoG_fullpath)

        lit_ecc_deg = spatial_DoG_data["Xdata"]  # ecc (deg)
        lit_cen_min_arc = spatial_DoG_data["Ydata"]  # rf center radius (min of arc)

        # Plot the results
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(ecc_deg_fit, cen_min_arc_fit, "o", label="Ellipse fit")
        plt.plot(ecc_deg_vae, cen_min_arc_vae, "o", label="VAE ellipse fit")
        plt.plot(ecc_deg_cr, cen_min_arc_cr, "o", label="VAE concentric rings fit")
        plt.plot(lit_ecc_deg, lit_cen_min_arc, "o", label="Schottdorf_2021_JPhysiol")
        plt.xlabel("Eccentricity (deg)")
        plt.ylabel("Center radius (min of arc)")
        plt.legend()
        plt.title(
            f"GC dendritic diameter vs eccentricity, {self.construct_retina.gc_type} type"
        )

        if savefigname:
            self._figsave(figurename=savefigname)

    def show_catplot(self, param_plot_dict):
        """
        Visualization of parameter values in different categories. Data is collected in _build_param_plot, and all plotting is here.

        Definitions for parametric plotting of multiple conditions/categories.

        First, define what data is going to be visualized in to_spa_dict.
        Second, define how it is visualized in param_plot_dict.

        Limitations:
            You cannot have analyzes as title AND inner_sub = True.
            For violinplot and inner_sub = True, N bin edges MUST be two (split view)

        outer : panel (distinct subplots) # analyzes, startpoints, parameters, controls
        inner : inside one axis (subplot) # startpoints, parameters, controls

        The dictionary xy_plot_dict contains:

        title : str
            Title-level of plot, e.g. "parameters". Multiple allowed => each in separate figure
        outer : str
            Panel-level of plot, e.g. "analyzes". Multiple allowed => plt subplot panels
        inner : str
            Inside one axis (subplot) level of plot, e.g. "startpoints". Multiple allowed => direct comparison
        inner_sub : bool
            Further subdivision by value, such as mean firing rate
        inner_sub_ana : str
            Name of ana. This MUST be included into to_spa_dict "analyzes". E.g. "Excitatory Firing Rate"
        bin_edges : list of lists
            Binning of data. E.g. [[0.001, 150], [150, 300]]
        plot_type : str
            Parametric plot type. Allowed types include "box", "violin", "strip", "swarm", "boxen", "point" and "bar".
        compiled_results : bool
            Data at compiled_results folder, mean over iterations
        sharey : bool
            Share y-axis between subplots
        inner_paths : bool
            Provide comparison from arbitrary paths, e.g. controls
        paths : list of tuples
            Provide list of tuples of full path parts to data folder.
            E.g. [(root_path, 'Single_narrow_iterations_control', 'Bacon_gL_compiled_results'),]
            The number of paths MUST be the same as the number of corresponding inner variables.
        """

        coll_ana_df = copy.deepcopy(self.coll_spa_dict["coll_ana_df"])
        to_spa_dict = copy.deepcopy(self.context.to_spa_dict)

        titles = to_spa_dict[param_plot_dict["title"]]

        if param_plot_dict["save_description"] is True:
            describe_df_list = []
            describe_df_columns_list = []
            describe_folder_full = Path.joinpath(self.context.path, "Descriptions")
            describe_folder_full.mkdir(parents=True, exist_ok=True)

        # If param_plot_dict["inner_paths"] is True, replace titles with and [""] .
        if param_plot_dict["inner_paths"] is True:
            titles = [""]

        # Recursive call for multiple titles => multiple figures
        for this_title in titles:
            this_title_list = [this_title]
            to_spa_dict[param_plot_dict["title"]] = this_title_list

            (
                data_list,
                data_name_list,
                data_sub_list,
                outer_name_list,
                sub_col_name,
            ) = self._build_param_plot(coll_ana_df, param_plot_dict, to_spa_dict)

            sharey = param_plot_dict["sharey"]
            palette = param_plot_dict["palette"]

            if param_plot_dict["display_optimal_values"] is True:
                optimal_value_foldername = param_plot_dict["optimal_value_foldername"]
                optimal_description_name = param_plot_dict["optimal_description_name"]

                # read optimal values to dataframe from path/optimal_values/optimal_unfit_description.csv
                optimal_df = pd.read_csv(
                    Path.joinpath(
                        self.context.path,
                        optimal_value_foldername,
                        optimal_description_name,
                    )
                )
                # set the first column as index
                optimal_df.set_index(optimal_df.columns[0], inplace=True)

            fig, [axs] = plt.subplots(1, len(data_list), sharey=sharey, squeeze=False)

            if (
                "divide_by_frequency" in param_plot_dict
                and param_plot_dict["divide_by_frequency"] is True
            ):
                # Divide by frequency

                frequency_names_list = [
                    "Excitatory Firing Rate",
                    "Inhibitory Firing Rate",
                ]

                # Get the index of the frequency names
                out_fr_idx = np.array([], dtype=int)
                for out_idx, this_name in enumerate(outer_name_list):
                    if this_name in frequency_names_list:
                        out_fr_idx = np.append(out_fr_idx, out_idx)

                # Sum the two frequency values, separately for each inner
                frequencies = np.zeros([len(data_list[0][0]), len(data_list[0])])
                # Make a numpy array of zeros, whose shape is length
                for this_idx in out_fr_idx:
                    this_fr_list = data_list[this_idx]
                    for fr_idx, this_fr in enumerate(this_fr_list):
                        frequencies[:, fr_idx] += this_fr.values

                # Drop the out_fr_idx from the data_list
                data_list = [
                    l for idx, l in enumerate(data_list) if idx not in out_fr_idx
                ]
                outer_name_list = [
                    l for idx, l in enumerate(outer_name_list) if idx not in out_fr_idx
                ]

                # Divide the values by the frequencies
                for out_idx, this_data_list in enumerate(data_list):
                    for in_idx, this_data in enumerate(this_data_list):
                        new_values = this_data.values / frequencies[:, in_idx]
                        # Assign this_data.values back to the data_list
                        data_list[out_idx][in_idx] = pd.Series(
                            new_values, name=this_data.name
                        )

            for out_idx, inner_data_list in enumerate(data_list):
                outer_name = outer_name_list[out_idx]
                inner_df_coll = pd.DataFrame()
                sub_df_coll = pd.DataFrame()
                for in_idx, inner_series in enumerate(inner_data_list):
                    inner_df_coll[data_name_list[out_idx][in_idx]] = inner_series
                    if param_plot_dict["inner_sub"] is True:
                        sub_df_coll[data_name_list[out_idx][in_idx]] = data_sub_list[
                            out_idx
                        ][in_idx]

                self.data_is_valid(inner_df_coll.values, accept_empty=False)

                # For backwards compatibility in FCN22 project 221209 SV
                if outer_name == "Coherence":
                    if inner_df_coll.max().max() > 1:
                        inner_df_coll = inner_df_coll / 34
                if param_plot_dict["save_description"] is True:
                    describe_df_list.append(inner_df_coll)  # for saving
                    describe_df_columns_list.append(f"{outer_name}")

                # We use axes level plots instead of catplot which is figure level plot.
                # This way we can control the plotting order and additional arguments
                if param_plot_dict["inner_sub"] is False:
                    # wide df--each column plotted
                    boxprops = dict(
                        linestyle="-", linewidth=1, edgecolor="black", facecolor=".7"
                    )

                    if param_plot_dict["plot_type"] == "box":
                        g1 = sns.boxplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            boxprops=boxprops,
                            whis=[0, 100],
                            showfliers=False,
                            showbox=True,
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "violin":
                        g1 = sns.violinplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "strip":
                        g1 = sns.stripplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "swarm":
                        g1 = sns.swarmplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "boxen":
                        g1 = sns.boxenplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "point":
                        g1 = sns.pointplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )
                    elif param_plot_dict["plot_type"] == "bar":
                        g1 = sns.barplot(
                            data=inner_df_coll,
                            ax=axs[out_idx],
                            palette=palette,
                        )

                elif param_plot_dict["inner_sub"] is True:
                    inner_df_id_vars = pd.DataFrame().reindex_like(inner_df_coll)
                    # Make a “long-form” DataFrame
                    for this_bin_idx, this_bin_limits in enumerate(
                        param_plot_dict["bin_edges"]
                    ):
                        # Apply bin edges to sub data
                        inner_df_id_vars_idx = sub_df_coll.apply(
                            lambda x: (x > this_bin_limits[0])
                            & (x < this_bin_limits[1]),
                            raw=True,
                        )
                        inner_df_id_vars[inner_df_id_vars_idx] = this_bin_idx

                    inner_df_id_values_vars = pd.concat(
                        [
                            inner_df_coll.stack(dropna=False),
                            inner_df_id_vars.stack(dropna=False),
                        ],
                        axis=1,
                    )

                    inner_df_id_values_vars = inner_df_id_values_vars.reset_index()
                    inner_df_id_values_vars.drop(columns="level_0", inplace=True)
                    inner_df_id_values_vars.columns = [
                        "Title",
                        outer_name,
                        sub_col_name,
                    ]

                    bin_legends = [
                        f"{m}-{n}" for [m, n] in param_plot_dict["bin_edges"]
                    ]
                    inner_df_id_values_vars[sub_col_name].replace(
                        to_replace=[*range(0, len(param_plot_dict["bin_edges"]))],
                        value=bin_legends,
                        inplace=True,
                    )

                    if param_plot_dict["plot_type"] == "box":
                        g1 = sns.boxplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            whis=[0, 100],
                            showfliers=False,
                            showbox=True,
                        )
                    elif param_plot_dict["plot_type"] == "violin":
                        g1 = sns.violinplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            split=True,
                        )
                    elif param_plot_dict["plot_type"] == "strip":
                        g1 = sns.stripplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "swarm":
                        g1 = sns.swarmplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "boxen":
                        g1 = sns.boxenplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "point":
                        g1 = sns.pointplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )
                    elif param_plot_dict["plot_type"] == "bar":
                        g1 = sns.barplot(
                            data=inner_df_id_values_vars,
                            x="Title",
                            y=outer_name,
                            hue=sub_col_name,
                            palette=palette,
                            ax=axs[out_idx],
                            hue_order=bin_legends,
                            dodge=True,
                        )

                g1.set(xlabel=None, ylabel=None)
                fig.suptitle(this_title, fontsize=16)

                labels = data_name_list[out_idx]
                axs[out_idx].set_xticklabels(labels, rotation=60)

                if param_plot_dict["display_optimal_values"] is True:
                    # Get column name from coll_ana_df
                    col_name = coll_ana_df.loc[outer_name, "csv_col_name"]
                    matching_column = [
                        c
                        for c in optimal_df.columns
                        if c.startswith(col_name) and c.endswith("_mean")
                    ]
                    if len(matching_column) > 0:
                        min_value = optimal_df.loc["min", matching_column[0]]
                        max_value = optimal_df.loc["max", matching_column[0]]
                        # draw a horizontal dashed line to axs[out_idx] at y=min_value and y=max_value
                        axs[out_idx].axhline(y=min_value, color="black", linestyle="--")
                        axs[out_idx].axhline(y=max_value, color="black", linestyle="--")

                # To get min max etc if necessary
                # print(inner_df_coll.describe())

                # If statistics is tested, set statistics value and name to each axs subplot
                if param_plot_dict["inner_stat_test"] is True:
                    """
                    Apply the statistical test to inner_df_coll
                    If len(inner_data_list) == 2, apply Wilcoxon signed-rank test.
                    Else if len(inner_data_list) > 2, apply Friedman test.
                    Set stat_name to the test name.
                    """
                    if len(inner_data_list) == 2:
                        stat_test_name = "Wilcoxon signed-rank test"
                        statistics, stat_p_value = self.ana.stat_tests.wilcoxon_test(
                            inner_df_coll.values[:, 0], inner_df_coll.values[:, 1]
                        )
                    elif len(inner_data_list) > 2:
                        stat_test_name = "Friedman test"
                        statistics, stat_p_value = self.ana.stat_tests.friedman_test(
                            inner_df_coll.values
                        )
                    else:
                        raise ValueError(
                            "len(inner_data_list) must be 2 or more for stat_test, aborting..."
                        )

                    # Find the column with largest median value, excluding nans
                    median_list = []
                    for this_idx, this_column in enumerate(inner_df_coll.columns):
                        median_list.append(
                            np.nanmedian(inner_df_coll.values[:, this_idx])
                        )
                    max_median_idx = np.argmax(median_list)

                    # If p-value is less than 0.05, append
                    # the str(max_median_idx) to stat_p_value
                    if stat_p_value < 0.05:
                        stat_corrected_p_value_str = f"{stat_p_value:.3f} (max median at {data_name_list[out_idx][max_median_idx]})"
                    else:
                        stat_corrected_p_value_str = f"{stat_p_value:.3f}"

                    axs[out_idx].set_title(
                        f"{outer_name}\n{stat_test_name} =\n{stat_corrected_p_value_str}\n{statistics:.1f}\nN = {inner_df_coll.shape[0]}"
                    )
                else:
                    axs[out_idx].set_title(outer_name)

            if param_plot_dict["save_description"] is True:
                describe_df_columns_list = [
                    c.replace(" ", "_") for c in describe_df_columns_list
                ]
                describe_df_all = pd.DataFrame()
                for this_idx, this_column in enumerate(describe_df_columns_list):
                    # Append the describe_df_all data describe_df_list[this_idx]
                    this_describe_df = describe_df_list[this_idx]
                    # Prepend the column names with this_column
                    this_describe_df.columns = [
                        this_column + "_" + c for c in this_describe_df.columns
                    ]

                    describe_df_all = pd.concat(
                        [describe_df_all, this_describe_df], axis=1
                    )

                filename_full = Path.joinpath(
                    describe_folder_full,
                    param_plot_dict["save_name"] + "_" + this_title + ".csv",
                )

                # Save the describe_df_all dataframe .to_csv(filename_full, index=False)
                describe_df_all_df = describe_df_all.describe()
                describe_df_all_df.insert(
                    0, "description", describe_df_all.describe().index
                )
                describe_df_all_df.to_csv(filename_full, index=False)
                describe_df_list = []
                describe_df_columns_list = []

            if self.save_figure_with_arrayidentifier is not None:
                id = "box"

                self._figsave(
                    figurename=f"{self.save_figure_with_arrayidentifier}_{id}_{this_title}",
                    myformat="svg",
                    subfolderpath=self.save_figure_to_folder,
                )
                self._figsave(
                    figurename=f"{self.save_figure_with_arrayidentifier}_{id}_{this_title}",
                    myformat="png",
                    subfolderpath=self.save_figure_to_folder,
                )
