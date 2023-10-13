# Builtin
from pathlib import Path
from argparse import ArgumentError
import copy
import pdb
import sys

# Analysis
import numpy as np
import pandas as pd

# Viz
import matplotlib.pyplot as plt


class ProjectUtilities:
    """
    Utilities for ProjectManager class. This class is not instantiated. It serves as a container for project independent helper functions.
    """

    def round_to_n_significant(self, value_in, significant_digits=3):
        boolean_test = value_in != 0

        if boolean_test and not np.isnan(value_in):
            int_to_subtract = significant_digits - 1
            value_out = round(
                value_in, -int(np.floor(np.log10(np.abs(value_in))) - int_to_subtract)
            )
        else:
            value_out = value_in

        return value_out

    def destroy_from_folders(self, path=None, dict_key_list=None):
        """
        Run destroy_data from root folder, deleting selected variables from data files one level towards leafs.
        """

        if path is None:
            p = Path(".")
        elif isinstance(path, Path):
            p = path
        elif isinstance(path, str):
            p = Path(path)
            if not p.is_dir():
                raise ArgumentError(f"path argument is not valid path, aborting...")

        folders = [x for x in p.iterdir() if x.is_dir()]
        metadata_full = []
        for this_folder in folders:
            for this_file in list(this_folder.iterdir()):
                if "metadata" in str(this_file):
                    metadata_full.append(this_file.resolve())

        for this_metadata in metadata_full:
            try:
                print(f"Updating {this_metadata}")
                updated_meta_full, foo_df = self.update_metadata(this_metadata)
                self.destroy_data(updated_meta_full, dict_key_list=dict_key_list)
            except FileNotFoundError:
                print(f"No files for {this_metadata}, nothing changed...")

    def destroy_data(self, meta_fname, dict_key_list=None):
        """
        Sometimes you have recorded too much and you want to reduce the filesize by removing some data.

        For not manipulating accidentally data in other folders (from path context), this method works only either at the metadata folder or with full path.

        :param meta_fname: str or pathlib object, metadata file name or full path
        :param dict_key_list: list, list of dict keys to remove from the file.
        example dict_key_list={'vm_all' : ['NG1_L4_CI_SS_L4', 'NG2_L4_CI_BC_L4']}

        Currently specific to destroying the second level of keys, as in above example.
        """
        if dict_key_list is None:
            raise ArgumentError(
                dict_key_list, "dict_key_list is None - nothing to do, aborting..."
            )

        if Path(meta_fname).is_file() and "metadata" in str(meta_fname):
            meta_df = self.data_io.get_data(meta_fname)
        else:
            raise FileNotFoundError(
                "The first argument must be valid metadata file name in current folder, or full path to metadata file"
            )

        def format(filename, dict_key_list):
            # This will destroy the selected data
            data_dict = self.data_io.get_data(filename)
            for key in dict_key_list.keys():
                for key2 in dict_key_list[key]:
                    try:
                        del data_dict[key][key2]
                    except KeyError:
                        print(
                            f"Key {key2} not found, assuming removed, nothing changed..."
                        )
                        return
            self.data_io.write_to_file(filename, data_dict)

        for filename in meta_df["Full path"]:
            if Path(filename).is_file():
                format(filename, dict_key_list)

    def pp_df_full(self, df):
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.max_colwidth",
            -1,
        ):
            print(df)

    def end2idx(self, t_idx_end, n_samples):
        if t_idx_end is None:
            t_idx_end = n_samples
        elif t_idx_end < 0:
            t_idx_end = n_samples + t_idx_end
        return t_idx_end

    def metadata_manipulator(
        self, meta_full=None, filename=None, multiply_rows=1, replace_dict={}
    ):
        """
        Replace strings in a metadata file.
        :param path: str or pathlib object
        :param filename: str, metadata filename, if empty, search most recent in path
        :param replace_dict: dict,
            keys: 'columns', 'find' and 'replace'
            values: lists of same length
            key: 'rows'
            values: list of row index values (as in df.loc) for the changes to apply
        """

        if meta_full is None:
            raise ArgumentError("Need full path to metadatafile, aborting...")

        if not replace_dict:
            raise ArgumentError("Missing replace dict, aborting...")

        data_df = self.data_io.load_from_file(meta_full)

        # multiply rows by factor multiply_rows
        multiply_rows = 2
        new_df = pd.DataFrame(
            np.repeat(data_df.values, multiply_rows, axis=0), columns=data_df.columns
        )

        for this_row in replace_dict["rows"]:
            for col_idx, this_column in enumerate(replace_dict["columns"]):
                f = replace_dict["find"][col_idx]
                r = replace_dict["replace"][col_idx]
                print(f"Replacing {f=} for {r=}, for {this_row=}, {this_column=}")
                new_df.loc[this_row][this_column] = str(
                    new_df.loc[this_row][this_column]
                ).replace(
                    f, r
                )  # str method

        self.pp_df_full(new_df)
        new_meta_full = self._write_updated_metadata_to_file(meta_full, new_df)
        print(f"Created {new_meta_full}")

    # Debugging
    def pp_df_memory(self, df):
        BYTES_TO_MB_DIV = 0.000001
        mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3)
        print("Memory usage is " + str(mem) + " MB")

    def pp_obj_size(self, obj):
        from IPython.lib.pretty import pprint

        pprint(obj)
        print(f"\nObject size is {sys.getsizeof(obj)} bytes")

    def get_added_attributes(self, obj1, obj2):
        XOR_attributes = set(dir(obj1)).symmetric_difference(dir(obj2))
        unique_attributes_list = [n for n in XOR_attributes if not n.startswith("_")]
        return unique_attributes_list

    def pp_attribute_types(self, obj, attribute_list=[]):
        if not attribute_list:
            attribute_list = dir(obj)

        for this_attribute in attribute_list:
            attribute_type = eval(f"type(obj.{this_attribute})")
            print(f"{this_attribute}: {attribute_type}")

    def countlines(self, startpath, lines=0, header=True, begin_start=None):
        # Counts lines in folder .py files.
        # From https://stackoverflow.com/questions/38543709/count-lines-of-code-in-directory-using-python

        startpath = Path(startpath).resolve()

        if header:
            print("{:>10} |{:>10} | {:<20}".format("ADDED", "TOTAL", "FILE"))
            print("{:->11}|{:->11}|{:->20}".format("", "", ""))

        for thing in Path.iterdir(startpath):
            thing = Path.joinpath(startpath, thing)
            if thing.is_file():
                if str(thing).endswith(".py"):
                    with open(thing, "r") as f:
                        newlines = f.readlines()
                        newlines = len(newlines)
                        lines += newlines

                        if begin_start is not None:
                            reldir_of_thing = "." + str(thing).replace(
                                str(begin_start), ""
                            )
                        else:
                            reldir_of_thing = "." + str(thing).replace(
                                str(startpath), ""
                            )

                        print(
                            "{:>10} |{:>10} | {:<20}".format(
                                newlines, lines, reldir_of_thing
                            )
                        )

        for thing in Path.iterdir(startpath):
            thing = Path.joinpath(startpath, thing)
            if Path.is_dir(thing):
                lines = self.countlines(
                    thing, lines, header=False, begin_start=startpath
                )

        return lines


class DataSampler:
    """
    Class for digitizing data points from published images.

    Attributes
    ----------
    filename : str
        Path to the image file.
    min_X : float
        Minimum x-axis value for calibration.
    max_X : float
        Maximum x-axis value for calibration.
    min_Y : float
        Minimum y-axis value for calibration.
    max_Y : float
        Maximum y-axis value for calibration.
    logX : bool
        If True, indicates x-axis is logarithmic.
    calibration_points : list
        List of calibration points selected from the image.
    data_points : list
        List of data points selected from the image.

    Methods
    -------
    add_calibration_point(point)
        Adds a point to the list of calibration points.
    add_data_point(point)
        Adds a point to the list of data points.
    validate_calibration()
        Validates calibration point input.
    to_data_units(x, y)
        Converts image units (pixels) to data units.
    to_image_units(x_data, y_data)
        Converts data units back to image units (pixels).
    save_data()
        Saves the digitized data to a file.
    quality_control()
        Displays the original image with calibration and data points.
    collect_and_save_points()
        Interactively collect calibration and data points from the image.
    """

    def __init__(self, filename, min_X, max_X, min_Y, max_Y, logX=False):
        self.filename = filename
        self.min_X = min_X
        self.max_X = max_X
        self.min_Y = min_Y
        self.max_Y = max_Y
        self.logX = logX

        self.calibration_points = []
        self.data_points = []

    def add_calibration_point(self, point):
        """Adds a point to the list of calibration points."""
        self.calibration_points.append(point)

    def add_data_point(self, point):
        """Adds a point to the list of data points."""
        Xdata, Ydata = self.to_data_units(point[0], point[1])
        self.data_points.append((Xdata, Ydata))

    def validate_calibration(self):
        """Validates the calibration point input."""
        assert (
            len(self.calibration_points) == 3
        ), "Three calibration points are required."
        x_min, y_min = self.calibration_points[0]
        _, y_max = self.calibration_points[1]
        x_max, _ = self.calibration_points[2]
        assert (
            x_min < x_max and y_min > y_max
        ), "The calibration set is not valid, 1. origo, 2. y max, 3. x max"

    def to_data_units(self, x, y):
        """Converts image units (pixels) to data units."""
        pix_x_min, pix_y_min = self.calibration_points[0]
        _, pix_y_max = self.calibration_points[1]
        pix_x_max, _ = self.calibration_points[2]
        x_range = pix_x_max - pix_x_min
        y_range = self.calibration_points[0][1] - pix_y_max  # inverted scale

        x_scaled = (x - pix_x_min) / x_range * (self.max_X - self.min_X) + self.min_X
        y_upright = (
            pix_y_min - y
        )  # From bottom upwards, y min is the largest pixel value
        y_scaled = (y_upright / y_range) * (self.max_Y - self.min_Y) + self.min_Y
        if self.logX:
            x_scaled = np.exp(x_scaled)
        return x_scaled, y_scaled

    def to_image_units(self, x_data, y_data):
        """Converts data units back to image units (pixels)."""
        pix_x_min, pix_y_min = self.calibration_points[0]
        _, pix_y_max = self.calibration_points[1]
        pix_x_max, _ = self.calibration_points[2]
        x_range = pix_x_max - pix_x_min
        y_range = pix_y_min - pix_y_max  # inverted scale

        if self.logX:
            x_data = np.log(x_data)
        x = (x_data - self.min_X) / (self.max_X - self.min_X) * x_range + pix_x_min
        y_offset = y_data - self.min_Y
        y_upright = y_offset / (self.max_Y - self.min_Y) * y_range
        y = pix_y_min - y_upright

        return x, y

    def save_data(self):
        """Saves the digitized data to a file."""
        Xdata, Ydata = zip(*[(x, y) for x, y in self.data_points])
        calib_x, calib_y = zip(*[(x, y) for x, y in self.calibration_points])
        nameout = self.filename.stem + "_c"
        filename_full = self.filename.parent / nameout
        np.savez(
            filename_full, Xdata=Xdata, Ydata=Ydata, calib_x=calib_x, calib_y=calib_y
        )
        print(f"Saved data into {filename_full}.npz")

    def quality_control(self, restore=False):
        """Displays the original image with calibration and data points."""
        imagedata = plt.imread(self.filename)

        if restore is True:
            namein = self.filename.stem + "_c.npz"
            data_filename_full = self.filename.parent / namein
            data = np.load(data_filename_full)
            self.data_points = [(x, y) for x, y in zip(*[data["Xdata"], data["Ydata"]])]
            self.calibration_points = [
                (x, y) for x, y in zip(*[data["calib_x"], data["calib_y"]])
            ]

        data_x, data_y = zip(*[self.to_image_units(x, y) for x, y in self.data_points])
        calib_x = [pt[0] for pt in self.calibration_points]
        calib_y = [pt[1] for pt in self.calibration_points]

        fig, ax = plt.subplots()
        ax.imshow(imagedata, cmap="gray")

        ax.scatter(calib_x, calib_y, color="red", s=50, label="Calibration Points")

        ax.scatter(data_x, data_y, color="blue", s=30, label="Data Points")

        ax.legend()

    def collect_and_save_points(self):
        """Interactively collect calibration and data points from the image."""
        imagedata = plt.imread(self.filename)
        fig, ax = plt.subplots()
        ax.imshow(imagedata, cmap="gray")

        print("Calibrate 1. origo, 2. y max, 3. x max")
        calib_points = plt.ginput(3, timeout=0)
        for point in calib_points:
            self.add_calibration_point(point)

        self.validate_calibration()

        print("And now the data points...")
        data_points = plt.ginput(-1, timeout=0)
        for point in data_points:
            self.add_data_point(point)
        plt.close(fig)

        self.save_data()
