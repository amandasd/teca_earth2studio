from collections import OrderedDict

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils.type import CoordSystem

import numpy
import torch
if torch.cuda.is_available():
    import cupy

import xarray as xr
import pandas as pd
from teca import *

class DetectNodes(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("path_buffer", torch.empty(0))
        self.step = 0

    def reset_path_buffer(self) -> None:
        """Resets the internal"""
        self.path_buffer = torch.empty(0)

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the diagnostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(["msl", "u10m", "v10m", "z", "z300", "z500"]),
                "lat": np.linspace(90, -90, 721, endpoint=True),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the diagnostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """

        # [batch, candidate_id, variable]
        return OrderedDict(
            [
                ("batch", input_coords["batch"]),
                ("candidate_id", np.empty(0)),
                ("variable", np.array(["step", "year", "month", "day", "hour", "minute", "ncandidates", "i", "j", "lat", "lon", "msl", "w10m", "zs"])),
            ]
        )

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs diagnostic model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system
        """
        out_coords = self.output_coords(coords)

        # get the device to run on
        np = numpy
        use_gpu = False
        if torch.cuda.is_available() and self._device == "gpu":
            np = cupy
            use_gpu = True

        def get_variable(x: torch.Tensor, var: str) -> torch.Tensor:
            index = ["msl", "u10m", "v10m", "z", "z300", "z500", ].index(var)
            return x[:, index]

        # Calculate wind speed
        u10m = get_variable(x, "u10m")
        v10m = get_variable(x, "v10m")
        w10m = torch.sqrt(torch.pow(u10m, 2) + torch.pow(v10m, 2))

        # Calculate thickness
        z300 = get_variable(x, "z300")
        z500 = get_variable(x, "z500")
        thickness = z300 - z500

        msl = get_variable(x, "msl")
        zs = get_variable(x, "z")
        new_x = torch.stack([msl, w10m, thickness, zs], dim=1)
        coords["variable"] = numpy.array(["msl", "w10m", "thickness", "zs"])

        batch, nvars, ny, nx = new_x.shape
        assert batch == 1, "Only batch size 1 is supported in this example"

        lon = teca_variant_array.New(coords["lon"])
        lat = teca_variant_array.New(coords["lat"])

        # Create and configure a teca mesh
        mesh = teca_cartesian_mesh.New()
        mesh.set_x_coordinates("x", lon)
        mesh.set_y_coordinates("y", lat)
        mesh.set_time_units("hours since "+str(self._current_time[0]).split('T')[0]+" "+str(self._current_time[0]).split('T')[1].split('.')[0])
        mesh.set_calendar("gregorian")

        # Convert a pytorch tensor to either a cupy or numpy array,
        # depending on which device
        for i, var in enumerate(coords["variable"]):
            arr = new_x[0, i].detach().to(torch.float32)
            if use_gpu:
                arr_np = cupy.asarray(arr)
            else:
                arr_np = arr.cpu().numpy()
            # Add the needed variables from the array into a teca mesh
            mesh.get_point_arrays().append(str(var), arr_np.ravel())

        coordinates_out = teca_metadata()
        coordinates_out["x"] = lon
        coordinates_out["y"] = lat
        coordinates_out["x_variable"] = "lon"
        coordinates_out["y_variable"] = "lat"

        # Create and configure the metadata
        md = teca_metadata()
        md["variables"] = {"msl", "w10m", "thickness", "zs"}
        md["coordinates"] = coordinates_out
        md["number_of_time_steps"] = 1
        md["index_initializer_key"] = "number_of_time_steps"
        md["index_request_key"] = "time_step"
        if not use_gpu:
            md["device_id"] = -1

        # Begin: TECA pipeline
        #
        # Create a teca_dataset_source from the mesh and metadata,
        # which will serve as input data to teca_detect_nodes
        source = teca_dataset_source.New()
        source.set_metadata(md)
        source.set_dataset(mesh)

        # Run teca_detect_nodes
        candidates = teca_detect_nodes.New()
        candidates.set_input_connection(source.get_output_port())
        candidates.set_search_by_min("msl")
        candidates.set_closed_contour_cmd("msl,200.0,5.5,0;thickness,-58.8,6.5,1.0")
        candidates.set_output_cmd("msl,min,0;w10m,max,2;zs,min,0")
        candidates.initialize();

        calendar = teca_table_calendar.New()
        calendar.set_input_connection(candidates.get_output_port())

        # Extract the output table with detected candidates
        output_table = teca_dataset_capture.New()
        output_table.set_input_connection(calendar.get_output_port())
        output_table.update()
        table = teca_table.New()
        table.shallow_copy(output_table.get_dataset())
        #
        # End: TECA pipeline

        if self.path_buffer.numel() != 0:
            self.step += 1

        # Convert teca output table to either a cupy or numpy array
        arrays = []
        columns = ["year", "month", "day", "hour", "minute", "step", "ncandidates", "i", "j", "lat", "lon", "msl", "w10m", "zs"]
        for original_name in columns:
            if "step" in original_name:
               arr = np.array(table.get_column(original_name)) + self.step
            else:
               arr = np.array(table.get_column(original_name))
            arrays.append(arr)
        data = np.stack(arrays, axis=1)

        # Convert a cupy or numpy array to a pytorch tensor
        tensor = torch.tensor(data)

        # Add a "batch" dimension
        out = tensor.unsqueeze(0)

        # Accumulate detected candidates over timesteps into path_buffer
        if self.path_buffer.numel() == 0:
            self.path_buffer = out.detach().clone()
        else:
            self.path_buffer = torch.cat((self.path_buffer, out.detach()), dim=1)

        out_coords["candidate_id"] = np.arange(self.path_buffer.shape[1])

        return self.path_buffer, out_coords

class StitchNodes:

    def __init__(self) -> None:
        super().__init__()

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the diagnostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "candidate_id": np.empty(0),
                "variable": np.array(["step", "year", "month", "day", "hour", "minute", "ncandidates", "i", "j", "lat", "lon", "msl", "w10m", "zs"]),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the diagnostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """

        # [batch, path_id, step, variable]
        return OrderedDict(
            [
                ("batch", input_coords["batch"]),
                ("path_id", np.empty(0)),
                ("step", np.empty(0)),
                ("variable", np.array(["tc_length", "tc_year", "tc_month", "tc_day", "tc_hour", "tc_i", "tc_j", "tc_lat", "tc_lon", "tc_msl", "tc_w10m", "tc_zs"])),
            ]
        )

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs diagnostic model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system
        """
        out_coords = self.output_coords(coords)

        np = numpy

        # Convert a pytorch tensor to a numpy array, then to a pandas dataframe
        columns = ["year", "month", "day", "hour", "minute", "step", "ncandidates", "i", "j", "lat", "lon", "msl", "w10m", "zs"]
        df = pd.DataFrame(x[0].detach().cpu().numpy(), columns=columns)

        # Convert a pandas dataframe to a teca table
        table_in = teca_table.New()
        for col in df.columns:
            if col in ["year", "month", "day", "hour", "minute", "i", "j"]:
                teca_array = teca_int_array.New()
            elif col in ["ncandidates"]:
                teca_array = teca_long_array.New()
            else:
                teca_array = teca_double_array.New()
            teca_array.resize(len(df[col].values))
            for i, v in enumerate(df[col].values):
                teca_array[i] = v
            table_in.append_column(col, teca_array)

        # Begin: TECA pipeline
        #
        # Create a teca_dataset_source from teca table,
        # which will serve as input data to teca_stitch_nodes
        source = teca_dataset_source.New()
        source.set_dataset(table_in)

        # Run teca_stitch_nodes
        tracks = teca_stitch_nodes.New()
        tracks.set_input_connection(source.get_output_port())
        tracks.set_in_fmt("step,i,j,lat,lon,msl,w10m,zs")
        tracks.set_track_threshold_cmd("w10m,>=,10.0,10;lat,<=,50.0,10;lat,>=,-50.0,10;zs,<=,15.0,10")
        tracks.initialize();

        # Extract the output table containing tropical cyclone trackings
        output_table = teca_dataset_capture.New()
        output_table.set_input_connection(tracks.get_output_port())
        output_table.update()
        table_out = teca_table.New()
        table_out.shallow_copy(output_table.get_dataset())
        #
        # End: TECA pipeline

        # Check if the output table is empty
        if table_out.get_number_of_rows() == 0:
            return torch.empty(0), out_coords

        def get_column_index(table, name):
            if not table.has_column(name):
                return -1
            for i in range(table.get_number_of_columns()):
                if table.get_column_name(i) == name:
                    return i
            return -1

        col_step = get_column_index(table_out, "step")
        col_storm_id = get_column_index(table_out, "storm_id")

        if col_step == -1 or col_storm_id == -1:
            return torch.empty(0), out_coords

        # Begin: Prepare the data to match the output format of existing Earth2Studio TC trackers
        #
        # Convert teca output table to a numpy array
        arrays = []
        columns = ["storm_id", "path_length", "year", "month", "day", "hour", "step", "i", "j", "lat", "lon", "msl", "w10m", "zs"]
        for original_name in columns:
            arr = np.array(table_out.get_column(original_name))
            arrays.append(arr)
        data = np.stack(arrays, axis=1)

        # Convert a numpy array to a pytorch tensor
        tensor = torch.tensor(data)

        # Extract path_ids and unique ones
        path_ids = tensor[:, col_storm_id].long()
        unique_paths = torch.unique(path_ids).tolist()
        num_paths = len(unique_paths)

        # Create output tensor: [num_paths, nsteps, variable] filled with nans
        output = torch.full((num_paths, self._nsteps, tensor.size(1)), float('nan'))

        # Fill data into the correct slots
        for i, path_id in enumerate(unique_paths):
            # Filter rows for this path_id
            mask = path_ids == path_id
            group = tensor[mask]

            # Get timesteps for this group
            timesteps = group[:, col_step].long()

            # Place each row into its correct timestep
            for row, t in zip(group, timesteps):
                output[i, t] = row

        # step and storm_id are redundant columns
        # as this information is already present in out_coords
        remove_cols = sorted([col_step, col_storm_id])

        # Create list of indices for the columns to keep
        keep_cols = [i for i in range(output.shape[2]) if i not in remove_cols]

        # Select only the desired columns
        # and add a "batch" dimension
        out = output[:, :, keep_cols].unsqueeze(0)
        #
        # End: Prepare the data to match the output format of existing Earth2Studio TC trackers

        # Update out_coords with path_id and step identifiers
        out_coords["path_id"] = np.arange(num_paths)
        out_coords["step"] = np.arange(self._nsteps)

        return out, out_coords

class teca_tempest_tc_detect(torch.nn.Module):
    """Custom diagnostic model"""

    def __init__(self):
        super().__init__()
        self.detect = DetectNodes()
        self.stitch = StitchNodes()
