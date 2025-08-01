from collections import OrderedDict

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import handshake_coords, handshake_dim
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

    def reset_step(self) -> None:
        """Resets the internal"""
        self.step = 0

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
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lon", 3)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "variable", 1)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")

        # [batch, candidate_id, variable]
        return OrderedDict(
            [
                ("batch", input_coords["batch"]),
                ("candidate_id", np.empty(0)),
                ("variable", np.array(["year", "month", "day", "hour", "minute", "step", "ncandidates", "i", "j", "lat", "lon", "msl", "w10m", "zs"])),
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

        # get the device to run on
        np = numpy
        use_gpu = False
        if torch.cuda.is_available() and self._device == "gpu":
            np = cupy
            use_gpu = True

        t_lon = torch.as_tensor(self.input_coords()["lon"], device=x.device)
        t_lat = torch.as_tensor(self.input_coords()["lat"], device=x.device)
        if use_gpu:
            lon = teca_variant_array.New(cupy_asarray(t_lon))
            lat = teca_variant_array.New(cupy_asarray(t_lat))
        else:
            lon = teca_variant_array.New(t_lon.cpu().numpy())
            lat = teca_variant_array.New(t_lat.cpu().numpy())

        # Create and configure a teca mesh
        mesh = teca_cartesian_mesh.New()
        mesh.set_x_coordinates("x", lon)
        mesh.set_y_coordinates("y", lat)
        mesh.set_calendar("gregorian")

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

        if self.path_buffer.numel() != 0:
            self.step += 1

        def get_variable(x: torch.Tensor, var: str) -> torch.Tensor:
            index = ["msl", "u10m", "v10m", "z", "z300", "z500", ].index(var)
            return x[index]

        outs = []
        for i in range(x.shape[0]):
            # Calculate wind speed
            u10m = get_variable(x[i], "u10m")
            v10m = get_variable(x[i], "v10m")
            w10m = torch.sqrt(torch.pow(u10m, 2) + torch.pow(v10m, 2))

            # Calculate thickness
            z300 = get_variable(x[i], "z300")
            z500 = get_variable(x[i], "z500")
            thickness = z300 - z500

            msl = get_variable(x[i], "msl")
            zs = get_variable(x[i], "z")
            new_x = torch.stack([msl, w10m, thickness, zs], dim=0)

            mesh.set_time_units("hours since "+str(self._current_time[i]).split('T')[0]+" "+str(self._current_time[i]).split('T')[1].split('.')[0])

            # Convert a pytorch tensor to either a cupy or numpy array,
            # depending on which device
            for v, var in enumerate(["msl", "w10m", "thickness", "zs"]):
                arr = new_x[v].detach().to(torch.float32)
                if use_gpu:
                    arr_np = cupy.asarray(arr)
                else:
                    arr_np = arr.cpu().numpy()
                # Add the needed variables from the array into a teca mesh
                mesh.get_point_arrays().append(str(var), arr_np.ravel())

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

            # Convert teca output table to either a cupy or numpy array
            arrays = []
            columns = ["year", "month", "day", "hour", "minute", "step", "ncandidates", "i", "j", "lat", "lon", "msl", "w10m", "zs"]
            for original_name in columns:
                if "step" in original_name:
                   arr = numpy.array(table.get_column(original_name)) + self.step
                else:
                   arr = numpy.array(table.get_column(original_name))
                arrays.append(arr)
            data = numpy.stack(arrays, axis=1)

            # Convert a cupy or numpy array to a pytorch tensor
            tensor = torch.tensor(data, device="cpu")
            outs.append(tensor)

        np = numpy
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
        out = torch.nn.utils.rnn.pad_sequence(outs, padding_value=np.nan, batch_first=True).to("cpu")

        # Accumulate detected candidates over timesteps into path_buffer
        if self.path_buffer.numel() == 0:
            self.path_buffer = out.detach().clone().to("cpu")
        else:
            self.path_buffer = torch.cat((self.path_buffer, out.detach().to("cpu")), dim=1)

        out_coords = self.output_coords(coords)
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
                "variable": np.array(["year", "month", "day", "hour", "minute", "step", "ncandidates", "i", "j", "lat", "lon", "msl", "w10m", "zs"]),
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
        np = numpy

        outs = []
        for i in range(x.shape[0]):

            # Convert a pytorch tensor to a numpy array, then to a pandas dataframe
            columns = ["year", "month", "day", "hour", "minute", "step", "ncandidates", "i", "j", "lat", "lon", "msl", "w10m", "zs"]
            df = pd.DataFrame(x[i].detach().cpu().numpy(), columns=columns)
            # Remove rows that contain any nans
            df = df.dropna(how='all')

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
                for c, v in enumerate(df[col].values):
                    teca_array[c] = v
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

            def get_column_index(table, name):
                for c in range(table.get_number_of_columns()):
                    if table.get_column_name(c) == name:
                        return c

            col_step = get_column_index(table_out, "step")
            col_storm_id = get_column_index(table_out, "storm_id")

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
            for ii, path_id in enumerate(unique_paths):
                # Filter rows for this path_id
                mask = path_ids == path_id
                group = tensor[mask]

                # Get timesteps for this group
                timesteps = group[:, col_step].long()

                # Place each row into its correct timestep
                for row, t in zip(group, timesteps):
                    output[ii, t] = row

            # step and storm_id are redundant columns
            # as this information is already present in out_coords
            remove_cols = sorted([col_step, col_storm_id])

            # Create list of indices for the columns to keep
            keep_cols = [ii for ii in range(output.shape[2]) if ii not in remove_cols]

            # Select only the desired columns
            outs.append(output[:, :, keep_cols])
            #
            # End: Prepare the data to match the output format of existing Earth2Studio TC trackers

        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
        out = torch.nn.utils.rnn.pad_sequence(outs, padding_value=np.nan, batch_first=True)

        # Update out_coords with path_id and step identifiers
        out_coords = self.output_coords(coords)
        out_coords["path_id"] = np.arange(out.shape[1])
        out_coords["step"] = np.arange(self._nsteps)

        return out, out_coords

class teca_tempest_tc_detect(torch.nn.Module):
    """Custom diagnostic model"""

    def __init__(self):
        super().__init__()
        self.detect = DetectNodes()
        self.stitch = StitchNodes()
