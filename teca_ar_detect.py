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

import time

LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]

class Detecting(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("path_buffer", torch.empty(0))
        self.register_buffer("path_info", torch.empty(0))
        self.step = 0

    def reset_path_buffer(self) -> None:
        """Resets the internal"""
        self.path_buffer = torch.empty(0)
        self.path_info = torch.empty(0)

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
                "variable": np.array([f"{var}{lev}" for var in ["q", "u", "v"] for lev in LEVELS]),
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

        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "variable": np.array(["ar_probability", "ivt", "ivt_u", "ivt_v"]),
                "lat": np.linspace(90, -90, 721, endpoint=True),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
                "info": {"ar_count": np.empty((0, 0), dtype=np.int32),
                         "parameter_table_row": np.empty((0, 0), dtype=np.int32),},
            }
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
        if self.path_buffer.numel() != 0:
            self.step += 1

        # get the device to run on
        if not torch.cuda.is_available() or str(self._device) == "cpu":
            np = numpy
            use_gpu = False
        else:
            np = cupy
            use_gpu = True

        t_lon = torch.as_tensor(self.input_coords()["lon"], device=x.device, dtype=torch.float32)
        t_lat = torch.as_tensor(self.input_coords()["lat"], device=x.device, dtype=torch.float32)
        t_z   = torch.as_tensor([lev * 100 for lev in LEVELS], device=x.device, dtype=torch.float32)
        nx = t_lon.shape[0]; ny = t_lat.shape[0]; nz = t_z.shape[0]
        wext = [0, nx - 1, 0, ny - 1, 0, nz - 1]
        if use_gpu:
            lon = teca_variant_array.New(cupy.asarray(t_lon))
            lat = teca_variant_array.New(cupy.asarray(t_lat))
            z   = teca_variant_array.New(cupy.asarray(t_z))
        else:
            lon = teca_variant_array.New(t_lon.cpu().numpy())
            lat = teca_variant_array.New(t_lat.cpu().numpy())
            z   = teca_variant_array.New(t_z.cpu().numpy())

        # Create and configure a teca mesh
        mesh = teca_cartesian_mesh.New()
        mesh.set_x_coordinates("lon", lon)
        mesh.set_y_coordinates("lat", lat)
        mesh.set_z_coordinates("plev", z)
        mesh.set_time(1.0)
        mesh.set_calendar("gregorian")
        mesh.set_whole_extent(wext)
        mesh.set_extent(wext)

        q_md = teca_metadata()
        FLOAT32 = 10
        q_md["type_code"] = FLOAT32
        q_md["centering"] = "point"
        q_md["units"] = "kg kg-1"
        q_md["standard_name"] = "specific humidity"

        u_md = teca_metadata()
        u_md["type_code"] = FLOAT32
        u_md["centering"] = "point"
        u_md["units"] = "m s-1"
        u_md["standard_name"] = "x-component of wind"

        v_md = teca_metadata()
        v_md["type_code"] = FLOAT32
        v_md["centering"] = "point"
        v_md["units"] = "m s-1"
        v_md["standard_name"] = "y-component of wind"

        plev_md = teca_metadata()
        plev_md["type_code"] = FLOAT32
        plev_md["centering"] = "point"
        plev_md["units"] = "Pa"
        plev_md["standard_name"] = "vertical_coordinate"

        attrs = teca_metadata()
        attrs.set("q", q_md)
        attrs.set("u", u_md)
        attrs.set("v", v_md)
        attrs.set("plev", plev_md)
        mesh.set_attributes(attrs)

        coordinates_out = teca_metadata()
        coordinates_out.set("x", lon)
        coordinates_out.set("y", lat)
        coordinates_out.set("z", z)
        coordinates_out.set("x_variable", "lon")
        coordinates_out.set("y_variable", "lat")
        coordinates_out.set("z_variable", "plev")

        md = teca_metadata()
        md.set("variables", "q,u,v")
        md.set("whole_extent", wext)
        md.set("attributes", attrs)
        md.set("coordinates", coordinates_out)
        md.set("number_of_time_steps", 1)
        md.set("index_initializer_key", "number_of_time_steps")
        md.set("index_request_key", "time_step")

        if not use_gpu:
            md.set("device_id", -1)

        def get_variable(x: torch.Tensor, var: str) -> torch.Tensor:
            index = [f"{var}{lev}" for var in ["q", "u", "v"] for lev in LEVELS].index(var)
            return x[index]

        out = []; out_info = []
        for i in range(x.shape[0]):
            q = torch.stack([get_variable(x[i], f"q{lev}") for lev in LEVELS])
            u = torch.stack([get_variable(x[i], f"u{lev}") for lev in LEVELS])
            v = torch.stack([get_variable(x[i], f"v{lev}") for lev in LEVELS])
            new_x = torch.stack([q, u, v]) # [variables, levels, lat, lon]

            mesh.set_time_units("hours since "+str(self._current_time[i]).split('T')[0]+" "+str(self._current_time[i]).split('T')[1].split('.')[0])

            # Convert a pytorch tensor to either a cupy or numpy array,
            # depending on which device
            for idx, var in enumerate(["q", "u", "v"]):
                arr = new_x[idx].detach().to(torch.float32)
                if use_gpu:
                    arr_np = cupy.asarray(arr)
                else:
                    arr_np = arr.cpu().numpy()
                # Add the needed variables from the array into a teca mesh
                mesh.get_point_arrays().append(str(var), arr_np)

            # Begin: TECA pipeline
            #
            # Create a teca_dataset_source from the mesh and metadata,
            # which will serve as input data to teca_bayesian_ar_detect
            #start = time.time()
            source = teca_dataset_source.New()
            source.set_metadata(md)
            source.set_dataset(mesh)

            norm_coords = teca_normalize_coordinates.New()
            norm_coords.set_input_connection(source.get_output_port())
            vv_mask = teca_valid_value_mask.New()
            vv_mask.set_input_connection(norm_coords.get_output_port())
            unpack = teca_unpack_data.New()
            unpack.set_input_connection(vv_mask.get_output_port())

            ivt_int = teca_integrated_vapor_transport.New()
            ivt_int.set_input_connection(unpack.get_output_port())
            ivt_int.set_specific_humidity_variable("q")
            ivt_int.set_wind_u_variable("u")
            ivt_int.set_wind_v_variable("v")
            ivt_int.set_ivt_u_variable("ivt_u")
            ivt_int.set_ivt_v_variable("ivt_v")

            l2_norm = teca_l2_norm.New()
            l2_norm.set_input_connection(ivt_int.get_output_port())
            l2_norm.set_component_0_variable("ivt_u")
            l2_norm.set_component_1_variable("ivt_v")
            l2_norm.set_l2_norm_variable("ivt")

            # Run teca_bayesian_ar_detect
            params = teca_bayesian_ar_detect_parameters.New()
            ar_detect = teca_bayesian_ar_detect.New()
            ar_detect.set_ivt_variable("ivt")
            ar_detect.set_ar_probability_variable("ar_probability")
            ar_detect.set_input_connection(0, params.get_output_port())
            ar_detect.set_input_connection(1, l2_norm.get_output_port())
            ar_detect.set_thread_pool_size(1)

            # Extract the output data
            output = teca_dataset_capture.New()
            output.set_input_connection(ar_detect.get_output_port())
            output.update()

            output_mesh = teca_cartesian_mesh.New()
            output_mesh.shallow_copy(output.get_dataset())
            ivt_array = output_mesh.get_point_arrays().get("ivt")
            ar_probability_array = output_mesh.get_point_arrays().get("ar_probability")
            ivt_u_array = output_mesh.get_point_arrays().get("ivt_u")
            ivt_v_array = output_mesh.get_point_arrays().get("ivt_v")
            info_arrays = output_mesh.get_information_arrays()
            ar_count_array = info_arrays.get("ar_count")
            dim_parameter_table_row_array = info_arrays.get("ar_count")
            #end = time.time()
            #print(f"Elapsed time(TECA): {end - start:.6f} seconds")
            #
            # End: TECA pipeline

            # Convert teca dataset to a numpy array
            info = numpy.stack([ar_count_array, dim_parameter_table_row_array]).reshape(1, 2, np.asarray(ar_count_array).shape[0])
            data = numpy.stack([ar_probability_array, ivt_array, ivt_u_array, ivt_v_array]).reshape(1, 4, ny, nx)

            # Convert a numpy array to a pytorch tensor
            tensor = torch.from_numpy(data).to(torch.device("cpu"))
            t_info = torch.from_numpy(info).to(torch.device("cpu"))
            if len(out) == 0:
                out = tensor.unsqueeze(0)
                out_info = t_info.unsqueeze(0)
            else:
                out = torch.cat([out, tensor.unsqueeze(0)], dim=0)
                out_info = torch.cat([out_info, t_info.unsqueeze(0)], dim=0)

        np = numpy

        # Accumulate detected candidates over timesteps into path_buffer
        if self.path_buffer.numel() == 0:
            self.path_buffer = out.detach().clone().to("cpu")
            self.path_info = out_info.detach().clone().to("cpu")
        else:
            self.path_buffer = torch.cat((self.path_buffer, out.detach().to("cpu")), dim=1)
            self.path_info = torch.cat((self.path_info, out_info.detach().to("cpu")), dim=1)

        out_coords = self.output_coords(coords)
        out_coords["time"] = np.arange(self.path_buffer.shape[1])
        out_coords["info"] = {"ar_count": self.path_info[:,:,0,:],
                              "parameter_table_row": self.path_info[:,:,1,:],}

        return self.path_buffer, out_coords

class teca_ar_detect(torch.nn.Module):
    """Custom diagnostic model"""

    def __init__(self):
        super().__init__()
        self.detect = Detecting()
