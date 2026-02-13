import torch
import xarray as xr
import numpy as np
from tqdm import tqdm

from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.data import NCAR_ERA5
from earth2studio.data import fetch_data, prep_data_array
from earth2studio.utils.time import to_time_array
from earth2studio.utils.coords import CoordSystem
from earth2studio.utils.coords import map_coords

from mpi4py import MPI
import os

import argparse
import time

from earth2studio.models.px import FCN3
from earth2studio.perturbation import Zero
from earth2studio.run import ensemble

from earth2studio.models.dx import teca_tempest_tc_detect
from earth2studio.models.dx import aews_detect

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FCN3 model for one initial condition.")
    parser.add_argument(
        "--initial-condition", type=str, required=True,
        help="Initial condition date in format YYYY-MM-DD"
    )
    parser.add_argument(
        "--nsteps", type=int, default=20,
        help="Number of forecast steps"
    )
    parser.add_argument(
        "--local-ensemble", type=int, default=1,
        help="Number of local ensemble members"
    )
    parser.add_argument(
        "--seed", type=int, default=333,
        help="Seed for each MPI process"
    )
    parser.add_argument(
        "--out_path", type=str, default="/pscratch/sd/a/asdufek/projects/teca/earth2studio",
        help="Output path for track files"
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Extract run configuration from arguments
    ic = args.initial_condition
    start_time = datetime.fromisoformat(ic)
    nsteps = args.nsteps # Number of forecast steps
    times = [start_time + timedelta(hours=6 * i) for i in range(nsteps+1)]

    nensemble = args.local_ensemble
    batch_size = 1
    seed = args.seed
    out_path = args.out_path

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print(f"Rank {rank}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Running FCN3 for initial condition {ic}, nsteps={nsteps}, local_ensemble={nensemble}")

    start_total = time.perf_counter()

    # Select compute device: CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the data source
    data = NCAR_ERA5()

    # ARCO surface geopotential
    arco = ARCO()
    da = arco([start_time], ['z'])
    z, coords_z = prep_data_array(da, device=device)

    # ERA5 surface geopotential
    #ds = xr.open_dataset("e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc")
    #ds = ds.rename({"Z": "z"})
    #da = ds["z"]
    #da = da.expand_dims(dim={"variable": ["z"]})
    #da = da.transpose("time", "variable", "latitude", "longitude")
    #z, coords_z = prep_data_array(da, device=device)

    # Load the default FCN3 model
    prognostic = FCN3.load_model(FCN3.load_default_package())
    prognostic = prognostic.to(device)

    # Create tropical cyclone tracker
    tc_tracker = teca_tempest_tc_detect()
    tc_tracker = tc_tracker.to(device)
    tc_tracker.detect._device = device

    # Create African Easterly Wave tracker
    aew_tracker = aews_detect()
    aew_tracker = aew_tracker.to(device)
    aew_tracker.detect._device = device

    # Load initial model state
    if rank == 0:
        xx, coords = fetch_data(
            source=data,
            time=to_time_array([start_time]),
            variable=prognostic.input_coords()["variable"],
            lead_time=prognostic.input_coords()["lead_time"],
            device="cpu",
        )
        shape = xx.shape
        dtype = xx.numpy().dtype
    else:
        xx = None
        coords = None
        shape = None
        dtype = None
        
    MPI.COMM_WORLD.Barrier()

    # Begin: Broadcast from rank 0
    #
    # Metadata
    shape = comm.bcast(shape, root=0)
    dtype = comm.bcast(dtype, root=0)
    coords = comm.bcast(coords, root=0)

    # Allocate tensor on non-root ranks
    if rank != 0:
        xx = torch.empty(shape, dtype=torch.from_numpy(np.empty((), dtype=dtype)).dtype)

    # Raw memory
    comm.Bcast(xx.numpy(), root=0)
    #
    # End: Broadcast from rank 0

    batch_ids_produce = list(range(0, int(np.ceil(nensemble / batch_size)),))
    # Loop over ensemble batches with a progress bar
    for batch_id in tqdm(
        batch_ids_produce,
        total=len(batch_ids_produce),
        desc="Total Ensemble Batches",
    ):
        # Begin: FCN3 + TC tracker + AEW tracker
        #
        start_fcn3 = time.perf_counter()
        # Move input data to device
        x = xx.to(device)

        # Determine how many ensemble members are in this batch
        num_batches_per_ic = int(np.ceil(nensemble / batch_size))
        mini_batch_sizes = [min((nensemble - ii * batch_size), batch_size) for ii in range(num_batches_per_ic)]
        batch_id_ic = batch_id % num_batches_per_ic
        mini_batch_size = mini_batch_sizes[batch_id_ic]

        # Add an ensemble dimension to coords
        coords = {
            "ensemble": np.array(
                [
                    sum(mini_batch_sizes[0 : batch_id % num_batches_per_ic]) + t
                    for t in range(0, mini_batch_size)
                ]
            )
        } | coords.copy()

        # Expand input tensor to include an ensemble dimension
        x = x.unsqueeze(0).repeat(mini_batch_size, *([1] * xx.ndim))

        # Reset trackers for a new forecast
        tc_tracker.detect.reset_path_buffer()
        tc_tracker.detect.reset_step()
        aew_tracker.detect.reset_path_buffer()

        # No perturbation required due to hidden Markov formulation of FCN3
        # FCN3 internally handles stochasticity
        perturbation = Zero()
        x, coords = perturbation(x, coords)

        # Set random seed
        # Unique seed per rank and per batch_id
        seed = seed + rank * 10 + batch_id
        prognostic.set_rng(seed)

        # Create FCN3 iterator
        model = prognostic.create_iterator(x, coords)

        # Loop over forecast steps with a progress bar
        with tqdm(
            total=nsteps + 1,
            desc="Running inference",
            leave=False
        ) as pbar:
            for step, (x_fcn, coords_fcn) in enumerate(model):
                # Begin: Tropical cyclone detection
                #
                # Delete lead_time, no need for it since steps are present in the tracks
                # Combine geopotential height with FCN3 output fields
                x_tc = torch.cat((z.unsqueeze(2), x_fcn.squeeze(1)), dim=2)
                coords_tc = coords_fcn.copy()
                coords_tc["variable"] = np.concatenate([coords_z["variable"], coords_fcn["variable"]])
                del coords_tc["lead_time"]
                x_tc, coords_tc = map_coords(x_tc, coords_tc, tc_tracker.detect.input_coords())
                # Set current time
                tc_tracker.detect._current_time = np.array([np.datetime64(times[step], 'ns')])
                # Detect TC candidates for the current forecast time
                tc_tensor, tc_coords = tc_tracker.detect(x_tc, coords_tc)
                #
                # End: Tropical cyclone detection

                # Begin: AEW detection
                #
                # Prepare inputs for AEW tracker
                x_aew = x_fcn.squeeze(1)
                coords_aew = coords_fcn.copy()
                del coords_aew["lead_time"]
                x_aew, coords_aew = map_coords(x_aew, coords_aew, aew_tracker.detect.input_coords())
                # Set current and next time (needed for AEW propagation)
                aew_tracker.detect._current_time = np.array([times[step]])
                if step < nsteps:
                   aew_tracker.detect._next_time = np.array([times[step+1]])
                else:
                   aew_tracker.detect._next_time = None
                # Detect AEW candidates for the current forecast time
                aew_tensor, aew_coords = aew_tracker.detect(x_aew, coords_aew)
                #
                # End: AEW detection

                pbar.update(1)
                if step == nsteps:
                    break

        # Begin: Post-process tracks
        #
        # Stitch TC detections
        tc_tracker.stitch._nsteps = nsteps+1
        tc_tracks_tensor, tc_track_coords = tc_tracker.stitch(tc_tensor, tc_coords)

        # Filter AEW detections
        aew_tracks_tensor, aew_track_coords = aew_tracker.filter(aew_tensor, aew_coords)
        #
        # End: Post-process tracks

        end_fcn3 = time.perf_counter()
        print(f"Elapsed time(fcn3[{batch_id}]): {end_fcn3 - start_fcn3:.6f} seconds")
        #
        # End: FCN3 + TC tracker + AEW tracker

        # Begin: Gather results across MPI ranks
        #
        start_write = time.perf_counter()
        # Move tensors to CPU
        tc_local_tensor = tc_tracks_tensor.detach().cpu()
        tc_local_coords = tc_track_coords

        aew_local_tensor = aew_tracks_tensor.detach().cpu()
        aew_local_coords = aew_track_coords

        # Collect results on rank 0
        tc_gathered = comm.gather((tc_local_tensor, tc_local_coords), root=0)
        aew_gathered = comm.gather((aew_local_tensor, aew_local_coords), root=0)
        #
        # End: Gather results across MPI ranks

        # Begin: Write NetCDF output
        #
        if rank == 0:
            os.makedirs(out_path, exist_ok=True)

            tc_file_name = f"tc_tracks_"+ic+"_seed_"+str(seed)+"_batch_"+str(batch_id)+".nc"
            tc_full_path = os.path.join(out_path, tc_file_name)

            aew_file_name = f"aew_tracks_"+ic+"_seed_"+str(seed)+"_batch_"+str(batch_id)+".nc"
            aew_full_path = os.path.join(out_path, aew_file_name)

            # Write each ensemble member into a NetCDF group
            tc_gathered_tensors, tc_gathered_coords = zip(*tc_gathered)
            for ens_id, (tensor, coords) in enumerate(zip(tc_gathered_tensors, tc_gathered_coords)):
                tracks_da = xr.DataArray(
                    data=tensor.numpy(),
                    coords=coords,
                    dims=list(coords.keys()),
                    name="tc_tracks",
                )

                tracks_da.to_netcdf(
                    tc_full_path,
                    group=f"ensemble_"+str(ens_id),
                    mode="a" if ens_id > 0 else "w",
                )

            # Write each ensemble member into a NetCDF group
            aew_gathered_tensors, aew_gathered_coords = zip(*aew_gathered)
            for ens_id, (tensor, coords) in enumerate(zip(aew_gathered_tensors, aew_gathered_coords)):
                tracks_da = xr.DataArray(
                    data=tensor.numpy(),
                    coords=coords,
                    dims=list(coords.keys()),
                    name="aew_tracks",
                )

                tracks_da.to_netcdf(
                    aew_full_path,
                    group=f"ensemble_"+str(ens_id),
                    mode="a" if ens_id > 0 else "w",
                )
        end_write = time.perf_counter()
        print(f"Elapsed time(write): {end_write - start_write:.6f} seconds")
        #
        # End: Write NetCDF output

    end_total = time.perf_counter()
    print(f"Elapsed time(total): {end_total - start_total:.6f} seconds")
