#!/usr/bin/env python3
"""
Integrate FCN3 ensemble with TC and AEW trackers in a typical Earth2Studio workflow.

This script tracks tropical cyclones (TC) and African easterly waves (AEW) using outputs from Earth2Studio's FCN3 model.
The tropical cyclone detection algorithm based on TempestExtremes runs on TECA.
The African easterly wave detection algorithm is based on original code by Emily Bercos-Hickey.

Usage:
    python run_fcn3.py --initial-condition INITIAL_CONDITION --nsteps NSTEPS --seed SEED --out_path OUT_PATH

Parameters:
    --initial-condition INITIAL_CONDITION: Initial condition date in YYYY-MM-DD format (e.g., "2024-01-01")
    --nsteps NSTEPS: Number of forecast steps (default: 60)
    --seed SEED: Seed for each MPI process (default: 333)
    --out_path OUT_PATH: Directory where output files will be written
"""
import argparse
import os
import time
from datetime import datetime, timedelta

import torch
import xarray as xr
import numpy as np
from tqdm import tqdm

# Earth2Studio imports
from earth2studio.data import ARCO
from earth2studio.data import NCAR_ERA5
from earth2studio.data import fetch_data, prep_data_array
from earth2studio.utils.time import to_time_array
from earth2studio.utils.coords import CoordSystem
from earth2studio.utils.coords import map_coords

from mpi4py import MPI

# Earth2Studio imports
from earth2studio.models.px import FCN3
from earth2studio.perturbation import Zero
from earth2studio.run import ensemble

# Local import
from earth2studio.models.dx.tempest_tc_detect import teca_tempest_tc_detect
from earth2studio.models.dx.aews_detect import aews_detect

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run FCN3 model for a single initial condition and "
                    "detect tropical cyclones and African easterly waves using its outputs.")

    parser.add_argument(
        "--initial-condition", type=str, required=True,
        help="Initial condition date in format YYYY-MM-DD"
    )
    parser.add_argument(
        "--nsteps", type=int, default=60,
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

    return parser.parse_args()

def model_init(comm, node_comm, local_size, local_rank, device, start_date):
    """Load the default FCN3 model and data source."""
    print(f"Loading FCN3 model...")

    if rank == 0:
        package = FCN3.load_default_package()
    comm.Barrier()

    # Serialize only inside node
    for r in range(local_size):
       if local_rank == r:
          package = FCN3.load_default_package()
          prognostic = FCN3.load_model(package).to(device)
       # Synchronize within node
       node_comm.Barrier()

    if rank == 0:
        print(f"Rank 0 loading initial model state...")
        # Create the data source
        data = NCAR_ERA5()

        xx, coords = fetch_data(
            source=data,
            time=to_time_array([start_date]),
            variable=prognostic.input_coords()["variable"],
            lead_time=prognostic.input_coords()["lead_time"],
            device="cpu",
        )
        shape = xx.shape
        dtype = xx.numpy().dtype

        print(f"Rank 0 loading ERA5 surface geopotential...")
        ds = xr.open_dataset("e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc")
        ds = ds.rename({"Z": "z"})
        da = ds["z"]
        da = da.expand_dims(dim={"variable": ["z"]})
        da = da.transpose("time", "variable", "latitude", "longitude")
        zz, coords_z = prep_data_array(da, device="cpu")
        shape_z = zz.shape
        dtype_z = zz.numpy().dtype
    else:
        xx = None
        coords = None
        shape = None
        dtype = None

        zz = None
        coords_z = None
        shape_z = None
        dtype_z = None

    comm.Barrier()

    # Begin: Broadcast from rank 0
    #
    # Metadata
    shape = comm.bcast(shape, root=0)
    dtype = comm.bcast(dtype, root=0)
    coords = comm.bcast(coords, root=0)

    shape_z = comm.bcast(shape_z, root=0)
    dtype_z = comm.bcast(dtype_z, root=0)
    coords_z = comm.bcast(coords_z, root=0)

    # Allocate tensor on non-root ranks
    if rank != 0:
        xx = torch.empty(shape, dtype=torch.from_numpy(np.empty((), dtype=dtype)).dtype)
        zz = torch.empty(shape_z, dtype=torch.from_numpy(np.empty((), dtype=dtype_z)).dtype)

    # Raw memory
    comm.Bcast(xx.numpy(), root=0)
    comm.Bcast(zz.numpy(), root=0)
    z = zz.to(device)
    #
    # End: Broadcast from rank 0

    return prognostic, xx, coords, z, coords_z

def run_tc_tracker(tc_tracker, x_fcn, coords_fcn, z, coords_z, step):
    """Run tropical cyclone tracker."""
    print(f"Running tc tracker...")
    # Delete lead_time, no need for it since steps are present in the tracks
    # Combine geopotential height with FCN3 output fields
    x_tc = torch.cat((z.unsqueeze(2), x_fcn.squeeze(1)), dim=2)
    coords_tc = coords_fcn.copy()
    coords_tc["variable"] = np.concatenate([coords_z["variable"], coords_fcn["variable"]])
    del coords_tc["lead_time"]
    x_tc, coords_tc = map_coords(x_tc, coords_tc, tc_tracker.detect.input_coords())

    # Set current time
    tc_tracker.detect._current_time = np.array([np.datetime64(step, 'ns')])

    # Detect TC candidates for the current forecast time
    tc_tensor, tc_coords = tc_tracker.detect(x_tc, coords_tc)

    return tc_tensor, tc_coords

def run_aew_tracker(aew_tracker, x_fcn, coords_fcn, current_time, next_time, step, nsteps):
    """Run African easterly wave tracker."""
    print(f"Running aew tracker...")
    x_aew = x_fcn.squeeze(1)
    coords_aew = coords_fcn.copy()
    del coords_aew["lead_time"]
    x_aew, coords_aew = map_coords(x_aew, coords_aew, aew_tracker.detect.input_coords())

    # Set current and next time (needed for AEW propagation)
    aew_tracker.detect._current_time = current_time
    aew_tracker.detect._next_time = next_time

    # Detect AEW candidates for the current forecast time
    aew_tensor, aew_coords = aew_tracker.detect(x_aew, coords_aew)

    return aew_tensor, aew_coords

def write_data(gathered, full_path):
    """Write track data to files."""
    print(f"Writing track data...")
    # Write each ensemble member into a NetCDF group
    gathered_tensors, gathered_coords = zip(*gathered)
    for ens_id, (tensor, coords) in enumerate(zip(gathered_tensors, gathered_coords)):
        tracks_da = xr.DataArray(
            data=tensor.numpy(),
            coords=coords,
            dims=list(coords.keys()),
            name="tracks",
        )

        tracks_da.to_netcdf(
            full_path,
            group=f"ensemble_"+str(ens_id),
            mode="a" if ens_id > 0 else "w",
        )

def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()

    # Begin: Extract run configuration from arguments
    #
    ic = args.initial_condition
    # Convert date string to datetime object
    start_date = datetime.fromisoformat(ic)

    # Number of forecast steps
    nsteps = args.nsteps
    times = [start_date + timedelta(hours=6 * i) for i in range(nsteps+1)]

    nensemble = args.local_ensemble
    batch_size = 1

    seed = args.seed
    out_path = args.out_path
    #
    # End: Extract run configuration from arguments

    # Begin: MPI setup
    #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = node_comm.Get_rank()
    local_size = node_comm.Get_size()
    #
    # End: MPI setup

    # Select compute device: CPU or GPU
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running FCN3 for initial condition {ic}, nsteps={nsteps}, local_ensemble={nensemble}")
    print(f"Rank {rank} of {size}, PID {os.getpid()}")
    print(f"Rank {rank}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, local_rank={local_rank}, local_size={local_size}")
    print(f"Output path for track files: {out_path}")
    start_total = time.perf_counter()

    start_init = time.perf_counter()
    # Load the default FCN3 model and data source
    prognostic, xx, coords, z, coords_z = model_init(comm, node_comm, local_size, local_rank, device, start_date)
    end_init = time.perf_counter()
    print(f"Elapsed time(fcn3-init[{rank}]): {end_init - start_init:.6f} seconds")

    # Create tropical cyclone tracker
    tc_tracker = teca_tempest_tc_detect()
    tc_tracker = tc_tracker.to(device)
    tc_tracker.detect._device = device

    # Create African Easterly Wave tracker
    aew_tracker = aews_detect()
    aew_tracker = aew_tracker.to(device)
    aew_tracker.detect._device = device

    batch_ids_produce = list(range(0, int(np.ceil(nensemble / batch_size)),))
    # Loop over ensemble batches with a progress bar
    for batch_id in tqdm(
        batch_ids_produce,
        total=len(batch_ids_produce),
        desc="Total Ensemble Batches",
    ):
        start_fcn3 = time.perf_counter()
        # Begin: FCN3 + TC tracker + AEW tracker
        #
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
                # Tropical cyclone detection
                tc_tensor, tc_coords = run_tc_tracker(tc_tracker, x_fcn, coords_fcn, z, coords_z, times[step])

                # AEW detection
                if step < nsteps:
                    aew_tensor, aew_coords = run_aew_tracker(aew_tracker, x_fcn, coords_fcn, np.array([times[step]]), np.array([times[step+1]]), step, nsteps)
                else:
                    aew_tensor, aew_coords = run_aew_tracker(aew_tracker, x_fcn, coords_fcn, np.array([times[step]]), None, step, nsteps)

                pbar.update(1)
                if step == nsteps:
                    break

        # Move tensors to CPU
        tc_local_tensor = tc_tensor.detach().cpu()
        tc_local_coords = tc_coords

        aew_local_tensor = aew_tensor.detach().cpu()
        aew_local_coords = aew_coords

        # Begin: Post-process tracks
        #
        # Stitch TC detections
        tc_tracker.stitch._nsteps = nsteps+1
        tc_tracks_tensor, tc_track_coords = tc_tracker.stitch(tc_local_tensor, tc_local_coords)

        # Filter AEW detections
        aew_tracks_tensor, aew_track_coords = aew_tracker.filter(aew_local_tensor, aew_local_coords)
        #
        # End: Post-process tracks

        end_fcn3 = time.perf_counter()
        print(f"Elapsed time(fcn3[{batch_id}]): {end_fcn3 - start_fcn3:.6f} seconds")
        #
        # End: FCN3 + TC tracker + AEW tracker

        start_write = time.perf_counter()
        # Begin: Gather results across MPI ranks
        #
        # Collect results on rank 0
        tc_gathered = comm.gather((tc_tracks_tensor, tc_track_coords), root=0)
        aew_gathered = comm.gather((aew_tracks_tensor, aew_track_coords), root=0)
        #
        # End: Gather results across MPI ranks

        # Begin: Write NetCDF output
        #
        if rank == 0:
            os.makedirs(out_path, exist_ok=True)

            # Write each ensemble member into a NetCDF group
            tc_file_name = f"tc_tracks_"+ic+"_seed_"+str(seed)+"_batch_"+str(batch_id)+"_nens_"+str(size)+".nc"
            tc_full_path = os.path.join(out_path, tc_file_name)
            write_data(tc_gathered, tc_full_path)

            # Write each ensemble member into a NetCDF group
            aew_file_name = f"aew_tracks_"+ic+"_seed_"+str(seed)+"_batch_"+str(batch_id)+"_nens_"+str(size)+".nc"
            aew_full_path = os.path.join(out_path, aew_file_name)
            write_data(aew_gathered, aew_full_path)
        #
        # End: Write NetCDF output
        end_write = time.perf_counter()
        print(f"Elapsed time(write): {end_write - start_write:.6f} seconds")

    end_total = time.perf_counter()
    print(f"Elapsed time(total): {end_total - start_total:.6f} seconds")

if __name__ == "__main__":
    main()
