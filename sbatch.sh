#!/bin/bash
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -A m1517
#SBATCH -N 8
#SBATCH -G 32
#SBATCH -n 32
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 00:09:00
#SBATCH -J fcn3_ens_32.out
#SBATCH --output=%x-%j.out
#SBATCH --exclusive

module load conda
module load cmake
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel

export EARTH2STUDIO_PACKAGE_TIMEOUT=2000
export EARTH2STUDIO_CACHE=/pscratch/sd/a/asdufek/earth2studio_cache

export PYTHONPATH=/pscratch/sd/a/asdufek/test/03202026/teca/install/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/pscratch/sd/a/asdufek/test/03202026/teca/install/lib:/pscratch/sd/a/asdufek/test/03202026/teca/install/lib64:/pscratch/sd/a/asdufek/en

conda activate /pscratch/sd/a/asdufek/envs/earth2studio-env-v12

BASE_SEED=333

total=$(date +%s%N)

i=0
while read -r IC; do
    echo "Running IC = $IC"

    # Each IC gets a different seed
    # Unique seed per rank and per batch_id
    # Within Python code: SEED + rank * 10 + batch_id
    SEED=$((BASE_SEED + i))

    tt=$(date +%s%N)
    srun -n 32 python -m mpi4py run_fcn3.py --initial-condition "$IC" --seed "$SEED" </dev/null
    echo "total runtime[$IC]: $(echo "scale=3;($(date +%s%N) - ${tt})/(1*10^09)" | bc) seconds"

    echo "Finished IC = $IC"
    i=$((i+1))
done < input.txt
echo "total runtime: $(echo "scale=3;($(date +%s%N) - ${total})/(1*10^09)" | bc) seconds"
