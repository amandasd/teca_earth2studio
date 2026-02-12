#!/bin/bash
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -A m1517
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -n 4
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 00:30:00
#SBATCH -J fcn3_ic_4_ens_4.out
#SBATCH --output=%x-%j.out
#SBATCH --exclusive

export SLURM_CPU_BIND="cores"

module load conda
module load gcc/12.2.0
module load cmake
module load cray-libsci/23.09.1.1
module load cray-mpich/8.1.27
module load cray-hdf5-parallel/1.12.2.7
module load cray-netcdf-hdf5parallel/4.9.0.7

export EARTH2STUDIO_PACKAGE_TIMEOUT=2000
export EARTH2STUDIO_CACHE=/pscratch/sd/a/asdufek/earth2studio_cache
export PYTHONPATH=/pscratch/sd/a/asdufek/test/teca/install/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/pscratch/sd/a/asdufek/test/teca/install/lib:/global/common/software/m1517/teca/perlmutter_gpu/develop-c3979b34/lib:/pscratch/sd/a/asdufek/test/teca/install/lib64:/global/homes/a/asdufek/.conda/envs/earth2studio-env/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/lib-abi-mpich:$LD_LIBRARY_PATH

conda activate /pscratch/sd/a/asdufek/envs/earth2studio-env-v9

export MPICH_GPU_SUPPORT_ENABLED=0

BASE_SEED=333

i=0
while read -r IC; do
    echo "Running IC = $IC"

    # Each IC gets a different seed
    # Unique seed per rank and per batch_id
    # Within Python code: SEED + rank * 10 + batch_id
    SEED=$((BASE_SEED + i))

    tt=$(date +%s%N)
    srun -n 4 python run_fcn3.py --initial-condition "$IC" --seed "$SEED" </dev/null
    echo "total runtime[$IC]: $(echo "scale=3;($(date +%s%N) - ${tt})/(1*10^09)" | bc) seconds"

    echo "Finished IC = $IC"
    i=$((i+1))
done < input.txt
