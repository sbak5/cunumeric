#!/bin/bash
#SBATCH -J legate
#SBATCH -t 60
#SBATCH -p batch

export LEGATE_FIELD_REUSE_FREQ=1
for size in 625 1250 2500 3750 5000 
do
    #strong scaling
    for nnodes in 1 2 4
    do    
        RESULT_DIR=./result/$SLURM_JOB_ID/${nnodes}_$(($size * 8))_strong
        mkdir -p $RESULT_DIR 
        /home/sbak/gpfs/legate.core/install/bin/legate --launcher mpirun --numamem 200000 --omps 2 --ompthreads 18 --cpus 1 --gpus 8 --sysmem 200000 --fbmem 14500 --verbose --logdir $RESULT_DIR --nodes $nnodes --ranks-per-node 1  --profile examples/$1 -n $size -N 1 -i 12 -G 8 -logfile $RESULT_DIR/%.log
    done
    #weak scailing
    for nnodes in 1 2 4
    do    
        RESULT_DIR=./result/$SLURM_JOB_ID/${nnodes}_$(($size * 8 ))_weak
        mkdir -p $RESULT_DIR 
        /home/sbak/gpfs/legate.core/install/bin/legate --launcher mpirun --numamem 200000 --omps 2 --ompthreads 18 --cpus 1 --gpus 8 --sysmem 200000 --fbmem 14500 --verbose --logdir $RESULT_DIR --nodes $nnodes --ranks-per-node 1  --profile examples/$1 -n $size -N $nnodes -G 8 -i 12 -logfile $RESULT_DIR/%.log
    done
    # weak scailing in a single node
    for ngpus in 1 2 4
    do    
        RESULT_DIR=./result/$SLURM_JOB_ID/1_${size}_weak_${ngpus}
        mkdir -p $RESULT_DIR 
        /home/sbak/gpfs/legate.core/install/bin/legate --launcher mpirun --numamem 200000 --omps 2 --ompthreads 18 --cpus 1 --gpus 8 --sysmem 200000 --fbmem 14500 --verbose --logdir $RESULT_DIR --nodes 1 --ranks-per-node 1  --profile examples/$1 -n $size -N 1 -G $ngpus -i 12 -logfile $RESULT_DIR/%.log
    done
done
