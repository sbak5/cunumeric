#!/bin/bash
#SBATCH -J legate
#SBATCH -t 60
#SBATCH -p batch


ITER=100
RUN_SCRIPT="LEGATE_DIR=${INSTALL_PATH} $HOME/quickstart.internal"
COMMON_OPT="--launcher mpirun --numamem 200000 --cpus 1 --sysmem 256 --verbose --log-to-file "
ONE_PROC="$COMMON_OPT --omps 2 --ompthreads 18 --gpus 8 --fbmem 14500 --ranks-per-node 1 --cpu-bind 0-79 --nic-bind mlx5_0,mlx5_1,mlx5_2,mlx5_3"
TWO_PROC="$COMMON_OPT --omps 1 --ompthreads 16 --gpus 4 --fbmem 7250 --ranks-per-node 2 --cpu-bind 0-19,40-59/20-39,60-79 --mem-bind 0/1 --nic-bind mlx5_0,mlx5_1/mlx5_2,mlx5_3 "

export LEGATE_FIELD_REUSE_FREQ=1
for size in 625 1250 2500 3750 5000 
do
    #weak scailing
    for nnodes in 1 2 4
    do    
        TOTAL_SIZE=$(($size * 8 ))
        RESULT_DIR=./result/$SLURM_JOB_ID/${nnodes}_${TOTAL_SIZE}_weak
        mkdir -p $RESULT_DIR 
        /home/sbak/gpfs/legate.core/install/bin/legate ${ONE_PROC} --nodes $nnodes  examples/$1 -n $size -N $nnodes -G 8 -i $ITER -logfile $RESULT_DIR/%.log > $RESULT_DIR/${TOTAL_SIZE}_${nnodes}_8_one 

        sleep 5
        /home/sbak/gpfs/legate.core/install/bin/legate ${TWO_PROC} --nodes $nnodes  examples/$1 -n $size -N $nnodes -G 8 -i $ITER -logfile $RESULT_DIR/%.log > $RESULT_DIR/${TOTAL_SIZE}_${nnodes}_8_two 
    done
    # weak scailing in a single node
    for ngpus in 1 2 4
    do 
        TOTAL_SIZE=$(($size * $ngpus ))
        RESULT_DIR=./result/$SLURM_JOB_ID/1_${TOTAL_SIZE}_${ngpus}_weak
        mkdir -p $RESULT_DIR 
        /home/sbak/gpfs/legate.core/install/bin/legate $COMMON_OPT --omps 1 --ompthreads 16 --gpus $ngpus --fbmem 14500 --ranks-per-node 1 --cpu-bind 0-19,40-59 --nic-bind mlx5_0,mlx5_1 --nodes 1 --ranks-per-node 1  examples/$1 -n $size -N 1 -G $ngpus -i $ITER -logfile $RESULT_DIR/%.log > $RESULT_DIR/${TOTAL_SIZE}_${nnodes}_${ngpus}_one
    done
done
