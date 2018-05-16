#!/bin/bash

module load intel/compiler/64/2017/17.0.2
module load intel/mkl/64/2017/2.174


HOME='/usr/users/fsimone'
ENV="${HOME}/myenv"
LOG_SUFFIX="gx_r2"
GXPRED="${HOME}/gxpred_annots/gxpred/gx_annotations_r2.py"
DESC="fullgenome"
DIR="/cbscratch/franco/datasets/gtex_models/gxpred_models_with_annots/z1/${DESC}"
# QUEUE='mpi mpi2 mpi3_all hh sa'
RUNTIME="24:00"
CONFIG="config_fullgenome"



for CHROM in `seq 2 22`
do

for MODEL in `ls $DIR`
    do
    echo "$CHROM - $MODEL"
    # bsub -n 8 -q mpi -R span[hosts=1] -a openmp 
    # bsub -x -q mpi -W ${RUNTIME} -R scratch\
    bsub -n 2 -a openmp -q mpi -W ${RUNTIME} -R span[hosts=1] -R cbscratch \
            -o ${DIR}/${MODEL}/${LOG_SUFFIX}.log \
            -e ${DIR}/${MODEL}/${LOG_SUFFIX}.err \
            $ENV/bin/python $GXPRED --dir ${DIR}/${MODEL} \
                                    --config ${CONFIG} \
                                    --chr ${CHROM}
done    

done
