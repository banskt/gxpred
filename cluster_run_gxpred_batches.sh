#!/bin/bash

# create conda env with
# conda create --prefix /usr/users/fsimone/myenv numpy scipy sklearn


module load intel/compiler/64/2017/17.0.2
module load intel/mkl/64/2017/2.174


SPLIT=$1

if [ -z ${SPLIT} ]; 
    then echo "SPLIT is not set"; 
    exit; 
elif [ $SPLIT -lt 1 ]; 
    then echo "SPLIT is less than 1"; 
    exit;
fi

# for CHROM in `seq 1 1`
# do

# CHROM="1"

HOME='/usr/users/fsimone'
ENV="${HOME}/myenv"
LOG_SUFFIX="chr12_testruns_1kgannots_fixedPI"
GXPRED="${HOME}/gxpred_annots/gxpred/learn_annotations_cluster.py"
OUTDIR="/cbscratch/franco/datasets/gtex_models/gxpred_models_with_annots"
# QUEUE='mpi mpi2 mpi3_all hh sa'
RUNTIME="24:00"
CONFIG="config_annots"


for SECTION in `seq 0 $(($SPLIT-1))`
    do
    echo "$SECTION"
    # bsub -n 8 -q mpi -R span[hosts=1] -a openmp 
    # bsub -x -q mpi -W ${RUNTIME} -R scratch\
    bsub -n 8 -a openmp -q mpi -W ${RUNTIME} -R span[hosts=1] -R cbscratch \
            -o ${OUTDIR}/section${SECTION}_${LOG_SUFFIX}.log \
            -e ${OUTDIR}/section${SECTION}_${LOG_SUFFIX}.err \
            $ENV/bin/python $GXPRED --out ${OUTDIR} \
                                    --split ${SPLIT} \
                                    --section ${SECTION} \
                                    --config ${CONFIG}
done    

# done
