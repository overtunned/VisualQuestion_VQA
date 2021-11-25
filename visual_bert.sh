#!/bin/bash
#PBS -l select=1:ncpus=10:mpiprocs=10
#PBS -N visual_bert
#PBS -l software=python
#PBS -V
#PBS -q workq
#EXECUTION SEQUENCE
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
echo "Visualizing BERT"
export OMP_NUM_THREADS=4; /usr/bin/python3 <Visual_Attention/bert_features_extract.py> "$PBSJOBID.log"