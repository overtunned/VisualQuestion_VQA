#!/bin/bash
#PBS -l select=2:ncpus=14:mem=32gb
#PBS -N visual_bert
#PBS -l software=python
#PBS -V
#PBS -q workq
#EXECUTION SEQUENCE
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
pip3 install --upgrade tensorflow
echo "Visualizing BERT"
export OMP_NUM_THREADS=8; /usr/bin/python3 <Visual_Attention/bert_features_extract.py> "$PBSJOBID.log"