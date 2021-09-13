#!/bin/bash
#
# This script trains the BRepNet model with hyper-parameters which reproduce the
# results in the paper 
# BRepNet: A Topological Message Passing System for Solid Models
# Joseph G. Lambourne, Karl D.D. Willis, Pradeep Kumar Jayaraman, Aditya Sanghi
# Peter Meltzer and Hooman Shayani
# Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
# June 2021.   Pages 12773-12782
#
# Args:  /path/to/where/you/downloaded/s2.0.0/
if [ $# -ne 1 ]
then
    echo "Call this script from the BRepNet folder"
    echo "Usage train/reproduce_paper_results.sh /path/to/where/you/downloaded/s2.0.0/"
    exit 1
fi

python -m train.train \
  --dataset_file $1/processed/dataset.json \
  --dataset_dir  $1/processed/ \
  --num_layers 2 \
  --use_face_grids 0 \
  --use_edge_grids 0 \
  --use_coedge_grids 0 \
  --use_face_features 1 \
  --use_edge_features 1 \
  --use_coedge_features 1 \
  --dropout 0.0 \
  --max_epochs 50