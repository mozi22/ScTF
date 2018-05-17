#!/bin/bash
PBS -N ClusterHelloWorld
PBS -S /bin/bash
PBS nodes=1:ppn=1,gpus=1,mem=10gb,walltime=01:00:00 -q default-gpu main.py
PBS -m a
PBS -M muazzama@informatik.uni-freiburg.de
PBS -j oe
sleep 10
echo "Hello World"
exit 0