#!/bin/bash
#PBS -N ClusterHelloWorld
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,mem=1000mb,nice=10,walltime=00:01:00
#PBS -m a
#PBS -j oe
sleep 10
echo "Hello World"
exit 0
