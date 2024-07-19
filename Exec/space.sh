#!/usr/bin/env bash

mpiexec -n 4 ./main2d.gnu.MPI.ex inputs_convergence fixed_dt=1.e-6 nsteps=100 plot_int=100 n_cell=64 max_grid_size=64 | grep BENCHMARKING
printf "\n"
mpiexec -n 4 ./main2d.gnu.MPI.ex inputs_convergence fixed_dt=1.e-6 nsteps=100 plot_int=100 n_cell=128 max_grid_size=128 | grep BENCHMARKING
printf "\n"
mpiexec -n 4 ./main2d.gnu.MPI.ex inputs_convergence fixed_dt=1.e-6 nsteps=100 plot_int=100 n_cell=256 max_grid_size=256 | grep BENCHMARKING
