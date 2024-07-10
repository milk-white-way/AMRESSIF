#!/usr/bin/env bash

mpiexec -n 4 ./main2d.gnu.MPI.ex inputs_convergence fixed_dt=1.e-5  nsteps=100 plot_int=100 n_cell=128 max_grid_size=128 | grep BENCHMARKING
printf "\n"
mpiexec -n 4 ./main2d.gnu.MPI.ex inputs_convergence fixed_dt=5.e-6  nsteps=200 plot_int=200 n_cell=128 max_grid_size=128 | grep BENCHMARKING
printf "\n"
mpiexec -n 4 ./main2d.gnu.MPI.ex inputs_convergence fixed_dt=2.5e-6 nsteps=400 plot_int=400 n_cell=128 max_grid_size=128 | grep BENCHMARKING
