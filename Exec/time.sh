mpiexec -n 4 ./main2d.gnu.MPI.ex inputs_convergence fixed_dt=1.e-5  nsteps=1 plot_int=1 n_cell=128 max_grid_size=64 | grep BENCHMARKING
printf "\n"
mpiexec -n 4 ./main2d.gnu.MPI.ex inputs_convergence fixed_dt=5.e-6  nsteps=2 plot_int=2 n_cell=128 max_grid_size=64 | grep BENCHMARKING
printf "\n"
mpiexec -n 4 ./main2d.gnu.MPI.ex inputs_convergence fixed_dt=2.5e-6 nsteps=4 plot_int=4 n_cell=128 max_grid_size=64 | grep BENCHMARKING
