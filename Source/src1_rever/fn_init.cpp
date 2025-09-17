/**
 * @file fn_init.cpp
 * @author milk-white-way (tam.thien.nguyen@ndsu.edu)
 * @brief 
 * @version 0.3
 * @date 2024-06-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <AMReX_MultiFabUtil.H>

#include "fn_init.H"
#include "fn_enforce_wall_bcs.H"
#include "kn_init.H"
#include "utilities.H"

using namespace amrex;
// ================================= MODULE | INITIALIZATION =================================

/**
 * @brief This function initializes the velocity components at face centers and the pressure components at cell centers.
 * 
 * @param userCtx 
 * @param velCont 
 * @param velContPrev 
 * @param velContDiff 
 * @param geom 
 */
void hybrid_grid_init ( MultiFab& userCtx,
                        Array<MultiFab, AMREX_SPACEDIM>& velCont,
                        Array<MultiFab, AMREX_SPACEDIM>& velContPrev,
                        MultiFab& velCart,
                        MultiFab& velCartPrev,
                        Geometry const& geom,
                        int const& Nghost,
                        Vector<int> const& phy_bc_lo,
                        Vector<int> const& phy_bc_hi,
						Vector<amrex::Real> const& inflow_waveform,
                        Real& time,
                        Real const& dt,
                        int const& n_cell )
{
    Box dom(geom.Domain());
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
	amrex::Print() << "INFO| dx = " << dx[0] << " dy = " << dx[1];
#if (AMREX_SPACEDIM > 2)
    amrex::Print() << " dz = " << dx[2] << "\n";
#else
    amrex::Print() << "\n";
#endif
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

// Initialize velocity components at cells' face centers
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        auto const& vel_cont_x = velCont[0].array(mfi);
        auto const& vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_z = velCont[2].array(mfi);
#endif

        auto const& vel_cont_prev_x = velContPrev[0].array(mfi);
        auto const& vel_cont_prev_y = velContPrev[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_prev_z = velContPrev[2].array(mfi);
#endif

		int lo = dom.smallEnd(0);
		int hi = dom.bigEnd(0);
        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            // CASE: 2D Taylor-Green vortex flow
            amrex::Real x = prob_lo[0] + (i + Real(0.0)) * dx[0];
            amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM > 2)
			amrex::Real z = prob_lo[2] + (k + Real(0.5)) * dx[2];
#endif

            //vel_cont_x(i, j, k) = std::sin(amrex::Real(2.0) * M_PI * x) * std::cos(amrex::Real(2.0) * M_PI * y);
            vel_cont_x(i, j, k) = std::sin(x) * std::cos(y);
			/*
#if (AMREX_SPACEDIM > 2)
				* std::cos(z);
#else
				;
#endif
			*/
            
			vel_cont_prev_x(i, j, k) = std::sin(x) * std::cos(y) * std::exp(-Real(2.0) * (time - dt));
			/*
#if (AMREX_SPACEDIM > 2)
				* std::cos(z) * std::exp(-Real(2.0) * (time - dt));
#else
				* std::exp(-Real(2.0) * (time - dt));
#endif
			*/

            // CASE: 2D lid-driven cavity flow
			vel_cont_x(i, j, k) = amrex::Real(0.0);
			vel_cont_prev_x(i, j, k) = amrex::Real(0.0);

            //const std::string &debug_file_1 = "ucont_x_at_init.txt";
            //write_exact_line_solution(time, x, y, vel_cont_x(i, j, k), vel_cont_prev_x(i, j, k), debug_file_1);
        });

        lo = dom.smallEnd(1);
        hi = dom.bigEnd(1);
        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            // CASE: 2D Taylor-Green vortex flow
            amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
            amrex::Real y = prob_lo[1] + (j + Real(0.0)) * dx[1];
#if (AMREX_SPACEDIM > 2)
			amrex::Real z = prob_lo[2] + (k + Real(0.5)) * dx[2];
#endif

            //vel_cont_y(i, j, k) = - std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y);

            vel_cont_y(i, j, k) = - std::cos(x) * std::sin(y);
			/*
#if (AMREX_SPACEDIM > 2)
				* std::cos(z);
#else
				;
#endif
			*/

            vel_cont_prev_y(i, j, k) = - std::cos(x) * std::sin(y) * std::exp(-Real(2.0) * (time - dt));
			/*
#if (AMREX_SPACEDIM > 2)
				* std::cos(z) * std::exp(-Real(2.0) * (time - dt));
#else
				* std::exp(-Real(2.0) * (time - dt));
#endif
			*/

            // CASE: 2D lid-driven cavity flow
			vel_cont_y(i, j, k) = amrex::Real(0.0);
			vel_cont_prev_y(i, j, k) = amrex::Real(0.0);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            vel_cont_z(i, j, k) = amrex::Real(0.0);
            vel_cont_prev_z(i, j, k) = amrex::Real(0.0);
        });
#endif
    }

    // Initialize cartesian velocity components at cell centers
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
	for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
	{		
		const Box &vbx = mfi.validbox();
		auto const &ucart_init = velCart.array(mfi);
		auto const &ucart_prev = velCartPrev.array(mfi);

		amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            // CASE: 2D Taylor-Green vortex flow
			amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
			amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM > 2)
			amrex::Real z = prob_lo[2] + (k + Real(0.5)) * dx[2];
#endif

            ucart_init(i, j, k, 0) = std::sin(x) * std::cos(y);
			/*
#if (AMREX_SPACEDIM > 2)
				* std::sin(z);
#else
				;
#endif
			*/

            ucart_init(i, j, k, 1) = - std::cos(x) * std::sin(y);
			/*
#if (AMREX_SPACEDIM > 2)
				* std::sin(z);
#else
				;
#endif
			*/

            // CASE: 2D lid-driven cavity flow
            ucart_init(i, j, k, 0) = amrex::Real(0.0);
            ucart_init(i, j, k, 1) = amrex::Real(0.0);


#if (AMREX_SPACEDIM > 2)
            // CASE: 3D lid-driven cavity flow
            ucart_init(i, j, k, 2) = amrex::Real(0.0);
#endif

            // CASE: 2D lid-driven cavity flow
            ucart_prev(i, j, k, 0) = amrex::Real(0.0);
            ucart_prev(i, j, k, 1) = amrex::Real(0.0);

            // CASE: 2D Taylor-Green vortex flow
            //ucart_prev(i, j, k, 0) = std::sin(x) * std::cos(y) * std::exp(-Real(2.0) * (time - dt));
            //ucart_prev(i, j, k, 1) = - std::cos(x) * std::sin(y) * std::exp(-Real(2.0) * (time - dt));
#if (AMREX_SPACEDIM > 2)
            // CASE: 3D lid-driven cavity flow
            ucart_prev(i, j, k, 2) = amrex::Real(0.0);

            // CASE: 2D Taylor-Green vortex flow
            //ucart_prev(i, j, k, 2) = amrex::Real(0.0);
#endif
					
			//const std::string &debug_file_2 = "ucart_x_at_init";
			//write_exact_line_solution(time, x, y, ucart_init(i, j, k, 0), ucart_prev(i, j, k, 0), debug_file_2);
		});
	}	

    // Fill ghost cells
    // Periodic boundary conditions
    // -- periodic: 111
    velCart.FillBoundary(geom.periodicity());
    velCartPrev.FillBoundary(geom.periodicity()); // Not used in the calculation, so don't bother
    // Physical boundary conditions
    // -- wall: 135 (no-slip), -135 (slip)
    // -- inlet: 165 (constant velocity), -165 (time-dependent velocity)
    // -- outlet: 195 (constant velocity), -195 (time-dependent velocity)
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   	for (MFIter mfi(velCart); mfi.isValid(); ++mfi) {
      	const Box &vbx = mfi.growntilebox(Nghost);
      	auto const &vel_cart = velCart.array(mfi);

		auto const &west_wall_bcs = phy_bc_lo[0]; 	// x- wall
		auto const &east_wall_bcs = phy_bc_hi[0]; 	// x+ wall
	 	auto const &south_wall_bcs = phy_bc_lo[1];	// y- wall
		auto const &north_wall_bcs = phy_bc_hi[1];	// y+ wall
#if (AMREX_SPACEDIM > 2)
		auto const &bakward_wall_bcs = phy_bc_lo[2]; // z- wall
		auto const &forward_wall_bcs = phy_bc_hi[2]; // z+ wall
#endif

		auto const &inflow_x = inflow_waveform[0];
		auto const &inflow_y = inflow_waveform[1];
#if (AMREX_SPACEDIM > 2)
		auto const &inflow_z = inflow_waveform[2];
#endif

		//amrex::Print() << "INFO| Uniflow velocity at inflow boundary: (" << inflow_x << ", " << inflow_y << ")\n";

		int lo = dom.smallEnd(0);
		int hi = dom.bigEnd(0);
		// Ghost cells to the left (of the West wall)
		if (vbx.smallEnd(0) < lo) {
			if (west_wall_bcs == 135) {
				amrex::ParallelFor(vbx,
                               	   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i < lo) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(-i - 1, j, k, dir);
						}
					}
				});
			} else if (west_wall_bcs == -135) {
				amrex::ParallelFor(vbx,
	 							   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i < lo) {
						vel_cart(i, j, k, 0) = -vel_cart(-i - 1, j, k, 0);
						vel_cart(i, j, k, 1) = vel_cart(-i - 1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
						vel_cart(i, j, k, 2) = vel_cart(-i - 1, j, k, 2);
#endif
					}
				});
			} else if (west_wall_bcs == 165) {
				amrex::ParallelFor(vbx,
										 [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i < lo) {
						vel_cart(i, j, k, 0) = 2 - vel_cart(-i - 1, j, k, 0);
						vel_cart(i, j, k, 1) = 2 - vel_cart(-i - 1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
						vel_cart(i, j, k, 2) = 2 - vel_cart(-i - 1, j, k, 2);
#endif
					}
				});
			} else if (west_wall_bcs == -165) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i < lo) {
						vel_cart(i, j, k, 0) = 2 - vel_cart(-i - 1, j, k, 0);
						vel_cart(i, j, k, 1) = 2 - vel_cart(-i - 1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
						vel_cart(i, j, k, 2) = 2 - vel_cart(-i - 1, j, k, 2);
#endif
					}
				});
			} else if (west_wall_bcs == 195) {
				amrex::Abort("Not yet implemented");
			} else if (west_wall_bcs == -195) {
				amrex::Abort("Not yet implemented");
			}
		}

		// Ghost cells to the right (of the East wall)
		if (vbx.bigEnd(0) > hi) {
			if (east_wall_bcs == 135) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i > hi) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(((n_cell - i) + (n_cell - 1)), j, k, dir);
						}
					}
				});
			} else if (east_wall_bcs == 1) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i > hi) {
						vel_cart(i, j, k, 0) = -vel_cart(((n_cell - i) + (n_cell - 1)), j, k, 0);
						vel_cart(i, j, k, 1) = vel_cart(((n_cell - i) + (n_cell - 1)), j, k, 1);
#if (AMREX_SPACEDIM > 2)
						vel_cart(i, j, k, 2) = vel_cart(((n_cell - i) + (n_cell - 1)), j, k, 2);
#endif
					}
				});
			} else if (east_wall_bcs == 165) {
				amrex::ParallelFor(vbx,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i > hi) {
						vel_cart(i, j, k, 0) = 2 - vel_cart(((n_cell - i) + (n_cell - 1)), j, k, 0);
						vel_cart(i, j, k, 1) = 2 - vel_cart(((n_cell - i) + (n_cell - 1)), j, k, 1);
#if (AMREX_SPACEDIM > 2)
						vel_cart(i, j, k, 2) = 2 - vel_cart(((n_cell - i) + (n_cell - 1)), j, k, 2);
#endif
					}
				});
			} else if (east_wall_bcs == -165) {
				amrex::Abort("Not yet implemented");
			} else if (east_wall_bcs == 195) {
				amrex::Abort("Not yet implemented");
			} else if (east_wall_bcs == -195) {
				amrex::Abort("Not yet implemented");
			}
		}

		lo = dom.smallEnd(1);
		hi = dom.bigEnd(1);
		// Ghost cells to the bottom (of the South wall)
		if (vbx.smallEnd(1) < lo) {
			if (south_wall_bcs == 135) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j < lo) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(i, -j - 1, k, dir);
						}
					}
				});
			} else if (south_wall_bcs == -135) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j < lo) {
						vel_cart(i, j, k, 0) = vel_cart(i, -j - 1, k, 0);
						vel_cart(i, j, k, 1) = -vel_cart(i, -j - 1, k, 1);
#if (AMREX_SPACEDIM > 2)
						vel_cart(i, j, k, 2) = vel_cart(i, -j - 1, k, 2);
#endif
					}
				});
			} else if (south_wall_bcs == 165) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j < lo) {
						vel_cart(i, j, k, 0) = 2 - vel_cart(i, -j - 1, k, 0);
						vel_cart(i, j, k, 1) = 2 - vel_cart(i, -j - 1, k, 1);
#if (AMREX_SPACEDIM > 2)
						vel_cart(i, j, k, 2) = 2 - vel_cart(i, -j - 1, k, 2);
#endif
					}
				});
			} else if (south_wall_bcs == -165) {
				amrex::Abort("Not yet implemented");
			} else if (south_wall_bcs == 195) {
				amrex::Abort("Not yet implemented");
			} else if (south_wall_bcs == -195) {
				amrex::Abort("Not yet implemented");
			}
		}

		// Ghost cells to the top (of the North wall)
		if (vbx.bigEnd(1) > hi) {
			if (north_wall_bcs == 135) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j > hi) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(i, ((n_cell - j) + (n_cell - 1)), k, dir);
						}
					}
				});
			} else if (north_wall_bcs == -135) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j > hi) {
						vel_cart(i, j, k, 0) = vel_cart(i, ((n_cell - j) + (n_cell - 1)), k, 0);
						vel_cart(i, j, k, 1) = -vel_cart(i, ((n_cell - j) + (n_cell - 1)), k, 1);
#if (AMREX_SPACEDIM > 2)
						vel_cart(i, j, k, 2) = vel_cart(i, ((n_cell - j) + (n_cell - 1)), k, 2);
#endif
					}
				});
			} else if (north_wall_bcs == 165) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j > hi) {
							vel_cart(i, j, k, 0) = 2*inflow_x - vel_cart(i, ((n_cell - j) + (n_cell - 1)), k, 0);
							vel_cart(i, j, k, 1) = 2*inflow_y - vel_cart(i, ((n_cell - j) + (n_cell - 1)), k, 1);
#if (AMREX_SPACEDIM > 2)
							vel_cart(i, j, k, 2) = 2*inflow_z - vel_cart(i, ((n_cell - j) + (n_cell - 1)), k, 2);
#endif
						//amrex::Print() << "INFO| Applied north wall BCs (y+): (i , j) = (" << i << ", " << j << ") "
						//		   	   << "x-vel = " << vel_cart(i, j, k, 0) 
						//		   	   << ", y-vel = " << vel_cart(i, j, k, 1) << "\n";
					}
				});
			} else if (north_wall_bcs == -165) {
				amrex::Abort("Not yet unlocked");
			} else if (north_wall_bcs == 195) {
				amrex::Abort("Not yet unlocked");
			} else if (north_wall_bcs == -195) {
				amrex::Abort("Not yet unlocked");
			}
		}

#if (AMREX_SPACEDIM > 2)
		lo = dom.smallEnd(2);
		hi = dom.bigEnd(2);

		if (vbx.smallEnd(2) < lo) {
			if (bakward_wall_bcs == 135) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k < lo) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(i, j, -k -1, dir);
						}
					}
				});
			} else if (bakward_wall_bcs == -135) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k < lo) {
						vel_cart(i, j, k, 0) = vel_cart(i, j, -k -1, 0);
						vel_cart(i, j, k, 1) = vel_cart(i, j, -k -1, 1);
						vel_cart(i, j, k, 2) = -vel_cart(i, j, -k -1, 2);
					}
				});
			} else if (bakward_wall_bcs == 165) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k < lo) {
						vel_cart(i, j, k, 0) = 2 - vel_cart(i, j, -k -1, 0);
						vel_cart(i, j, k, 1) = 2 - vel_cart(i, j, -k -1, 1);
						vel_cart(i, j, k, 2) = 2 - vel_cart(i, j, -k -1, 2);
					}
				});
			} else if (bakward_wall_bcs == -165) {
				amrex::Abort("Not yet unlocked");
			} else if (bakward_wall_bcs == 195) {
				amrex::Abort("Not yet unlocked");
			} else if (bakward_wall_bcs == -195) {
				amrex::Abort("Not yet unlocked");
			}
		}

		if (vbx.bigEnd(2) > hi) {
			if (forward_wall_bcs == 135) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k > hi) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(i, j, ((n_cell - k) + (n_cell - 1)), dir);
						}
					}
				});
			} else if (forward_wall_bcs == -135) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k > hi)	{
						vel_cart(i, j, k, 0) = vel_cart(i, j, ((n_cell - k) + (n_cell - 1)), 0);
						vel_cart(i, j, k, 1) = vel_cart(i, j, ((n_cell - k) + (n_cell - 1)), 1);
						vel_cart(i, j, k, 2) = -vel_cart(i, j, ((n_cell - k) + (n_cell - 1)), 2);
					}
				});
			} else if (forward_wall_bcs == 165) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k > hi) {
						vel_cart(i, j, k, 0) = 2 - vel_cart(i, j, ((n_cell - k) + (n_cell - 1)), 0);
						vel_cart(i, j, k, 1) = 2 - vel_cart(i, j, ((n_cell - k) + (n_cell - 1)), 1);
						vel_cart(i, j, k, 2) = 2 - vel_cart(i, j, ((n_cell - k) + (n_cell - 1)), 2);
					}
				});
			} else if (forward_wall_bcs == -165) {
				amrex::Abort("Not yet unlocked");
			} else if (forward_wall_bcs == 195) {
				amrex::Abort("Not yet unlocked");
			} else if (forward_wall_bcs == -195) {
				amrex::Abort("Not yet unlocked");
			}
		}
#endif
	}

// Initialize pressure components at celll centers
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(userCtx); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx = userCtx.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_userCtx(i, j, k, ctx, dx, prob_lo);
        });
    }

    userCtx.FillBoundary(geom.periodicity());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
	for (MFIter mfi(userCtx); mfi.isValid(); ++mfi)
	{
		const Box& vbx = mfi.growntilebox(1);
		auto const& ctx = userCtx.array(mfi);

		int lo = dom.smallEnd(0); //amrex::Print() << lo << "\n";
		int hi = dom.bigEnd(0);   //amrex::Print() << hi << "\n";

		if (vbx.smallEnd(0) < lo) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( i < lo ) {
					ctx(i, j, k, 0) = ctx(-i - 1, j, k, 0);
					ctx(i, j, k, 1) = ctx(-i - 1, j, k, 1);
				}
			});
		}

		if (vbx.bigEnd(0) > hi) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( i > hi ) {
					ctx(i, j, k, 0) = ctx(((n_cell - i) + (n_cell - 1)), j, k, 0);
					ctx(i, j, k, 1) = ctx(((n_cell - i) + (n_cell - 1)), j, k, 1);
				}
			});
		}

		lo = dom.smallEnd(1);
		hi = dom.bigEnd(1);

		if (vbx.smallEnd(1) < lo) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( j < lo ) {
					ctx(i, j, k, 0) = ctx(i, -j - 1, k, 0);
					ctx(i, j, k, 1) = ctx(i, -j - 1, k, 1);
				}
			});
		}

		if (vbx.bigEnd(1) > hi) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( j > hi ) {
					ctx(i, j, k, 0) = ctx(i, ((n_cell - j) + (n_cell - 1)), k, 0);
					ctx(i, j, k, 1) = ctx(i, ((n_cell - j) + (n_cell - 1)), k, 1);
				}
			});
		}

#if (AMREX_SPACEDIM > 2)
		lo = dom.smallEnd(2);
		hi = dom.bigEnd(2);

		if (vbx.smallEnd(2) < lo) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( k < lo ) {
					ctx(i, j, k, 0) = ctx(i, j, -k - 1, 0);
					ctx(i, j, k, 1) = ctx(i, j, -k - 1, 1);
				}
			});
		}

		if (vbx.bigEnd(2) > hi) {
			amrex::ParallelFor(vbx,
							   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
				if ( k > hi ) {
					ctx(i, j, k, 0) = ctx(i, j, ((n_cell - k) + (n_cell - 1)), 0);
					ctx(i, j, k, 1) = ctx(i, j, ((n_cell - k) + (n_cell - 1)), 1);
				}
			});
		}
#endif
	}
}

void init (MultiFab& userCtx,
           MultiFab& velCart,
           MultiFab& velCartDiff,
           Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
           Geometry const& geom)
{

    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(userCtx); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx = userCtx.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_userCtx(i, j, k, ctx, dx, prob_lo);
        });
    }
    userCtx.FillBoundary(geom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vcart = velCart.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_cartesian_velocity(i, j, k, vcart, dx, prob_lo);
        });
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCartDiff); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vcart_diff = velCartDiff.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_cartesian_velocity_difference(i, j, k, vcart_diff);
        });
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velContDiff[0]); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        auto const& xcont_diff = velContDiff[0].array(mfi);
        auto const& ycont_diff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zcont_diff = velContDiff[2].array(mfi);
#endif
        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            xcont_diff(i, j, k) = Real(0.0);
        });
        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            ycont_diff(i, j, k) = Real(0.0);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            zcont_diff(i, j, k) = Real(0.0);
        });
#endif
    }
}
