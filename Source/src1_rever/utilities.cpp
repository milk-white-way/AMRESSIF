#include <AMReX_MultiFabUtil.H>

#include "utilities.H"

using namespace amrex;

namespace {
	void GotoNextLine (std::istream& is)
	{
		 constexpr std::streamsize bl_ignore_max { 100000 };
		 is.ignore(bl_ignore_max, '\n');
	}
}

// ===================== UTILITY | CONVERSION  =====================
void cont2cart(MultiFab &velCart,
               Array<MultiFab, AMREX_SPACEDIM> &velCont,
               const Geometry &geom,
               int const &Nghost,
               Vector<int> const &phy_bc_lo,
               Vector<int> const &phy_bc_hi,
			   Vector<amrex::Real> const& inflow_waveform,
			   Real &time,
               int const &n_cell) {
	BL_PROFILE_VAR("cont2cart()", cont2cart);

    Box dom(geom.Domain());
	// Interpolate boundary conditions on contravariant velocity components on the wall
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

	// Interpolate contravariant velocity components to cell-centered velocity components
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(velCart); mfi.isValid(); ++mfi) {
      	const Box &vbx = mfi.validbox();
      	auto const &vel_cart = velCart.array(mfi);

      	auto const &vel_cont_x = velCont[0].array(mfi);
      	auto const &vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
      	auto const &vel_cont_z = velCont[2].array(mfi);
#endif

      	// Average to interior cell-centered velocity
      	amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
     		vel_cart(i, j, k, 0) = amrex::Real(0.5) * (vel_cont_x(i, j, k) + vel_cont_x(i + 1, j, k));
			vel_cart(i, j, k, 1) = amrex::Real(0.5) * (vel_cont_y(i, j, k) + vel_cont_y(i, j + 1, k));
#if (AMREX_SPACEDIM > 2)
			vel_cart(i, j, k, 2) = amrex::Real(0.5) * (vel_cont_z(i, j, k) + vel_cont_z(i, j, k + 1));
#endif
       	});
    }

    // Fill Ghost cells according to physical boundary conditions
    // Step 1: Periodic BCs
    // -- periodic: 111
    velCart.FillBoundary(geom.periodicity());
    // Step 2: Physical BCs
    // -- wall: 135 (no-slip), -135 (slip)
    // -- inlet: 165 (constant velocity), -165 (time-dependent velocity)
    // -- outlet: 195 (constant velocity), -195 (time-dependent velocity)
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   	for (MFIter mfi(velCart); mfi.isValid(); ++mfi) {
      	const Box &vbx = mfi.growntilebox(Nghost);
      	auto const &vel_cart = velCart.array(mfi);

		auto const &west_wall_bcs = phy_bc_lo[0]; 	// west wall
		auto const &east_wall_bcs = phy_bc_hi[0]; 	// east wall
	 	auto const &south_wall_bcs = phy_bc_lo[1];	// south wall
		auto const &north_wall_bcs = phy_bc_hi[1];	// north wall
#if (AMREX_SPACEDIM > 2)
		auto const &bakward_wall_bcs = phy_bc_lo[2]; // z- wall
		auto const &forward_wall_bcs = phy_bc_hi[2]; // z+ wall
#endif

		auto const &inflow_x = inflow_waveform[0];
		auto const &inflow_y = inflow_waveform[1];
#if (AMREX_SPACEDIM > 2)
		auto const &inflow_z = inflow_waveform[2];
#endif

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
		// IMPORTANT: CURRENTLY SET TO LID-DRIVEN CAVITY
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
					}
				});
			} else if (north_wall_bcs == -165) {
				amrex::Abort("Not yet implemented");
			} else if (north_wall_bcs == 195) {
				amrex::Abort("Not yet implemented");
			} else if (north_wall_bcs == -195) {
				amrex::Abort("Not yet implemented");
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
				amrex::Abort("Not yet implemented");
			} else if (bakward_wall_bcs == 195) {
				amrex::Abort("Not yet implemented");
			} else if (bakward_wall_bcs == -195) {
				amrex::Abort("Not yet implemented");
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
				amrex::Abort("Not yet implemented");
			} else if (forward_wall_bcs == 195) {
				amrex::Abort("Not yet implemented");
			} else if (forward_wall_bcs == -195) {
				amrex::Abort("Not yet implemented");
			}
		}
#endif
	}
}

void shift_face_to_center(MultiFab &cell_centre,
                          Array<MultiFab, AMREX_SPACEDIM> &cell_face) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(cell_centre); mfi.isValid(); ++mfi) {
        const Box &vbx = mfi.validbox();
        auto const &cc = cell_centre.array(mfi);

        auto const &cf_x = cell_face[0].array(mfi);
        auto const &cf_y = cell_face[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &cf_z = cell_face[2].array(mfi);
#endif
        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               cc(i, j, k, 0) = cf_x(i + 1, j, k);
                               cc(i, j, k, 1) = cf_y(i, j + 1, k);
#if (AMREX_SPACEDIM > 2)
                               cc(i, j, k, 2) = cf_z(i, j, k + 1);
#endif
         });
    }
}

// ===================== UTILITY | EXTRACT LINE SOLUTION  =====================
void write_interp_line_solution(Real const &interp_sol,
                                std::string const &filename) {
    // Construct the filename for this iteration
    std::string interp_filename = "interp_" + filename;

    // Open a file for writing
    std::ofstream outfile(interp_filename, std::ios::app);

    // Check if the file was opened successfully
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file for writing\n";
    }

    // Write data to the file
    outfile << interp_sol << "\n";

    // Close the file
    outfile.close();
}

void write_exact_line_solution(Real const& time,
                               Real const& x,
                               Real const& y,
							   Real const& z,
                               Real const& numerical_sol,
                               std::string const& filename) {
    // Open a file for writing
	std::string full_filename = filename + ".txt";
    std::ofstream outfile(full_filename, std::ios::app);

    // Check if the file was opened successfully
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file for writing\n";
    }

    // Write data to the file
    outfile << time << "," << x << "," << y << "," << z << "," << numerical_sol << "\n";

    // Close the file
    outfile.close();
}

// ===================== UTILITY | ERROR NORM  =====================
amrex::Real Error_Computation(Array<MultiFab, AMREX_SPACEDIM> &velCont,
                              Array<MultiFab, AMREX_SPACEDIM> &velStar,
                              Array<MultiFab, AMREX_SPACEDIM> &velStarDiff,
                              Geometry const &geom) {
    amrex::Real normError;
    amrex::Real npts;
    Box my_domain = geom.Domain();

#if (AMREX_SPACEDIM == 2)
    npts = (static_cast<amrex::Real>(my_domain.length(0)) * static_cast<amrex::Real>(my_domain.length(1)));
#elif (AMREX_SPACEDIM == 3)
    npts = (static_cast<amrex::Real>(my_domain.length(0)) * static_cast<amrex::Real>(my_domain.length(1)) * static_cast<amrex::Real>(my_domain.length(2)));
#endif
	//amrex::Print() << "INFO | Number of points in the domain: " << npts << "\n";

    for (MFIter mfi(velStarDiff[0]); mfi.isValid(); ++mfi) {

        const Box &xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1, 0, 0)));
        const Box &ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0, 1, 0)));
#if (AMREX_SPACEDIM > 2)
        const Box &zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0, 0, 1)));
#endif

        auto const &xprev = velCont[0].array(mfi);
        auto const &yprev = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &zprev = velCont[2].array(mfi);
#endif

        auto const &xnext = velStar[0].array(mfi);
        auto const &ynext = velStar[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &znext = velStar[2].array(mfi);
#endif

        auto const &xdiff = velStarDiff[0].array(mfi);
        auto const &ydiff = velStarDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &zdiff = velStarDiff[2].array(mfi);
#endif

        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			xdiff(i, j, k) = xprev(i, j, k) - xnext(i, j, k);
		});

        amrex::ParallelFor(ybx,
						   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			ydiff(i, j, k) = yprev(i, j, k) - ynext(i, j, k);
		});

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {	
			zdiff(i, j, k) = zprev(i, j, k) - znext(i, j, k);
		});
#endif
    }// End of all loops for Multi-Fabs
    //Real xerror = velStarDiff[0].norm0();
    //Real yerror = velStarDiff[1].norm0();

    Vector<Real> l2_sum(AMREX_SPACEDIM);
    StagL2Norm(velStarDiff, 0, l2_sum);
    Real xerror = l2_sum[0] / std::sqrt(npts);
    Real yerror = l2_sum[1] / std::sqrt(npts);
    normError = std::max(xerror, yerror);
#if (AMREX_SPACEDIM > 2)
    Real zerror = l2_sum[2] / std::sqrt(npts);
    normError = std::max(normError, zerror);
#endif

    return normError;
}

// ===================== UTILITY | EXPORT  =====================
void Export_Fluxes(MultiFab &fluxConvect,
                   MultiFab &fluxViscous,
                   MultiFab &fluxPrsGrad,
                   BoxArray const &ba,
                   DistributionMapping const &dm,
                   Geometry const &geom,
                   Real const &time,
                   int const &timestep) {

    MultiFab plt(ba, dm, 3 * AMREX_SPACEDIM, 0);

#if (AMREX_SPACEDIM > 2)
    MultiFab::Copy(plt, fluxConvect, 0, 0, 1, 0);
    MultiFab::Copy(plt, fluxConvect, 1, 1, 1, 0);
    MultiFab::Copy(plt, fluxConvect, 2, 2, 1, 0);
    MultiFab::Copy(plt, fluxViscous, 0, 3, 1, 0);
    MultiFab::Copy(plt, fluxViscous, 1, 4, 1, 0);
    MultiFab::Copy(plt, fluxViscous, 2, 5, 1, 0);
    MultiFab::Copy(plt, fluxPrsGrad, 0, 6, 1, 0);
    MultiFab::Copy(plt, fluxPrsGrad, 1, 7, 1, 0);
    MultiFab::Copy(plt, fluxPrsGrad, 2, 8, 1, 0);
#else
    MultiFab::Copy(plt, fluxConvect, 0, 0, 1, 0);
    MultiFab::Copy(plt, fluxConvect, 1, 1, 1, 0);
    MultiFab::Copy(plt, fluxViscous, 0, 2, 1, 0);
    MultiFab::Copy(plt, fluxViscous, 1, 3, 1, 0);
    MultiFab::Copy(plt, fluxPrsGrad, 0, 4, 1, 0);
    MultiFab::Copy(plt, fluxPrsGrad, 1, 5, 1, 0);
#endif

    const std::string &plt_flux = amrex::Concatenate("pltFlux", timestep, 5);
#if (AMREX_SPACEDIM > 2)
    WriteSingleLevelPlotfile(plt_flux, plt, {"conv_flux_x", "conv_flux_y", "conv_flux_z", "visc_flux_x", "visc_flux_y", "visc_flux_z", "press_grad_x", "press_grad_y", "press_grad_z"}, geom, time, timestep);
#else
    WriteSingleLevelPlotfile(plt_flux, plt, {"conv_flux_x", "conv_flux_y", "visc_flux_x", "visc_flux_y", "press_grad_x", "press_grad_y"}, geom, time, timestep);
#endif
}

void Export_Flow_Field(std::string const &nameofFile,
                       MultiFab &userCtx,
                       MultiFab &velCart,
                       BoxArray const &ba,
                       DistributionMapping const &dm,
                       Geometry const &geom,
                       Real const &time,
                       int const &timestep) {
    // Depending on the dimensions the MultiFab needs to store enough
    // components 4 : (u,v,w, p) for flow fields in 3D
    // components = 3 (u,v,p) for flow fields in 2D
#if (AMREX_SPACEDIM > 2)
    MultiFab plt(ba, dm, 4, 0);
#else
    MultiFab plt(ba, dm, 3, 0);
#endif

    // Copy the pressure and velocity fields to the 'plt' Multifab
    // Note the component sequence
    // userCtx [0] --> pressure
    // velCart [1] --> u
    // velCart [2] --> v
    // velCart [3] --> w
    MultiFab::Copy(plt, userCtx, 0, 0, 1, 0);
    MultiFab::Copy(plt, velCart, 0, 1, 1, 0);
    MultiFab::Copy(plt, velCart, 1, 2, 1, 0);
#if (AMREX_SPACEDIM > 2)
    MultiFab::Copy(plt, velCart, 2, 3, 1, 0);
#endif

    const std::string &pltfile = amrex::Concatenate(nameofFile, timestep, 5);//5 spaces
#if (AMREX_SPACEDIM > 2)
    WriteSingleLevelPlotfile(pltfile, plt, {"pressure", "U", "V", "W"}, geom, time, timestep);
#else
    WriteSingleLevelPlotfile(pltfile, plt, {"pressure", "U", "V"}, geom, time, timestep);
#endif
}

void array_analytical_vel_calc(Array<MultiFab, AMREX_SPACEDIM> &array_analytical_vel,
                               Geometry const &geom,
                               Real const &time) {
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(array_analytical_vel[0]); mfi.isValid(); ++mfi) {
        const Box &xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1, 0, 0)));
        const Box &ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0, 1, 0)));
#if (AMREX_SPACEDIM > 2)
        const Box &zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0, 0, 1)));
#endif

        auto const &vel_cont_exact_x = array_analytical_vel[0].array(mfi);
        auto const &vel_cont_exact_y = array_analytical_vel[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const &vel_cont_exact_z = array_analytical_vel[2].array(mfi);
#endif

        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			amrex::Real x = prob_lo[0] + (i + Real(0.0)) * dx[0];
			amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];

			//vel_cont_exact_x(i, j, k) = std::sin(amrex::Real(2.0) * M_PI * x) * std::cos(amrex::Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);
			vel_cont_exact_x(i, j, k) = std::sin(x) * std::cos(y) * std::exp(-Real(2.0) * time);
		});

        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
			amrex::Real y = prob_lo[1] + (j + Real(0.0)) * dx[1];

			//vel_cont_exact_y(i, j, k) = -std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);
			vel_cont_exact_y(i, j, k) = -std::cos(x) * std::sin(y) * std::exp(-Real(2.0) * time);
		});

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			vel_cont_exact_z(i, j, k) = Real(0.0);
		});
#endif
    }
}

void cc_analytical_calc(MultiFab &analytical_sol,
                        Geometry const &geom,
                        Real const &time) {
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    for (MFIter mfi(analytical_sol); mfi.isValid(); ++mfi) {
        const Box &vbx = mfi.validbox();
        auto const &exact_sol = analytical_sol.array(mfi);

        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
			amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];

			exact_sol(i, j, k, 1) = std::sin(x) * std::cos(y) * std::exp(-Real(2.0) * time);

			exact_sol(i, j, k, 2) = -std::cos(x) * std::sin(y) * std::exp(-Real(2.0) * time);

			exact_sol(i, j, k, 0) = Real(0.25) * (std::cos(Real(2.0) * x) + std::cos(Real(2.0) * y)) * std::exp(-Real(4.0) * time);
		});
    }
}

void cc_spectral_analysis(MultiFab &kinetic_energy,
						  MultiFab &analytical_sol,
						  Geometry const &geom)
{
	amrex::Print() << "INFO| Calculating total kinetic energy = ";
	kinetic_energy.setVal(0.0);

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    for (MFIter mfi(analytical_sol); mfi.isValid(); ++mfi) {
        const Box &vbx = mfi.validbox();
        auto const &vel = analytical_sol.array(mfi);
		auto const &kin = kinetic_energy.array(mfi);

        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			kin(i, j, k, 0) = amrex::Real(0.5)* ( vel(i, j, k, 0) * vel(i, j, k, 0) + vel(i, j, k, 1) * vel(i, j, k, 1) ) * ( dx[0] * dx[1] );
        });
    }

	amrex::Print() << kinetic_energy.sum() << "\n";

}

void SumAbsStag(const std::array<MultiFab,
                                 AMREX_SPACEDIM> &m1,
                amrex::Vector<amrex::Real> &sum) {
    BL_PROFILE_VAR("SumAbsStag()", SumAbsStag);

    // Initialize to zero
    std::fill(sum.begin(), sum.end(), 0.);

    ReduceOps<ReduceOpSum> reduce_op;

    ReduceData<Real> reduce_datax(reduce_op);
    using ReduceTuple = typename decltype(reduce_datax)::Type;

    for (MFIter mfi(m1[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box &bx = mfi.tilebox();
        const Box &bx_grid = mfi.validbox();

        auto const &fab = m1[0].array(mfi);

        int xlo = bx_grid.smallEnd(0);
        int xhi = bx_grid.bigEnd(0);

        reduce_op.eval(bx, reduce_datax,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                           Real weight = (i > xlo && i < xhi) ? 1.0 : 0.5;
                           return {std::abs(fab(i, j, k) * weight)};
                       });
    }

    sum[0] = amrex::get<0>(reduce_datax.value());
    ParallelDescriptor::ReduceRealSum(sum[0]);

    ReduceData<Real> reduce_datay(reduce_op);

    for (MFIter mfi(m1[1], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box &bx = mfi.tilebox();
        const Box &bx_grid = mfi.validbox();

        auto const &fab = m1[1].array(mfi);

        int ylo = bx_grid.smallEnd(1);
        int yhi = bx_grid.bigEnd(1);

        reduce_op.eval(bx, reduce_datay,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                           Real weight = (j > ylo && j < yhi) ? 1.0 : 0.5;
                           return {std::abs(fab(i, j, k) * weight)};
                       });
    }

    sum[1] = amrex::get<0>(reduce_datay.value());
    ParallelDescriptor::ReduceRealSum(sum[1]);

#if (AMREX_SPACEDIM == 3)

    ReduceData<Real> reduce_dataz(reduce_op);

    for (MFIter mfi(m1[2], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box &bx = mfi.tilebox();
        const Box &bx_grid = mfi.validbox();

        auto const &fab = m1[2].array(mfi);

        int zlo = bx_grid.smallEnd(2);
        int zhi = bx_grid.bigEnd(2);

        reduce_op.eval(bx, reduce_dataz,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                           Real weight = (k > zlo && k < zhi) ? 1.0 : 0.5;
                           return {std::abs(fab(i, j, k) * weight)};
                       });
    }

    sum[2] = amrex::get<0>(reduce_dataz.value());
    ParallelDescriptor::ReduceRealSum(sum[2]);

#endif
}

void StagL2Norm(const std::array<MultiFab, AMREX_SPACEDIM> &m1,
                const int &comp,
                amrex::Vector<amrex::Real> &inner_prod) {

    BL_PROFILE_VAR("StagL2Norm()", StagL2Norm);

    Array<MultiFab, AMREX_SPACEDIM> mscr;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
        mscr[dir].define(m1[dir].boxArray(), m1[dir].DistributionMap(), 1, 0);
    }

    StagInnerProd(m1, comp, mscr, inner_prod);
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
        inner_prod[dir] = std::sqrt(inner_prod[dir]);
    }
}

void StagInnerProd(const std::array<MultiFab, AMREX_SPACEDIM> &m1,
                   const int &comp1,
                   std::array<MultiFab, AMREX_SPACEDIM> &mscr,
                   amrex::Vector<amrex::Real> &prod_val) {
    BL_PROFILE_VAR("StagInnerProd()", StagInnerProd);

    for (int d = 0; d < AMREX_SPACEDIM; d++) {
        MultiFab::Copy(mscr[d], m1[d], comp1, 0, 1, 0);
        MultiFab::Multiply(mscr[d], m1[d], comp1, 0, 1, 0);
    }

    std::fill(prod_val.begin(), prod_val.end(), 0.);
    SumAbsStag(mscr, prod_val);
}

// create a checkpoint directory
// write out time and BoxArray to a Header file
// write out multifab data
/*
void SaveCheckpoint(MultiFab const& pressure,
                    MultiFab const& vel_xCont,
						  MultiFab const& vel_yCont,
						  MultiFab const& vel_xContPrev,
						  MultiFab const& vel_yContPrev,
						  Real const& time,
						  int const& step) {
*/
void SaveCheckpoint(BoxArray const& ba,
					DistributionMapping const& dm,
					MultiFab const& userCtx,
					Array<MultiFab, AMREX_SPACEDIM> const& velCont,
					Array<MultiFab, AMREX_SPACEDIM> const& velContPrev,
					Real const& time,
					int const& chk_in) {
	// define flow fields to be saved in checkpoint file
	// Nghost = number of ghost cells for each array
	int Nghost = 0;

	// Ncomp = number of components for each array
	int Ncomp = 1;

	MultiFab pressure(ba, dm, Ncomp, Nghost);

	BoxArray edge_ba = ba;
	edge_ba.surroundingNodes(0);
	MultiFab vel_xCont(edge_ba, dm, Ncomp, Nghost);
	MultiFab vel_xContPrev(edge_ba, dm, Ncomp, Nghost);
		
	edge_ba = ba;
	edge_ba.surroundingNodes(1);
	MultiFab vel_yCont(edge_ba, dm, Ncomp, Nghost);
	MultiFab vel_yContPrev(edge_ba, dm, Ncomp, Nghost);

#if (AMREX_SPACEDIM > 2)
	edge_ba = ba;
	edge_ba.surroundingNodes(2);
	MultiFab vel_zCont(edge_ba, dm, Ncomp, Nghost);
	MultiFab vel_zContPrev(edge_ba, dm, Ncomp, Nghost);
#endif

	// transfer data from solver flow fields to checkpoint flow fields
	MultiFab::Copy(pressure, userCtx, 0, 0, 1, 0);
	MultiFab::Copy(vel_xCont, velCont[0], 0, 0, 1, 0);
	MultiFab::Copy(vel_yCont, velCont[1], 0, 0, 1, 0);
	MultiFab::Copy(vel_xContPrev, velContPrev[0], 0, 0, 1, 0);
	MultiFab::Copy(vel_yContPrev, velContPrev[1], 0, 0, 1, 0);
#if (AMREX_SPACEDIM > 2)
	MultiFab::Copy(vel_zCont, velCont[2], 0, 0, 1, 0);
	MultiFab::Copy(vel_zContPrev, velContPrev[2], 0, 0, 1, 0);
#endif
	
	// checkpoint file name, e.g., chk00010
	const std::string& checkpointname = Concatenate("checkpoint", chk_in, 5);

	//BoxArray ba = pressure.boxArray();

	// single level problem
	int nlevels = 1;

	// ---- prebuild a hierarchy of directories
	// ---- dirName is built first.  if dirName exists, it is renamed.  then build
	// ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
	// ---- if callBarrier is true, call ParallelDescriptor::Barrier()
	// ---- after all directories are built
	// ---- ParallelDescriptor::IOProcessor() creates the directories
	PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);

	VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

	// write Header file to store time and BoxArray
	if (ParallelDescriptor::IOProcessor()) {

		std::ofstream HeaderFile;
		HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
		std::string HeaderFileName(checkpointname + "/Header");
		HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out |
			std::ofstream::trunc |
			std::ofstream::binary);

		if( !HeaderFile.good()) {
			FileOpenFailed(HeaderFileName);
		}

		HeaderFile.precision(15);

		// write out title line
		HeaderFile << "Checkpoint file for AMRESSIF\n";

		// write out time
		HeaderFile << time << "\n";

		// write the BoxArray
		ba.writeOn(HeaderFile);
		HeaderFile << '\n';
	}

	// write the MultiFab data to, e.g., chk00010/Level_0/
	VisMF::Write(pressure, MultiFabFileFullPrefix(0, checkpointname, "Level_", "pressure"));
	VisMF::Write(vel_xCont, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_xCont"));
	VisMF::Write(vel_yCont, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_yCont"));
	VisMF::Write(vel_xContPrev, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_xContPrev"));
	VisMF::Write(vel_yContPrev, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_yContPrev"));
#if (AMREX_SPACEDIM > 2)
	VisMF::Write(vel_zCont, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_zCont"));
	VisMF::Write(vel_zContPrev, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_zContPrev"));
#endif

}

// read in the time and BoxArray, then create a DistributionMapping
// Define phi and fill it with data from the checkpoint file
void LoadCheckpoint(BoxArray& ba,
					DistributionMapping& dm,
					MultiFab& userCtx,
					Array<MultiFab, AMREX_SPACEDIM>& velCont,
					Array<MultiFab, AMREX_SPACEDIM>& velContPrev,
					Real& time,
					int const& chk_out) {
	// declare flow fields to be read in checkpoint file
	// NOTE: input fields are from the solver and are undefined
	// Nghost = number of ghost cells for each array
	int Nghost = 0;

	// Ncomp = number of components for each array
	int Ncomp = 1;

	BoxArray ba_from_ckpt;
	DistributionMapping dm_from_ckpt;
	BoxArray edge_ba;

	MultiFab pressure;
	MultiFab vel_xCont;
	MultiFab vel_xContPrev;
	MultiFab vel_yCont;
	MultiFab vel_yContPrev;
#if (AMREX_SPACEDIM > 2)
	MultiFab vel_zCont;
	MultiFab vel_zContPrev;
#endif

	// checkpoint file name, e.g., chk00010
	const std::string& checkpointname = Concatenate("checkpoint", chk_out, 5);

	Print() << "AMRESSIF restarts from checkpoint " << checkpointname << "\n";

	VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

	std::string line, word;

	// Header
	{
		std::string File(checkpointname + "/Header");
		Vector<char> fileCharPtr;
		ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
		std::string fileCharPtrString(fileCharPtr.dataPtr());
		std::istringstream is(fileCharPtrString, std::istringstream::in);

		// read in title line
		std::getline(is, line);

		// read in time
		is >> time;
		GotoNextLine(is);

		// read in BoxArray from Header
		ba_from_ckpt.readFrom(is);
		GotoNextLine(is);

		// create a distribution mapping
		dm_from_ckpt.define(ba_from_ckpt, ParallelDescriptor::NProcs());

		// define checkpoint+solver flow fields 
		pressure.define(ba_from_ckpt, dm_from_ckpt, Ncomp, Nghost);

		edge_ba = ba_from_ckpt;
		edge_ba.surroundingNodes(0);
		vel_xCont.define(edge_ba, dm_from_ckpt, Ncomp, Nghost);
		vel_xContPrev.define(edge_ba, dm_from_ckpt, Ncomp, Nghost);
		
		edge_ba = ba;
		edge_ba.surroundingNodes(1);
		vel_yCont.define(edge_ba, dm_from_ckpt, Ncomp, Nghost);
		vel_yContPrev.define(edge_ba, dm_from_ckpt, Ncomp, Nghost);

#if (AMREX_SPACEDIM > 2)
		edge_ba = ba_from_ckpt;
		edge_ba.surroundingNodes(2);
		vel_zCont.define(edge_ba, dm_from_ckpt, Ncomp, Nghost);
		vel_zContPrev.define(edge_ba, dm_from_ckpt, Ncomp, Nghost);
#endif
	}

	// read in the MultiFab data
	VisMF::Read(pressure, MultiFabFileFullPrefix(0, checkpointname, "Level_", "pressure"));
	VisMF::Read(vel_xCont, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_xCont"));
	VisMF::Read(vel_yCont, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_yCont"));
	VisMF::Read(vel_xContPrev, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_xContPrev"));
	VisMF::Read(vel_yContPrev, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_yContPrev"));
#if (AMREX_SPACEDIM > 2)
	VisMF::Read(vel_zCont, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_zCont"));
	VisMF::Read(vel_zContPrev, MultiFabFileFullPrefix(0, checkpointname, "Level_", "vel_zContPrev"));
#endif

	// print debugging MultiFab content	
	/*
	for ( MFIter mfi(vel_xCont); mfi.isValid(); ++mfi )
	{
        const Box& vbx = mfi.validbox();
        auto const& deb = vel_xCont.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Print() << "DEBUG | vel_xCont at i=" << i << " ; j=" << j << " ; k=" << k << " = " << deb(i, j, k) << "\n";
        });
	}
	*/

	// define solver flow fields
	userCtx.define(ba, dm, 2, 1);

	for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
		edge_ba = ba;
		edge_ba.surroundingNodes(dir);
		velCont[dir].define(edge_ba, dm, 1, 0);
		velContPrev[dir].define(edge_ba, dm, 1, 0);
	}

	// transfer data from checkpoint flow fields to solver flow fields
	MultiFab::ParallelCopy(userCtx, pressure, 0, 0, 1, 0);
	MultiFab::ParallelCopy(velCont[0], vel_xCont, 0, 0, 1, 0);
	MultiFab::ParallelCopy(velCont[1], vel_yCont, 0, 0, 1, 0);
	MultiFab::ParallelCopy(velContPrev[0], vel_xContPrev, 0, 0, 1, 0);
	MultiFab::ParallelCopy(velContPrev[1], vel_yContPrev, 0, 0, 1, 0);
#if (AMREX_SPACEDIM > 2)
	MultiFab::ParallelCopy(velCont[2], vel_zCont, 0, 0, 1, 0);
	MultiFab::ParallelCopy(velContPrev[2], vel_zContPrev, 0, 0, 1, 0);
#endif

}