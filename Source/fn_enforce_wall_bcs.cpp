#include <AMReX_MultiFabUtil.H>

#include "fn_enforce_wall_bcs.H"
#include "kn_enforce_wall_bcs.H"

using namespace amrex;

// ============================== UTILITY | BOUNDARY CONDITIONS ==============================
void enforce_boundary_conditions (MultiFab& inputMultiFab,
                                  std::string const& typeMultiFab,
                                  int const& Nghost,
                                  Vector<int> const& phy_bc_lo,
                                  Vector<int> const& phy_bc_hi,
                                  int const& n_cell)
{
    for (MFIter mfi(inputMultiFab); mfi.isValid(); ++mfi)
    {
        auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
        auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

        auto const& south_wall_bcs = phy_bc_lo[1]; // south wall
        auto const& north_wall_bcs = phy_bc_hi[1]; // north wall
#if (AMREX_SPACEDIM > 2)
        auto const& fron_wall_bcs = phy_bc_lo[2]; // front wall
        auto const& back_wall_bcs = phy_bc_hi[2]; // back wall
#endif
        const Box& gbx = mfi.growntilebox(Nghost);
        auto const& gmf = inputMultiFab.array(mfi);

        if ( west_wall_bcs != 0 ) {
            // amrex::Print() << "========================== west wall  ========================= \n";

            if ( typeMultiFab == "pressure" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_pres_bcs_west(i, j, k, n_cell, west_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "phi" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_phi_bcs_west(i, j, k, n_cell, west_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "velocity" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_vel_bcs_west(i, j, k, n_cell, west_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "flux" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_flux_bcs_west(i, j, k, n_cell, west_wall_bcs, gmf);
                });
            } else { amrex::Abort("ABORTING| NOT a Valid Type to enforce boundary conditions! \n"); }

        }

        if ( east_wall_bcs != 0 ) {
            // amrex::Print() << "========================== east wall  ========================= \n";

            if ( typeMultiFab == "pressure" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_pres_bcs_east(i, j, k, n_cell, east_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "phi" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_phi_bcs_east(i, j, k, n_cell, east_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "velocity" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_vel_bcs_east(i, j, k, n_cell, east_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "flux" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_flux_bcs_east(i, j, k, n_cell, east_wall_bcs, gmf);
                });
            } else { amrex::Abort("ABORTING| NOT a Valid Type to enforce boundary conditions! \n"); }

        }

        if ( south_wall_bcs != 0 ) {
            // amrex::Print() << "========================= south wall  ========================= \n";

            if ( typeMultiFab == "pressure" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_pres_bcs_south(i, j, k, n_cell, south_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "phi" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_phi_bcs_south(i, j, k, n_cell, south_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "velocity" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_vel_bcs_south(i, j, k, n_cell, south_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "flux" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_flux_bcs_south(i, j, k, n_cell, south_wall_bcs, gmf);
                });
            } else { amrex::Abort("ABORTING| NOT a Valid Type to enforce boundary conditions! \n"); }

        }

        if ( north_wall_bcs != 0 ) {
            // amrex::Print() << "========================= north wall  ========================= \n";

            if ( typeMultiFab == "pressure" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_pres_bcs_north(i, j, k, n_cell, north_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "phi" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_phi_bcs_north(i, j, k, n_cell, north_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "velocity" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_vel_bcs_north(i, j, k, n_cell, north_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "flux" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_flux_bcs_north(i, j, k, n_cell, north_wall_bcs, gmf);
                });
            } else { amrex::Abort("ABORTING| NOT a Valid Type to enforce boundary conditions! \n"); }

        }
#if (AMREX_SPACEDIM > 2)
        if ( front_wall_bcs != 0 ) {
            // amrex::Print() << "========================= front wall  ========================= \n";

            if ( typeMultiFab == "pressure" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_pres_bcs_front(i, j, k, n_cell, front_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "phi" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_phi_bcs_front(i, j, k, n_cell, front_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "velocity" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_vel_bcs_front(i, j, k, n_cell, front_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "flux" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_flux_bcs_front(i, j, k, n_cell, front_wall_bcs, gmf);
                });
            } else { amrex::Abort("ABORTING| NOT a Valid Type to enforce boundary conditions! \n"); }

        }

        if ( back_wall_bcs != 0 ) {
            // amrex::Print() << "========================== back wall  ========================= \n";

            if ( typeMultiFab == "pressure" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_pres_bcs_back(i, j, k, n_cell, back_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "phi" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_phi_bcs_back(i, j, k, n_cell, back_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "velocity" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_vel_bcs_back(i, j, k, n_cell, back_wall_bcs, gmf);
                });
            } else if ( typeMultiFab == "flux" ) {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    enforce_flux_bcs_back(i, j, k, n_cell, back_wall_bcs, gmf);
                });
            } else { amrex::Abort("ABORTING| NOT a Valid Type to enforce boundary conditions! \n"); }

        }
#endif
    }
}
