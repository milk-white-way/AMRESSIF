#include <AMReX_MultiFabUtil.H>

#include "fn_enforce_wall_bcs.H"
#include "kn_enforce_wall_bcs.H"

using namespace amrex;

// ============================== UTILITY | BOUNDARY CONDITIONS ==============================
void enforce_boundary_conditions (MultiFab& velCart,
                                  Geometry const& geom,
                                  int const& Nghost,
                                  Vector<int> const& phy_bc_lo,
                                  Vector<int> const& phy_bc_hi,
                                  int const& n_cell)
{
    Box dom(geom.Domain());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
    {

        const Box& vbx = mfi.growntilebox(Nghost);
        auto const& vel_cart = velCart.array(mfi);

        auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
        auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

        auto const& south_wall_bcs = phy_bc_lo[1]; // south wall
        auto const& north_wall_bcs = phy_bc_hi[1]; // north wall
#if (AMREX_SPACEDIM > 2)
        auto const& fron_wall_bcs = phy_bc_lo[2]; // front wall
        auto const& back_wall_bcs = phy_bc_hi[2]; // back wall
#endif

        int lo = dom.smallEnd(0);
        int hi = dom.bigEnd(0);

// DEBUGGING
/*        
        amrex::Print() << "INFO| lo: " << lo << "\n";
        amrex::Print() << "INFO| hi: " << hi << "\n";

*/
        amrex::Print() << "INFO| x-smallEnd: " << vbx.smallEnd(0) << "\n";
        amrex::Print() << "INFO| x-bigEnd: " << vbx.bigEnd(0) << "\n";

        amrex::Print() << "INFO| y-smallEnd: " << vbx.smallEnd(1) << "\n";
        amrex::Print() << "INFO| y-bigEnd: " << vbx.bigEnd(1) << "\n"; 
    
        if (vbx.smallEnd(0) < lo) {
            if ( west_wall_bcs == -1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( i < lo ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            vel_cart(i, j, k, dir) = - vel_cart(-i-1, j, k, dir);
                        }
                        
                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });
            } else if ( west_wall_bcs == 1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( i < lo ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        vel_cart(i, j, k, 0) = - vel_cart(-i-1, j, k, 0);
                        vel_cart(i, j, k, 1) = vel_cart(-i-1, j, k, 1);
    #if (AMREX_SPACEDIM > 2)
                        vel_cart(i, j, k, 2) = vel_cart(-i-1, j, k, 2);
    #endif

                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });
            }
        }

        if (vbx.bigEnd(0) > hi) {
            if ( east_wall_bcs == -1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( i > hi ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            vel_cart(i, j, k, dir) = - vel_cart(( (n_cell-i) + (n_cell-1) ), j, k, dir);
                        }

                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });

            } else if ( east_wall_bcs == 1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( i > hi ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        vel_cart(i, j, k, 0) = - vel_cart(( (n_cell-i) + (n_cell-1) ), j, k, 0);
                        vel_cart(i, j, k, 1) = vel_cart(( (n_cell-i) + (n_cell-1) ), j, k, 1);
    #if (AMREX_SPACEDIM > 2)
                        vel_cart(i, j, k, 2) = vel_cart(( (n_cell-i) + (n_cell-1) ), j, k, 2);
    #endif

                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });
            }
        }

        lo = dom.smallEnd(1);
        hi = dom.bigEnd(1);

        if (vbx.smallEnd(1) < lo) {
            if ( south_wall_bcs == -1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( j < lo ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            vel_cart(i, j, k, dir) = - vel_cart(i, -j-1, k, dir);
                        }

                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });
            } else if ( south_wall_bcs == 1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( j < lo ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        vel_cart(i, j, k, 0) = vel_cart(i, -j-1, k, 0);
                        vel_cart(i, j, k, 1) = - vel_cart(i, -j-1, k, 1);
    #if (AMREX_SPACEDIM > 2)
                        vel_cart(i, j, k, 2) = vel_cart(i, -j-1, k, 2);
    #endif

                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });
            }
        }

        if (vbx.bigEnd(1) > hi) {
            if ( north_wall_bcs == -1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( j > hi ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            vel_cart(i, j, k, dir) = - vel_cart(i, ( (n_cell-j) + (n_cell-1) ), k, dir);
                        }

                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });
            } else if ( north_wall_bcs == 1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( j > hi ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        vel_cart(i, j, k, 0) = vel_cart(i, ( (n_cell-j) + (n_cell-1) ), k, 0);
                        vel_cart(i, j, k, 1) = - vel_cart(i, ( (n_cell-j) + (n_cell-1) ), k, 1);
    #if (AMREX_SPACEDIM > 2)
                        vel_cart(i, j, k, 2) = vel_cart(i, ( (n_cell-j) + (n_cell-1) ), k, 2);
    #endif

                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });
            }
        }

#if (AMREX_SPACEDIM > 2)
        lo = dom.smallEnd(2);
        hi = dom.bigEnd(2);

        if (vbx.smallEnd(2) < lo) {
            if ( front_wall_bcs == -1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( k < lo ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            vel_cart(i, j, k, dir) = - vel_cart(i, j, -k-1, dir);
                        }

                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });
            } else if ( front_wall_bcs == 1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( k < lo ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        vel_cart(i, j, k, 0) = vel_cart(i, j, -k-1, 0);
                        vel_cart(i, j, k, 1) = vel_cart(i, j, -k-1, 1);
                        vel_cart(i, j, k, 2) = - vel_cart(i, j, -k-1, 2);

                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });
            }
        }

        if (vbx.bigEnd(2) > hi) {
            if ( back_wall_bcs == -1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( k > hi ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            vel_cart(i, j, k, dir) = - vel_cart(i, j, ( (n_cell-k) + (n_cell-1) ), dir);
                        }

                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });
            } else if ( back_wall_bcs == 1 ) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( k > hi ) {
                        //amrex::Print() << "FILLING | Ghost Cell at i=" << i << " ; j=" << j << " ; k=" << k;
                        vel_cart(i, j, k, 0) = vel_cart(i, j, ( (n_cell-k) + (n_cell-1) ), 0);
                        vel_cart(i, j, k, 1) = vel_cart(i, j, ( (n_cell-k) + (n_cell-1) ), 1);
                        vel_cart(i, j, k, 2) = - vel_cart(i, j, ( (n_cell-k) + (n_cell-1) ), 2);

                        //for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        //    amrex::Print() << " | " << vel_cart(i, j, k, dir);
                        //}
                        //amrex::Print() << "\n";
                    }
                });
            }
        }
#endif
    }
}