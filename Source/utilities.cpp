#include <AMReX_MultiFabUtil.H>

#include "utilities.H"

using namespace amrex;

// ===================== UTILITY | CONVERSION  =====================
void cont2cart (MultiFab& velCart,
                Array<MultiFab, AMREX_SPACEDIM>& velCont,
                const Geometry& geom, 
                int const& Nghost,
                Vector<int> const& phy_bc_lo,
                Vector<int> const& phy_bc_hi,
                int const& n_cell)
{
    Box dom(geom.Domain());
    //average_face_to_cellcenter(velCart, amrex::GetArrOfConstPtrs(velCont), geom);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vel_cart  = velCart.array(mfi);

        auto const& vel_cont_x = velCont[0].array(mfi);
        auto const& vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_z = velCont[2].array(mfi);
#endif

        // Average to interior cell-centered velocity
        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            vel_cart(i, j, k, 0) = amrex::Real(0.5)*( vel_cont_x(i, j, k) + vel_cont_x(i+1, j, k) );
            vel_cart(i, j, k, 1) = amrex::Real(0.5)*( vel_cont_y(i, j, k) + vel_cont_y(i, j+1, k) );
#if (AMREX_SPACEDIM > 2)
            vel_cart(i, j, k, 2) = amrex::Real(0.5)*( vel_cont_z(i, j, k) + vel_cont_z(i, j, k+1) );
#endif
        });
    }

    // Periodic BCs
    velCart.FillBoundary(geom.periodicity());
    // Non-periodic BCs
    // -- wall: -1, 1
    // -- inlet: -2
    // -- outlet: 2
    // -- Direclet: 10, 11
    if ( phy_bc_lo[0] == 10 || phy_bc_lo[0] == 10 || phy_bc_hi[0] == 10 || phy_bc_hi[0] == 10 ) {
        //Print() << "INFO| Applying Direclet conditions on the volume centers on ghost node\n";
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
        {

            const Box& vbx = mfi.growntilebox(Nghost);
            auto const& vel_cart = velCart.array(mfi);

            int lo = dom.smallEnd(0);
            int hi = dom.bigEnd(0);

            if (vbx.smallEnd(0) < lo) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( i < lo ) {
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            vel_cart(i, j, k, dir) = Real(1.0);
                        }
                    }
                });
            }
            if (vbx.bigEnd(0) > hi) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( i > hi ) {
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            vel_cart(i, j, k, dir) = Real(1.0);
                        }
                    }
                });
            }

            lo = dom.smallEnd(1);
            hi = dom.bigEnd(1);

            if (vbx.smallEnd(1) < lo) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( j < lo ) {
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            vel_cart(i, j, k, dir) = Real(1.0);
                        }
                    }
                });
            }
            if (vbx.bigEnd(1) > hi) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( j > hi ) {
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            vel_cart(i, j, k, dir) = Real(1.0);
                        }
                    }
                });
            }
        }
    }
    /**
     * @brief Direclet boundary for steady flow with Taylor-Green Vortex boundary
     * 
     */
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    if ( phy_bc_lo[0] == 11 || phy_bc_lo[0] == 11 || phy_bc_hi[0] == 11 || phy_bc_hi[0] == 11 ) {
        //Print() << "INFO| Applying Direclet conditions on the volume centers on ghost node\n";
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
        {

            const Box& vbx = mfi.growntilebox(Nghost);
            auto const& vel_cart = velCart.array(mfi);

            int lo = dom.smallEnd(0);
            int hi = dom.bigEnd(0);

            if (vbx.smallEnd(0) < lo) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( i < lo ) {
                        amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                        vel_cart(i, j, k, 0) = Real(0.0);// std::sin(amrex::Real(2.0) * M_PI * x) * std::cos(amrex::Real(2.0) * M_PI * y);
                        vel_cart(i, j, k, 1) = Real(0.0);//-std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y);
                    }
                });
            }
            if (vbx.bigEnd(0) > hi) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( i > hi ) {
                        amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                        vel_cart(i, j, k, 0) = Real(0.0);// std::sin(amrex::Real(2.0) * M_PI * x) * std::cos(amrex::Real(2.0) * M_PI * y);
                        vel_cart(i, j, k, 1) = Real(0.0);//-std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y);
                    }
                });
            }

            lo = dom.smallEnd(1);
            hi = dom.bigEnd(1);

            if (vbx.smallEnd(1) < lo) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( j < lo ) {
                        amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                        vel_cart(i, j, k, 0) = Real(0.0);// std::sin(amrex::Real(2.0) * M_PI * x) * std::cos(amrex::Real(2.0) * M_PI * y);
                        vel_cart(i, j, k, 1) = Real(0.0);//-std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y);
                    }
                });
            }
            if (vbx.bigEnd(1) > hi) {
                amrex::ParallelFor(vbx, 
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if ( j > hi ) {
                        amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                        amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                        vel_cart(i, j, k, 0) = Real(0.0);// std::sin(amrex::Real(2.0) * M_PI * x) * std::cos(amrex::Real(2.0) * M_PI * y);
                        vel_cart(i, j, k, 1) = Real(0.0);//-std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y);
                    }
                });
            }
        }
    }


    if ( phy_bc_lo[0] == -1 || phy_bc_lo[0] == 1 || phy_bc_hi[0] == -1 || phy_bc_hi[0] == 1 ) {
        Print() << "INFO| Applying wall conditions on the x-physical boundaries (west-east)\n";
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
        {

            const Box& vbx = mfi.growntilebox(Nghost);
            auto const& vel_cart = velCart.array(mfi);

            auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
            auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

            int lo = dom.smallEnd(0);
            int hi = dom.bigEnd(0);

            if (vbx.smallEnd(0) < lo) {
                if ( west_wall_bcs == -1 ) {
                    amrex::ParallelFor(vbx, 
                                       [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        if ( i < lo ) {
                            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                                vel_cart(i, j, k, dir) = - vel_cart(-i-1, j, k, dir);
                            }
                        }
                    });
                } else if ( west_wall_bcs == 1 ) {
                    amrex::ParallelFor(vbx, 
                                       [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        if ( i < lo ) {
                            vel_cart(i, j, k, 0) = - vel_cart(-i-1, j, k, 0);
                            vel_cart(i, j, k, 1) = vel_cart(-i-1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
                            vel_cart(i, j, k, 2) = vel_cart(-i-1, j, k, 2);
#endif
                        }
                    });
                }
            }
            if (vbx.bigEnd(0) > hi) {
                if ( east_wall_bcs == -1 ) {
                    amrex::ParallelFor(vbx, 
                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        if ( i > hi ) {
                            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                                vel_cart(i, j, k, dir) = - vel_cart(( (n_cell-i) + (n_cell-1) ), j, k, dir);
                            }
                        }
                    });

                } else if ( east_wall_bcs == 1 ) {
                    amrex::ParallelFor(vbx, 
                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        if ( i > hi ) {
                            vel_cart(i, j, k, 0) = - vel_cart(( (n_cell-i) + (n_cell-1) ), j, k, 0);
                            vel_cart(i, j, k, 1) = vel_cart(( (n_cell-i) + (n_cell-1) ), j, k, 1);
        #if (AMREX_SPACEDIM > 2)
                            vel_cart(i, j, k, 2) = vel_cart(( (n_cell-i) + (n_cell-1) ), j, k, 2);
        #endif
                        }
                    });
                }
            }
        }

        for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
        {
            int const& box_id = mfi.LocalIndex();

            const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
            auto const& vel_cont_x = velCont[0].array(mfi);

            auto const& vel_cart = velCart.array(mfi);

            int lo = dom.smallEnd(0);
            int hi = dom.bigEnd(0)+1;

            amrex::ParallelFor(xbx,
                              [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if ( i == lo ) {
                    vel_cont_x(i, j, k) = Real(0.5)*( vel_cart(i, j, k, 0) + vel_cart(i-1, j, k, 0) );
                }
            });

            amrex::ParallelFor(xbx,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if ( i == hi ) {
                    vel_cont_x(i, j, k) = Real(0.5)*( vel_cart(i, j, k, 0) + vel_cart(i-1, j, k, 0) );
                }
            });
        }
        //enforce_wall_bcs_for_cell_centered_velocity_on_ghost_cells(velCart, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
    } else if ( phy_bc_lo[0] == -2 || phy_bc_hi[0] == -2 ) {
        Print() << "INFO| Applying inlet boundary conditions on the x-physical boundaries\n";
    } else if ( phy_bc_lo[0] == 2 || phy_bc_hi[0] == 2 ) {
        Print() << "INFO| Applying outlet boundary conditions on the x-physical boundaries\n";
    }
    
    if ( phy_bc_lo[1] == -1 || phy_bc_lo[1] == 1 || phy_bc_hi[1] == -1 || phy_bc_hi[1] == 1 ) {
        Print() << "INFO| Applying wall boundary conditions on the y-physical boundaries\n";
        for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
        {

            const Box& vbx = mfi.growntilebox(Nghost);
            auto const& vel_cart = velCart.array(mfi);

            auto const& south_wall_bcs = phy_bc_lo[1]; // west wall
            auto const& north_wall_bcs = phy_bc_hi[1]; // east wall

            int lo = dom.smallEnd(1);
            int hi = dom.bigEnd(1);

            if (vbx.smallEnd(1) < lo) {
                if ( south_wall_bcs == -1 ) {
                    amrex::ParallelFor(vbx, 
                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        if ( j < lo ) {
                            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                                vel_cart(i, j, k, dir) = - vel_cart(i, -j-1, k, dir);
                            }
                        }
                    });
                } else if ( south_wall_bcs == 1 ) {
                    amrex::ParallelFor(vbx, 
                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        if ( j < lo ) {
                            vel_cart(i, j, k, 0) = vel_cart(i, -j-1, k, 0);
                            vel_cart(i, j, k, 1) = - vel_cart(i, -j-1, k, 1);
        #if (AMREX_SPACEDIM > 2)
                            vel_cart(i, j, k, 2) = vel_cart(i, -j-1, k, 2);
        #endif
                        }
                    });
                }
            }

            if (vbx.bigEnd(1) > hi) {
                if ( north_wall_bcs == -1 ) {
                    amrex::ParallelFor(vbx, 
                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        if ( j > hi ) {
                            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                                vel_cart(i, j, k, dir) = - vel_cart(i, ( (n_cell-j) + (n_cell-1) ), k, dir);
                            }
                        }
                    });
                } else if ( north_wall_bcs == 1 ) {
                    amrex::ParallelFor(vbx, 
                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        if ( j > hi ) {
                            vel_cart(i, j, k, 0) = vel_cart(i, ( (n_cell-j) + (n_cell-1) ), k, 0);
                            vel_cart(i, j, k, 1) = - vel_cart(i, ( (n_cell-j) + (n_cell-1) ), k, 1);
        #if (AMREX_SPACEDIM > 2)
                            vel_cart(i, j, k, 2) = vel_cart(i, ( (n_cell-j) + (n_cell-1) ), k, 2);
        #endif
                        }
                    });
                }
            }
        }

        for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
        {
            const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
            auto const& vel_cont_y = velCont[1].array(mfi);
            auto const& vel_cart = velCart.array(mfi);

            int lo = dom.smallEnd(1);
            int hi = dom.bigEnd(1)+1;

            amrex::ParallelFor(ybx,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if ( j == lo ) {
                    vel_cont_y(i, j, k) = Real(0.5)*( vel_cart(i, j, k, 1) + vel_cart(i, j-1, k, 1) );
                }
            });

            amrex::ParallelFor(ybx,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if ( j == hi ) {
                    vel_cont_y(i, j, k) = Real(0.5)*( vel_cart(i, j, k, 1) + vel_cart(i, j-1, k, 1) );
                }
            });
        }
        //enforce_wall_bcs_for_cell_centered_velocity_on_ghost_cells(velCart, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
    } else if ( phy_bc_lo[1] == -2 || phy_bc_hi[1] == -2 ) {
        Print() << "INFO| Applying inlet boundary conditions on the y-physical boundaries\n";
    } else if ( phy_bc_lo[1] == 2 || phy_bc_hi[1] == 2 ) {
        Print() << "INFO| Applying outlet boundary conditions on the y-physical boundaries\n";
    }

#if (AMREX_SPACEDIM > 2)
    if ( phy_bc_lo[2] == -1 || phy_bc_lo[2] == 1 || phy_bc_hi[2] == -1 || phy_bc_hi[2] == 1 ) {
        Print() << "INFO| Applying wall boundary conditions on the z-physical boundaries\n";
        enforce_wall_bcs_for_cell_centered_velocity_on_ghost_cells(velCart, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
    } else if ( phy_bc_lo[2] == -2 || phy_bc_hi[2] == -2 ) {
        Print() << "INFO| Applying inlet boundary conditions on the z-physical boundaries\n";
    } else if ( phy_bc_lo[2] == 2 || phy_bc_hi[2] == 2 ) {
        Print() << "INFO| Applying outlet boundary conditions on the z-physical boundaries\n";
    }
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

        auto const& vel_cart = velCart.array(mfi);

        int lo = dom.smallEnd(0);
        int hi = dom.bigEnd(0)+1;

        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if ( i == lo || i == hi ) {
                //amrex::Print() << "FILLING | X-Contravariant velocity at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
                vel_cont_x(i, j, k) = Real(0.5)*( vel_cart(i, j, k, 0) + vel_cart(i-1, j, k, 0) );
            }
        });

        lo = dom.smallEnd(1);
        hi = dom.bigEnd(1)+1;

        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if ( j == lo || j == hi ) {
                //amrex::Print() << "FILLING | Y-Contravariant velocity at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
                vel_cont_y(i, j, k) = Real(0.5)*( vel_cart(i, j, k, 1) + vel_cart(i, j-1, k, 1) );
            }
        });

#if (AMREX_SPACEDIM > 2)
        lo = dom.smallEnd(2);
        hi = dom.bigEnd(2)+1;

        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if ( k == lo || k == hi ) {
                //amrex::Print() << "FILLING | Z-Contravariant velocity at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
                vel_cont_z(i, j, k) = Real(0.5)*( vel_cart(i, j, k, 2) + vel_cart(i, j, k-1, 2) );
            }
        });
#endif
    }
}

void shift_face_to_center (MultiFab& cell_centre,
                           Array<MultiFab, AMREX_SPACEDIM>& cell_face)
{ 
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(cell_centre); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& cc = cell_centre.array(mfi);

        auto const& cf_x = cell_face[0].array(mfi);
        auto const& cf_y = cell_face[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& cf_z = cell_face[2].array(mfi);
#endif    
        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            cc(i, j, k, 0) = cf_x(i+1, j, k);
            cc(i, j, k, 1) = cf_y(i, j+1, k);
#if (AMREX_SPACEDIM > 2)
            cc(i, j, k, 2) = cf_z(i, j, k+1);
#endif
        });
    }
}

// ===================== UTILITY | EXTRACT LINE SOLUTION  =====================
void write_interp_line_solution (Real const& interp_sol,
                                 std::string const& filename)
{
    // Construct the filename for this iteration
    std::string interp_filename = "interp_" + filename;

    // Open a file for writing
    std::ofstream outfile(interp_filename, std::ios::app);

    // Check if the file was opened successfully
    if (!outfile.is_open())
    {
        std::cerr << "Failed to open file for writing\n";
    }

    // Write data to the file
    outfile << interp_sol << "\n";

    // Close the file
    outfile.close();
}

void write_exact_line_solution (Real const& x,
                                Real const& y,
                                Real const& numerical_sol,
                                Real const& analytical_sol,
                                std::string const& filename)
{
    // Open a file for writing
    std::ofstream outfile(filename, std::ios::app);

    // Check if the file was opened successfully
    if (!outfile.is_open())
    {
        std::cerr << "Failed to open file for writing\n";
    }

    // Write data to the file
    outfile << x << " " << y << " " << numerical_sol << " " << analytical_sol << "\n";

    // Close the file
    outfile.close();
}

// ===================== UTILITY | ERROR NORM  =====================
amrex::Real Error_Computation (Array<MultiFab, AMREX_SPACEDIM>& velCont,
                               Array<MultiFab, AMREX_SPACEDIM>& velStar,
                               Array<MultiFab, AMREX_SPACEDIM>& velStarDiff,
                               Geometry const& geom)
{
    amrex::Real normError;

    long npts;
    Box my_domain = geom.Domain();
#if (AMREX_SPACEDIM == 2)
    npts = (my_domain.length(0) * my_domain.length(1));
#elif (AMREX_SPACEDIM == 3)
    npts = (my_domain.length(0) * my_domain.length(1) * my_domain.length(2));
#endif

    for ( MFIter mfi(velStarDiff[0]); mfi.isValid(); ++mfi )
    {

        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif

        auto const& xprev = velCont[0].array(mfi);
        auto const& yprev = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zprev = velCont[2].array(mfi);
#endif

        auto const& xnext = velStar[0].array(mfi);
        auto const& ynext = velStar[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& znext = velStar[2].array(mfi);
#endif

        auto const& xdiff = velStarDiff[0].array(mfi);
        auto const& ydiff = velStarDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zdiff = velStarDiff[2].array(mfi);
#endif

        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            xdiff(i, j, k) = xprev(i, j, k) - xnext(i, j, k);
        });

        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            ydiff(i, j, k) = yprev(i, j, k) - ynext(i, j, k);
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            zdiff(i, j, k) = zprev(i, j, k) - znext(i, j, k);
        });
#endif
    }// End of all loops for Multi-Fabs
    //Real xerror = velStarDiff[0].norm0();
    //Real yerror = velStarDiff[1].norm0();

    Vector<Real> l2_sum(AMREX_SPACEDIM);
    StagL2Norm(velStarDiff, 0, l2_sum);
    Real xerror = l2_sum[0]/std::sqrt(npts);
    Real yerror = l2_sum[1]/std::sqrt(npts);
    normError = std::max(xerror, yerror);
#if (AMREX_SPACEDIM > 2)
    Real zerror = l2_sum[2]/std::sqrt(npts);
    normError = std::max(normError, zerror);
#endif

    return normError;
}

// ===================== UTILITY | EXPORT  =====================
void Export_Fluxes( MultiFab& fluxConvect,
                    MultiFab& fluxViscous,
                    MultiFab& fluxPrsGrad,
                    BoxArray const& ba,
                    DistributionMapping const& dm,
                    Geometry const& geom,
                    Real const& time,
                    int const& timestep)
{

    MultiFab plt(ba, dm, 3*AMREX_SPACEDIM, 0);

    MultiFab::Copy(plt, fluxConvect, 0, 0, 1, 0);
    MultiFab::Copy(plt, fluxConvect, 1, 1, 1, 0);
    MultiFab::Copy(plt, fluxViscous, 0, 2, 1, 0);
    MultiFab::Copy(plt, fluxViscous, 1, 3, 1, 0);
    MultiFab::Copy(plt, fluxPrsGrad, 0, 4, 1, 0);
    MultiFab::Copy(plt, fluxPrsGrad, 1, 5, 1, 0);

    const std::string& plt_flux = amrex::Concatenate("pltFlux", timestep, 5);
    WriteSingleLevelPlotfile(plt_flux, plt, {"conv_fluxx", "conv_fluxy", "visc_fluxx", "visc_fluxy", "press_gradx", "press_grady"}, geom, time, timestep);
}

void Export_Flow_Field (std::string const& nameofFile,
                        MultiFab& userCtx,
                        MultiFab& velCart,
                        BoxArray const& ba,
                        DistributionMapping const& dm,
                        Geometry const& geom,
                        Real const& time,
                        int const& timestep)
{
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

    const std::string& pltfile = amrex::Concatenate(nameofFile, timestep, 5); //5 spaces
#if (AMREX_SPACEDIM > 2)
    WriteSingleLevelPlotfile(pltfile, plt, {"pressure", "U", "V", "W"}, geom, time, timestep);
#else
    WriteSingleLevelPlotfile(pltfile, plt, {"pressure", "U", "V"}, geom, time, timestep);
#endif
}

void array_analytical_vel_calc (Array<MultiFab, AMREX_SPACEDIM>& array_analytical_vel,
                                Geometry const& geom,
                                Real const& time)
{
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(array_analytical_vel[0]); mfi.isValid(); ++mfi) {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif

        auto const& vel_cont_exact_x = array_analytical_vel[0].array(mfi);
        auto const& vel_cont_exact_y = array_analytical_vel[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_exact_z = array_analytical_vel[2].array(mfi);
#endif

        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            amrex::Real x = prob_lo[0] + (i + Real(0.0)) * dx[0];
            amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
            
            vel_cont_exact_x(i, j, k) = std::sin(amrex::Real(2.0) * M_PI * x) * std::cos(amrex::Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);
        });
        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
            amrex::Real y = prob_lo[1] + (j + Real(0.0)) * dx[1];

            vel_cont_exact_y(i, j, k) = - std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            vel_cont_exact_z(i, j, k) = Real(0.0);
        });
#endif
    }
}

void cc_analytical_calc (MultiFab& analytical_sol,
                         Geometry const& geom,
                         Real const& time)
{
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // Initialize the analytical pressure field
    for ( MFIter mfi(analytical_sol); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& exact_sol = analytical_sol.array(mfi);

        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
            amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];

            exact_sol(i, j, k, 0) = std::sin(Real(2.0) * M_PI * x) * std::cos(Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);

            exact_sol(i, j, k, 1) = - std::cos(Real(2.0) * M_PI * x) * std::sin(Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);

            exact_sol(i, j, k, 2) = Real(0.25) * ( std::cos(Real(4.0) * M_PI * x) + std::cos(Real(4.0) * M_PI * y) ) * std::exp(-Real(16.0) * M_PI * M_PI * time);
        });
    }
}


void SumAbsStag(const std::array<MultiFab, 
                AMREX_SPACEDIM>& m1,
	            amrex::Vector<amrex::Real>& sum)
{
  BL_PROFILE_VAR("SumAbsStag()", SumAbsStag);

  // Initialize to zero
  std::fill(sum.begin(), sum.end(), 0.);

  ReduceOps<ReduceOpSum> reduce_op;

  //////// x-faces

  ReduceData<Real> reduce_datax(reduce_op);
  using ReduceTuple = typename decltype(reduce_datax)::Type;

  for (MFIter mfi(m1[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      const Box& bx = mfi.tilebox();
      const Box& bx_grid = mfi.validbox();

      auto const& fab = m1[0].array(mfi);

      int xlo = bx_grid.smallEnd(0);
      int xhi = bx_grid.bigEnd(0);

      reduce_op.eval(bx, reduce_datax,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real weight = (i>xlo && i<xhi) ? 1.0 : 0.5;
          return {std::abs(fab(i,j,k)*weight)};
      });
  }

  sum[0] = amrex::get<0>(reduce_datax.value());
  ParallelDescriptor::ReduceRealSum(sum[0]);

  //////// y-faces

  ReduceData<Real> reduce_datay(reduce_op);

  for (MFIter mfi(m1[1],TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      const Box& bx = mfi.tilebox();
      const Box& bx_grid = mfi.validbox();

      auto const& fab = m1[1].array(mfi);

      int ylo = bx_grid.smallEnd(1);
      int yhi = bx_grid.bigEnd(1);

      reduce_op.eval(bx, reduce_datay,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real weight = (j>ylo && j<yhi) ? 1.0 : 0.5;
          return {std::abs(fab(i,j,k)*weight)};
      });
  }

  sum[1] = amrex::get<0>(reduce_datay.value());
  ParallelDescriptor::ReduceRealSum(sum[1]);

#if (AMREX_SPACEDIM == 3)

  //////// z-faces

  ReduceData<Real> reduce_dataz(reduce_op);

  for (MFIter mfi(m1[2],TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      const Box& bx = mfi.tilebox();
      const Box& bx_grid = mfi.validbox();

      auto const& fab = m1[2].array(mfi);

      int zlo = bx_grid.smallEnd(2);
      int zhi = bx_grid.bigEnd(2);

      reduce_op.eval(bx, reduce_dataz,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real weight = (k>zlo && k<zhi) ? 1.0 : 0.5;
          return {std::abs(fab(i,j,k)*weight)};
      });
  }

  sum[2] = amrex::get<0>(reduce_dataz.value());
  ParallelDescriptor::ReduceRealSum(sum[2]);

#endif

}

void StagL2Norm(const std::array<MultiFab, AMREX_SPACEDIM>& m1,
		        const int& comp,
                amrex::Vector<amrex::Real>& inner_prod)
{

    BL_PROFILE_VAR("StagL2Norm()", StagL2Norm);

    Array<MultiFab, AMREX_SPACEDIM> mscr;
    for (int dir=0; dir < AMREX_SPACEDIM; dir++) {
        mscr[dir].define(m1[dir].boxArray(), m1[dir].DistributionMap(), 1, 0);
    }

    StagInnerProd(m1, comp, mscr, inner_prod);
    for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
        inner_prod[dir] = std::sqrt(inner_prod[dir]);
    }
}

void StagInnerProd(const std::array<MultiFab, AMREX_SPACEDIM>& m1,
                   const int& comp1,
                   std::array<MultiFab, AMREX_SPACEDIM>& mscr,
                   amrex::Vector<amrex::Real>& prod_val)
{
  BL_PROFILE_VAR("StagInnerProd()", StagInnerProd);

  for (int d=0; d<AMREX_SPACEDIM; d++) {
    MultiFab::Copy(mscr[d], m1[d], comp1, 0, 1, 0);
    MultiFab::Multiply(mscr[d], m1[d], comp1, 0, 1, 0);
  }

  std::fill(prod_val.begin(), prod_val.end(), 0.);
  SumAbsStag(mscr, prod_val);
}
