#ifndef MY_FLUX_H_
#define MY_FLUX_H_

#include <AMReX_FArrayBox.H>

// ================================= KERNEL | FLUXES CALCULATION =================================
// +++++++++++++++ CASE: PERIODIC BOUNADRY CONDITIONS +++++++++++++++
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_halfnode_convective_flux_x_contrib_periodic (int i, int j, int k,
                                                          amrex::Array4<amrex::Real> const& fluxx_xcont,
                                                          amrex::Array4<amrex::Real> const& fluxy_xcont,
                                                          amrex::Array4<amrex::Real> const& fluxz_xcont,
                                                          amrex::Array4<amrex::Real> const& xcont,
                                                          amrex::Array4<amrex::Real> const& vcart,
                                                          amrex::Real const& coef)
{
    using amrex::Real;

    // amrex::Print() << "COMPUTING| X-Contributing Half-node Flux | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
    //Contribution of x-directional terms:
    if ( xcont(i, j, k) < 0 ) { // down stream
        fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 0) - Real(2.0) * vcart((i+1)-1, j, k, 0) + Real(3.0) * vcart((i)-1, j, k, 0) ) + vcart((i+1)-1, j, k, 0) );

        fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 1) - Real(2.0) * vcart(i, (j+1)-1, k, 1) + Real(3.0) * vcart(i, (j)-1, k, 1) ) + vcart(i, (j+1)-1, k, 1) );

        fluxz_xcont(i, j, k) = Real(0.0)
#if (AMREX_SPACEDIM > 2)
            + xcont(i, j, k) * ( coef * ( - vcart(i, j, (k+2)-1, 2) - Real(2.0) * vcart(i, j, (k+1)-1, 2) + Real(3.0) * vcart(i, j, (k)-1, 2) ) + vcart(i, j, (k+1)-1, 2) );
#else
        ;
#endif
    } else if ( xcont(i, j, k) > 0) { // up stream
        fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 0) - Real(2.0) * vcart((i)-1, j, k, 0) + Real(3.0) * vcart((i+1)-1, j, k, 0) ) + vcart((i)-1, j, k, 0) );

        fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 1) - Real(2.0) * vcart(i, (j)-1, k, 1) + Real(3.0) * vcart(i, (j+1)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

        fluxz_xcont(i, j, k) = Real(0.0)
#if (AMREX_SPACEDIM > 2)
            + xcont(i, j, k) * ( coef * ( - vcart(i, j, (k-1)-1, 2) - Real(2.0) * vcart(i, j, (k)-1, 2) + Real(3.0) * vcart(i, j, (k+1)-1, 2) ) + vcart(i, j, (k)-1, 2) );
#else
        ;
#endif
    } else { // zero xcont(i, j, k)
        fluxx_xcont(i, j, k) = Real(0.0);
        fluxy_xcont(i, j, k) = Real(0.0);
        fluxz_xcont(i, j, k) = Real(0.0);
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_halfnode_convective_flux_y_contrib_periodic (int i, int j, int k,
                                                          amrex::Array4<amrex::Real> const& fluxx_ycont,
                                                          amrex::Array4<amrex::Real> const& fluxy_ycont,
                                                          amrex::Array4<amrex::Real> const& fluxz_ycont,
                                                          amrex::Array4<amrex::Real> const& ycont,
                                                          amrex::Array4<amrex::Real> const& vcart,
                                                          amrex::Real const& coef)
{
    using amrex::Real;

    // amrex::Print() << "COMPUTING| Y-Contributing Half-node Flux | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
    // Contribution of y-directional terms: flux_ycont_xface, flux_ycont_yface
    if ( ycont(i, j, k) < 0 ) { // down stream
        fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 0) - Real(2.0) * vcart((i+1)-1, j, k, 0) + Real(3.0) * vcart((i)-1, j, k, 0) ) + vcart((i+1)-1, j, k, 0) );

        fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 1) - Real(2.0) * vcart(i, (j+1)-1, k, 1) + Real(3.0) * vcart(i, (j)-1, k, 1) ) + vcart(i, (j+1)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
        fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j, (k+2)-1, 2) - Real(2.0) * vcart(i, j, (k+1)-1, 2) + Real(3.0) * vcart(i, j, (k)-1, 2) ) + vcart(i, j, (k+1)-1, 2) );
#endif
    } else if ( ycont(i, j, k) > 0) { // up stream
        fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 0) - Real(2.0) * vcart((i)-1, j, k, 0) + Real(3.0) * vcart((i+1)-1, j, k, 0) ) + vcart((i)-1, j, k, 0) );

        fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 1) - Real(2.0) * vcart(i, (j)-1, k, 1) + Real(3.0) * vcart(i, (j+1)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

        fluxz_ycont(i, j, k) = Real(0.0)
#if (AMREX_SPACEDIM > 2)
            + ycont(i, j, k) * ( coef * ( - vcart(i, j, (k-1)-1, 2) - Real(2.0) * vcart(i, j, (k)-1, 2) + Real(3.0) * vcart(i, j, (k+1)-1, 2) ) + vcart(i, j, (k)-1, 2) );
#else
        ;

#endif
    } else { // zero ycont(i, j, k)
        fluxx_ycont(i, j, k) = Real(0.0);
        fluxy_ycont(i, j, k) = Real(0.0);
        fluxz_ycont(i, j, k) = Real(0.0);
    }
}

#if (AMREX_SPACEDIM > 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_half_node_convective_flux_z_contrib_periodic (int i, int j, int k,
                                                           amrex::Array4<amrex::Real> const& fluxx_zcont,
                                                           amrex::Array4<amrex::Real> const& fluxy_zcont,
                                                           amrex::Array4<amrex::Real> const& fluxz_zcont,
                                                           amrex::Array4<amrex::Real> const& zcont,
                                                           amrex::Array4<amrex::Real> const& vcart,
                                                           amrex::Real const& coef,
                                                           int const& n_cell)
{
    using amrex::Real;

    amrex::Print() << "COMPUTING| Z-Contributing Half-node Flux | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
    // Contribution of z-directional terms:
    fluxx_zcont(i, j, k) = Real(0.0);
    fluxy_zcont(i, j, k) = real(0.0);
    fluxz_zcont(i, j, k) = real(0.0);
}
#endif

// +++++++++++++++ CASE: PERIODIC BOUNADRY CONDITIONS +++++++++++++++
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_halfnode_convective_flux_x_contrib_wall (int i, int j, int k,
                                                      amrex::Array4<amrex::Real> const& fluxx_xcont,
                                                      amrex::Array4<amrex::Real> const& fluxy_xcont,
                                                      amrex::Array4<amrex::Real> const& fluxz_xcont,
                                                      amrex::Array4<amrex::Real> const& xcont,
                                                      amrex::Array4<amrex::Real> const& vcart,
                                                      amrex::Real const& coef,
                                                      int const& n_cell)
{
    using amrex::Real;

    // amrex::Print() << "COMPUTING| Half-node Flux | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
    //Contribution of x-directional terms:
    if ( xcont(i, j, k) < 0 ) { // down stream
        if ( i==0 ) { // west wall (if any)
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 0) - Real(2.0) * vcart((i+1)-1, j, k, 0) + Real(3.0) * vcart((i)-1, j, k, 0) ) + vcart((i+1)-1, j, k, 0) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 1) - Real(2.0) * vcart(i, (j+1)-1, k, 1) + Real(3.0) * vcart(i, (j)-1, k, 1) ) + vcart(i, (j+1)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, j, (k+2)-1, 2) - Real(2.0) * vcart(i, j, (k+1)-1, 2) + Real(3.0) * vcart(i, j, (k)-1, 2) ) + vcart(i, j, (k+1)-1, 2) );
#endif
        } else if ( i==(n_cell-1) ) { // east wall (if any)
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+1)-1, j, k, 0) - Real(2.0) * vcart((i+1)-1, j, k, 0) + Real(3.0) * vcart((i)-1, j, k, 0) ) + vcart((i+1)-1, j, k, 0) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, (j+1)-1, k, 1) - Real(2.0) * vcart(i, (j+1)-1, k, 1) + Real(3.0) * vcart(i, (j)-1, k, 1) ) + vcart(i, (j+1)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, j, (k+1)-1, 2) - Real(2.0) * vcart(i, j, (k+1)-1, 2) + Real(3.0) * vcart(i, j, (k)-1, 2) ) + vcart(i, j, (k+1)-1, 2) );
#endif
        } else { // inner domain
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 0) - Real(2.0) * vcart((i+1)-1, j, k, 0) + Real(3.0) * vcart((i)-1, j, k, 0) ) + vcart((i+1)-1, j, k, 0) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 1) - Real(2.0) * vcart(i, (j+1)-1, k, 1) + Real(3.0) * vcart(i, (j)-1, k, 1) ) + vcart(i, (j+1)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, j, (k+2)-1, 2) - Real(2.0) * vcart(i, j, (k+1)-1, 2) + Real(3.0) * vcart(i, j, (k)-1, 2) ) + vcart(i, j, (k+1)-1, 2) );
#endif
        }
    } else if ( xcont(i, j, k) > 0) { // up stream
        if ( i==0 ) { // west wall (if any)
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i)-1, j, k, 0) - Real(2.0) * vcart((i)-1, j, k, 0) + Real(3.0) * vcart((i+1)-1, j, k, 0) ) + vcart((i)-1, j, k, 0) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, (j)-1, k, 1) - Real(2.0) * vcart(i, (j)-1, k, 1) + Real(3.0) * vcart(i, (j+1)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, j, (k)-1, 2) - Real(2.0) * vcart(i, j, (k)-1, 2) + Real(3.0) * vcart(i, j, (k+1)-1, 2) ) + vcart(i, j, (k)-1, 2) );
#endif
        } else if ( i==(n_cell-1) ) { // east wall (if any)
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 1) - Real(2.0) * vcart((i)-1, j, k, 1) + Real(3.0) * vcart((i+1)-1, j, k, 1) ) + vcart((i)-1, j, k, 1) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 1) - Real(2.0) * vcart(i, (j)-1, k, 1) + Real(3.0) * vcart(i, (j+1)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, j, (k-1)-1, 0) - Real(2.0) * vcart(i, j, (k)-1, 0) + Real(3.0) * vcart(i, j, (k+1)-1, 0) ) + vcart(i, j, (k)-1, 0) );
#endif
        } else { // inner domain
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 0) - Real(2.0) * vcart((i)-1, j, k, 0) + Real(3.0) * vcart((i+1)-1, j, k, 0) ) + vcart((i)-1, j, k, 0) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 1) - Real(2.0) * vcart(i, (j)-1, k, 1) + Real(3.0) * vcart(i, (j+1)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart(i, j, (k-1)-1, 2) - Real(2.0) * vcart(i, j, (k)-1, 2) + Real(3.0) * vcart(i, j, (k+1)-1, 2) ) + vcart(i, j, (k)-1, 2) );
#endif
        }
    } else { // zero
        fluxx_xcont(i, j, k) = Real(0.0);
        fluxy_xcont(i, j, k) = Real(0.0);
        fluxz_xcont(i, j, k) = Real(0.0);
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_halfnode_convective_flux_y_contrib_wall (int i, int j, int k,
                                                      amrex::Array4<amrex::Real> const& fluxx_ycont,
                                                      amrex::Array4<amrex::Real> const& fluxy_ycont,
                                                      amrex::Array4<amrex::Real> const& fluxz_ycont,
                                                      amrex::Array4<amrex::Real> const& ycont,
                                                      amrex::Array4<amrex::Real> const& vcart,
                                                      amrex::Real const& coef,
                                                      int const& n_cell)
{
    using amrex::Real;

    // amrex::Print() << "COMPUTING| Half-node Flux | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
    // Contribution of y-directional terms: flux_ycont_xface, flux_ycont_yface
    if ( ycont(i, j, k) < 0 ) { // down stream
        if ( j==0 ) { // south wall (if any)
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 0) - Real(2.0) * vcart((i+1)-1, j, k, 0) + Real(3.0) * vcart((i)-1, j, k, 0) ) + vcart((i+1)-1, j, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 1) - Real(2.0) * vcart(i, (j+1)-1, k, 1) + Real(3.0) * vcart(i, (j)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j, (k+2)-1, 2) - Real(2.0) * vcart(i, j, (k+1)-1, 2) + Real(3.0) * vcart(i, j, (k)-1, 2) ) + vcart(i, j, (k)-1, 2) );
#endif
        } else if ( j==(n_cell-1) ) { // north wall (if any)
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart((i+1)-1, j, k, 0) - Real(2.0) * vcart((i+1)-1, j, k, 0) + Real(3.0) * vcart((i)-1, j, k, 0) ) + vcart((i+1)-1, j, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+1)-1, k, 1) - Real(2.0) * vcart(i, (j+1)-1, k, 1) + Real(3.0) * vcart(i, (j)-1, k, 1) ) + vcart(i, (j+1)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j, (k+1)-1, 2) - Real(2.0) * vcart(i, j, (k+1)-1, 2) + Real(3.0) * vcart(i, j, (k)-1, 2) ) + vcart(i, j, (k+1)-1, 2) );
#endif
        } else { // inner domain
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 0) - Real(2.0) * vcart((i+1)-1, j, k, 0) + Real(3.0) * vcart((i)-1, j, k, 0) ) + vcart((i+1)-1, j, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 1) - Real(2.0) * vcart(i, (j+1)-1, k, 1) + Real(3.0) * vcart(i, (j)-1, k, 1) ) + vcart(i, (j+1)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j, (k+2)-1, 2) - Real(2.0) * vcart(i, j, (k+1)-1, 2) + Real(3.0) * vcart(i, j, (k)-1, 2) ) + vcart(i, j, (k+1)-1, 2) );
#endif
        }
    } else if ( ycont(i, j, k) > 0) { // up stream
        if ( j==0 ) { // south wall (if any)
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart((i)-1, j, k, 0) - Real(2.0) * vcart((i)-1, j, k, 0) + Real(3.0) * vcart((i+1)-1, j, k, 0) ) + vcart((i)-1, j, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j)-1, k, 1) - Real(2.0) * vcart(i, (j)-1, k, 1) + Real(3.0) * vcart(i, (j+1)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j, (k)-1, 2) - Real(2.0) * vcart(i, j, (k)-1, 2) + Real(3.0) * vcart(i, j, (k+1)-1, 2) ) + vcart(i, j, (k)-1, 2) );
#endif
        } else if ( j==(n_cell-1) ) { // north wall (if any)
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 0) - Real(2.0) * vcart((i)-1, j, k, 0) + Real(3.0) * vcart((i+1)-1, j, k, 0) ) + vcart((i)-1, j, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 1) - Real(2.0) * vcart(i, (j)-1, k, 1) + Real(3.0) * vcart(i, (j+1)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j, (k-1)-1, 2) - Real(2.0) * vcart(i, j, (k)-1, 2) + Real(3.0) * vcart(i, j, (k+1)-1, 2) ) + vcart(i, j, (k)-1, 2) );
#endif
        } else { // inner domain
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 0) - Real(2.0) * vcart((i)-1, j, k, 0) + Real(3.0) * vcart((i+1)-1, j, k, 0) ) + vcart((i)-1, j, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 1) - Real(2.0) * vcart(i, (j)-1, k, 1) + Real(3.0) * vcart(i, (j+1)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, j, (k-1)-1, 2) - Real(2.0) * vcart(i, j, (k)-1, 2) + Real(3.0) * vcart(i, j, (k+1)-1, 2) ) + vcart(i, j, (k)-1, 2) );
#endif
        }
    } else { // zero
        fluxx_ycont(i, j, k) = Real(0.0);
        fluxy_ycont(i, j, k) = Real(0.0);
        fluxz_ycont(i, j, k) = Real(0.0);
    }
}

#if (AMREX_SPACEDIM > 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_halfnode_convective_flux_z_contrib_wall (int i, int j, int k,
                                                      amrex::Array4<amrex::Real> const& fluxx_zcont,
                                                      amrex::Array4<amrex::Real> const& fluxy_zcont,
                                                      amrex::Array4<amrex::Real> const& fluxz_zcont,
                                                      amrex::Array4<amrex::Real> const& zcont,
                                                      amrex::Array4<amrex::Real> const& vcart,
                                                      amrex::Real const& coef,
                                                      int const& n_cell)
{
    using amrex::Real;

    amrex::Print() << "COMPUTING| Z-Contributing Half-node Flux | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
    // Contribution of z-directional terms:
    fluxx_zcont(i, j, k) = Real(0.0);
    fluxy_zcont(i, j, k) = real(0.0);
    fluxz_zcont(i, j, k) = real(0.0);
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_viscous_flux_periodic (int i, int j, int k,
                                    amrex::Array4<amrex::Real> const& visc_flux,
                                    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                                    amrex::Array4<amrex::Real> const& vcart,
                                    amrex::Real const& ren)
{
    for ( int dir=0; dir < AMREX_SPACEDIM; ++dir )
    {
        auto const& centerMAC = vcart(i, j, k, dir);
        auto const& northMAC = vcart(i, j+1, k, dir);
        auto const& southMAC = vcart(i, j-1, k, dir);
        auto const& westMAC = vcart(i-1, j, k, dir);
        auto const& eastMAC = vcart(i+1, j, k, dir);

        visc_flux(i, j, k, dir) = ( (westMAC - 2*centerMAC + eastMAC)/(dx[0]*dx[0]) + (southMAC - 2*centerMAC + northMAC)/(dx[1]*dx[1]) )/ren;
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_pressure_gradient_periodic (int i, int j, int k,
                                         amrex::Array4<amrex::Real> const& presgrad_flux,
                                         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                                         amrex::Array4<amrex::Real> const& pressurefield)
{
    using amrex::Real;

    presgrad_flux(i, j, k, 0) = (pressurefield(i+1, j, k, 0) - pressurefield(i-1, j, k, 0))/(Real(2.0)*dx[0]);
    presgrad_flux(i, j, k, 1) = (pressurefield(i, j+1, k, 0) - pressurefield(i, j-1, k, 0))/(Real(2.0)*dx[1]);
#if (AMREX_SPACEDIM > 2)
    presgrad_flux(i, j, k, 2) = (pressurefield(i, j, k+1, 0) - pressurefield(i, j, k-1, 0))/(Real(2.0)*dx[2]);
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_total_flux (int i, int j, int k,
                         amrex::Array4<amrex::Real> const& total_flux,
                         amrex::Array4<amrex::Real> const& conv_flux,
                         amrex::Array4<amrex::Real> const& visc_flux,
                         amrex::Array4<amrex::Real> const& prsgrad_flux)
{
    for ( int dir=0; dir < AMREX_SPACEDIM; ++dir )
    {
        total_flux(i, j, k, dir) = - conv_flux(i, j, k, dir) + visc_flux(i, j, k, dir) - prsgrad_flux(i, j, k, dir);
    }
}


#endif
