#ifndef MY_KERNEL_H_
#define MY_KERNEL_H_

#include <AMReX_FArrayBox.H>

// ================================= KERNEL | INITIALIZATION =================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_userCtx (int i, int j, int k,
                   amrex::Array4<amrex::Real> const& ctx,
                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo)
{
    using amrex::Real;

    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0];
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1];
    // ctx[0] is pressure
    // ctx[1] is intermediate Phi
    ctx(i, j, k, 0) = - 0.25 * ( std::cos(Real(2.0) * M_PI * x) + std::cos(Real(2.0) * M_PI * y) );
    ctx(i, j, k, 1) = 0.0;
#if (AMREX_SPACEDIM > 2)
    Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_cartesian_velocity (int i, int j, int k,
                              amrex::Array4<amrex::Real> const& vcart,
                              amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                              amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo)
{
    using amrex::Real;

    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0];
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1];
    // vcart is volume-centered Cartesian velocity
    vcart(i, j, k, 0) = - std::cos(Real(2.0) * M_PI * x) * std::sin(Real(2.0) * M_PI * y);
    vcart(i, j, k, 1) = std::sin(Real(2.0) * M_PI * x) * std::cos(Real(2.0) * M_PI * y);
#if (AMREX_SPACEDIM > 2)
    Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
    vcart(i, j, k, 2) = Real(0.0);
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_cartesian_velocity_difference (int i, int j, int k,
                                         amrex::Array4<amrex::Real> const& vcart_diff)
{
    using amrex::Real;

    // vcart_diff live in volume center
    vcart_diff(i, j, k, 0) = Real(0.0);
    vcart_diff(i, j, k, 1) = Real(0.0);
#if (AMREX_SPACEDIM > 2)
    vcart_diff(i ,j k, 2) = Real(0.0);
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_contravariant_velocity_difference (int i, int j, int k,
                                             amrex::Array4<amrex::Real> const& cf_cont_diff)
{
    using amrex::Real;

    // cf_cont_diff live in face center
    cf_cont_diff(i, j, k) = Real(0.0);
}

// ============================== UTILITY | DEBUGGING ==============================
void print_box (int const& box_id)
{
    amrex::Print() << "======================== Entering Box " << box_id << " ======================= \n";
}

// ==================================== MODULE | ADVANCE =====================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_half_node_convective_flux_x_contrib (int i, int j, int k,
                                                  amrex::Array4<amrex::Real> const& fluxx_xcont,
                                                  amrex::Array4<amrex::Real> const& fluxy_xcont,
                                                  amrex::Array4<amrex::Real> const& fluxz_xcont,
                                                  amrex::Array4<amrex::Real> const& xcont,
                                                  amrex::Array4<amrex::Real> const& vcart,
                                                  amrex::Real const& coef,
                                                  int const& n_cell,
                                                  int const& box_id)
{
    using amrex::Real;

    // amrex::Print() << "COMPUTING| Half-node Flux | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
    //Contribution of x-directional terms:
    if ( xcont(i, j, k) < 0 ) { // down stream
        if ( i==0 ) { // west wall (if any)
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 0) - Real(2.0) * vcart((i+1)-1, j, k, 0) + Real(3.0) * vcart((i)-1, j, k, 0) ) + vcart((i+1)-1, j, k, 0) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 1) - Real(2.0) * vcart((i+1)-1, j, k, 1) + Real(3.0) * vcart((i)-1, j, k, 1) ) + vcart((i+1)-1, j, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 2) - Real(2.0) * vcart((i+1)-1, j, k, 2) + Real(3.0) * vcart((i)-1, j, k, 2) ) + vcart((i+1)-1, j, k, 2) );
#endif
        } else if ( i==(n_cell-1) ) { // east wall (if any)
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+1)-1, j, k, 0) - Real(2.0) * vcart((i+1)-1, j, k, 0) + Real(3.0) * vcart((i)-1, j, k, 0) ) + vcart((i+1)-1, j, k, 0) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+1)-1, j, k, 1) - Real(2.0) * vcart((i+1)-1, j, k, 1) + Real(3.0) * vcart((i)-1, j, k, 1) ) + vcart((i+1)-1, j, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+1)-1, j, k, 2) - Real(2.0) * vcart((i+1)-1, j, k, 2) + Real(3.0) * vcart((i)-1, j, k, 2) ) + vcart((i+1)-1, j, k, 2) );
#endif
        } else { // inner domain
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 0) - Real(2.0) * vcart((i+1)-1, j, k, 0) + Real(3.0) * vcart((i)-1, j, k, 0) ) + vcart((i+1)-1, j, k, 0) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 1) - Real(2.0) * vcart((i+1)-1, j, k, 1) + Real(3.0) * vcart((i)-1, j, k, 1) ) + vcart((i+1)-1, j, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i+2)-1, j, k, 2) - Real(2.0) * vcart((i+1)-1, j, k, 2) + Real(3.0) * vcart((i)-1, j, k, 2) ) + vcart((i+1)-1, j, k, 2) );
#endif
        }
    } else if ( xcont(i, j, k) > 0) { // up stream
        if ( i==0 ) { // west wall (if any)
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i)-1, j, k, 0) - Real(2.0) * vcart((i)-1, j, k, 0) + Real(3.0) * vcart((i+1)-1, j, k, 0) ) + vcart((i)-1, j, k, 0) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i)-1, j, k, 1) - Real(2.0) * vcart((i)-1, j, k, 1) + Real(3.0) * vcart((i+1)-1, j, k, 1) ) + vcart((i)-1, j, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i)-1, j, k, 2) - Real(2.0) * vcart((i)-1, j, k, 2) + Real(3.0) * vcart((i+1)-1, j, k, 2) ) + vcart((i)-1, j, k, 2) );
#endif
        } else if ( i==(n_cell-1) ) { // east wall (if any)
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 0) - Real(2.0) * vcart((i)-1, j, k, 0) + Real(3.0) * vcart((i+1)-1, j, k, 0) ) + vcart((i)-1, j, k, 0) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 1) - Real(2.0) * vcart((i)-1, j, k, 1) + Real(3.0) * vcart((i+1)-1, j, k, 1) ) + vcart((i)-1, j, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 1) - Real(2.0) * vcart((i)-1, j, k, 1) + Real(3.0) * vcart((i+1)-1, j, k, 1) ) + vcart((i)-1, j, k, 1) );
#endif
        } else { // inner domain
            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 0) - Real(2.0) * vcart((i)-1, j, k, 0) + Real(3.0) * vcart((i+1)-1, j, k, 0) ) + vcart((i)-1, j, k, 0) );

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 1) - Real(2.0) * vcart((i)-1, j, k, 1) + Real(3.0) * vcart((i+1)-1, j, k, 1) ) + vcart((i)-1, j, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_xcont(i, j, k) = xcont(i, j, k) * ( coef * ( - vcart((i-1)-1, j, k, 2) - Real(2.0) * vcart((i)-1, j, k, 2) + Real(3.0) * vcart((i+1)-1, j, k, 2) ) + vcart((i)-1, j, k, 2) );
#endif
        }
    } else { // zero
        fluxx_xcont(i, j, k) = Real(0.0);
        fluxy_xcont(i, j, k) = Real(0.0);
        fluxz_xcont(i, j, k) = Real(0.0);
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_half_node_convective_flux_y_contrib (int i, int j, int k,
                                                  amrex::Array4<amrex::Real> const& fluxx_ycont,
                                                  amrex::Array4<amrex::Real> const& fluxy_ycont,
                                                  amrex::Array4<amrex::Real> const& fluxz_ycont,
                                                  amrex::Array4<amrex::Real> const& ycont,
                                                  amrex::Array4<amrex::Real> const& vcart,
                                                  amrex::Real const& coef,
                                                  int const& n_cell,
                                                  int const& box_id)
{
    using amrex::Real;

    // amrex::Print() << "COMPUTING| Half-node Flux | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
    // Contribution of y-directional terms: flux_ycont_xface, flux_ycont_yface
    if ( ycont(i, j, k) < 0 ) { // down stream
        if ( j==0 ) { // south wall (if any)
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 0) - Real(2.0) * vcart(i, (j+1)-1, k, 0) + Real(3.0) * vcart(i, (j)-1, k, 0) ) + vcart(i, (j+1)-1, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 1) - Real(2.0) * vcart(i, (j+1)-1, k, 1) + Real(3.0) * vcart(i, (j)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 2) - Real(2.0) * vcart(i, (j+1)-1, k, 2) + Real(3.0) * vcart(i, (j)-1, k, 2) ) + vcart(i, (j)-1, k, 2) );
#endif
        } else if ( j==(n_cell-1) ) { // north wall (if any)
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+1)-1, k, 0) - Real(2.0) * vcart(i, (j+1)-1, k, 0) + Real(3.0) * vcart(i, (j)-1, k, 0) ) + vcart(i, (j+1)-1, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+1)-1, k, 1) - Real(2.0) * vcart(i, (j+1)-1, k, 1) + Real(3.0) * vcart(i, (j)-1, k, 1) ) + vcart(i, (j+1)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+1)-1, k, 2) - Real(2.0) * vcart(i, (j+1)-1, k, 2) + Real(3.0) * vcart(i, (j)-1, k, 2) ) + vcart(i, (j+1)-1, k, 2) );
#endif
        } else { // inner domain
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 0) - Real(2.0) * vcart(i, (j+1)-1, k, 0) + Real(3.0) * vcart(i, (j)-1, k, 0) ) + vcart(i, (j+1)-1, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 1) - Real(2.0) * vcart(i, (j+1)-1, k, 1) + Real(3.0) * vcart(i, (j)-1, k, 1) ) + vcart(i, (j+1)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j+2)-1, k, 2) - Real(2.0) * vcart(i, (j+1)-1, k, 2) + Real(3.0) * vcart(i, (j)-1, k, 2) ) + vcart(i, (j+1)-1, k, 2) );
#endif
        }
    } else if ( ycont(i, j, k) > 0) { // up stream
        if ( j==0 ) { // south wall (if any)
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j)-1, k, 0) - Real(2.0) * vcart(i, (j)-1, k, 0) + Real(3.0) * vcart(i, (j+1)-1, k, 0) ) + vcart(i, (j)-1, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j)-1, k, 1) - Real(2.0) * vcart(i, (j)-1, k, 1) + Real(3.0) * vcart(i, (j+1)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j)-1, k, 2) - Real(2.0) * vcart(i, (j)-1, k, 2) + Real(3.0) * vcart(i, (j+1)-1, k, 2) ) + vcart(i, (j)-1, k, 2) );
#endif
        } else if ( j==(n_cell-1) ) { // north wall (if any)
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 0) - Real(2.0) * vcart(i, (j)-1, k, 0) + Real(3.0) * vcart(i, (j+1)-1, k, 0) ) + vcart(i, (j)-1, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 1) - Real(2.0) * vcart(i, (j)-1, k, 1) + Real(3.0) * vcart(i, (j+1)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 2) - Real(2.0) * vcart(i, (j)-1, k, 2) + Real(3.0) * vcart(i, (j+1)-1, k, 2) ) + vcart(i, (j)-1, k, 2) );
#endif
        } else { // inner domain
            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 0) - Real(2.0) * vcart(i, (j)-1, k, 0) + Real(3.0) * vcart(i, (j+1)-1, k, 0) ) + vcart(i, (j)-1, k, 0) );

            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 1) - Real(2.0) * vcart(i, (j)-1, k, 1) + Real(3.0) * vcart(i, (j+1)-1, k, 1) ) + vcart(i, (j)-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            fluxz_ycont(i, j, k) = ycont(i, j, k) * ( coef * ( - vcart(i, (j-1)-1, k, 2) - Real(2.0) * vcart(i, (j)-1, k, 2) + Real(3.0) * vcart(i, (j+1)-1, k, 2) ) + vcart(i, (j)-1, k, 2) );
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
void compute_half_node_convective_flux_z_contrib (int i, int j, int k,
                                                  amrex::Array4<amrex::Real> const& fluxx_zcont,
                                                  amrex::Array4<amrex::Real> const& fluxy_zcont,
                                                  amrex::Array4<amrex::Real> const& fluxz_zcont,
                                                  amrex::Array4<amrex::Real> const& zcont,
                                                  amrex::Array4<amrex::Real> const& vcart,
                                                  amrex::Real const& coef,
                                                  int const& n_cell,
                                                  int const& box_id)
{
    using amrex::Real;

    amrex::Print() << "COMPUTING| Y-Contributing Half-node Flux | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
    // Contribution of z-directional terms:
    fluxx_zcont(i, j, k) = Real(0.0);
    fluxy_zcont(i, j, k) = real(0.0);
    fluxz_zcont(i, j, k) = real(0.0);
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_x (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxx,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dxinv)
{

    // Pressure flux in x-axis
    fluxx(i,j,k) = (phi(i,j,k)-phi(i-1,j,k)) * dxinv;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_y (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxy,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dyinv)
{
    // Pressure flux in y-axis
    fluxy(i,j,k) = (phi(i,j,k)-phi(i,j-1,k)) * dyinv;
}

#if (AMREX_SPACEDIM > 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_z (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxz,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dzinv)
{
    fluxz(i,j,k) = (phi(i,j,k)-phi(i,j,k-1)) * dzinv;
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void update_phi (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& phiold,
                 amrex::Array4<amrex::Real      > const& phinew,
                 AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fluxx,
                              amrex::Array4<amrex::Real const> const& fluxy,
                              amrex::Array4<amrex::Real const> const& fluxz),
                 amrex::Real dt,
                 AMREX_D_DECL(amrex::Real dxinv,
                              amrex::Real dyinv,
                              amrex::Real dzinv))
{
    phinew(i,j,k) = phiold(i,j,k)
        + dt * dxinv * (fluxx(i+1,j  ,k  ) - fluxx(i,j,k))
        + dt * dyinv * (fluxy(i  ,j+1,k  ) - fluxy(i,j,k))
#if (AMREX_SPACEDIM > 2)
        + dt * dzinv * (fluxz(i  ,j  ,k+1) - fluxz(i,j,k));
#else
        ;
#endif
}

#endif

// ==================================== UTILITY | CONVERSION  ================================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void convert_contravariant_velocity_x (int i, int j, int k,
                                       amrex::Array4<amrex::Real> const& xcont,
                                       amrex::Array4<amrex::Real const> const& vcart)
{
    using amrex::Real;
    xcont(i, j, k) = Real(0.5) * ( vcart(i-1, j, k, 0) + vcart(i, j, k, 0) ) ;
    // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
    // amrex::Print() << "DEBUGGING| contravariant velocity in x-direction " << xcont(i, j, k) << "\n";
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void convert_contravariant_velocity_y (int i, int j, int k,
                                       amrex::Array4<amrex::Real> const& ycont,
                                       amrex::Array4<amrex::Real const> const& vcart)
{
    using amrex::Real;
    ycont(i, j, k) = Real(0.5) * ( vcart(i, j-1, k, 1) + vcart(i, j, k, 1) ) ;
}

#if (AMREX_SPACEDIM > 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void convert_contravariant_velocity_z (int i, int j, int k,
                                       amrex::Array4<amrex::Real> const& zcont,
                                       amrex::Array4<amrex::Real const> const& vcart)
{
    using amrex:Real;
    zcont(i, j, k) = Real(0.5) * ( vcart(i, j, k-1, 2) + vcart(i, j, k, 2) ) ;
}
#endif

// ============================== UTILITY | BOUNDARY CONDITIONS ==============================
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void filling_ghost_west (int i, int j, int k, int n_cell, int bcs,
                         amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( i < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";

        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                vcart(i, j, k, dir) = Real(0.0);
            }
        } else {
            if ( bcs == -1 ) {
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    vcart(i, j, k, dir) = - vcart(-i-1, j, k, dir);
                }
            } else if ( bcs == 1 ) {
                vcart(i, j, k, 0) = - vcart(-i-1, j, k, 0);
                vcart(i, j, k, 1) = vcart(-i-1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
                vcart(i, j, k, 2) = vcart(-i-1, j, k, 2);
#endif
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void filling_ghost_east (int i, int j, int k, int n_cell, int bcs,
                         amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( i > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";

        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                vcart(i, j, k, dir) = Real(0.0);
            }
        } else {
            if ( bcs == -1 ) {
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    vcart(i, j, k, dir) = - vcart(( (n_cell-i) + (n_cell-1) ), j, k, dir);
                }
            } else if ( bcs == 1 ) {
                vcart(i, j, k, 0) = - vcart(( (n_cell-i) + (n_cell-1) ), j, k, 0);
                vcart(i, j, k, 1) = vcart(( (n_cell-i) + (n_cell-1) ), j, k, 1);
#if (AMREX_SPACEDIM > 2)
                vcart(i, j, k, 2) = vcart(( (n_cell-i) + (n_cell-1) ), j, k, 2);
#endif
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void filling_ghost_south (int i, int j, int k, int n_cell, int bcs,
                          amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( j < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";

        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                vcart(i, j, k, dir) = Real(0.0);
            }
        } else {
            if ( bcs == -1 ) {
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    vcart(i, j, k, dir) = - vcart(i, -j-1, k, dir);
                }
            } else if ( bcs == 1 ) {
                vcart(i, j, k, 0) = vcart(i, -j-1, k, 0);
                vcart(i, j, k, 1) = - vcart(i, -j-1, k, 1);
#if (AMREX_SPACEDIM > 2)
                vcart(i, j, k, 2) = vcart(i, -j-1, k, 2);
#endif
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void filling_ghost_north (int i, int j, int k, int n_cell, int bcs,
                          amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( j > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";

        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                vcart(i, j, k, dir) = Real(0.0);
            }
        } else {
            if ( bcs == -1 ) {
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    vcart(i, j, k, dir) = - vcart(i, ( (n_cell-j) + (n_cell-1) ), k, dir);
                }
            } else if ( bcs == 1 ) {
                vcart(i, j, k, 0) = vcart(i, ( (n_cell-j) + (n_cell-1) ), k, 0);
                vcart(i, j, k, 1) = - vcart(i, ( (n_cell-j) + (n_cell-1) ), k, 1);
#if (AMREX_SPACEDIM > 2)
                vcart(i, j, k, 2) = vcart(i, ( (n_cell-j) + (n_cell-1) ), k, 2);
#endif
            }
        }
    }
}
#if (AMREX_SPACEDIM > 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void filling_ghost_front (int i, int j, int k, int n_cell, int bcs,
                          amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( k < 0 ) {
        amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";

        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                vcart(i, j, k, dir) = Real(0.0);
            }
        } else {
            if ( bcs == -1 ) {
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    vcart(i, j, k, dir) = - vcart(i, j, -k-1, dir);
                }
            } else if ( bcs == 1 ) {
                vcart(i, j, k, 0) = vcart(i, j, -k-1, 0);
                vcart(i, j, k, 1) = vcart(i, j, -k-1, 1);
#if (AMREX_SPACEDIM > 2)
                vcart(i, j, k, 2) = - vcart(i, j, -k-1, 2);
#endif
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void filling_ghost_back (int i, int j, int k, int n_cell, int bcs,
                         amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( k > (n_cell - 1) ) {
        amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";

        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                vcart(i, j, k, dir) = Real(0.0);
            }
        } else {
            if ( bcs == -1 ) {
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    vcart(i, j, k, dir) = - vcart(i, j, ( (n_cell-k) + (n_cell-1) ), dir);
                }
            } else if ( bcs == 1 ) {
                vcart(i, j, k, 0) = vcart(i, j, ( (n_cell-k) + (n_cell-1) ), 0);
                vcart(i, j, k, 1) = vcart(i, j, ( (n_cell-k) + (n_cell-1) ), 1);
#if (AMREX_SPACEDIM > 2)
                vcart(i, j, k, 2) = - vcart(i, j, ( (n_cell-k) + (n_cell-1) ), 2);
#endif
            }
        }
    }
}
#endif

