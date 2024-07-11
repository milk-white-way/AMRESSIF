#include <AMReX_MultiFabUtil.H>

#include "fn_enforce_wall_bcs.H"
#include "fn_flux_calc.H"
#include "kn_flux_calc.H"
#include "kn_poisson.H"

using namespace amrex;
int const& UMIST = 0;

// ++++++++++++++++++++++++++++++ Convective Flux ++++++++++++++++++++++++++++++
void convective_flux_calc ( MultiFab& fluxConvect,
                            Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                            Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                            Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN3,
                            MultiFab& velCart,
                            Array<MultiFab, AMREX_SPACEDIM>& velCont,
                            Vector<int> const& phy_bc_lo,
                            Vector<int> const& phy_bc_hi,
                            Geometry const& geom,
                            int const& n_cell )
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // UMIST (Upstream Monotonic Interpolation for Scalar Transport) is a scheme within the flux-limited method. (Lien and Leschziner, 1993)
    // It is a limited variant of QUICK scheme, and has 3rd order accuracy where monotonic.
    //Real flux_limited_r, u_over_deltx, v_over_delty;
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
        auto const& xcont = velCont[0].array(mfi);
        auto const& ycont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
       auto const& zcont = velCont[2].array(mfi);
#endif
        auto const& fluxx_xcont = fluxHalfN1[0].array(mfi);
        auto const& fluxy_xcont = fluxHalfN2[0].array(mfi);
        auto const& fluxz_xcont = fluxHalfN3[0].array(mfi);

        auto const& fluxx_ycont = fluxHalfN1[1].array(mfi);
        auto const& fluxy_ycont = fluxHalfN2[1].array(mfi);
        auto const& fluxz_ycont = fluxHalfN3[1].array(mfi);

#if (AMREX_SPACEDIM > 2)
        auto const& fluxx_zcont = fluxHalfN1[2].array(mfi);
        auto const& fluxy_zcont = fluxHalfN2[2].array(mfi);
        auto const& fluxz_zcont = fluxHalfN3[2].array(mfi);
#endif
        auto const& vcart = velCart.array(mfi);

        auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
        auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

        auto const& south_wall_bcs = phy_bc_lo[1]; // south wall
        auto const& north_wall_bcs = phy_bc_hi[1]; // north wall
#if (AMREX_SPACEDIM > 2)
        auto const& fron_wall_bcs = phy_bc_lo[2]; // front wall
        auto const& back_wall_bcs = phy_bc_hi[2]; // back wall
#endif

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            //amrex::Print() << "MOMENTUM | Calculating half-node flux at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
            // Note that the half-node flux (face-centered) are calculated in the east face
            // This leaves the first half-node flux on the west boundary to be prescribed by the boundary conditions
            // Step 1: Assign local cell-centre nodes
            auto const& xcart_W  = vcart(i-2, j, k, 0);
            auto const& xcart_P  = vcart(i-1, j, k, 0);
            auto const& xcart_E  = vcart(i  , j, k, 0);
            auto const& xcart_EE = vcart(i+1, j, k, 0);

            auto const& ycart_W  = vcart(i-2, j, k, 1);
            auto const& ycart_P  = vcart(i-1, j, k, 1);
            auto const& ycart_E  = vcart(i  , j, k, 1);
            auto const& ycart_EE = vcart(i+1, j, k, 1);

            // Step 2: Detect direction of flow
            auto const& ucon = Real(0.5) * xcont(i, j, k);
            auto const& fldr = ( ucon - std::abs(ucon) )/xcont(i, j, k); // Print() << "fldr: " << fldr << "\n"; // DEBUGGING
            //  ucon - |ucon|     -- 0 if ucon > 0, ==> xcont(i, j, k) > 0 (flow to the right)
            //---------------- = |
            // xcont(i, j, k)    -- 1 if ucon < 0 ==> xcont(i, j, k) < 0 (flow to the left)

            // Default that the flow is to the left
            Real xcart_UU = xcart_EE;
            Real xcart_U  = xcart_E;
            Real xcart_D  = xcart_P;

            Real ycart_UU = ycart_EE;
            Real ycart_U  = ycart_E;
            Real ycart_D  = ycart_P;
           if ( fldr == 0 ) {
                // Flow to the right
                xcart_UU = xcart_W;
                xcart_U  = xcart_P;
                xcart_D  = xcart_E;

                ycart_UU = ycart_W;
                ycart_U  = ycart_P;
                ycart_D  = ycart_E;
            } 

            Real psi = Real(2.0);
            Real flux_limited_ratio = psi;

            fluxx_xcont(i, j, k) = xcont(i, j, k) * ( - xcart_UU/8 + 3*xcart_U/4 + 3*xcart_D/8);
#if (UMIST == 1)
            if ( xcart_D != xcart_U ) {
                flux_limited_ratio = ( xcart_U - xcart_UU ) / ( xcart_D - xcart_U );
                if ( flux_limited_ratio < 0) {
                    // non-monotonic 
                    fluxx_xcont(i, j, k) = xcont(i, j, k) * xcart_U;
                } else {
                    // monotonic
                    psi = std::min(psi, 2*flux_limited_ratio);
                    psi = std::min(psi, (1 + 3*flux_limited_ratio)/4);
                    psi = std::min(psi, (3 + flux_limited_ratio)/4);
                    Print() << "psi: " << psi << "\n";
                    fluxx_xcont(i, j, k) = xcont(i, j, k) * ( xcart_U + Real(0.5)*psi*(xcart_D - xcart_U) );
                }
                Print() << "flux_limited_ratio: " << flux_limited_ratio << "\n";
            }
#endif

            fluxy_xcont(i, j, k) = xcont(i, j, k) * ( - ycart_UU/8 + 3*ycart_U/4 + 3*ycart_D/8);
#if (UMIST == 1)
            if ( ycart_D != ycart_U ) {
                flux_limited_ratio = ( ycart_U - ycart_UU ) / ( ycart_D - ycart_U );
                if ( flux_limited_ratio < 0) {
                    // non-monotonic 
                    fluxy_xcont(i, j, k) = xcont(i, j, k) * ycart_U;
                } else {
                    // monotonic
                    psi = std::min(psi, 2*flux_limited_ratio);
                    psi = std::min(psi, (1 + 3*flux_limited_ratio)/4);
                    psi = std::min(psi, (3 + flux_limited_ratio)/4);
                    Print() << "psi: " << psi << "\n";
                    fluxy_xcont(i, j, k) = xcont(i, j, k) * ( ycart_U + Real(0.5)*psi*(ycart_D - ycart_U) );
                }
                Print() << "flux_limited_ratio: " << flux_limited_ratio << "\n";
            }
#endif
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Note that the half-node flux (face-centered) are calculated in the north face
            // This leaves the first half-node flux on the south boundary to be prescribed by the boundary conditions
            auto const& xcart_S  = vcart(i, j-2, k, 0);
            auto const& xcart_P  = vcart(i, j-1, k, 0);
            auto const& xcart_N  = vcart(i, j  , k, 0);
            auto const& xcart_NN = vcart(i, j+1, k, 0);

            auto const& ycart_S  = vcart(i, j-2, k, 1);
            auto const& ycart_P  = vcart(i, j-1, k, 1);
            auto const& ycart_N  = vcart(i, j  , k, 1);
            auto const& ycart_NN = vcart(i, j+1, k, 1);

            // Step 2: Detect direction of flow
            auto const& ucon = Real(0.5) * ycont(i, j, k);
            auto const& fldr = ( ucon - std::abs(ucon) )/ycont(i, j, k);
            //  ucon - |ucon|     -- 0 if ucon > 0, ==> ycont(i, j, k) > 0 (flow to the top)
            //---------------- = |
            // ycont(i, j, k)    -- 1 if ucon < 0 ==> ycont(i, j, k) < 0 (flow to the bottom)

            // Default that the flow is to the bottom
            Real xcart_UU = xcart_NN;
            Real xcart_U  = xcart_N;
            Real xcart_D  = xcart_P;

            Real ycart_UU = ycart_NN;
            Real ycart_U  = ycart_N;
            Real ycart_D  = ycart_P;
           if ( fldr == 0 ) {
                // Flow to the top
                xcart_UU = xcart_N;
                xcart_U  = xcart_P;
                xcart_D  = xcart_S;

                ycart_UU = ycart_N;
                ycart_U  = ycart_P;
                ycart_D  = ycart_S;
            } 

            Real psi = Real(2.0);
            Real flux_limited_ratio = psi;

            fluxx_ycont(i, j, k) = ycont(i, j, k) * ( - xcart_UU/8 + 3*xcart_U/4 + 3*xcart_D/8);
#if (UMIST == 1)
            if ( xcart_D != xcart_U ) {
                flux_limited_ratio = ( xcart_U - xcart_UU ) / ( xcart_D - xcart_U );
                if ( flux_limited_ratio < 0) {
                    // non-monotonic 
                    fluxx_ycont(i, j, k) = ycont(i, j, k) * xcart_U;
                } else {
                    // monotonic
                    psi = std::min(psi, Real(2.0)*flux_limited_ratio);
                    psi = std::min(psi, (1 + 3*flux_limited_ratio)/4);
                    psi = std::min(psi, (3 + flux_limited_ratio)/4);
                    Print() << "psi: " << psi << "\n";
                    fluxx_xcont(i, j, k) = ycont(i, j, k) * ( xcart_U + Real(0.5)*psi*(xcart_D - xcart_U) );
                }
                Print() << "flux_limited_ratio: " << flux_limited_ratio << "\n";
            }
#endif
            
            fluxy_ycont(i, j, k) = ycont(i, j, k) * ( - ycart_UU/8 + 3*ycart_U/4 + 3*ycart_D/8);
#if (UMIST == 1)
            if ( ycart_D != ycart_U ) {
                flux_limited_ratio = ( ycart_U - ycart_UU ) / ( ycart_D - ycart_U );
                if ( flux_limited_ratio < 0) {
                    // non-monotonic 
                    fluxy_ycont(i, j, k) = ycont(i, j, k) * ycart_U;
                } else {
                    // monotonic
                    psi = std::min(psi, 2*flux_limited_ratio);
                    psi = std::min(psi, (1 + 3*flux_limited_ratio)/4);
                    psi = std::min(psi, (3 + flux_limited_ratio)/4);
                    Print() << "psi: " << psi << "\n";
                    fluxy_xcont(i, j, k) = ycont(i, j, k) * ( ycart_U + Real(0.5)*psi*(ycart_D - ycart_U) );
                }
                Print() << "flux_limited_ratio: " << flux_limited_ratio << "\n";
            }
#endif
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Not implemented
        });
#endif
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxConvect); mfi.isValid(); ++mfi ) {
        const Box& vbx = mfi.validbox();
        auto const& conv_flux = fluxConvect.array(mfi);

        auto const& fluxx_xcont = fluxHalfN1[0].array(mfi);
        auto const& fluxy_xcont = fluxHalfN2[0].array(mfi);
        auto const& fluxz_xcont = fluxHalfN3[0].array(mfi);

        auto const& fluxx_ycont = fluxHalfN1[1].array(mfi);
        auto const& fluxy_ycont = fluxHalfN2[1].array(mfi);
        auto const& fluxz_ycont = fluxHalfN3[1].array(mfi);

#if (AMREX_SPACEDIM > 2)
        auto const& fluxx_zcont = fluxHalfN1[2].array(mfi);
        auto const& fluxy_zcont = fluxHalfN2[2].array(mfi);
        auto const& fluxz_zcont = fluxHalfN3[2].array(mfi);
#endif
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            conv_flux(i, j, k, 0) = (fluxx_xcont(i+1, j, k) - fluxx_xcont(i, j, k))/(dx[0]) + (fluxx_ycont(i, j+1, k) - fluxx_ycont(i, j, k))/(dx[1])
#if (AMREX_SPACEDIM > 2)
                + (fluxx_zcont(i, j, k+1) - fluxx_zcont(i, j, k))/(dx[2]);
#else
            ;
#endif

            conv_flux(i, j, k, 1) = (fluxy_xcont(i+1, j, k) - fluxy_xcont(i, j, k))/(dx[0]) + (fluxy_ycont(i, j+1, k) - fluxy_ycont(i, j, k))/(dx[1])
#if (AMREX_SPACEDIM > 2)
                + (fluxy_zcont(i, j, k+1) - fluxy_zcont(i, j, k))/(dx[2]);
#else
            ;
#endif

#if (AMREX_SPACEDIM > 2)
            conv_flux(i, j, k, 2) = (fluxz_xcont(i+1, j, k) - fluxz_xcont(i, j, k))/(dx[0]) + (fluxz_ycont(i, j+1, k) - fluxz_ycont(i, j, k))/(dx[1]) + (fluxz_zcont(i, j, k+1) - fluxz_zcont(i, j, k))/(dx[2]);
#endif
        });
    }
}

// ++++++++++++++++++++++++++++++ Viscous Flux ++++++++++++++++++++++++++++++
void viscous_flux_calc ( MultiFab& fluxViscous,
                         MultiFab& velCart,
                         Geometry const& geom,
                         Real const& ren )
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxViscous); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vel_cart = velCart.array(mfi);
        auto const& visc_flux = fluxViscous.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // 2D PERIODIC ONLY
            compute_viscous_flux_periodic(i, j, k, visc_flux, dx, vel_cart, ren);
        });
    }
}

// +++++++++++++++++++++++++ Gradient Flux  +++++++++++++++++++++++++
void gradient_calc_approach1 ( MultiFab& fluxPrsGrad,
                               MultiFab& cc_grad_phi,
                               MultiFab& userCtx,
                               Geometry const& geom,
                               int const& Nghost,
                               Vector<int> const& phy_bc_lo,
                               Vector<int> const& phy_bc_hi,
                               int const& n_cell )
{
    enforce_wall_bcs_for_cell_centered_userCtx_on_ghost_cells(userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxPrsGrad); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx = userCtx.array(mfi);
        auto const& presgrad_flux = fluxPrsGrad.array(mfi);
        auto const& grad_phi = cc_grad_phi.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // PERIODIC ONLY
            compute_pressure_gradient_periodic(i, j, k, presgrad_flux, dx, ctx);
            compute_phi_gradient_periodic(i, j, k, grad_phi, dx, ctx);
        });
    }
    
    enforce_wall_bcs_for_cell_centered_flux_on_ghost_cells(cc_grad_phi, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
}

void gradient_calc_approach2 ( Array<MultiFab, AMREX_SPACEDIM>& array_grad_p,
                               Array<MultiFab, AMREX_SPACEDIM>& array_grad_phi,
                               MultiFab& userCtx,
                               Geometry const& geom,
                               int const& Nghost,
                               Vector<int> const& phy_bc_lo,
                               Vector<int> const& phy_bc_hi,
                               int const& n_cell )
{
    userCtx.FillBoundary(geom.periodicity());
    //enforce_wall_bcs_for_cell_centered_userCtx_on_ghost_cells(userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    Box dom(geom.Domain());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(array_grad_phi[1]); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif

        auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
        auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

        auto const& south_wall_bcs = phy_bc_lo[1]; // south wall
        auto const& north_wall_bcs = phy_bc_hi[1]; // north wall
#if (AMREX_SPACEDIM > 2)
        auto const& fron_wall_bcs = phy_bc_lo[2]; // front wall
        auto const& back_wall_bcs = phy_bc_hi[2]; // back wall
#endif

        auto const& grad_p_x = array_grad_p[0].array(mfi);
        auto const& grad_p_y = array_grad_p[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& grad_p_z = array_grad_p[2].array(mfi);
#endif

        auto const& grad_phi_x = array_grad_phi[0].array(mfi);
        auto const& grad_phi_y = array_grad_phi[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& grad_phi_z = array_grad_phi[2].array(mfi);
#endif

        auto const& ctx = userCtx.array(mfi);

        // Avoiding boundary face-centered gradients
        int lo = dom.smallEnd(0);
        int hi = dom.bigEnd(0)+1;
        
        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            grad_p_x(i, j, k) = ( ctx(i, j, k, 0) - ctx(i-1, j, k, 0) )/dx[0];

            grad_phi_x(i, j, k) = ( ctx(i, j, k, 1) - ctx(i-1, j, k, 1) )/dx[0];

            if ( (i == lo && west_wall_bcs != 0) || (i == hi && east_wall_bcs != 0) ){
                grad_p_x(i, j, k) = Real(0.0);
                grad_phi_x(i, j, k) = Real(0.0);
            }
        });

        lo = dom.smallEnd(1);
        hi = dom.bigEnd(1)+1;

        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            grad_p_y(i, j, k) = ( ctx(i, j, k, 0) - ctx(i, j-1, k, 0) )/dx[1];

            grad_phi_y(i, j, k) = ( ctx(i, j, k, 1) - ctx(i, j-1, k, 1) )/dx[1];

            if ( (j == lo && south_wall_bcs != 0) || (j == hi && north_wall_bcs !=0) ){
                grad_p_y(i, j, k) = Real(0.0);
                grad_phi_y(i, j, k) = Real(0.0);
            }
        });

#if (AMREX_SPACEDIM > 2)
        lo = dom.smallEnd(2);
        hi = dom.bigEnd(2)+1;

        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            grad_p_z(i, j, k) = ( ctx(i, j, k, 0) - ctx(i, j, k-1, 0) )/dx[2];

            grad_phi_z(i, j, k) = ( ctx(i, j, k, 1) - ctx(i, j, k-1, 1) )/dx[2];

            if ( (k == lo && fron_wall_bcs != 0) || (k == hi && back_wall_bcs != 0) ){
                grad_p_z(i, j, k) = Real(0.0);
                grad_phi_z(i, j, k) = Real(0.0);
            }
        });
#endif
    }
}

// +++++++++++++++++++++++++ Total Flux  +++++++++++++++++++++++++
void total_flux_calc ( MultiFab& fluxTotal,
                       MultiFab& fluxConvect,
                       MultiFab& fluxViscous,
                       MultiFab& fluxPrsGrad,
                       Array<MultiFab, AMREX_SPACEDIM>& rhs,
                       Array<MultiFab, AMREX_SPACEDIM>& array_grad_p,
                       Vector<int> const& phy_bc_lo,
                       Vector<int> const& phy_bc_hi,
                       const Geometry& geom )
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxTotal); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& conv_flux = fluxConvect.array(mfi);
        auto const& visc_flux = fluxViscous.array(mfi);
        auto const& prsgrad_flux = fluxPrsGrad.array(mfi);
        auto const& total_flux = fluxTotal.array(mfi);

        // Only components inside physical domain
        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            compute_total_flux(i, j, k, total_flux, conv_flux, visc_flux, prsgrad_flux);
        });
    }

    fluxTotal.FillBoundary(geom.periodicity());

    Box dom(geom.Domain());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(rhs[0]); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif

        auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
        auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

        auto const& south_wall_bcs = phy_bc_lo[1]; // south wall
        auto const& north_wall_bcs = phy_bc_hi[1]; // north wall
#if (AMREX_SPACEDIM > 2)
        auto const& fron_wall_bcs = phy_bc_lo[2]; // front wall
        auto const& back_wall_bcs = phy_bc_hi[2]; // back wall
#endif

        auto const& xrhs = rhs[0].array(mfi);
        auto const& yrhs = rhs[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zrhs = rhs[2].array(mfi);
#endif

        auto const& grad_p_x = array_grad_p[0].array(mfi);
        auto const& grad_p_y = array_grad_p[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& grad_p_z = array_grad_p[2].array(mfi);
#endif
        auto const& total_flux = fluxTotal.array(mfi);

        int lo = dom.smallEnd(0);
        int hi = dom.bigEnd(0)+1;
        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            xrhs(i, j, k) = amrex::Real(0.5)*( total_flux(i-1, j, k, 0) + total_flux(i, j, k, 0) ) - grad_p_x(i, j, k);
            if ( (i == lo && west_wall_bcs != 0) || (i == hi && east_wall_bcs != 0) ){
                xrhs(i, j, k) = amrex::Real(0.0);
            }
        });

        lo = dom.smallEnd(1);
        hi = dom.bigEnd(1)+1;
        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            yrhs(i, j, k) = amrex::Real(0.5)*( total_flux(i, j-1, k, 1) + total_flux(i, j, k, 1) ) - grad_p_y(i, j, k);
            if ( (j == lo && south_wall_bcs != 0) || (j == hi && north_wall_bcs != 0) ){
                yrhs(i, j, k) = amrex::Real(0.0);
            }
        });

#if (AMREX_SPACEDIM > 2)
        lo = dom.smallEnd(2);
        hi = dom.bigEnd(2)+1;
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            zrhs(i, j, k) = amrex::Real(0.5)*( total_flux(i, j, k-1, 2) + total_flux(i, j, k, 2) ) - grad_p_z(i, j, k);
            if ( (k == lo && fron_wall_bcs != 0) || (k == hi && back_wall_bcs != 0) ){
                zrhs(i, j, k) = amrex::Real(0.0);
            }
        });
#endif
    }
}
