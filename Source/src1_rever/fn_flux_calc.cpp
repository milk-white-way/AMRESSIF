#include <AMReX_MultiFabUtil.H>

#include "fn_enforce_wall_bcs.H"
#include "fn_flux_calc.H"
#include "kn_flux_calc.H"
#include "kn_poisson.H"

using namespace amrex;
int const& UMIST = 0;
// ++++++++++++++++++++++++++++++ Convective Flux ++++++++++++++++++++++++++++++
void convective_flux_calc_new_quick ( MultiFab& fluxTotal,
                                      MultiFab& fluxConvect,
                                      Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                                      Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                                      Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN3,
                                      MultiFab& velCart,
                                      Array<MultiFab, AMREX_SPACEDIM>& velCont,
                                      Vector<int> const& phy_bc_lo,
                                      Vector<int> const& phy_bc_hi,
                                      Geometry const& geom)
{
    BL_PROFILE_VAR("convective_flux_calc_new_quick()", convective_flux_calc_new_quick);

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    Box dom(geom.Domain());

    //amrex::Print() << "DEBUGGING| Start Convective Flux: \n";
    BL_PROFILE_VAR("half_flux_x_face_calc()", half_flux_x_face_calc);
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

        auto const& vel_cart = velCart.array(mfi);

        auto const& ucont = velCont[0].array(mfi);
        auto const& vcont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& wcont = velCont[2].array(mfi);
#endif

        auto const& uhalf_xface = fluxHalfN1[0].array(mfi);
        auto const& vhalf_xface = fluxHalfN2[0].array(mfi);
        auto const& whalf_xface = fluxHalfN3[0].array(mfi);

        auto const& uhalf_yface = fluxHalfN1[1].array(mfi);
        auto const& vhalf_yface = fluxHalfN2[1].array(mfi);
        auto const& whalf_yface = fluxHalfN3[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& uhalf_zface = fluxHalfN1[2].array(mfi);
        auto const& vhalf_zface = fluxHalfN2[2].array(mfi);
        auto const& whalf_zface = fluxHalfN3[2].array(mfi);
#endif

        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // WEST FACE
            //if ( i == lo ) {
            auto const& ucon = Real(0.5) * ucont(i, j, k);
            auto const& ud = ucon + std::abs(ucon);
            auto const& up = ucon - std::abs(ucon);
            //amrex::Print() << "DEBUG| (i, j) = (" << i << ", " << j << ") | ucon = " << ucon << " ; ud = " << ud << " ; up = " << up << "\n";

            uhalf_xface(i, j, k) = up * ( amrex::Real(1.0/8.0) * ( - vel_cart(i+1, j, k, 0) - 2*vel_cart(i, j, k, 0) + 3*vel_cart(i-1, j, k, 0) ) + vel_cart(i, j, k, 0) ) + ud * ( amrex::Real(1.0/8.0) * ( - vel_cart(i-2, j, k, 0) - 2*vel_cart(i-1, j, k, 0) + 3*vel_cart(i, j, k, 0) ) + vel_cart(i-1, j, k, 0) );  

            vhalf_xface(i, j, k) = up * ( amrex::Real(1.0/8.0) * ( - vel_cart(i+1, j, k, 1) - 2*vel_cart(i, j, k, 1) + 3*vel_cart(i-1, j, k, 1) ) + vel_cart(i, j, k, 1) ) + ud * ( amrex::Real(1.0/8.0) * ( - vel_cart(i-2, j, k, 1) - 2*vel_cart(i-1, j, k, 1) + 3*vel_cart(i, j, k, 1) ) + vel_cart(i-1, j, k, 1) );

#if (AMREX_SPACEDIM > 2)
            whalf_xface(i, j, k) = up * ( amrex::Real(1.0/8.0) * ( - vel_cart(i+1, j, k, 2) - 2*vel_cart(i, j, k, 2) + 3*vel_cart(i-1, j, k, 2) ) + vel_cart(i, j, k, 2) ) + ud * ( amrex::Real(1.0/8.0) * ( - vel_cart(i-2, j, k, 2) - 2*vel_cart(i-1, j, k, 2) + 3*vel_cart(i, j, k, 2) ) + vel_cart(i-1, j, k, 2) );
#endif
            //}

            /*
            // EAST FACE
            auto const& ucon = Real(0.5) * ucont(i+1, j, k);
            auto const& ud = ucon + std::abs(ucon);
            auto const& up = ucon - std::abs(ucon);

            uhalf_xface(i+1, j, k) = up * ( amrex::Real(1.0/8.0) * ( - vel_cart(i+2, j, k, 0) - 2*vel_cart(i+1, j, k, 0) + 3*vel_cart(i, j, k, 0) ) + vel_cart(i+1, j, k, 0) ) + ud * ( amrex::Real(1.0/8.0) * ( - vel_cart(i-1, j, k, 0) - 2*vel_cart(i, j, k, 0) + 3*vel_cart(i+1, j, k, 0) ) + vel_cart(i, j, k, 0) );

            vhalf_xface(i+1, j, k) = up * ( amrex::Real(1.0/8.0) * ( - vel_cart(i+2, j, k, 1) - 2*vel_cart(i+1, j, k, 1) + 3*vel_cart(i, j, k, 1) ) + vel_cart(i+1, j, k, 1) ) + ud * ( amrex::Real(1.0/8.0) * ( - vel_cart(i-1, j, k, 1) - 2*vel_cart(i, j, k, 1) + 3*vel_cart(i+1, j, k, 1) ) + vel_cart(i, j, k, 1) );

#if (AMREX_SPACEDIM > 2)
            whalf_xface(i+1, j, k) = up * ( amrex::Real(1.0/8.0) * ( - vel_cart(i+2, j, k, 2) - 2*vel_cart(i+1, j, k, 2) + 3*vel_cart(i, j, k, 2) ) + vel_cart(i+1, j, k, 2) ) + ud * ( amrex::Real(1.0/8.0) * ( - vel_cart(i-1, j, k, 2) - 2*vel_cart(i, j, k, 2) + 3*vel_cart(i+1, j, k, 2) ) + vel_cart(i, j, k, 2) );
#endif
            }
            */
        });

        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            auto const& vcon = Real(0.5) * vcont(i, j, k);
            auto const& vd = vcon + std::abs(vcon);
            auto const& vp = vcon - std::abs(vcon);

            uhalf_yface(i, j, k) = vp * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j+1, k, 0) - 2*vel_cart(i, j, k, 0) + 3*vel_cart(i, j-1, k, 0) ) + vel_cart(i, j, k, 0) ) + vd * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j-2, k, 0) - 2*vel_cart(i, j-1, k, 0) + 3*vel_cart(i, j, k, 0) ) + vel_cart(i, j-1, k, 0) );

            vhalf_yface(i, j, k) = vp * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j+1, k, 1) - 2*vel_cart(i, j, k, 1) + 3*vel_cart(i, j-1, k, 1) ) + vel_cart(i, j, k, 1) ) + vd * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j-2, k, 1) - 2*vel_cart(i, j-1, k, 1) + 3*vel_cart(i, j, k, 1) ) + vel_cart(i, j-1, k, 1) );

#if (AMREX_SPACEDIM > 2)
            whalf_yface(i, j, k) = vp * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j+1, k, 2) - 2*vel_cart(i, j, k, 2) + 3*vel_cart(i, j-1, k, 2) ) + vel_cart(i, j, k, 2) ) + vd * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j-2, k, 2) - 2*vel_cart(i, j-1, k, 2) + 3*vel_cart(i, j, k, 2) ) + vel_cart(i, j-1, k, 2) );
#endif
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            auto const& wcon = Real(0.5) * wcont(i, j, k);
            auto const& wd = wcon + std::abs(wcon);
            auto const& wp = wcon - std::abs(wcon);

            uhalf_zface(i, j, k) = wp * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j, k+1, 0) - 2*vel_cart(i, j, k, 0) + 3*vel_cart(i, j, k-1, 0) ) + vel_cart(i, j, k, 0) ) + wd * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j, k-2, 0) - 2*vel_cart(i, j, k-1, 0) + 3*vel_cart(i, j, k, 0) ) + vel_cart(i, j, k-1, 0) );

            vhalf_zface(i, j, k) = wp * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j, k+1, 1) - 2*vel_cart(i, j, k, 1) + 3*vel_cart(i, j, k-1, 1) ) + vel_cart(i, j, k, 1) ) + wd * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j, k-2, 1) - 2*vel_cart(i, j, k-1, 1) + 3*vel_cart(i, j, k, 1) ) + vel_cart(i, j, k-1, 1) );

#if (AMREX_SPACEDIM > 2)
            whalf_zface(i, j, k) = wp * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j, k+1, 2) - 2*vel_cart(i, j, k, 2) + 3*vel_cart(i, j, k-1, 2) ) + vel_cart(i, j, k, 2) ) + wd * ( amrex::Real(1.0/8.0) * ( - vel_cart(i, j, k-2, 2) - 2*vel_cart(i, j, k-1, 2) + 3*vel_cart(i, j, k, 2) ) + vel_cart(i, j, k-1, 2) );
#endif
        });

#endif
    }
    BL_PROFILE_VAR_STOP(half_flux_x_face_calc);

    BL_PROFILE_VAR("half_flux_y_face_calc()", half_flux_y_face_calc);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxConvect); mfi.isValid(); ++mfi ) {
        const Box& vbx = mfi.validbox();
        auto const& conv_flux = fluxConvect.array(mfi);
        auto const& total_flux = fluxTotal.array(mfi);
        auto const& vel_cart = velCart.array(mfi);

        auto const& ucont = velCont[0].array(mfi);
        auto const& vcont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& wcont = velCont[2].array(mfi);
#endif

        auto const& uhalf_xface = fluxHalfN1[0].array(mfi);
        auto const& vhalf_xface = fluxHalfN2[0].array(mfi);
        auto const& whalf_xface = fluxHalfN3[0].array(mfi);

        auto const& uhalf_yface = fluxHalfN1[1].array(mfi);
        auto const& vhalf_yface = fluxHalfN2[1].array(mfi);
        auto const& whalf_yface = fluxHalfN3[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& uhalf_zface = fluxHalfN1[2].array(mfi);
        auto const& vhalf_zface = fluxHalfN2[2].array(mfi);
        auto const& whalf_zface = fluxHalfN3[2].array(mfi);
#endif

        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            conv_flux(i, j, k, 0) = ( uhalf_xface(i+1, j, k) - uhalf_xface(i, j, k) )/dx[0] + ( uhalf_yface(i, j+1, k) - uhalf_yface(i, j, k) )/dx[1] 
#if (AMREX_SPACEDIM > 2)
                + ( uhalf_zface(i, j, k+1) - uhalf_zface(i, j, k) )/dx[2];
#else
            ;
#endif

            conv_flux(i, j, k, 1) = ( vhalf_xface(i+1, j, k) - vhalf_xface(i, j, k) )/dx[0] + ( vhalf_yface(i, j+1, k) - vhalf_yface(i, j, k) )/dx[1]
#if (AMREX_SPACEDIM > 2)
                + ( vhalf_zface(i, j, k+1) - vhalf_zface(i, j, k) )/dx[2];
#else
            ;
#endif

#if (AMREX_SPACEDIM > 2)
            conv_flux(i, j, k, 2) = ( whalf_xface(i+1, j, k) - whalf_xface(i, j, k) )/dx[0] + ( whalf_yface(i, j+1, k) - whalf_yface(i, j, k) )/dx[1] + ( whalf_zface(i, j, k+1) - whalf_zface(i, j, k) )/dx[2];
#endif
        });
    }
    BL_PROFILE_VAR_STOP(half_flux_y_face_calc);
    //amrex::Print() << "DEBUGGING| End Convective Flux: \n";

    BL_PROFILE_VAR("convective_update_total_flux()", convective_update_total_flux);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(fluxTotal); mfi.isValid(); ++mfi ) {
        const Box& vbx = mfi.validbox();
        auto const& total_flux = fluxTotal.array(mfi);
        auto const& conv_flux = fluxConvect.array(mfi);

        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                total_flux(i, j, k, dir) = total_flux(i, j, k, dir) - conv_flux(i, j, k, dir);
            }
        });
    }
    BL_PROFILE_VAR_STOP(convective_update_total_flux);
}

void convective_flux_calc_quick ( MultiFab& fluxTotal,
                                  MultiFab& fluxConvect,
                                  Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN1,
                                  Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN2,
                                  Array<MultiFab, AMREX_SPACEDIM>& fluxHalfN3,
                                  MultiFab& velCart,
                                  Array<MultiFab, AMREX_SPACEDIM>& velCont,
                                  Vector<int> const& phy_bc_lo,
                                  Vector<int> const& phy_bc_hi,
                                  Geometry const& geom)
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    Box dom(geom.Domain());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& conv_flux = fluxConvect.array(mfi);
        auto const& total_flux = fluxTotal.array(mfi);
        auto const& vel_cart = velCart.array(mfi);

        auto const& ucont = velCont[0].array(mfi);
        auto const& vcont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& wcont = velCont[2].array(mfi);
#endif

        auto const& ucart_xface = fluxHalfN1[0].array(mfi);
        auto const& vcart_xface = fluxHalfN2[0].array(mfi);
        auto const& wcart_xface = fluxHalfN3[0].array(mfi);

        auto const& ucart_yface = fluxHalfN1[1].array(mfi);
        auto const& vcart_yface = fluxHalfN2[1].array(mfi);
        auto const& wcart_yface = fluxHalfN3[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& ucart_zface = fluxHalfN1[2].array(mfi);
        auto const& vcart_zface = fluxHalfN2[2].array(mfi);
        auto const& wcart_zface = fluxHalfN3[2].array(mfi);
#endif

        // Half-note velocity for QUICK scheme
        // using west face for consistency in indexing
        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            int hi = dom.bigEnd(0);
            if ( ucont(i, j, k) > 0 ) {
                ucart_xface(i, j, k) = vel_cart(i-1, j, k, 0) + Real(1/8)*( - vel_cart(i-2, j, k, 0) - 2*vel_cart(i-1, j, k, 0) + 3*vel_cart(i, j, k, 0) );

                vcart_xface(i, j, k) = vel_cart(i-1, j, k, 1) + Real(1/8)*( - vel_cart(i-2, j, k, 1) - 2*vel_cart(i-1, j, k, 1) + 3*vel_cart(i, j, k, 1) );

#if (AMREX_SPACEDIM > 2)
                wcart_xface(i, j, k) = vel_cart(i-1, j, k, 2) + Real(1/8)*( - vel_cart(i-2, j, k, 2) - 2*vel_cart(i-1, j, k, 2) + 3*vel_cart(i, j, k, 2) );
#endif
                if ( i == hi ) {
                    ucart_xface(i+1, j, k) = ucart_xface(0, j, k);
                    vcart_xface(i+1, j, k) = vcart_xface(0, j, k);
#if (AMREX_SPACEDIM > 2)
                    wcart_xface(i+1, j, k) = wcart_xface(0, j, k);
#endif
                }
            } else {
                ucart_xface(i, j, k) = vel_cart(i, j, k, 0) + Real(1/8)*( 3*vel_cart(i-1, j, k, 0) - 2*vel_cart(i, j, k, 0) - vel_cart(i+1, j, k, 0) );

                vcart_xface(i, j, k) = vel_cart(i, j, k, 1) + Real(1/8)*( 3*vel_cart(i-1, j, k, 1) - 2*vel_cart(i, j, k, 1) - vel_cart(i+1, j, k, 1) );

#if (AMREX_SPACEDIM > 2)
                wcart_xface(i, j, k) = vel_cart(i, j, k, 2) + Real(1/8)*( 3*vel_cart(i-1, j, k, 2) - 2*vel_cart(i, j, k, 2) - vel_cart(i+1, j, k, 2) );
#endif

                if ( i == hi ) {
                    ucart_xface(i+1, j, k) = ucart_xface(0, j, k);
                    vcart_xface(i+1, j, k) = vcart_xface(0, j, k);
#if (AMREX_SPACEDIM > 2)
                    wcart_xface(i+1, j, k) = wcart_xface(0, j, k);
#endif
                }
            }

            hi = dom.bigEnd(1);
            if ( vcont(i, j, k) > 0 ) {
                ucart_yface(i, j, k) = vel_cart(i, j-1, k, 0) + Real(1/8)*( - vel_cart(i, j-2, k, 0) - 2*vel_cart(i, j-1, k, 0) + 3*vel_cart(i, j, k, 0) );

                vcart_yface(i, j, k) = vel_cart(i, j-1, k, 1) + Real(1/8)*( - vel_cart(i, j-2, k, 1) - 2*vel_cart(i, j-1, k, 1) + 3*vel_cart(i, j, k, 1) );

#if (AMREX_SPACEDIM > 2)
                wcart_yface(i, j, k) = vel_cart(i, j-1, k, 2) + Real(1/8)*( - vel_cart(i, j-2, k, 2) - 2*vel_cart(i, j-1, k, 2) + 3*vel_cart(i, j, k, 2) );
#endif

                if ( j == hi ) {
                    ucart_yface(i, j+1, k) = ucart_yface(i, 0, k);
                    vcart_yface(i, j+1, k) = vcart_yface(i, 0, k);
#if (AMREX_SPACEDIM > 2)
                    wcart_yface(i, j+1, k) = wcart_yface(i, 0, k);
#endif
                }
            } else {
                ucart_yface(i, j, k) = vel_cart(i, j, k, 0) + Real(1/8)*( 3*vel_cart(i, j-1, k, 0) - 2*vel_cart(i, j, k, 0) - vel_cart(i, j+1, k, 0) );

                vcart_yface(i, j, k) = vel_cart(i, j, k, 1) + Real(1/8)*( 3*vel_cart(i, j-1, k, 1) - 2*vel_cart(i, j, k, 1) - vel_cart(i, j+1, k, 1) );

#if (AMREX_SPACEDIM > 2)
                wcart_yface(i, j, k) = vel_cart(i, j, k, 2) + Real(1/8)*( 3*vel_cart(i, j-1, k, 2) - 2*vel_cart(i, j, k, 2) - vel_cart(i, j+1, k, 2) );
#endif

                if ( j == hi ) {
                    ucart_yface(i, j+1, k) = ucart_yface(i, 0, k);
                    vcart_yface(i, j+1, k) = vcart_yface(i, 0, k);
#if (AMREX_SPACEDIM > 2)
                    wcart_yface(i, j+1, k) = wcart_yface(i, 0, k);
#endif
                }
            }

#if (AMREX_SPACEDIM > 2)
            hi = dom.bigEnd(2);
            if ( wcont(i, j, k) > 0 ) {
                ucart_zface(i, j, k) = vel_cart(i, j, k-1, 0) + Real(1/8)*( - vel_cart(i, j, k-2, 0) - 2*vel_cart(i, j, k-1, 0) + 3*vel_cart(i, j, k, 0) );

                vcart_zface(i, j, k) = vel_cart(i, j, k-1, 1) + Real(1/8)*( - vel_cart(i, j, k-2, 1) - 2*vel_cart(i, j, k-1, 1) + 3*vel_cart(i, j, k, 1) );

                wcart_zface(i, j, k) = vel_cart(i, j, k-1, 2) + Real(1/8)*( - vel_cart(i, j, k-2, 2) - 2*vel_cart(i, j, k-1, 2) + 3*vel_cart(i, j, k, 2) );

                if ( k == hi ) {
                    ucart_zface(i, j, k+1, k) = ucart_zface(i, j, 0, k);
                    vcart_zface(i, j, k+1, k) = vcart_zface(i, j, 0, k);
#if (AMREX_SPACEDIM > 2)
                    wcart_zface(i, j, k+1, k) = wcart_zface(i, j, 0, k);
#endif
                }
            } else {
                ucart_zface(i, j, k) = vel_cart(i, j, k, 0) + Real(1/8)*( 3*vel_cart(i, j, k-1, 0) - 2*vel_cart(i, j, k, 0) - vel_cart(i, j, k+1, 0) );

                vcart_zface(i, j, k) = vel_cart(i, j, k, 1) + Real(1/8)*( 3*vel_cart(i, j, k-1, 1) - 2*vel_cart(i, j, k, 1) - vel_cart(i, j, k+1, 1) );

                wcart_zface(i, j, k) = vel_cart(i, j, k, 2) + Real(1/8)*( 3*vel_cart(i, j, k-1, 2) - 2*vel_cart(i, j, k, 2) - vel_cart(i, j, k+1, 2) );

                if ( k == hi ) {
                    ucart_zface(i, j, k+1, k) = ucart_zface(i, j, 0, k);
                    vcart_zface(i, j, k+1, k) = vcart_zface(i, j, 0, k);
#if (AMREX_SPACEDIM > 2)
                    wcart_zface(i, j, k+1, k) = wcart_zface(i, j, 0, k);
#endif
                }
            }
#endif
        });


        // Convective flux calculation
        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            //conv_flux(i, j, k, 0) = ( vel_cart(i, j, k, 0) * (ucart_xface(i+1, j, k) - ucart_xface(i, j, k)) )/(dx[0]) + ( vel_cart(i, j, k, 1) * (ucart_yface(i+1, j, k) - ucart_yface(i, j, k)) )/(dx[1])
            conv_flux(i, j, k, 0) = ( ucont(i+1, j, k) * ucart_xface(i+1, j, k) - ucont(i, j, k) * ucart_xface(i, j, k) )/(dx[0]) + ( vcont(i+1, j, k) * ucart_yface(i+1, j, k) - vcont(i, j, k) * ucart_yface(i, j, k) )/(dx[1])
#if (AMREX_SPACEDIM > 2)
                //+ ( vel_cart(i, j, k, 2) * (ucart_zface(i+1, j, k) - ucart_zface(i, j, k)) )/(dx[2]);
                + ( wcont(i, j, k+1) * ucart_zface(i, j, k+1) - wcont(i, j, k) * ucart_zface(i, j, k) )/(dx[2]);
#else
            ;
#endif

            //amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
            //amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
            //amrex::Print() << x << ";" << y << ";" << conv_flux(i, j, k, 0) << "\n";

            conv_flux(i, j, k, 1) = ( vel_cart(i, j, k, 0) * (vcart_xface(i+1, j, k) - vcart_xface(i, j, k)) )/(dx[0]) + ( vel_cart(i, j, k, 1) * (vcart_yface(i+1, j, k) - vcart_yface(i, j, k)) )/(dx[1])
#if (AMREX_SPACEDIM > 2)
                + ( vel_cart(i, j, k, 2) * (vcart_zface(i+1, j, k) - vcart_zface(i, j, k)) )/(dx[2]);
#else
            ;
#endif

#if (AMREX_SPACEDIM > 2)
            conv_flux(i, j, k, 2) = ( vel_cart(i, j, k, 0) * (wcart_xface(i+1, j, k) - wcart_xface(i, j, k)) )/(dx[0]) + ( vel_cart(i, j, k, 1) * (wcart_yface(i+1, j, k) - wcart_yface(i, j, k)) )/(dx[1])
                + ( vel_cart(i, j, k, 2) * (wcart_zface(i+1, j, k) - wcart_zface(i, j, k)) )/(dx[2]);
#endif

        });

        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                total_flux(i, j, k, dir) = total_flux(i, j, k, dir) - conv_flux(i, j, k, dir);
            }
        });
    }
}

void convective_flux_calc ( MultiFab& fluxTotal,
                            MultiFab& fluxConvect,
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

        auto const& vel_cart = velCart.array(mfi);

        auto const& ucont = velCont[0].array(mfi); 
        auto const& vcont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& wcont = velCont[2].array(mfi);
#endif

        auto const& ucart_xface = fluxHalfN1[0].array(mfi);
        auto const& vcart_xface = fluxHalfN2[0].array(mfi);
        auto const& wcart_xface = fluxHalfN3[0].array(mfi);

        auto const& ucart_yface = fluxHalfN1[1].array(mfi);
        auto const& vcart_yface = fluxHalfN2[1].array(mfi);
        auto const& wcart_yface = fluxHalfN3[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& ucart_zface = fluxHalfN1[2].array(mfi);
        auto const& vcart_zface = fluxHalfN2[2].array(mfi);
        auto const& wcart_zface = fluxHalfN3[2].array(mfi);
#endif

        auto const& west_wall_bcs = phy_bc_lo[0]; // west wall
        auto const& east_wall_bcs = phy_bc_hi[0]; // east wall

        auto const& south_wall_bcs = phy_bc_lo[1]; // south wall
        auto const& north_wall_bcs = phy_bc_hi[1]; // north wall
#if (AMREX_SPACEDIM > 2)
        auto const& fron_wall_bcs = phy_bc_lo[2]; // front wall
        auto const& back_wall_bcs = phy_bc_hi[2]; // back wall
#endif

        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            //amrex::Print() << "MOMENTUM | Calculating half-node flux at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
            // Note that the half-node flux (face-centered) are calculated in the east face
            // This leaves the first half-node flux on the west boundary to be prescribed by the boundary conditions
            // Step 1: Assign local cell-centre nodes
            auto const& vcart_W  = vel_cart(i-2, j, k, 1);
            auto const& vcart_P  = vel_cart(i-1, j, k, 1);
            auto const& vcart_E  = vel_cart(i  , j, k, 1);
            auto const& vcart_EE = vel_cart(i+1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
            auto const& wcart_W  = vel_cart(i-2, j, k, 2);
            auto const& wcart_P  = vel_cart(i-1, j, k, 2);
            auto const& wcart_E  = vel_cart(i  , j, k, 2);
            auto const& wcart_EE = vel_cart(i+1, j, k, 2);
#endif

            // Step 2: Detect direction of flow
            auto const& ucon = Real(0.5) * ucont(i, j, k);
            auto const& fldr = ( ucon - std::abs(ucon) ); // Print() << "fldr: " << fldr << "\n"; // DEBUGGING
            //  ucon - |ucon| = |-- 0 if ucon > 0, ==> ucont(i, j, k) >= 0 (flow to the right)
            //                  |-- 1 if ucon < 0, ==> ucont(i, j, k) <  0 (flow to the left)

            // Default that the flow is to the left
            Real vcart_UU = vcart_EE;
            Real vcart_U  = vcart_E;
            Real vcart_D  = vcart_P;
            if ( fldr == 0 ) {
                // Flow to the right
                vcart_UU = vcart_W;
                vcart_U  = vcart_P;
                vcart_D  = vcart_E;
            } 
#if (AMREX_SPACEDIM > 2)
            Real wcart_UU = wcart_EE;
            Real wcart_U  = wcart_E;
            Real wcart_D  = wcart_P;
            if ( fldr == 0 ) {
                // Flow to the right
                wcart_UU = wcart_W;
                wcart_U  = wcart_P;
                wcart_D  = wcart_E;
            } 
#endif
            
#if (UMIST == 1)
            Real psi = Real(2.0);
            Real flux_limited_ratio = psi;
#endif

            vcart_xface(i, j, k) = - vcart_UU/8 + 3*vcart_U/4 + 3*vcart_D/8;
#if (UMIST == 1)
            if ( vcart_D != vcart_U ) {
                flux_limited_ratio = ( vcart_U - vcart_UU ) / ( vcart_D - vcart_U );
                if ( flux_limited_ratio < 0) {
                    // non-monotonic 
                    vcart_xface(i, j, k) = vcart_U;
                } else {
                    // monotonic
                    psi = std::min(psi, 2*flux_limited_ratio);
                    psi = std::min(psi, (1 + 3*flux_limited_ratio)/4);
                    psi = std::min(psi, (3 + flux_limited_ratio)/4);
                    //Print() << "psi: " << psi << "\n";
                    vcart_xface(i, j, k) = vcart_U + Real(0.5)*psi*(vcart_D - vcart_U);
                }
                //Print() << "flux_limited_ratio: " << flux_limited_ratio << "\n";
            }
#endif
#if (AMREX_SPACEDIM > 2)
#if (UMIST == 1)
            wcart_xface(i, j, k) = - wcart_UU/8 + 3*wcart_U/4 + 3*wcart_D/8;
            if ( wcart_D != wcart_U ) {
                flux_limited_ratio = ( wcart_U - wcart_UU ) / ( wcart_D - wcart_U );
                if ( flux_limited_ratio < 0) {
                    // non-monotonic 
                    wcart_xface(i, j, k) = wcart_U;
                } else {
                    // monotonic
                    psi = std::min(psi, 2*flux_limited_ratio);
                    psi = std::min(psi, (1 + 3*flux_limited_ratio)/4);
                    psi = std::min(psi, (3 + flux_limited_ratio)/4);
                    //Print() << "psi: " << psi << "\n";
                    wcart_xface(i, j, k) = wcart_U + Real(0.5)*psi*(wcart_D - wcart_U);
                }
                //Print() << "flux_limited_ratio: " << flux_limited_ratio << "\n";
            }
#endif
#endif
        });

        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Note that the half-node flux (face-centered) are calculated in the north face
            // This leaves the first half-node flux on the south boundary to be prescribed by the boundary conditions
            auto const& ucart_S  = vel_cart(i, j-2, k, 0);
            auto const& ucart_P  = vel_cart(i, j-1, k, 0);
            auto const& ucart_N  = vel_cart(i, j  , k, 0);
            auto const& ucart_NN = vel_cart(i, j+1, k, 0);
#if (AMREX_SPACEDIM > 2)
            auto const& wcart_S  = vel_cart(i, j-2, k, 2);
            auto const& wcart_P  = vel_cart(i, j-1, k, 2);
            auto const& wcart_N  = vel_cart(i, j  , k, 2);
            auto const& wcart_NN = vel_cart(i, j+1, k, 2);
#endif

            // Step 2: Detect direction of flow
            auto const& vcon = Real(0.5) * vcont(i, j, k);
            auto const& fldr = ( vcon - std::abs(vcon) );
            //  vcon - |vcon| = |-- 0 if ucon >= 0, ==> vcont(i, j, k) > 0 (flow to the top)
            //                  |-- 1 if ucon <  0, ==> vcont(i, j, k) < 0 (flow to the bottom)

            // Default that the flow is to the bottom
            Real ucart_UU = ucart_NN;
            Real ucart_U  = ucart_N;
            Real ucart_D  = ucart_P;
           if ( fldr == 0 ) {
                // Flow to the top
                ucart_UU = ucart_S;
                ucart_U  = ucart_P;
                ucart_D  = ucart_N;
            }
#if (AMREX_SPACEDIM > 2)
            Real wcart_UU = wcart_NN;
            Real wcart_U  = wcart_N;
            Real wcart_D  = wcart_P;
           if ( fldr == 0 ) {
                // Flow to the top
                wcart_UU = wcart_S;
                wcart_U  = wcart_P;
                wcart_D  = wcart_N;
            }
#endif

#if (UMIST == 1)
            Real psi = Real(2.0);
            Real flux_limited_ratio = psi;
#endif

            ucart_yface(i, j, k) = - ucart_UU/8 + 3*ucart_U/4 + 3*ucart_D/8;
#if (UMIST == 1)
            if ( ucart_D != ucart_U ) {
                flux_limited_ratio = ( ucart_U - ucart_UU ) / ( ucart_D - ucart_U );
                if ( flux_limited_ratio < 0) {
                    // non-monotonic 
                    ucart_yface(i, j, k) = ucart_U;
                } else {
                    // monotonic
                    psi = std::min(psi, Real(2.0)*flux_limited_ratio);
                    psi = std::min(psi, (1 + 3*flux_limited_ratio)/4);
                    psi = std::min(psi, (3 + flux_limited_ratio)/4);
                    //Print() << "psi: " << psi << "\n";
                    ucart_yface(i, j, k) = ucart_U + Real(0.5)*psi*(ucart_D - ucart_U);
                }
                //Print() << "flux_limited_ratio: " << flux_limited_ratio << "\n";
            }
#endif
            
#if (AMREX_SPACEDIM > 2)
            wcart_yface(i, j, k) = - wcart_UU/8 + 3*wcart_U/4 + 3*wcart_D/8;
#if (UMIST == 1)
            if ( wcart_D != wcart_U ) {
                flux_limited_ratio = ( wcart_U - wcart_UU ) / ( wcart_D - wcart_U );
                if ( flux_limited_ratio < 0) {
                    // non-monotonic 
                    wcart_yface(i, j, k) = wcart_U;
                } else {
                    // monotonic
                    psi = std::min(psi, 2*flux_limited_ratio);
                    psi = std::min(psi, (1 + 3*flux_limited_ratio)/4);
                    psi = std::min(psi, (3 + flux_limited_ratio)/4);
                    //Print() << "psi: " << psi << "\n";
                    wcart_yface(i, j, k) = vcart_U + Real(0.5)*psi*(vcart_D - vcart_U);
                }
                //Print() << "flux_limited_ratio: " << flux_limited_ratio << "\n";
            }
#endif
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
        auto const& total_flux = fluxTotal.array(mfi);
        auto const& vel_cart = velCart.array(mfi);

        auto const& ucont = velCont[0].array(mfi);
        auto const& vcont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& wcont = velCont[2].array(mfi);
#endif

        auto const& ucart_xface = fluxHalfN1[0].array(mfi);
        auto const& vcart_xface = fluxHalfN2[0].array(mfi);
        auto const& wcart_xface = fluxHalfN3[0].array(mfi);

        auto const& ucart_yface = fluxHalfN1[1].array(mfi);
        auto const& vcart_yface = fluxHalfN2[1].array(mfi);
        auto const& wcart_yface = fluxHalfN3[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& ucart_zface = fluxHalfN1[2].array(mfi);
        auto const& vcart_zface = fluxHalfN2[2].array(mfi);
        auto const& wcart_zface = fluxHalfN3[2].array(mfi);
#endif

        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            conv_flux(i, j, k, 0) = ( ucont(i+1, j, k) * ucont(i+1, j, k) - ucont(i, j, k) * ucont(i, j, k) )/( dx[0] ) + ( vcont(i, j+1, k) * ucart_yface(i, j+1, k) - vcont(i, j, k) * ucart_yface(i, j, k) )/( dx[1] )
            //conv_flux(i, j, k, 0) = ( vcont(i, j+1, k) * ucart_yface(i, j+1, k) - vcont(i, j, k) * ucart_yface(i, j, k) )/( dx[1] )
#if (AMREX_SPACEDIM > 2)
                + ( wcont(i, j, k+1) * ucart_zface(i, j, k+1) - wcont(i, j, k) * ucart_zface(i, j, k+1) )/( dx[2] );
#else
            ;
#endif
            

            conv_flux(i, j, k, 1) = ( vcont(i, j+1, k) * vcont(i, j+1, k) - vcont(i, j, k) * vcont(i, j, k) )/( dx[1] ) + ( ucont(i+1, j, k) * vcart_xface(i+1, j, k) - ucont(i, j, k) * vcart_xface(i, j, k) )/( dx[0] )
            //conv_flux(i, j, k, 1) = ( ucont(i+1, j, k) * vcart_xface(i+1, j, k) - ucont(i, j, k) * vcart_xface(i, j, k) )/( dx[0] )
#if (AMREX_SPACEDIM > 2)
                + ( wcont(i, j, k+1) * vcart_zface(i, j, k+1) - wcont(i, j, k) * vcart_zface(i, j, k) )/( dx[2] );
#else
            ;
#endif

#if (AMREX_SPACEDIM > 2)
            conv_flux(i, j, k, 2) = ( ucont(i+1, j, k) * wcart_xface(i+1, j, k) - ucont(i, j, k) * wcart_xface(i, j, k) )/( dx[0] ) + ( vcont(i, j+1, k) * wcart_yface(i, j+1, k) - vcont(i, j, k) * wcart_yface(i, j, k) )/( dx[1] ) + ( wcont(i, j, k+1) * wcont(i, j, k+1) - wcont(i, j, k) * wcont(i, j, k) )/( dx[2] );
#endif
        });
    }
}

// ++++++++++++++++++++++++++++++ Viscous Flux ++++++++++++++++++++++++++++++
void viscous_flux_calc ( MultiFab& fluxTotal,
                         MultiFab& fluxViscous,
                         MultiFab& velCart,
                         Real const& ren,
                         Geometry const& geom)
{
    BL_PROFILE_VAR("viscous_flux_calc()", viscous_flux_calc);

    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    //amrex::Print() << "DEBUGGING| Start Viscous Flux: \n";
    for ( MFIter mfi(fluxViscous); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vel_cart = velCart.array(mfi);
        auto const& visc_flux = fluxViscous.array(mfi);
        auto const& total_flux = fluxTotal.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            for ( int dir=0; dir < AMREX_SPACEDIM; ++dir )
            {
                auto const& centerMAC = vel_cart(i  , j  , k, dir);
                auto const& southMAC  = vel_cart(i  , j-1, k, dir);
                auto const& northMAC  = vel_cart(i  , j+1, k, dir);
                auto const& westMAC   = vel_cart(i-1, j  , k, dir);
                auto const& eastMAC   = vel_cart(i+1, j  , k, dir);
#if (AMREX_SPACEDIM > 2)
                auto const& frontMAC  = vel_cart(i  , j  , k-1, dir);
                auto const& backMAC   = vel_cart(i  , j  , k+1, dir);
#endif

                visc_flux(i, j, k, dir) = ( (eastMAC - 2*centerMAC + westMAC)/(dx[0]*dx[0]) + (northMAC - 2*centerMAC + southMAC)/(dx[1]*dx[1]) )/ren; 
#if (AMREX_SPACEDIM > 2)
                    + ( (backMAC - 2*centerMAC + frontMAC)/(dx[2]*dx[2]) )/ren;
#else
                    ;
#endif
                /*
                if (dir == 0) {
			        amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
			        amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];

                    amrex::Print() << x << ";" << y << ";" << visc_flux(i, j, k, dir) << "\n";
                }
                */
            }
        });
    }
    //amrex::Print() << "DEBUGGING| End Viscous Flux: \n";

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
	for (MFIter mfi(fluxTotal); mfi.isValid(); ++mfi)
	{
		const Box& vbx = mfi.validbox();
		auto const& total_flux = fluxTotal.array(mfi);
		auto const& visc_flux = fluxViscous.array(mfi);

		amrex::ParallelFor(vbx,
						   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
			for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
				total_flux(i, j, k, dir) = total_flux(i, j, k, dir) + visc_flux(i, j, k, dir);
			}
		});
	}
}

// +++++++++++++++++++++++++ Gradient Flux  +++++++++++++++++++++++++
void gradient_calc_approach1 ( MultiFab& fluxTotal,
                               MultiFab& fluxPrsGrad,
                               MultiFab& cc_grad_phi,
                               MultiFab& userCtx,
                               Geometry const& geom,
                               int const& Nghost,
                               Vector<int> const& phy_bc_lo,
                               Vector<int> const& phy_bc_hi,
                               int const& n_cell )
{
    BL_PROFILE_VAR("gradient_calc_approach1()", gradient_calc_approach1);

    //enforce_wall_bcs_for_cell_centered_userCtx_on_ghost_cells(userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    //amrex::Print() << "DEBUGGING| Pressure Gradient: \n";
    for ( MFIter mfi(fluxPrsGrad); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx = userCtx.array(mfi);
        auto const& pres_grad = fluxPrsGrad.array(mfi);
        auto const& phi_grad = cc_grad_phi.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // PERIODIC ONLY
            compute_pressure_gradient_periodic(i, j, k, pres_grad, dx, ctx);
            compute_phi_gradient_periodic(i, j, k, phi_grad, dx, ctx);
        });
    }
    
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
	for (MFIter mfi(fluxTotal); mfi.isValid(); ++mfi)
	{
		const Box& vbx = mfi.validbox();
		auto const& total_flux = fluxTotal.array(mfi);
		auto const& prs_grad = fluxPrsGrad.array(mfi);

		amrex::ParallelFor(vbx,
						   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
			for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
				total_flux(i, j, k, dir) = - prs_grad(i, j, k, dir);
			}
		});
	}

    //enforce_wall_bcs_for_cell_centered_flux_on_ghost_cells(cc_grad_phi, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

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
    BL_PROFILE_VAR("gradient_calc_approach2()", gradient_calc_approach2);

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
		//amrex::Print() << "West wall bc = " << west_wall_bcs << "\n";
		//amrex::Print() << "East wall bc = " << east_wall_bcs << "\n";
		//amrex::Print() << "South wall bc = " << south_wall_bcs << "\n";
		//amrex::Print() << "North wall bc = " << north_wall_bcs << "\n";

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
		});

		lo = dom.smallEnd(1);
		hi = dom.bigEnd(1)+1;

		amrex::ParallelFor(ybx,
								 [=] AMREX_GPU_DEVICE (int i, int j, int k){
			grad_p_y(i, j, k) = ( ctx(i, j, k, 0) - ctx(i, j-1, k, 0) )/dx[1];

			grad_phi_y(i, j, k) = ( ctx(i, j, k, 1) - ctx(i, j-1, k, 1) )/dx[1];
		});

#if (AMREX_SPACEDIM > 2)
		lo = dom.smallEnd(2);
		hi = dom.bigEnd(2)+1;

		amrex::ParallelFor(zbx,
								 [=] AMREX_GPU_DEVICE (int i, int j, int k){
			grad_p_z(i, j, k) = ( ctx(i, j, k, 0) - ctx(i, j, k-1, 0) )/dx[2];

			grad_phi_z(i, j, k) = ( ctx(i, j, k, 1) - ctx(i, j, k-1, 1) )/dx[2];
		});
#endif
	}
}
