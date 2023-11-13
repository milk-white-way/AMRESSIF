#include <AMReX_MultiFabUtil.H>

#include "utilities.H"
#include "kn_conversion.H"

using namespace amrex;

// ===================== UTILITY | CONVERSION  =====================
void cart2cont (MultiFab& velCart,
                Array<MultiFab, AMREX_SPACEDIM>& velCont)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
    {

        Box xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        Box ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
        auto const& xcont = velCont[0].array(mfi);
        auto const& ycont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        Box zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
        auto const& zcont = velCont[2].array(mfi);
#endif
        auto const& vcart = velCart.array(mfi);

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){ cart2cont_x(i, j, k, xcont, vcart); });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){ cart2cont_y(i, j, k, ycont, vcart); });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k){ cart2cont_z(i, j, k, zcont, vcart); });
#endif
    }
}

void cont2cart (MultiFab& velCart,
                Array<MultiFab, AMREX_SPACEDIM>& velCont,
                const Geometry& geom)
{
    average_face_to_cellcenter(velCart, amrex::GetArrOfConstPtrs(velCont), geom);
}

// ===================== UTILITY | EXTRACT LINE SOLUTION  =====================
void write_midline_solution (Real const& midx,
                             Real const& midy,
                             Real const& mdlu,
                             Real const& mdlv,
                             Real const& mdlp,
                             double const& anau,
                             double const& anav,
                             double const& anap,
                             int const& current_step)
{
    // Construct the filename for this iteration
    std::string filename = "midline_" + std::to_string(current_step) + ".txt";

    // Open a file for writing
    std::ofstream outfile(filename, std::ios::app);

    // Check if the file was opened successfully
    if (!outfile.is_open())
    {
        std::cerr << "Failed to open file for writing\n";
    }

    // Write data to the file
    outfile << midx << " " << midy << " " << mdlu << " " << mdlv << " " << mdlp << " " << anau << " " << anav << " " << anap << "\n";

    // Close the file
    outfile.close();
}

// ===================== UTILITY | ERROR NORM  =====================
amrex::Real Error_Computation (Array<MultiFab, AMREX_SPACEDIM>& velHat,
                               Array<MultiFab, AMREX_SPACEDIM>& velStar,
                               Array<MultiFab, AMREX_SPACEDIM>& velStarDiff,
                               Geometry const& geom)
{
    amrex::Real normError;

    for ( MFIter mfi(velStarDiff[0]); mfi.isValid(); ++mfi )
    {

        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        auto const& xnext = velHat[0].array(mfi);
        auto const& ynext = velHat[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& znext = velHat[2].array(mfi);
#endif
        auto const& xprev = velStar[0].array(mfi);
        auto const& yprev = velStar[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zprev = velStar[2].array(mfi);
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

    Real xerror = velStarDiff[0].norm2(0, geom.periodicity());
    Real yerror = velStarDiff[1].norm2(0, geom.periodicity());
    normError = std::max(xerror, yerror);
#if (AMREX_SPACEDIM > 2)
    Real zerror = velStarDiff[2].norminf(0, geom.periodicity());
    // Real zerror = velImDiff[2].norminf(0, 0);
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

void analytic_solution_calc (MultiFab& analyticSol,
                             Geometry const& geom,
                             Real const& time)
{
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(analyticSol); mfi.isValid(); ++mfi) {
        const Box &vbx = mfi.validbox();
        auto const &analytic = analyticSol.array(mfi);
        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
          // Real coordinates of the cell center
          Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
          Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];

          // u velocity
          analytic(i, j, k, 0) = std::sin(Real(2.0) * M_PI * x) * std::cos(Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);
          // v velocity
          analytic(i, j, k, 1) = -std::cos(Real(2.0) * M_PI * x) * std::sin(Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);
          // pressure
          analytic(i, j, k, 2) = -Real(0.25) * (std::cos(Real(4.0) * M_PI * x) + std::cos(Real(4.0) * M_PI * y)) * std::exp(-Real(16.0) * M_PI * M_PI * time);
        });
    }
}
