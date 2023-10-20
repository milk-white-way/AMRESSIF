#include <AMReX_MultiFabUtil.H>

#include "utilities.H"
#include "kn_conversion.H"

using namespace amrex;

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

// =========== UTILITY | CONVERSION  =====================
void cont2cart (MultiFab& velCart,
                Array<MultiFab, AMREX_SPACEDIM>& velCont,
                const Geometry& geom)
{
    average_face_to_cellcenter(velCart, amrex::GetArrOfConstPtrs(velCont), geom);
}

void write_midline_solution (Real const& midx,
                             Real const& v_analytic,
                             double const& velx,
                             double const& vely,
                             int const& timestep)
{
    // Construct the filename for this iteration
    std::string filename = "midline_" + std::to_string(timestep) + ".txt";

    // Open a file for writing
    std::ofstream outfile(filename, std::ios::app);

    // Check if the file was opened successfully
    if (!outfile.is_open())
    {
        std::cerr << "Failed to open file for writing\n";
    }

    // Write data to the file
    outfile << midx << ";" << velx << ";" << vely << ";" << v_analytic << "\n";

    // Close the file
    outfile.close();
}

void line_extract (MultiFab& velCart,
                   int const& n_cell,
                   int const& timestep)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    // auto const& midy = n_cell/2;
    for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vcart = velCart.array(mfi);
        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            if (j == n_cell / 2) {
                amrex::Real const& midx = 0.03125*i;
                amrex::Real const& v_analytic = -std::sin(2.0 * M_PI * midx)*std::exp(-2*timestep*1e-4);
                write_midline_solution(midx, v_analytic, vcart(i, j, k, 0), vcart(i, j, k, 1), timestep);
            }

        });
    }
}
