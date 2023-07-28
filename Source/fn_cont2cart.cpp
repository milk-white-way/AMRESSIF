#include <AMReX_MultiFabUtil.H>

#include "fn_cont2cart.H"

using namespace amrex;

// ==================================== UTILITY | CONVERSION  ================================
void cont2cart (MultiFab& velCart,
                Array<MultiFab, AMREX_SPACEDIM>& velCont,
                const Geometry& geom)
{
    average_face_to_cellcenter(velCart, amrex::GetArrOfConstPtrs(velCont), geom);
}
