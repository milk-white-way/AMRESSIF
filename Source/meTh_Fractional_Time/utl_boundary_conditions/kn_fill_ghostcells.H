#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void fill_non_slip_ghost_cells (int i, int j, int k,
                   amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( i<0 ) // west wall
    {
        for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
        {
            vcart(i, j, k, dir) = - vcart(-i-1, j, k, dir);
        }
    }

    if ( i>(n_cell-1) ) // east wall
    {
        for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
        {
            vcart(i, j, k, dir) = - vcart(-i+1, j, k, dir);
        }
    }

    if ( j<0 ) // south wall
    {
        for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
        {
            vcart(i, j, k, dir) = - vcart(i, -j-1, k, dir);
        }
    }

    if ( j>(n_cell-1) ) // north wall
    {
        for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
        {
            vcart(i, j, k, dir) = - vcart(i, -j+1, k, dir);
        }
    }
#if (AMREX_SPACEDIM > 2)
    if ( k<0 ) // up wall
    {
        for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
        {
            vcart(i, j, k, dir) = - vcart(i, j, -k-1, dir);
        }
    }

    if ( k>(n_cell-1) ) // bottom wall
    {
        for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
        {
            vcart(i, j, k, dir) = - vcart(i, j, -k+1, dir);
        }
    }
#endif
}

void fill_slip_ghost_cells (int i, int j, int k,
                   amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( i<0 ) // west wall
    {
        // Modified for >= 2 layer of ghost cell
        vcart(i, j, k, 0) = - vcart(-i-1, j, k, 0);
        vcart(i, j, k, 1) = vcart(-i-1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
        vcart(i, j, k, 2) = vcart(-i-1, j, k, 2);
#endif
    }

    if ( i>(n_cell-1) ) // east wall
    {
        vcart(i, j, k, 0) = - vcart(-i+1, j, k, 0);
        vcart(i, j, k, 1) = vcart(-i+1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
        vcart(i, j, k, 2) = vcart(-i+1, j, k, 2);
#endif
    }

    if ( j<0 ) // south wall
    {
        vcart(i, j, k, 0) = vcart(i, -j-1, k, 0);
        vcart(i, j, k, 1) = - vcart(i, -j-1, k, 1);
#if (AMREX_SPACEDIM > 2)
        vcart(i, j, k, 2) = vcart(i, -j-1, k, 2);
#endif
    }

    if ( j>(n_cell-1) ) // north wall
    {
        vcart(i, j, k, 0) = vcart(i, -j+1, k, 0);
        vcart(i, j, k, 1) = - vcart(i, -j+1, k, 1);
#if (AMREX_SPACEDIM > 2)
        vcart(i, j, k, 2) = vcart(i, -j+1, k, 2);
#endif
    }

#if (AMREX_SPACEDIM > 2)
    if ( k<0 ) // up wall
    {
        vcart(i, j, k, 0) = vcart(i, j, -k-1, 0);
        vcart(i, j, k, 1) = vcart(i, j, -k-1, 1);
        vcart(i, j, k, 2) = - vcart(i, j, -k-1, 2);
    }

    if ( j>(n_cell-1) ) // north wall
    {
        vcart(i, j, k, 0) = vcart(i, j, -k+1, 0);
        vcart(i, j, k, 1) = vcart(i, j, -k+1, 1);
        vcart(i, j, k, 2) = - vcart(i, j, -k+1, 2);
#endif
    }
}
