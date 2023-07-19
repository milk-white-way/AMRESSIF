#include <AMReX_MultiFabUtil.H>

#include "myfunc.H"
#include "mykernel.H"

using namespace amrex;

void fill_physical_ghost_cells (MultiFab& velCart,
                                int const& Nghost,
                                int const& n_cell,
                                Vector<int> const& bc_lo,
                                Vector<int> const& bc_hi)
{
   for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
   {
       const Box& gbx = mfi.growntilebox(Nghost);
       auto const& vcart = velCart.array(mfi);

       // bc = 0 means periodic boundary condition
       // bc = -1 means non-slip wall
       // bc = 1 means slip wall
       // Is bc_lo[0] west wall?
       // Is bc_hi[0] east wall?
       // Is bc_lo[1] south wall?
       // Is bc_hi[1] north wall?

       if ( bc_lo[0] != 0 )
       {
           if ( bc_lo[0] == -1 )
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( i<0 ) // west wall
                   {
                       for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
                       {
                           vcart(i, j, k, dir) = - vcart(-i-1, j, k, dir);
                       }
                   }
               });
           }
           else if (bc_lo[0] == 1)
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( i<0 )
                   {
                       vcart(i, j, k, 0) = - vcart(-i-1, j, k, 0);
                       vcart(i, j, k, 1) = vcart(-i-1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
                       vcart(i, j, k, 2) = vcart(-i-1, j, k, 2);
#endif
                   }
               });
           }
       }

       if ( bc_hi[0] != 0 )
       {
           if ( bc_hi[0] == -1 )
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( i>(n_cell-1) ) // east wall
                   {
                       for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
                       {
                           vcart(i, j, k, dir) = - vcart(-i+1, j, k, dir);
                       }
                   }
               });
           }
           else if (bc_hi[0] == 1)
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( i>(n_cell-1) ) // east wall
                   {
                       vcart(i, j, k, 0) = - vcart(-i+1, j, k, 0);
                       vcart(i, j, k, 1) = vcart(-i+1, j, k, 1);
#if (AMREX_SPACEDIM > 2)
                       vcart(i, j, k, 2) = vcart(-i+1, j, k, 2);
#endif
                   }
               });
           }
       }

       if ( bc_lo[1] != 0 )
       {
           if ( bc_lo[1] == -1 )
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( j<0 ) // south wall
                   {
                       for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
                       {
                           vcart(i, j, k, dir) = - vcart(i, -j-1, k, dir);
                       }
                   }
               });
           }
           else if (bc_lo[1] == 1)
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( j<0 ) // south wall
                   {
                       vcart(i, j, k, 0) = vcart(i, -j-1, k, 0);
                       vcart(i, j, k, 1) = - vcart(i, -j-1, k, 1);
#if (AMREX_SPACEDIM > 2)
                       vcart(i, j, k, 2) = vcart(i, -j-1, k, 2);
#endif
                   }
               });
           }
       }

       if ( bc_hi[1] != 0 )
       {
           if ( bc_hi[1] == -1 )
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( j>(n_cell-1) ) // north wall
                   {
                       for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
                       {
                           vcart(i, j, k, dir) = - vcart(i, -j+1, k, dir);
                       }
                   }
               });
           }
           else if (bc_hi[1] == 1)
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( j>(n_cell-1) ) // north wall
                   {
                       vcart(i, j, k, 0) = vcart(i, -j+1, k, 0);
                       vcart(i, j, k, 1) = - vcart(i, -j+1, k, 1);
#if (AMREX_SPACEDIM > 2)
                       vcart(i, j, k, 2) = vcart(i, -j+1, k, 2);
#endif
                   }
               });
           }
       }
#if (AMREX_SPACEDIM > 2)
       if ( bc_lo[2] != 0 )
       {
           if ( bc_lo[2] == -1 )
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( k<0 ) // up wall
                   {
                       for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
                       {
                           vcart(i, j, k, dir) = - vcart(i, j, -k-1, dir);
                       }
                   }
               });
           }
           else if (bc_lo[2] == 1)
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( k<0 ) // up wall
                   {
                       vcart(i, j, k, 0) = vcart(i, j, -k-1, 0);
                       vcart(i, j, k, 1) = vcart(i, j, -k-1, 1);
                       vcart(i, j, k, 2) = - vcart(i, j, -k-1, 2);
                   }
               });
           }
       }

       if ( bc_hi[2] != 0 )
       {
           if ( bc_hi[2] == -1 )
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( k>(n_cell-1) ) // bottom wall
                   {
                       for (int dir=0; dir < AMREX_SPACEDIM; ++dir)
                       {
                           vcart(i, j, k, dir) = - vcart(i, j, -k+1, dir);
                       }
                   }
               });
           }
           else if (bc_hi[2] == 1)
           {
               amrex::ParallelFor(gbx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k)
               {
                   if ( j>(n_cell-1) ) // north wall
                   {
                       vcart(i, j, k, 0) = vcart(i, j, -k+1, 0);
                       vcart(i, j, k, 1) = vcart(i, j, -k+1, 1);
                       vcart(i, j, k, 2) = - vcart(i, j, -k+1, 2);
                   }
               });
           }
       }
#endif
   }
}
