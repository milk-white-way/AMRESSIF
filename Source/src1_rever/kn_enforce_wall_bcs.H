#ifndef KERNEL_ENFORCE_BCS_H_
#define KERNEL_ENFORCE_BCS_H_

#include <AMReX_FArrayBox.H>

// ============================== UTILITY | BOUNDARY CONDITIONS ==============================
// +++++++++++++++++++++++++ Type: Pressure +++++++++++++++++++++++++
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_pres_bcs_west (int i, int j, int k, int n_cell, int bcs,
                            amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( i < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 0) = Real(0.0); } else {
            ctx(i, j, k, 0) = - ctx(-i-1, j, k, 0);
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_pres_bcs_east (int i, int j, int k, int n_cell, int bcs,
                            amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( i > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 0) = Real(0.0); } else {
            ctx(i, j, k, 0) = - ctx ( ( (n_cell-i) + (n_cell-1) ), j, k, 0);
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_pres_bcs_south (int i, int j, int k, int n_cell, int bcs,
                             amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( j < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 0) = Real(0.0); } else {
            ctx(i, j, k, 0) = - ctx(i, -j-1, k, 0);
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_pres_bcs_north (int i, int j, int k, int n_cell, int bcs,
                             amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( j > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 0) = Real(0.0); } else {
            ctx(i, j, k, 0) = - ctx(i, ( (n_cell-j) + (n_cell-1) ), k, 0);
        }
    }
}
#if (AMREX_SPACEDIM > 2)

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_pres_bcs_front (int i, int j, int k, int n_cell, int bcs,
                             amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( k < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 0) = Real(0.0); } else {
            ctx(i, j, k, 0) = - ctx (i, j, -k-1, 0);
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_pres_bcs_back (int i, int j, int k, int n_cell, int bcs,
                            amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( k > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 0) = Real(0.0); } else {
            ctx(i, j, k, 0) = - ctx (i, j, ( (n_cell-k) + (n_cell-1) ), 0);
        }
    }
}

#endif

// +++++++++++++++++++++++++ Type: Phi (Projection Method) +++++++++++++++++++++++++
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_phi_bcs_west (int i, int j, int k, int n_cell, int bcs,
                           amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( i < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 1) = Real(0.0); } else {
            ctx(i, j, k, 1) = - ctx(-i-1, j, k, 1);
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_phi_bcs_east (int i, int j, int k, int n_cell, int bcs,
                           amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( i > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 1) = Real(0.0); } else {
            ctx(i, j, k, 1) = - ctx ( ( (n_cell-i) + (n_cell-1) ), j, k, 1);
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_phi_bcs_south (int i, int j, int k, int n_cell, int bcs,
                            amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( j < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 1) = Real(0.0); } else {
            ctx(i, j, k, 1) = - ctx(i, -j-1, k, 1);
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_phi_bcs_north (int i, int j, int k, int n_cell, int bcs,
                            amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( j > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 1) = Real(0.0); } else {
            ctx(i, j, k, 1) = - ctx(i, ( (n_cell-j) + (n_cell-1) ), k, 1);
        }
    }
}
#if (AMREX_SPACEDIM > 2)

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_phi_bcs_front (int i, int j, int k, int n_cell, int bcs,
                           amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( k < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 1) = Real(0.0); } else {
            ctx(i, j, k, 1) = - ctx (i, j, -k-1, 1);
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_phi_bcs_back (int i, int j, int k, int n_cell, int bcs,
                          amrex::Array4<amrex::Real> const& ctx)
{
    using amrex::Real;

    if ( k > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) { ctx(i, j, k, 1) = Real(0.0); } else {
            ctx(i, j, k, 1) = - ctx (i, j, ( (n_cell-k) + (n_cell-1) ), 1);
        }
    }
}

#endif

// +++++++++++++++++++++++++ Type: Velocity +++++++++++++++++++++++++
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_vel_bcs_west (int i, int j, int k, int n_cell, int bcs,
                           amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( i < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j >= 0 && j <= (n_cell - 1) ) {
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
        } else {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                vcart(i, j, k, dir) = Real(2222.0);
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_vel_bcs_east (int i, int j, int k, int n_cell, int bcs,
                           amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( i > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j >= 0 && j <= (n_cell - 1) ) {
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
        } else {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                vcart(i, j, k, dir) = Real(2222.0);
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_vel_bcs_south (int i, int j, int k, int n_cell, int bcs,
                            amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( j < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( i >= 0 && i <= (n_cell - 1) ) {
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
        } else {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                vcart(i, j, k, dir) = Real(2222.0);
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_vel_bcs_north (int i, int j, int k, int n_cell, int bcs,
                            amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( j > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( i >= 0 && i <= (n_cell - 1) ) {
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
        } else {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                vcart(i, j, k, dir) = Real(2222.0);
            }
        }
    }
}

#if (AMREX_SPACEDIM > 2)

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_vel_bcs_front (int i, int j, int k, int n_cell, int bcs,
                           amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( k < 0 ) {
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
void enforce_vel_bcs_back (int i, int j, int k, int n_cell, int bcs,
                          amrex::Array4<amrex::Real> const& vcart)
{
    using amrex::Real;

    if ( k > (n_cell - 1) ) {
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

// +++++++++++++++++++++++++ Type: Flux +++++++++++++++++++++++++
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_flux_bcs_west (int i, int j, int k, int n_cell, int bcs,
                            amrex::Array4<amrex::Real> const& input_flux)
{
    using amrex::Real;

    if ( i < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = Real(0.0);
            }
        } else {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = - input_flux(-i-1, j, k, dir);
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_flux_bcs_east (int i, int j, int k, int n_cell, int bcs,
                            amrex::Array4<amrex::Real> const& input_flux)
{
    using amrex::Real;

    if ( i > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = Real(0.0);
            }
        } else {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = - input_flux(( (n_cell-i) + (n_cell-1) ), j, k, dir);
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_flux_bcs_south (int i, int j, int k, int n_cell, int bcs,
                             amrex::Array4<amrex::Real> const& input_flux)
{
    using amrex::Real;

    if ( j < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = Real(0.0);
            }
        } else {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = - input_flux(i, -j-1, k, dir);
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_flux_bcs_north (int i, int j, int k, int n_cell, int bcs,
                             amrex::Array4<amrex::Real> const& input_flux)
{
    using amrex::Real;

    if ( j > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = Real(0.0);
            }
        } else {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = - input_flux(i, ( (n_cell-j) + (n_cell-1) ), k, dir);
            }
        }
    }
}

#if (AMREX_SPACEDIM > 2)

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_bcs_front (int i, int j, int k, int n_cell, int bcs,
                        amrex::Array4<amrex::Real> const& input_flux)
{
    using amrex::Real;

    if ( k < 0 ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = Real(0.0);
            }
        } else {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = - input_flux(i, -j-1, k, dir);
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_bcs_back (int i, int j, int k, int n_cell, int bcs,
                       amrex::Array4<amrex::Real> const& input_flux)
{
    using amrex::Real;

    if ( k > (n_cell - 1) ) {
        // amrex::Print() << "FILLING| Ghost Cell | at i=" << i << " ; j=" << j << " ; k=" << k << "\n";
        if ( j == i || j == (n_cell - i) ) {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = Real(0.0);
            }
        } else {
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                input_flux(i, j, k, dir) = - input_flux(i, -j-1, k, dir);
            }
        }
    }
}

#endif

#endif
