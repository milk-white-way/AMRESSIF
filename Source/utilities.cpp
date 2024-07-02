#include <AMReX_MultiFabUtil.H>

#include "utilities.H"
#include "fn_enforce_wall_bcs.H"

using namespace amrex;

// ===================== UTILITY | CONVERSION  =====================
void cont2cart (MultiFab& velCart,
                Array<MultiFab, AMREX_SPACEDIM>& velCont,
                const Geometry& geom, 
                int const& Nghost,
                Vector<int> const& phy_bc_lo,
                Vector<int> const& phy_bc_hi,
                int const& n_cell)
{
    //average_face_to_cellcenter(velCart, amrex::GetArrOfConstPtrs(velCont), geom);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vel_cart  = velCart.array(mfi);

        auto const& vel_cont_x = velCont[0].array(mfi);
        auto const& vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_z = velCont[2].array(mfi);
#endif

        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            vel_cart(i, j, k, 0) = amrex::Real(0.5)*( vel_cont_x(i, j, k) + vel_cont_x(i+1, j, k) );
            vel_cart(i, j, k, 1) = amrex::Real(0.5)*( vel_cont_y(i, j, k) + vel_cont_y(i, j+1, k) );
#if (AMREX_SPACEDIM > 2)
            vel_cart(i, j, k, 2) = amrex::Real(0.5)*( vel_cont_z(i, j, k) + vel_cont_z(i, j, k+1) );
#endif
        });
    }

    enforce_wall_bcs_for_cell_centered_velocity_on_ghost_cells(velCart, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
    enforce_wall_bcs_for_face_centered_velocity_on_physical_boundaries(velCart, velCont, geom, phy_bc_lo, phy_bc_hi);
}

// ===================== UTILITY | EXTRACT LINE SOLUTION  =====================
void write_interp_line_solution (Real const& interp_sol,
                                 std::string const& filename)
{
    // Construct the filename for this iteration
    std::string interp_filename = "interp_" + filename;

    // Open a file for writing
    std::ofstream outfile(interp_filename, std::ios::app);

    // Check if the file was opened successfully
    if (!outfile.is_open())
    {
        std::cerr << "Failed to open file for writing\n";
    }

    // Write data to the file
    outfile << interp_sol << "\n";

    // Close the file
    outfile.close();
}

void write_exact_line_solution (Real const& x,
                                Real const& y,
                                Real const& numerical_sol,
                                Real const& analytical_sol,
                                std::string const& filename)
{
    // Open a file for writing
    std::ofstream outfile(filename, std::ios::app);

    // Check if the file was opened successfully
    if (!outfile.is_open())
    {
        std::cerr << "Failed to open file for writing\n";
    }

    // Write data to the file
    outfile << x << " " << y << " " << numerical_sol << " " << analytical_sol << "\n";

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
    Real zerror = velStarDiff[2].norm2(0, geom.periodicity());
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

void array_analytical_vel_calc (Array<MultiFab, AMREX_SPACEDIM>& array_analytical_vel,
                                Geometry const& geom,
                                Real const& time)
{
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(array_analytical_vel[0]); mfi.isValid(); ++mfi) {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif

        auto const& vel_cont_exact_x = array_analytical_vel[0].array(mfi);
        auto const& vel_cont_exact_y = array_analytical_vel[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_exact_z = array_analytical_vel[2].array(mfi);
#endif

        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            amrex::Real x = prob_lo[0] + (i + Real(0.0)) * dx[0];
            amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
            
            vel_cont_exact_x(i, j, k) = std::sin(amrex::Real(2.0) * M_PI * x) * std::cos(amrex::Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);
        });
        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
            amrex::Real y = prob_lo[1] + (j + Real(0.0)) * dx[1];

            vel_cont_exact_y(i, j, k) = - std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            vel_cont_exact_y(i, j, k) = - std::cos(amrex::Real(2.0) * M_PI * x) * std::sin(amrex::Real(2.0) * M_PI * y) * std::exp(-Real(8.0) * M_PI * M_PI * time);
        });
#endif
    }
}

void cc_analytical_press_calc (MultiFab& cc_analytical_press,
                               Geometry const& geom,
                               Real const& time)
{
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // Initialize the analytical pressure field
    for ( MFIter mfi(cc_analytical_press); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& press_exact = cc_analytical_press.array(mfi);

        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
            amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];

            press_exact(i, j, k, 0) = Real(0.25) * ( std::cos(Real(4.0) * M_PI * x) + std::cos(Real(4.0) * M_PI * y) ) * std::exp(-Real(8.0) * M_PI * M_PI * time) * std::exp(-Real(8.0) * M_PI * M_PI * time);
        });
    }
}


void SumAbsStag(const std::array<MultiFab, 
                AMREX_SPACEDIM>& m1,
	            amrex::Vector<amrex::Real>& sum)
{
  BL_PROFILE_VAR("SumAbsStag()", SumAbsStag);

  // Initialize to zero
  std::fill(sum.begin(), sum.end(), 0.);

  ReduceOps<ReduceOpSum> reduce_op;

  //////// x-faces

  ReduceData<Real> reduce_datax(reduce_op);
  using ReduceTuple = typename decltype(reduce_datax)::Type;

  for (MFIter mfi(m1[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      const Box& bx = mfi.tilebox();
      const Box& bx_grid = mfi.validbox();

      auto const& fab = m1[0].array(mfi);

      int xlo = bx_grid.smallEnd(0);
      int xhi = bx_grid.bigEnd(0);

      reduce_op.eval(bx, reduce_datax,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real weight = (i>xlo && i<xhi) ? 1.0 : 0.5;
          return {std::abs(fab(i,j,k)*weight)};
      });
  }

  sum[0] = amrex::get<0>(reduce_datax.value());
  ParallelDescriptor::ReduceRealSum(sum[0]);

  //////// y-faces

  ReduceData<Real> reduce_datay(reduce_op);

  for (MFIter mfi(m1[1],TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      const Box& bx = mfi.tilebox();
      const Box& bx_grid = mfi.validbox();

      auto const& fab = m1[1].array(mfi);

      int ylo = bx_grid.smallEnd(1);
      int yhi = bx_grid.bigEnd(1);

      reduce_op.eval(bx, reduce_datay,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real weight = (j>ylo && j<yhi) ? 1.0 : 0.5;
          return {std::abs(fab(i,j,k)*weight)};
      });
  }

  sum[1] = amrex::get<0>(reduce_datay.value());
  ParallelDescriptor::ReduceRealSum(sum[1]);

#if (AMREX_SPACEDIM == 3)

  //////// z-faces

  ReduceData<Real> reduce_dataz(reduce_op);

  for (MFIter mfi(m1[2],TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      const Box& bx = mfi.tilebox();
      const Box& bx_grid = mfi.validbox();

      auto const& fab = m1[2].array(mfi);

      int zlo = bx_grid.smallEnd(2);
      int zhi = bx_grid.bigEnd(2);

      reduce_op.eval(bx, reduce_dataz,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real weight = (k>zlo && k<zhi) ? 1.0 : 0.5;
          return {std::abs(fab(i,j,k)*weight)};
      });
  }

  sum[2] = amrex::get<0>(reduce_dataz.value());
  ParallelDescriptor::ReduceRealSum(sum[2]);

#endif

}

void StagL2Norm(const std::array<MultiFab, AMREX_SPACEDIM>& m1,
		        const int& comp,
                amrex::Vector<amrex::Real>& inner_prod)
{

    BL_PROFILE_VAR("StagL2Norm()", StagL2Norm);

    Array<MultiFab, AMREX_SPACEDIM> mscr;
    for (int dir=0; dir < AMREX_SPACEDIM; dir++) {
        mscr[dir].define(m1[dir].boxArray(), m1[dir].DistributionMap(), 1, 0);
    }

    StagInnerProd(m1, comp, mscr, inner_prod);
    for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
        inner_prod[dir] = std::sqrt(inner_prod[dir]);
    }
}

void StagInnerProd(const std::array<MultiFab, AMREX_SPACEDIM>& m1,
                   const int& comp1,
                   std::array<MultiFab, AMREX_SPACEDIM>& mscr,
                   amrex::Vector<amrex::Real>& prod_val)
{
  BL_PROFILE_VAR("StagInnerProd()", StagInnerProd);

  for (int d=0; d<AMREX_SPACEDIM; d++) {
    MultiFab::Copy(mscr[d], m1[d], comp1, 0, 1, 0);
    MultiFab::Multiply(mscr[d], m1[d], comp1, 0, 1, 0);
  }

  std::fill(prod_val.begin(), prod_val.end(), 0.);
  SumAbsStag(mscr, prod_val);
}
