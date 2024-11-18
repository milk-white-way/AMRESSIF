
#include <fstream>
#include <iostream>

#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

using namespace amrex;

static
void
PrintUsage (const char* progName)
{
    Print() << std::endl
            << "This utility performs a diff operation between two"           << std::endl
            << "plotfiles that have the same geometrical domain and nodality" << std::endl
            << "(supports all nodality types; cell, face, edge, node)"        << std::endl
            << "but possibly a factor of refinement between the cells,"       << std::endl
            << "and outputs the L0, L1, and L2 norms"                         << std::endl
            << "L1 = sum(|diff_ijk|)/npts_coarsedomain"                       << std::endl
            << "L2 = sqrt[sum(diff_ijk^2)]/sqrt(npts_coarsedomain)"           << std::endl
            <<  "(only single-level supported)"                               << std::endl << std::endl;

    Print() << "Usage:" << '\n';
    Print() << progName << '\n';
    Print() << "    infile1 = inputFileName1" << '\n';
    Print() << "    reffile = refinedPlotFile" << '\n';
    Print() << "    diffile = differenceFileName" << '\n';
    Print() << "              (If not specified no file is written)" << '\n' << '\n';

    Print() << "You can either point to the plotfile base directory itself, e.g."      << std::endl
            << "  infile=plt00000"                                                     << std::endl
            << "Or the raw data itself, e.g."                                          << std::endl
            << "  infile=plt00000/Level_0/Cell"                                        << std::endl
            << "the latter is useful for some applications that dump out raw"          << std::endl
            << "nodal data within a plotfile directory."                               << std::endl
            << "The program will first try appending 'Level_0/Cell'"                   << std::endl
            << "onto the specified filenames."                                         << std::endl
            << "If that _H file doesn't exist, it tries using the full specified name" << std::endl << std::endl;

    exit(1);
}

int
main (int   argc,
      char* argv[])
{
    amrex::Initialize(argc,argv);
    {

        if (argc == 1) {
            PrintUsage(argv[0]);
        }

        const std::string farg = amrex::get_command_argument(1);
        if (farg == "-h" || farg == "--help")
        {
            PrintUsage(argv[0]);
        }

        // plotfile names for the coarse, fine, and subtracted output
        std::string pltFile, tkeFile="total_kinetic_energy.csv";
        int nstart, nfinal, nstep;

        // read in parameters from inputs file
        ParmParse pp;

        // coarse MultiFab
        pp.query("plotfile", pltFile);
        if (pltFile.empty())
            amrex::Abort("You must specify `filename'");

        // subtracted output (optional)
        pp.query("textfile", tkeFile);

        pp.query("start", nstart);
        pp.query("final", nfinal);
        pp.query("step", nstep);

        int n = 0;
        while ( n < nfinal )
        {
            n += nstep;
            std::string frame = Concatenate(pltFile, n, 5);

            //Print() << "INFO | reading from file: " << frame << std::endl;

            if (amrex::FileExists(frame+"/Level_0/Cell_H")) {
                frame += "/Level_0/Cell";
            }

            // storage for the input Flow MultiFabs (mf_f)
            MultiFab mf_f;

            VisMF::Read(mf_f, frame);

            if (mf_f.contains_nan()) {
                Abort("First plotfile contains NaN(s)");
            }

            int ncomp = mf_f.nComp();
            //Print() << "INFO | number of components in flow data = " << ncomp << std::endl;

            BoxArray ba_f = mf_f.boxArray();

            // minimalBox() computes a single box to enclose all the boxes
            // enclosedCells() converts it to a cell-centered Box
            Box bx_f = ba_f.minimalBox().enclosedCells();
            //Print() << "INFO | number of cells in flow domain = " << bx_f.numPts() << std::endl;

            // grab the distribution map from the coarse MultiFab
            DistributionMapping dm = mf_f.DistributionMap();

            MultiFab tke(ba_f, dm, 1, 0);
            tke.setVal(0.0);
            for ( MFIter mfi(mf_f, TilingIfNotGPU()); mfi.isValid(); ++mfi ) 
            {
                const Box& bx = mfi.tilebox();
                auto const& ff = mf_f.array(mfi);
                auto const& lke = tke.array(mfi);

                amrex::ParallelFor(bx, 
                                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    lke(i, j, k, 0) = 0.5 * (ff(i, j, k, 1)*ff(i, j, k, 1) + ff(i, j, k, 2)*ff(i, j, k, 2));
                });
            } // end MFIter

            // write out the value of the total kinetic energy
            if (tkeFile != "") {
                //std::ofstream outfile(tkeFile, std::ios::app);

                //if (!outfile.is_open()) {
                //    std::cerr << "Failed to open file for writing\n";
                //}

                Real total_kinetic_energy = tke.sum(0);
                amrex::Print() << total_kinetic_energy << "\n";
                //outfile << total_kinetic_energy << "\n";

                //outfile.close();
            }
        }
    }
    amrex::Finalize();
}
