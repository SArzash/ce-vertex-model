/**********************************************************************************
 * external_shear.cpp
 *
 * Example driver for the *externally imposed* pure-shear protocol in a
 * vertex-model tissue -- the passive baseline of the paper, against which the
 * tuning processes are compared.
 *
 * Protocol:
 *   1) Build a force-balanced tissue with uniform preferences.
 *   2) Search over (p0 distribution, active-tension distribution) for an initial
 *      state whose average cell shape and fraction of vertical edges match the
 *      experimental values at the onset of convergent extension (Fig. 3).
 *   3) Step through a list of imposed strains. At each step the box and all
 *      vertices are deformed affinely by the pure shear
 *
 *          Lx = sqrt(N) (1 + eps),      Ly = sqrt(N) / (1 + eps),
 *
 *      and the tissue is relaxed at fixed strain.
 *
 * The essential difference from the tuning drivers: the active tensions
 * Lambda_ij are held fixed throughout, and the box strain is *imposed* rather
 * than relaxed. The total tensions still
 * evolve, but only passively, through the perimeter-elasticity contribution
 * Lambda^P as cell shapes change.
 *
 * Dependencies:
 *   cellGPU (Daniel Sussman group) plus project-specific extensions to the
 *   vertex-model classes.  https://github.com/sussmanLab/cellGPU
 *
 * Last updated: August 7, 2026
 **********************************************************************************/

#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "voronoiQuadraticEnergy.h"
#include "selfPropelledParticleDynamics.h"
#include "selfPropelledCellVertexDynamics.h"
#include "vertexQuadraticEnergy.h"
#include "DatabaseNetCDFSPV.h"
#include "DatabaseNetCDFAVM.h"
#include "EnergyMinimizerFIRE2D.h"

using std::cout;
using std::endl;
using std::ofstream;
using std::string;

// ---------------------------------------------------------------------------
// Driver-local helpers.
//
// These are repeated verbatim in each of the three example drivers so that
// every file can be read on its own; they are not part of the vertex-model
// library.
// ---------------------------------------------------------------------------

static std::string intToString(int value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

//! Convenience wrapper for the FIRE minimizer settings.
static void setFIREParameters(shared_ptr<EnergyMinimizerFIRE> fire,
                              Dscalar deltaT,
                              Dscalar alphaStart,
                              Dscalar deltaTMax,
                              Dscalar deltaTInc,
                              Dscalar deltaTDec,
                              Dscalar alphaDec,
                              int nMin,
                              Dscalar forceCutoff)
{
    fire->setDeltaT(deltaT);
    fire->setAlphaStart(alphaStart);
    fire->setDeltaTMax(deltaTMax);
    fire->setDeltaTInc(deltaTInc);
    fire->setDeltaTDec(deltaTDec);
    fire->setAlphaDec(alphaDec);
    fire->setNMin(nMin);
    fire->setForceCutoff(forceCutoff);
}

/*!
 * Build a sweep of n values on the half-open grid
 *      low, low + d, ..., high - d      with d = (high - low)/n,
 * i.e. the upper bound is approached but not reached. This is the grid used for
 * the parameter and strain scans in the paper; if you want the endpoint
 * included, use (high - low)/(n - 1) instead.
 */
static std::vector<double> buildLinearGrid(int n, Dscalar low, Dscalar high)
{
    std::vector<double> values;
    if (n <= 0)
        return values;

    values.resize(n);
    for (int i = 0; i < n; ++i)
        values[i] = low + ((high - low) / n) * i;
    return values;
}

/*!
 * Run FIRE on an already-configured minimizer/simulation pair until the tissue
 * is force balanced. The minimizer is restarted
 * `outerRounds` times with an increasing iteration budget, which is how the
 * project code gets past configurations FIRE initially struggles with; the loop
 * exits as soon as max_m |f_m| drops below forceTol.
 *
 * earlyAbortForce > 0 enables an additional escape hatch used during the
 * initial-condition search: a sample still far from force balance after a third
 * of the rounds is abandoned rather than pursued, since drawing another random
 * initial state is cheaper than forcing this one to converge.
 *
 * Returns the final max residual force.
 */
static Dscalar runFireLoop(shared_ptr<EnergyMinimizerFIRE> fire,
                           SimulationPtr sim,
                           Dscalar fireDt,
                           Dscalar forceTol,
                           int outerRounds,
                           int innerIterations,
                           Dscalar earlyAbortForce = -1.0)
{
    Dscalar maxForce = 0.0;
    for (int round = 0; round < outerRounds; ++round)
    {
        setFIREParameters(fire, fireDt, 0.15, 10 * fireDt, 1.1, 0.5, .99, 5, forceTol);
        fire->setMaximumIterations(innerIterations * (round + 1));
        sim->performTimestep();

        maxForce = fire->getMaxForce();
        if (maxForce < forceTol)
            break;

        if ((earlyAbortForce > 0.0) && (round > outerRounds / 3) && (maxForce > earlyAbortForce))
            break;
    }
    return maxForce;
}

/*!
 * Build a fresh FIRE minimizer for the tissue and relax it to force balance.
 */
static Dscalar runFireMinimization(shared_ptr<VertexQuadraticEnergy> avm,
                                   bool useGPU,
                                   Dscalar fireDt,
                                   Dscalar forceTol,
                                   int outerRounds,
                                   int innerIterations,
                                   Dscalar earlyAbortForce = -1.0)
{
    shared_ptr<EnergyMinimizerFIRE> fire = make_shared<EnergyMinimizerFIRE>(avm);
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(avm);
    sim->addUpdater(fire, avm);
    sim->setIntegrationTimestep(fireDt);
    sim->setCPUOperation(!useGPU);

    return runFireLoop(fire, sim, fireDt, forceTol, outerRounds, innerIterations, earlyAbortForce);
}

/*!
 * Append one record of tissue observables to the summary file. The column
 * layout is identical in all three example drivers:
 *
 *    1  step index (tension-update iteration or strain step; -1 = reference state)
 *    2  mean target perimeter p0 of the cell population
 *    3  total energy E
 *    4  area contribution to E
 *    5  perimeter contribution to E
 *    6  active tension contribution to E,  sum_ij Lambda_ij l_ij
 *    7  mean target shape index p0
 *    8  variance of the target shape index
 *    9  mean cell shape index  p = P / sqrt(A)     
 *   10  variance of the cell shape index
 *   11  tissue pure-shear strain eps
 *   12  max residual force at the end of the relaxation
 */
static void writeSummaryLine(ofstream &outFile,
                             int step,
                             Dscalar p0Mean,
                             Dscalar strain,
                             shared_ptr<VertexQuadraticEnergy> avm,
                             Dscalar maxForce)
{
    Dscalar3 virialStress = avm->computeStressComponents();
    Dscalar2 energyTerms = avm->computeEnergyTerms();
    Dscalar tensionEnergy = avm->computeEdgeTensionEnergyTerm();

    outFile << step << " " << p0Mean << " " << avm->computeEnergy() << " "
            << energyTerms.x << " " << energyTerms.y << " " << tensionEnergy << " "
             << " " << avm->reportq0() << " " << avm->reportVarq0() << " "
            << avm->reportq() << " " << avm->reportVarq() << " "
            << strain << " " << maxForce << endl;
}

int main(int argc, char *argv[])
{
    // =========================
    // 1) Parse command line args
    // =========================
    const int expectedArgs = 36; // program name + 35 parameters
    if (argc < expectedArgs)
    {
        cout << "Usage: external_shear outputDir seed numCells A0 KA KP dt FireTol initSteps "
             << "T1Threshold KA_init KP_init KA_shear KP_shear T1EdgeTensionReset "
             << "InitialAvgP ThetaEdgeFraction InitialVerticalEdgeFraction InitConfTol "
             << "NumP0Mean P0MeanLow P0MeanHigh NumP0STD P0STDLow P0STDHigh "
             << "NumTensionMean TensionMeanLow TensionMeanHigh NumTensionSTD TensionSTDLow TensionSTDHigh "
             << "FireTolInitConf NumEpsSteps EpsLow EpsHigh"
             << endl;
        return 1;
    }

    string outputDir = argv[1];                          // output directory for the text and NetCDF files
    int rngSeed = atoi(argv[2]);                         // base RNG seed / sample index
    int numCells = atoi(argv[3]);                        // number of cells N

    Dscalar areaPref = atof(argv[4]);                    // target cell area A0
    Dscalar areaModulus = atof(argv[5]);                 // K_A for the initial base relaxation
    Dscalar perimeterModulus = atof(argv[6]);            // K_P for the initial base relaxation

    Dscalar fireDt = atof(argv[7]);                      // FIRE time step
    Dscalar fireTol = atof(argv[8]);                     // force tolerance for the production minimizations
    int fireOuterRounds = atoi(argv[9]);                 // number of FIRE restart rounds

    Dscalar t1Threshold = atof(argv[10]);                // edge length below which a T1 is performed (SI: 1e-2)

    Dscalar areaModulusInitSearch = atof(argv[11]);      // K_A used while searching for the initial state
    Dscalar perimeterModulusInitSearch = atof(argv[12]); // K_P used while searching for the initial state
    Dscalar areaModulusShear = atof(argv[13]);           // K_A used during the shear protocol
    Dscalar perimeterModulusShear = atof(argv[14]);      // K_P used during the shear protocol

    Dscalar t1EdgeTensionResetValue = atof(argv[15]);    // optional fixed post-T1 tension (see note below)

    Dscalar targetAvgCellShape = atof(argv[16]);         // experimental average cell shape at CE onset
    Dscalar thetaEdgeFractionDeg = atof(argv[17]);       // half-window (degrees) defining a "vertical" edge
    Dscalar targetVerticalEdgeFraction = atof(argv[18]); // experimental vertical-edge fraction at CE onset
    Dscalar initConfTolerance = atof(argv[19]);          // tolerance for accepting the initial state

    int numP0Mean = atoi(argv[20]);                      // p0 mean sweep: number of values ...
    Dscalar p0MeanLow = atof(argv[21]);                  // ... lower bound ...
    Dscalar p0MeanHigh = atof(argv[22]);                 // ... upper bound

    int numP0Std = atoi(argv[23]);                       // p0 std sweep
    Dscalar p0StdLow = atof(argv[24]);
    Dscalar p0StdHigh = atof(argv[25]);

    int numTensionMean = atoi(argv[26]);                 // active-tension mean sweep
    Dscalar tensionMeanLow = atof(argv[27]);
    Dscalar tensionMeanHigh = atof(argv[28]);

    int numTensionStd = atoi(argv[29]);                  // active-tension std sweep
    Dscalar tensionStdLow = atof(argv[30]);
    Dscalar tensionStdHigh = atof(argv[31]);

    Dscalar fireTolInitConf = atof(argv[32]);            // looser force tolerance for the initial-state search

    int numEpsSteps = atoi(argv[33]);                    // number of imposed strain steps
    Dscalar epsLow = atof(argv[34]);                     // first imposed strain
    Dscalar epsHigh = atof(argv[35]);                    // upper bound of the strain sweep (not reached, see grid)

    // =========================
    // 2) Runtime setup
    // =========================
    // These drivers are quasistatic and dominated by many short FIRE runs on
    // small systems, so they are run on the CPU
    int gpuIndex = -1;
    bool useGPU = false;
    if (gpuIndex >= 0)
    {
        if (!chooseGPU(gpuIndex))
            return 1;
        cudaSetDevice(gpuIndex);
        useGPU = true;
    }

    int fireInnerIterations = 1000; // FIRE iterations per restart round
    bool reproducible = true;       // fixed RNG stream, so runs can be repeated
    bool runSPV = false;            // initialize from a Voronoi tessellation, not an SPV run

    Dscalar thetaEdgeFractionRad = thetaEdgeFractionDeg * (M_PI / 180.0);

    // Parsed but unused: this protocol resets the post-T1 tension to the running
    // network average <T> rather than to a fixed value. The argument is kept in
    // the interface so that a fixed reset can be switched on where marked below.
    (void)t1EdgeTensionResetValue;

    // =========================
    // 3) Sweep lists
    // =========================
    std::vector<double> p0MeanList = buildLinearGrid(numP0Mean, p0MeanLow, p0MeanHigh);
    std::vector<double> p0StdList = buildLinearGrid(numP0Std, p0StdLow, p0StdHigh);
    std::vector<double> tensionMeanList = buildLinearGrid(numTensionMean, tensionMeanLow, tensionMeanHigh);
    std::vector<double> tensionStdList = buildLinearGrid(numTensionStd, tensionStdLow, tensionStdHigh);
    std::vector<double> epsList = buildLinearGrid(numEpsSteps, epsLow, epsHigh);

    if (p0MeanList.empty() || p0StdList.empty() || tensionMeanList.empty() ||
        tensionStdList.empty() || epsList.empty())
    {
        cout << "Each of NumP0Mean, NumP0STD, NumTensionMean, NumTensionSTD, NumEpsSteps must be >= 1." << endl;
        return 1;
    }

    // =========================
    // 4) Output files
    // =========================
    // outFile        : one line of observables per strain step (columns above)
    // outInitSearch  : one line per (p0, tension) tuple visited by the initial-state search
    ofstream outFile((outputDir + "vertex_Ncell" + intToString(numCells) + "sample" + intToString(rngSeed) + ".txt").c_str());
    outFile.precision(17);

    ofstream outInitSearch((outputDir + "vertex_Ncell" + intToString(numCells) + "sample" + intToString(rngSeed) + "_initial_conf_steps_info.txt").c_str());
    outInitSearch.precision(17);

    string dataname = outputDir + "vertex_Ncell" + intToString(numCells) + "sample" + intToString(rngSeed) + ".nc";
    AVMDatabaseNetCDF ncdat(2 * numCells, dataname, NcFile::Replace);

    // Snapshot of the force-balanced reference tissue; the initial-state search
    // reloads it before every trial so that each (p0, tension) draw starts from
    // the same configuration.
    string datanameOriginal = outputDir + "vertex_Ncell" + intToString(numCells) + "sample" + intToString(rngSeed) + "_original.nc";
    AVMDatabaseNetCDF ncdatOriginal(2 * numCells, datanameOriginal, NcFile::Replace);

    // -------------------------
    // 5) Construct the tissue
    // -------------------------
    Dscalar baseP0 = p0MeanList[0];
    Dscalar selectedP0Mean = baseP0;

    shared_ptr<VertexQuadraticEnergy> avm =
        make_shared<VertexQuadraticEnergy>(numCells, 1.0, baseP0, reproducible, 1001 * rngSeed, runSPV);

    avm->setCellPreferencesUniform(areaPref, baseP0);
    avm->setModuliUniform(areaModulus, perimeterModulus);
    avm->setT1Threshold(t1Threshold);

    // The box shape is prescribed by the protocol, so it is *not* a relaxed
    // degree of freedom here. This is the one setting that distinguishes the
    // external-shear driver from the two tuning drivers.
    avm->setBoxStrainXStrainY_DOF(false);

    // --------------------------------------
    // 6) Relax the passive tissue
    // --------------------------------------
    Dscalar maxForce = runFireMinimization(avm, useGPU, fireDt, fireTol,
                                           fireOuterRounds, fireInnerIterations);

    avm->ComputeEdgeCharacteristics();
    avm->ComputeEdgeSusceptibility();
    ncdat.WriteState(avm);

    writeSummaryLine(outFile, -1, baseP0, 0.0, avm, maxForce);

    // -------------------------------------------------
    // 7) Reference state for the initial-state search
    // -------------------------------------------------
    // Switch on the active tension term of Eq. (1) and relax once more; the
    // resulting configuration is the common starting point of the search.
    avm->InitializeEdgeTensions();
    avm->setAddEdgeTensionEnergy(true);
    avm->setModuliUniform(areaModulusInitSearch, perimeterModulusInitSearch);

    maxForce = runFireMinimization(avm, useGPU, fireDt, fireTol,
                                   fireOuterRounds, fireInnerIterations);

    ncdatOriginal.WriteState(avm);

    Dscalar avgShape = avm->reportq();
    Dscalar verticalEdgeFraction = avm->ComputeFractionVerticalEdges(thetaEdgeFractionRad);

    outInitSearch << -1 << " " << -1 << " " << -1 << " " << -1 << " "
                  << -1 << " " << -1 << " " << -1 << " " << -1 << " "
                  << 0.0 << " " << avgShape << " " << verticalEdgeFraction << endl;

    // ------------------------------------------------------
    // 8) Search for an initial state matching experiment
    //
    // Identical to the search in local_rule.cpp and gd.cpp, so that all
    // protocols start from statistically equivalent initial states.
    // ------------------------------------------------------
    avm->SetResetT1EdgePropertiesAfterTransition(true);

    const int maxRandomTrials = 20;
    bool foundInitialConf = false;

    for (int p0MeanIter = 0; p0MeanIter < numP0Mean && !foundInitialConf; ++p0MeanIter)
    {
        for (int p0StdIter = 0; p0StdIter < numP0Std && !foundInitialConf; ++p0StdIter)
        {
            Dscalar p0Mean = p0MeanList[p0MeanIter];
            Dscalar p0Std = p0StdList[p0StdIter];

            for (int tensionMeanIter = 0; tensionMeanIter < numTensionMean && !foundInitialConf; ++tensionMeanIter)
            {
                for (int tensionStdIter = 0; tensionStdIter < numTensionStd && !foundInitialConf; ++tensionStdIter)
                {
                    Dscalar tensionMean = tensionMeanList[tensionMeanIter];
                    Dscalar tensionStd = tensionStdList[tensionStdIter];

                    bool foundGoodSample = false;
                    for (int randomTrial = 0; randomTrial < maxRandomTrials; ++randomTrial)
                    {
                        // Restart from the common reference configuration.
                        ncdatOriginal.ReadState(avm, 0, false);

                        int localSeed = 1001 * rngSeed * (randomTrial + 1);
                        avm->InitializeEdgeTensionsNormalRand(tensionMean, tensionStd, localSeed);
                        avm->setCellPreferencesNormalRand(areaPref, p0Mean, p0Std, localSeed);

                        // Reset the box to the undeformed square. Here the box
                        // shape is prescribed, so this is the only reset needed.
                        avm->Box->setGeneral(sqrt(numCells), 0.0, 0.0, sqrt(numCells));

                        // Post-T1 tension reset value, <T> for the current network (SI, Fig. S1).
                        // Replace with t1EdgeTensionResetValue for a fixed reset instead.
                        avm->SetT1EdgeTension(avm->ReturnAverageEdgeTensions());

                        maxForce = runFireMinimization(avm, useGPU, fireDt, fireTolInitConf,
                                                       fireOuterRounds, fireInnerIterations,
                                                       1e-3 /* abort samples that stall */);

                        if (maxForce < fireTolInitConf)
                        {
                            foundGoodSample = true;
                            break;
                        }
                    }

                    avgShape = avm->reportq();
                    verticalEdgeFraction = avm->ComputeFractionVerticalEdges(thetaEdgeFractionRad);

                    outInitSearch << p0MeanIter << " " << p0Mean << " "
                                  << p0StdIter << " " << p0Std << " "
                                  << tensionMeanIter << " " << tensionMean << " "
                                  << tensionStdIter << " " << tensionStd << " "
                                  << 0.0 << " " << avgShape << " "
                                  << verticalEdgeFraction << endl;

                    if (foundGoodSample &&
                        (fabs(avgShape - targetAvgCellShape) < initConfTolerance) &&
                        (fabs(verticalEdgeFraction - targetVerticalEdgeFraction) < initConfTolerance))
                    {
                        selectedP0Mean = p0Mean;
                        foundInitialConf = true;
                    }
                }
            }
        }
    }

    if (!foundInitialConf)
        cout << "Warning: no (p0, tension) tuple matched the experimental targets within tolerance; "
             << "continuing from the last state visited." << endl;

    // ===================================
    // 9) Imposed pure-shear protocol
    // ===================================
    avm->setModuliUniform(areaModulusShear, perimeterModulusShear);
    avm->SetResetT1EdgePropertiesAfterTransition(true);

    for (int epsIter = 0; epsIter < numEpsSteps; ++epsIter)
    {
        Dscalar eps = epsList[epsIter];
        Dscalar epsPrevious = (epsIter == 0) ? 0.0 : epsList[epsIter - 1];

        // Keep the post-T1 reset value equal to the current network average.
        avm->SetT1EdgeTension(avm->ReturnAverageEdgeTensions());

        // Affine pure shear taking the tissue from epsPrevious to eps.
        avm->AffinePureShearStrainBoxReshapeStepwise(epsPrevious, eps);

        // Relax at fixed strain. The active tensions are untouched; only the
        // vertex positions respond, and the total tensions change only through
        // the passive term Lambda^P as cell perimeters change.
        maxForce = runFireMinimization(avm, useGPU, fireDt, fireTol,
                                       fireOuterRounds, fireInnerIterations);

        avm->ComputeEdgeCharacteristics();
        avm->ComputeEdgeSusceptibility();
        ncdat.WriteState(avm);

        writeSummaryLine(outFile, epsIter, selectedP0Mean, eps, avm, maxForce);

        cout << "strain step " << epsIter
             << " // p0 mean " << selectedP0Mean
             << " // strain " << eps
             << " // energy " << avm->computeEnergy() << endl;
    }

    outInitSearch.close();
    outFile.close();

    if (useGPU)
        cudaDeviceReset();

    return 0;
}
