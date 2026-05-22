/**********************************************************************************
 * external_shear.cpp
 *
 * Template for simulating passive external shear in a vertex model tissue.
 *
 * This file demonstrates how to:
 *   1) Build an initial condition by searching over distributions of p0 and
 *      edge tensions to match experimental targets at CE onset:
 *        - target average cell shape
 *        - target fraction of vertical edges
 *   2) Apply passive (externally imposed) pure shear and measure the response.
 *
 * Important:
 *   - During the shear protocol, deformation is externally imposed.
 *   - This is not an active-tension-driven shear protocol.
 *
 * Dependencies:
 *   This file relies on cellGPU (Daniel Sussman group) and local extensions.
 *   https://github.com/sussmanLab/cellGPU
 *
 * Last updated: May 20, 2026
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

static std::string intToString(int value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

// Convenience wrapper for FIRE settings.
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

// Build a linear sweep
static std::vector<double> buildLinearList(int n, Dscalar low, Dscalar high)
{
    std::vector<double> values(n);
    for (int i = 0; i < n; ++i)
        values[i] = low + ((high - low) / n) * i;
    return values;
}

int main(int argc, char *argv[])
{
    // =========================
    // 1) Parse command line args
    // =========================
    // Argument map (argv index -> meaning) is documented directly below.
    if (argc < 36)
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

    string outputDir = argv[1];                   // Output directory for text and NetCDF files
    int rngSeed = atoi(argv[2]);                  // Base RNG seed for reproducible random draws
    int numCells = atoi(argv[3]);                 // Number of cells in the tissue

    Dscalar areaPref = atof(argv[4]);             // Preferred cell area A0
    Dscalar areaModulus = atof(argv[5]);          // Area stiffness KA for initial base relaxation
    Dscalar perimeterModulus = atof(argv[6]);     // Perimeter stiffness KP for initial base relaxation

    Dscalar fireDt = atof(argv[7]);               // FIRE integration timestep
    Dscalar fireTol = atof(argv[8]);              // FIRE force tolerance for standard minimizations
    int fireInitSteps = atoi(argv[9]);            // Number of outer minimization rounds (each up to tSteps iterations)

    Dscalar t1Threshold = atof(argv[10]);         // T1 edge-length threshold for neighbor exchange

    Dscalar areaModulusInitSearch = atof(argv[11]);      // KA used while searching for target initial morphology
    Dscalar perimeterModulusInitSearch = atof(argv[12]); // KP used while searching for target initial morphology

    Dscalar areaModulusShear = atof(argv[13]);           // KA used during passive shear protocol
    Dscalar perimeterModulusShear = atof(argv[14]);      // KP used during passive shear protocol

    Dscalar t1EdgeTensionResetValue = atof(argv[15]);    // Optional fixed T1-reset tension (currently not enforced)

    Dscalar targetAvgCellShape = atof(argv[16]);         // Target average cell shape (experimental CE onset)
    Dscalar thetaEdgeFractionDeg = atof(argv[17]);       // Angle cutoff (degrees) for counting vertical edges
    Dscalar targetVerticalEdgeFraction = atof(argv[18]); // Target fraction of vertical edges (experimental CE onset)
    Dscalar initConfTolerance = atof(argv[19]);          // Tolerance for accepting initial morphology match

    int numP0Mean = atoi(argv[20]);              // Number of sampled means for p0 distribution
    Dscalar p0MeanLow = atof(argv[21]);          // Lower bound of p0 mean sweep
    Dscalar p0MeanHigh = atof(argv[22]);         // Upper bound of p0 mean sweep

    int numP0Std = atoi(argv[23]);               // Number of sampled std values for p0 distribution
    Dscalar p0StdLow = atof(argv[24]);           // Lower bound of p0 std sweep
    Dscalar p0StdHigh = atof(argv[25]);          // Upper bound of p0 std sweep

    int numTensionMean = atoi(argv[26]);         // Number of sampled means for edge-tension distribution
    Dscalar tensionMeanLow = atof(argv[27]);     // Lower bound of edge-tension mean sweep
    Dscalar tensionMeanHigh = atof(argv[28]);    // Upper bound of edge-tension mean sweep

    int numTensionStd = atoi(argv[29]);          // Number of sampled std values for edge-tension distribution
    Dscalar tensionStdLow = atof(argv[30]);      // Lower bound of edge-tension std sweep
    Dscalar tensionStdHigh = atof(argv[31]);     // Upper bound of edge-tension std sweep

    Dscalar fireTolInitConf = atof(argv[32]);    // FIRE tolerance specifically for initial-condition search loop

    int numEpsSteps = atoi(argv[33]);            // Number of passive shear strain steps
    Dscalar epsLow = atof(argv[34]);             // Minimum applied shear strain in the sweep
    Dscalar epsHigh = atof(argv[35]);            // Maximum applied shear strain in the sweep

    // =========================
    // 2) Runtime setup
    // =========================
    int useGPU = -1;
    int tSteps = 1000;
    bool reproducible = true;
    bool runSPV = false;

    bool initializeGPU = true;
    if (useGPU >= 0)
    {
        bool gpu = chooseGPU(useGPU);
        if (!gpu)
            return 1;
        cudaSetDevice(useGPU);
    }
    else
    {
        initializeGPU = false;
    }

    std::vector<int> momentList = {1, 2};

    // =========================
    // 3) Sweep lists
    // =========================
    std::vector<double> p0MeanList = buildLinearList(numP0Mean, p0MeanLow, p0MeanHigh);
    std::vector<double> p0StdList = buildLinearList(numP0Std, p0StdLow, p0StdHigh);
    std::vector<double> tensionMeanList = buildLinearList(numTensionMean, tensionMeanLow, tensionMeanHigh);
    std::vector<double> tensionStdList = buildLinearList(numTensionStd, tensionStdLow, tensionStdHigh);
    std::vector<double> epsList = buildLinearList(numEpsSteps, epsLow, epsHigh);

    // =========================
    // 4) Output files
    // =========================
    ofstream outFile((outputDir + "vertex_Ncell" + intToString(numCells) + "sample" + intToString(rngSeed) + ".txt").c_str());
    outFile.precision(17);

    ofstream outInitSearch((outputDir + "vertex_Ncell" + intToString(numCells) + "sample" + intToString(rngSeed) + "_initial_conf_steps_info.txt").c_str());
    outInitSearch.precision(17);

    // ==========================================
    // 5) Main run (single outer p0 run for template
    // ==========================================
    for (int p0OuterIter = 0; p0OuterIter < 1; ++p0OuterIter)
    {
        string dataname = outputDir + "vertex_p0_sweep_p0_num" + intToString(p0OuterIter) + "sample" + intToString(rngSeed) + ".nc";
        AVMDatabaseNetCDF ncdat(2 * numCells, dataname, NcFile::Replace);

        string datanameOriginal = outputDir + "vertex_p0_sweep_p0_num" + intToString(p0OuterIter) + "sample" + intToString(rngSeed) + "_original.nc";
        AVMDatabaseNetCDF ncdatOriginal(2 * numCells, datanameOriginal, NcFile::Replace);

        Dscalar baseP0 = p0MeanList[p0OuterIter];
        Dscalar selectedP0Mean = baseP0;

        // -------------------------
        // 5a) Construct AVM instance
        // -------------------------
        shared_ptr<VertexQuadraticEnergy> avm = make_shared<VertexQuadraticEnergy>(numCells, 1.0, baseP0, reproducible, 1001 * rngSeed, runSPV);

        avm->setCellPreferencesUniform(areaPref, baseP0);
        avm->setModuliUniform(areaModulus, perimeterModulus);
        avm->setBoxStrainXStrainY_DOF(false);
        avm->setT1Threshold(t1Threshold);

        // --------------------------------------
        // 5b) First minimization of base tissue
        // --------------------------------------
        shared_ptr<EnergyMinimizerFIRE> fireMinimizer = make_shared<EnergyMinimizerFIRE>(avm);
        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(avm);
        sim->addUpdater(fireMinimizer, avm);
        sim->setIntegrationTimestep(fireDt);
        sim->setCPUOperation(!initializeGPU);

        avm->SetNumberT1s(0);

        Dscalar maxForce = 0.0;
        for (int i = 0; i < fireInitSteps; ++i)
        {
            setFIREParameters(fireMinimizer, fireDt, 0.15, 10 * fireDt, 1.1, 0.5, .99, 5, fireTol);
            fireMinimizer->setMaximumIterations(tSteps * (i + 1));
            sim->performTimestep();
            maxForce = fireMinimizer->getMaxForce();
            if (maxForce < fireTol)
                break;
        }

        avm->ComputeEdgeCharacteristics();
        avm->ComputeEdgeSusceptibility();
        ncdat.WriteState(avm);

        Dscalar3 virialStress = avm->computeStressComponents();
        Dscalar2 energyTerms = avm->computeEnergyTerms();
        Dscalar2 boxStrains = avm->ReturnStrainXStrainY();
        Dscalar tensionEnergy = avm->computeEdgeTensionEnergyTerm();

        outFile << -1 << " " << baseP0 << " " << avm->computeEnergy() << " "
                << energyTerms.x << " " << energyTerms.y << " " << tensionEnergy << " "
                << virialStress.x << " " << virialStress.y << " " << virialStress.z << " "
                << avm->computeShearModulus() << " "
                << avm->reportq0() << " " << avm->reportVarq0() << " "
                << avm->reportq() << " " << avm->reportVarq() << " "
                << boxStrains.x << " " << boxStrains.y << " "
                << avm->GetNumberT1s() << " " << maxForce << endl;

        // -------------------------------------------------
        // 5c) Prepare initial condition search with tensions
        // -------------------------------------------------
        avm->InitializeEdgeTensions();
        avm->setAddEdgeTensionEnergy(true);
        avm->setModuliUniform(areaModulusInitSearch, perimeterModulusInitSearch);

        // One short minimization before snapshotting the search baseline.
        {
            shared_ptr<EnergyMinimizerFIRE> fireLocal = make_shared<EnergyMinimizerFIRE>(avm);
            SimulationPtr simLocal = make_shared<Simulation>();
            simLocal->setConfiguration(avm);
            simLocal->addUpdater(fireLocal, avm);
            simLocal->setIntegrationTimestep(fireDt);
            simLocal->setCPUOperation(!initializeGPU);

            for (int i = 0; i < fireInitSteps; ++i)
            {
                setFIREParameters(fireLocal, fireDt, 0.15, 10 * fireDt, 1.1, 0.5, .99, 5, fireTol);
                fireLocal->setMaximumIterations(tSteps * (i + 1));
                simLocal->performTimestep();
                maxForce = fireLocal->getMaxForce();
                if (maxForce < fireTol)
                    break;
            }
        }

        ncdatOriginal.WriteState(avm);

        Dscalar thetaEdgeFractionRad = thetaEdgeFractionDeg * (M_PI / 180.0);
        Dscalar avgShape = avm->reportq();
        Dscalar verticalEdgeFraction = avm->ComputeFractionVerticalEdges(thetaEdgeFractionRad);
        boxStrains = avm->ReturnStrainXStrainY();

        outInitSearch << -1 << " " << -1 << " " << -1 << " " << -1 << " "
                      << -1 << " " << -1 << " " << -1 << " " << -1 << " "
                      << boxStrains.x << " " << avgShape << " " << verticalEdgeFraction << endl;

        // ------------------------------------------------------
        // 5d) Search (p0 distribution, tension distribution) for
        //     target initial morphology
        // ------------------------------------------------------
        avm->SetResetT1EdgePropertiesAfterTransition(true);

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
                        for (int rndSeedTrial = 0; rndSeedTrial < 20; ++rndSeedTrial)
                        {
                            ncdatOriginal.ReadState(avm, 0, false);

                            int localSeed = 1001 * rngSeed * (rndSeedTrial + 1);
                            avm->InitializeEdgeTensionsNormalRand(tensionMean, tensionStd, localSeed);
                            avm->setCellPreferencesNormalRand(areaPref, p0Mean, p0Std, localSeed);

                            avm->Box->setGeneral(sqrt(numCells), 0.0, 0.0, sqrt(numCells));

                            // For this protocol, T1 reset tension is set to current mean network tension.
                            Dscalar avgTension = avm->ReturnAverageEdgeTensions();
                            avm->SetT1EdgeTension(avgTension);

                            shared_ptr<EnergyMinimizerFIRE> fireSearch = make_shared<EnergyMinimizerFIRE>(avm);
                            SimulationPtr simSearch = make_shared<Simulation>();
                            simSearch->setConfiguration(avm);
                            simSearch->addUpdater(fireSearch, avm);
                            simSearch->setIntegrationTimestep(fireDt);
                            simSearch->setCPUOperation(!initializeGPU);

                            for (int i = 0; i < fireInitSteps; ++i)
                            {
                                setFIREParameters(fireSearch, fireDt, 0.15, 10 * fireDt, 1.1, 0.5, .99, 5, fireTolInitConf);
                                fireSearch->setMaximumIterations(tSteps * (i + 1));
                                simSearch->performTimestep();
                                maxForce = fireSearch->getMaxForce();

                                if (maxForce < fireTolInitConf)
                                {
                                    foundGoodSample = true;
                                    break;
                                }

                                // Abort "bad" samples early to keep search practical.
                                if ((i > fireInitSteps / 3) && (maxForce > 1e-3))
                                    break;
                            }

                            if (foundGoodSample)
                                break;
                        }

                        avgShape = avm->reportq();
                        verticalEdgeFraction = avm->ComputeFractionVerticalEdges(thetaEdgeFractionRad);
                        boxStrains = avm->ReturnStrainXStrainY();

                        outInitSearch << p0MeanIter << " " << p0Mean << " "
                                      << p0StdIter << " " << p0Std << " "
                                      << tensionMeanIter << " " << tensionMean << " "
                                      << tensionStdIter << " " << tensionStd << " "
                                      << boxStrains.x << " " << avgShape << " "
                                      << verticalEdgeFraction << endl;

                        if ((fabs(avgShape - targetAvgCellShape) < initConfTolerance) &&
                            (fabs(verticalEdgeFraction - targetVerticalEdgeFraction) < initConfTolerance))
                        {
                            selectedP0Mean = p0Mean;
                            foundInitialConf = true;
                            break;
                        }
                    }
                }
            }
        }

        // Keep this parsed argument as an optional manual override if desired.
        // If you prefer a fixed reset value, replace avg network tension with this:
        // avm->SetT1EdgeTension(t1EdgeTensionResetValue);
        (void)t1EdgeTensionResetValue;

        // ===================================
        // 5e) Passive external shear protocol
        // ===================================
        avm->setModuliUniform(areaModulusShear, perimeterModulusShear);
        avm->SetNumberT1s(0);
        avm->SetResetT1EdgePropertiesAfterTransition(true);

        for (int epsIter = 0; epsIter < numEpsSteps; ++epsIter)
        {
            string datanameStrain = outputDir + "vertex_p0_sweep_p0_num" + intToString(p0OuterIter) + "eps_num" + intToString(epsIter) + "sample" + intToString(rngSeed) + ".nc";
            AVMDatabaseNetCDF ncdatStrain(2 * numCells, datanameStrain, NcFile::Replace);

            Dscalar eps = epsList[epsIter];

            avm->ComputeEdgeCharacteristics();
            avm->ComputeEdgeSusceptibility();
            ncdatStrain.WriteState(avm);

            Dscalar avgTension = avm->ReturnAverageEdgeTensions();
            avm->SetT1EdgeTension(avgTension);

            if (epsIter == 0)
                avm->AffinePureShearStrainBoxReshapeStepwise(0.0, epsList[epsIter]);
            else
                avm->AffinePureShearStrainBoxReshapeStepwise(epsList[epsIter - 1], epsList[epsIter]);

            avm->ComputeEdgeCharacteristics();
            avm->ComputeEdgeSusceptibility();
            ncdatStrain.WriteState(avm);

            shared_ptr<EnergyMinimizerFIRE> fireStep = make_shared<EnergyMinimizerFIRE>(avm);
            SimulationPtr simStep = make_shared<Simulation>();
            simStep->setConfiguration(avm);
            simStep->addUpdater(fireStep, avm);
            simStep->setIntegrationTimestep(fireDt);
            simStep->setCPUOperation(!initializeGPU);

            for (int i = 0; i < fireInitSteps; ++i)
            {
                setFIREParameters(fireStep, fireDt, 0.1, 10 * fireDt, 1.1, 0.5, .99, 5, fireTol);
                fireStep->setMaximumIterations(tSteps * (i + 1));
                simStep->performTimestep();
                maxForce = fireStep->getMaxForce();
                if (maxForce < fireTol)
                    break;
            }

            avm->ComputeEdgeCharacteristics();
            avm->ComputeEdgeSusceptibility();
            ncdatStrain.WriteState(avm);

            virialStress = avm->computeStressComponents();
            energyTerms = avm->computeEnergyTerms();
            boxStrains = avm->ReturnStrainXStrainY();
            tensionEnergy = avm->computeEdgeTensionEnergyTerm();

            outFile << selectedP0Mean << " " << eps << " " << avm->computeEnergy() << " "
                    << energyTerms.x << " " << energyTerms.y << " " << tensionEnergy << " "
                    << virialStress.x << " " << virialStress.y << " " << virialStress.z << " "
                    << avm->computeShearModulus() << " "
                    << avm->reportq0() << " " << avm->reportVarq0() << " "
                    << avm->reportq() << " " << avm->reportVarq() << " "
                    << boxStrains.x << " " << boxStrains.y << " "
                    << avm->GetNumberT1s() << " " << maxForce << endl;
        }
    }

    outInitSearch.close();
    outFile.close();

    if (initializeGPU)
        cudaDeviceReset();

    return 0;
}
