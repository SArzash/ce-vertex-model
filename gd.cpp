/**********************************************************************************
 * gd.cpp
 *
 * Template for tuning edge tensions by gradient descent to approach a target
 * pure-shear strain in a vertex model tissue.
 *
 * This file demonstrates how to:
 *   1) Build an initial condition by searching over p0 and edge-tension
 *      distributions to match target CE-onset morphology.
 *   2) Evolve edge tensions using finite-difference gradients of a cost
 *      function C = (epsilon_target - epsilon_x)^2.
 *
 * Dependencies:
 *   This file relies on cellGPU (Daniel Sussman group) and local extensions.
 *   https://github.com/sussmanLab/cellGPU
 *
 * Last updated: May 22, 2026
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

// Build a linear sweep.
static std::vector<double> buildLinearList(int n, Dscalar low, Dscalar high)
{
    std::vector<double> values(n);
    for (int i = 0; i < n; ++i)
        values[i] = low + ((high - low) / n) * i;
    return values;
}

// Shared FIRE minimization routine used throughout the protocol.
static Dscalar runFireMinimization(shared_ptr<VertexQuadraticEnergy> avm,
                                   bool initializeGPU,
                                   Dscalar fireDt,
                                   Dscalar forceTol,
                                   int outerSteps,
                                   int innerTSteps)
{
    shared_ptr<EnergyMinimizerFIRE> fire = make_shared<EnergyMinimizerFIRE>(avm);
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(avm);
    sim->addUpdater(fire, avm);
    sim->setIntegrationTimestep(fireDt);
    sim->setCPUOperation(!initializeGPU);

    Dscalar maxForce = 0.0;
    for (int i = 0; i < outerSteps; ++i)
    {
        setFIREParameters(fire, fireDt, 0.15, 10 * fireDt, 1.1, 0.5, .99, 5, forceTol);
        fire->setMaximumIterations(innerTSteps * (i + 1));
        sim->performTimestep();

        maxForce = fire->getMaxForce();
        if (maxForce < forceTol)
            break;
    }
    return maxForce;
}

// Write one line of standard observables to the summary output file.
static void writeSummaryLine(ofstream &outFile,
                             int iterValue,
                             Dscalar p0Value,
                             shared_ptr<VertexQuadraticEnergy> avm,
                             Dscalar maxForce)
{
    Dscalar3 virialStress = avm->computeStressComponents();
    Dscalar2 energyTerms = avm->computeEnergyTerms();
    Dscalar2 boxStrains = avm->ReturnStrainXStrainY();
    Dscalar tensionEnergy = avm->computeEdgeTensionEnergyTerm();

    outFile << iterValue << " " << p0Value << " " << avm->computeEnergy() << " "
            << energyTerms.x << " " << energyTerms.y << " " << tensionEnergy << " "
            << virialStress.x << " " << virialStress.y << " " << virialStress.z << " "
            << avm->computeShearModulus() << " "
            << avm->reportq0() << " " << avm->reportVarq0() << " "
            << avm->reportq() << " " << avm->reportVarq() << " "
            << boxStrains.x << " " << boxStrains.y << " "
            << avm->GetNumberT1s() << " " << maxForce << endl;
}

// Find reverse edge direction: if edge is v1 -> v2, this returns k such that
// vertexNeighbors[3*v2 + k] == v1. Returns -1 if not found.
static int findReverseDirection(const ArrayHandle<int> &h_vn, int v1, int v2)
{
    for (int k = 0; k < 3; ++k)
    {
        if (h_vn.data[3 * v2 + k] == v1)
            return k;
    }
    return -1;
}

// Compute finite-difference gradient of C = (epsilon_target - epsilon_x)^2
// with respect to each unique edge tension.
static void computeEdgeTensionGradientsFiniteDifference(shared_ptr<VertexQuadraticEnergy> avm,
                                                        int numEdges,
                                                        Dscalar epsilonTarget,
                                                        Dscalar h)
{
    ArrayHandle<Dscalar> h_etGrad(avm->EdgeTensionGrad, access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar> h_et(avm->EdgeTension, access_location::host, access_mode::readwrite);
    ArrayHandle<int> h_vn(avm->vertexNeighbors, access_location::host, access_mode::read);

    int numVertices = numEdges / 3;
    std::vector<std::vector<int>> seenEdge(numVertices, std::vector<int>(numVertices, -1));

    std::vector<Dscalar> edgeTensionsTemp(numEdges);
    for (int i = 0; i < numEdges; ++i)
    {
        edgeTensionsTemp[i] = h_et.data[i];
        h_etGrad.data[i] = 0.0;
    }

    Dscalar fireDt = 0.0005;
    Dscalar fireTol = 1e-10;
    int fireOuterSteps = 1000;
    int tSteps = 10000;

    avm->setBoxStrainXStrainY_DOF(true);

    shared_ptr<EnergyMinimizerFIRE> fire = make_shared<EnergyMinimizerFIRE>(avm);
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(avm);
    sim->addUpdater(fire, avm);
    sim->setIntegrationTimestep(fireDt);
    sim->setCPUOperation(true);

    for (int v1 = 0; v1 < numVertices; ++v1)
    {
        for (int dir = 0; dir < 3; ++dir)
        {
            int v2 = h_vn.data[3 * v1 + dir];
            if ((v2 < 0) || (v2 >= numVertices))
                continue;

            if ((seenEdge[v1][v2] != -1) || (seenEdge[v2][v1] != -1))
                continue;

            int reverseDir = findReverseDirection(h_vn, v1, v2);
            if (reverseDir < 0)
                continue;

            int edgeIndexForward = 3 * v1 + dir;
            int edgeIndexReverse = 3 * v2 + reverseDir;

            Dscalar epsMinus = 0.0;
            Dscalar epsPlus = 0.0;

            // Perturb by -h.
            h_et.data[edgeIndexForward] = edgeTensionsTemp[edgeIndexForward] - h;
            h_et.data[edgeIndexReverse] = edgeTensionsTemp[edgeIndexReverse] - h;

            Dscalar maxForce = 0.0;
            for (int i = 0; i < fireOuterSteps; ++i)
            {
                setFIREParameters(fire, fireDt, 0.15, 10 * fireDt, 1.1, 0.5, .99, 5, fireTol);
                fire->setMaximumIterations(tSteps * (i + 1));
                sim->performTimestep();
                maxForce = fire->getMaxForce();
                if (maxForce < fireTol)
                    break;
            }
            (void)maxForce;

            Dscalar2 boxStrains = avm->ReturnStrainXStrainY();
            epsMinus = boxStrains.x;

            // Perturb by +h.
            h_et.data[edgeIndexForward] = edgeTensionsTemp[edgeIndexForward] + h;
            h_et.data[edgeIndexReverse] = edgeTensionsTemp[edgeIndexReverse] + h;

            for (int i = 0; i < fireOuterSteps; ++i)
            {
                setFIREParameters(fire, fireDt, 0.15, 10 * fireDt, 1.1, 0.5, .99, 5, fireTol);
                fire->setMaximumIterations(tSteps * (i + 1));
                sim->performTimestep();
                maxForce = fire->getMaxForce();
                if (maxForce < fireTol)
                    break;
            }

            boxStrains = avm->ReturnStrainXStrainY();
            epsPlus = boxStrains.x;

            Dscalar cMinus = (epsilonTarget - epsMinus) * (epsilonTarget - epsMinus);
            Dscalar cPlus = (epsilonTarget - epsPlus) * (epsilonTarget - epsPlus);
            Dscalar grad = (cPlus - cMinus) / (2 * h);

            h_etGrad.data[edgeIndexForward] = grad;
            h_etGrad.data[edgeIndexReverse] = grad;

            // Reset edge tension to original value.
            h_et.data[edgeIndexForward] = edgeTensionsTemp[edgeIndexForward];
            h_et.data[edgeIndexReverse] = edgeTensionsTemp[edgeIndexReverse];

            // Re-minimize after restoring original tension state.
            for (int i = 0; i < fireOuterSteps; ++i)
            {
                setFIREParameters(fire, fireDt, 0.15, 10 * fireDt, 1.1, 0.5, .99, 5, fireTol);
                fire->setMaximumIterations(tSteps * (i + 1));
                sim->performTimestep();
                maxForce = fire->getMaxForce();
                if (maxForce < fireTol)
                    break;
            }

            seenEdge[v1][v2] = 1;
            seenEdge[v2][v1] = 1;
        }
    }
}




int main(int argc, char *argv[])
{
    // =========================
    // 1) Parse command line args
    // =========================
    if (argc < 38)
    {
        cout << "Usage: gd outputDir seed numCells A0 KA KP dt FireTol initSteps "
             << "TensionDynamicsSteps TensionDynamicsDamping SamplingRate T1Threshold "
             << "KA_init KP_init KA_dyn KP_dyn T1EdgeTensionReset "
             << "InitialAvgP ThetaEdgeFraction InitialVerticalEdgeFraction InitConfTol "
             << "NumP0Mean P0MeanLow P0MeanHigh NumP0STD P0STDLow P0STDHigh "
             << "NumTensionMean TensionMeanLow TensionMeanHigh NumTensionSTD TensionSTDLow TensionSTDHigh "
             << "FireTolInitConf TargetEps FiniteDiffStep"
             << endl;
        return 1;
    }

    string outputDir = argv[1];
    int rngSeed = atoi(argv[2]);
    int numCells = atoi(argv[3]);

    Dscalar areaPref = atof(argv[4]);
    Dscalar areaModulus = atof(argv[5]);
    Dscalar perimeterModulus = atof(argv[6]);

    Dscalar fireDt = atof(argv[7]);
    Dscalar fireTol = atof(argv[8]);
    int fireOuterSteps = atoi(argv[9]);

    int tensionDynamicsSteps = atoi(argv[10]);
    Dscalar tensionDynamicsDamping = atof(argv[11]);
    int samplingRate = atoi(argv[12]);

    Dscalar t1Threshold = atof(argv[13]);

    Dscalar areaModulusInitSearch = atof(argv[14]);
    Dscalar perimeterModulusInitSearch = atof(argv[15]);
    Dscalar areaModulusDynamics = atof(argv[16]);
    Dscalar perimeterModulusDynamics = atof(argv[17]);

    Dscalar t1EdgeTensionResetValue = atof(argv[18]);

    Dscalar targetAvgCellShape = atof(argv[19]);
    Dscalar thetaEdgeFractionDeg = atof(argv[20]);
    Dscalar targetVerticalEdgeFraction = atof(argv[21]);
    Dscalar initConfTolerance = atof(argv[22]);

    int numP0Mean = atoi(argv[23]);
    Dscalar p0MeanLow = atof(argv[24]);
    Dscalar p0MeanHigh = atof(argv[25]);

    int numP0Std = atoi(argv[26]);
    Dscalar p0StdLow = atof(argv[27]);
    Dscalar p0StdHigh = atof(argv[28]);

    int numTensionMean = atoi(argv[29]);
    Dscalar tensionMeanLow = atof(argv[30]);
    Dscalar tensionMeanHigh = atof(argv[31]);

    int numTensionStd = atoi(argv[32]);
    Dscalar tensionStdLow = atof(argv[33]);
    Dscalar tensionStdHigh = atof(argv[34]);

    Dscalar fireTolInitConf = atof(argv[35]);

    Dscalar targetEps = atof(argv[36]);
    Dscalar finiteDiffStep = atof(argv[37]);

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

    // Keep this argument visible for optional manual override.
    (void)t1EdgeTensionResetValue;

    // =========================
    // 3) Sweep lists
    // =========================
    std::vector<double> p0MeanList = buildLinearList(numP0Mean, p0MeanLow, p0MeanHigh);
    std::vector<double> p0StdList = buildLinearList(numP0Std, p0StdLow, p0StdHigh);
    std::vector<double> tensionMeanList = buildLinearList(numTensionMean, tensionMeanLow, tensionMeanHigh);
    std::vector<double> tensionStdList = buildLinearList(numTensionStd, tensionStdLow, tensionStdHigh);

    Dscalar thetaEdgeFractionRad = thetaEdgeFractionDeg * (M_PI / 180.0);
    int numVertices = 2 * numCells;
    int numEdges = 3 * numVertices;

    // =========================
    // 4) Output files
    // =========================
    ofstream outFile((outputDir + "vertex_Ncell" + intToString(numCells) + "sample" + intToString(rngSeed) + ".txt").c_str());
    outFile.precision(17);

    ofstream outInitSearch((outputDir + "vertex_Ncell" + intToString(numCells) + "sample" + intToString(rngSeed) + "_initial_conf_steps_info.txt").c_str());
    outInitSearch.precision(17);

    // ==========================================
    // 5) Main run (single outer p0 run for template)
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
        avm->setBoxStrainXStrainY_DOF(true);
        avm->setT1Threshold(t1Threshold);

        // --------------------------------------
        // 5b) First minimization of base tissue
        // --------------------------------------
        avm->SetNumberT1s(0);
        Dscalar maxForce = runFireMinimization(avm,
                                               initializeGPU,
                                               fireDt,
                                               fireTol,
                                               fireOuterSteps,
                                               tSteps);

        avm->ComputeEdgeCharacteristics();
        avm->ComputeEdgeSusceptibility();
        ncdat.WriteState(avm);

        writeSummaryLine(outFile, -1, baseP0, avm, maxForce);

        // -------------------------------------------------
        // 5c) Prepare initial condition search with tensions
        // -------------------------------------------------
        avm->InitializeEdgeTensions();
        avm->setAddEdgeTensionEnergy(true);
        avm->setModuliUniform(areaModulusInitSearch, perimeterModulusInitSearch);

        maxForce = runFireMinimization(avm,
                                       initializeGPU,
                                       fireDt,
                                       fireTol,
                                       fireOuterSteps,
                                       tSteps);

        ncdatOriginal.WriteState(avm);

        Dscalar avgShape = avm->reportq();
        Dscalar verticalEdgeFraction = avm->ComputeFractionVerticalEdges(thetaEdgeFractionRad);
        Dscalar2 boxStrains = avm->ReturnStrainXStrainY();

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

                            // For this protocol, reset tension after T1 to current mean edge tension.
                            Dscalar avgTension = avm->ReturnAverageEdgeTensions();
                            avm->SetT1EdgeTension(avgTension);

                            maxForce = runFireMinimization(avm,
                                                           initializeGPU,
                                                           fireDt,
                                                           fireTolInitConf,
                                                           fireOuterSteps,
                                                           tSteps);

                            if (maxForce < fireTolInitConf)
                            {
                                foundGoodSample = true;
                                break;
                            }
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

                        if (foundGoodSample)
                            cout << "Found converged sample for current (p0, tension) tuple" << endl;
                    }
                }
            }
        }

        // =========================================
        // 5e) Gradient-descent edge-tension dynamics
        // =========================================
        avm->setModuliUniform(areaModulusDynamics, perimeterModulusDynamics);
        avm->SetNumberT1s(0);
        avm->SetResetT1EdgePropertiesAfterTransition(true);

        for (int tensionIter = 0; tensionIter < tensionDynamicsSteps; ++tensionIter)
        {
            Dscalar avgTension = avm->ReturnAverageEdgeTensions();
            avm->SetT1EdgeTension(avgTension);

            computeEdgeTensionGradientsFiniteDifference(avm, numEdges, targetEps, finiteDiffStep);

            ArrayHandle<Dscalar> h_et(avm->EdgeTension, access_location::host, access_mode::readwrite);
            ArrayHandle<Dscalar> h_etGrad(avm->EdgeTensionGrad, access_location::host, access_mode::read);
            for (int i = 0; i < numEdges; ++i)
                h_et.data[i] += - dt * tensionDynamicsDamping * h_etGrad.data[i];

            maxForce = runFireMinimization(avm,
                                           initializeGPU,
                                           fireDt,
                                           fireTol,
                                           fireOuterSteps,
                                           tSteps);

            if (tensionIter % samplingRate == 0)
            {
                avm->ComputeEdgeCharacteristics();
                avm->ComputeEdgeSusceptibility();
                ncdat.WriteState(avm);

                writeSummaryLine(outFile, tensionIter, selectedP0Mean, avm, maxForce);

                cout << "----------" << endl;
                cout << "selected p0 mean: " << selectedP0Mean
                     << " // energy: " << avm->computeEnergy() << endl;
                cout << "----------" << endl;
            }
        }
    }

    outInitSearch.close();
    outFile.close();

    if (initializeGPU)
        cudaDeviceReset();

    return 0;
}
