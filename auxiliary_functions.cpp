/**********************************************************************************
 * auxiliary_functions.cpp
 *
 * Simplified versions of a small set of functions used in the CE project
 *
 * The implementations are shortened to highlight the main algorithmic idea.
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

void setCellPreferencesUniform(shared_ptr<VertexQuadraticEnergy> avm,
                               Dscalar A0,
                               Dscalar P0)
{
    avm->AreaPeriPreferences.resize(avm->Ncells);
    ArrayHandle<Dscalar2> h_pref(avm->AreaPeriPreferences, access_location::host, access_mode::overwrite);

    for (int cell = 0; cell < avm->Ncells; ++cell)
    {
        h_pref.data[cell].x = A0;
        h_pref.data[cell].y = P0;
    }
}

void setCellPreferencesNormalRand(shared_ptr<VertexQuadraticEnergy> avm,
                                  Dscalar A0,
                                  Dscalar mean_P0,
                                  Dscalar std_P0,
                                  int randSeed)
{
    avm->AreaPeriPreferences.resize(avm->Ncells);
    ArrayHandle<Dscalar2> h_pref(avm->AreaPeriPreferences, access_location::host, access_mode::overwrite);

    noiseSource randGen;
    randGen.setReproducible(true);
    randGen.setReproducibleSeed(randSeed);

    for (int cell = 0; cell < avm->Ncells; ++cell)
    {
        Dscalar sampledP0 = randGen.getRealNormal(mean_P0, std_P0);
        while (sampledP0 < 0.0)
            sampledP0 = randGen.getRealNormal(mean_P0, std_P0);

        h_pref.data[cell].x = A0;
        h_pref.data[cell].y = sampledP0;
    }
}

void setModuliUniform(shared_ptr<VertexQuadraticEnergy> avm,
                      Dscalar newKA,
                      Dscalar newKP)
{
    avm->KA = newKA;
    avm->KP = newKP;
    avm->Moduli.resize(avm->Ncells);

    ArrayHandle<Dscalar2> h_moduli(avm->Moduli, access_location::host, access_mode::overwrite);
    for (int cell = 0; cell < avm->Ncells; ++cell)
    {
        h_moduli.data[cell].x = newKA;
        h_moduli.data[cell].y = newKP;
    }
}

void setAddEdgeTensionEnergy(shared_ptr<VertexQuadraticEnergy> avm, bool temp = true)
{
    avm->AddEdgeTensionEnergy = temp;
}

void ComputeEdgeCharacteristics(shared_ptr<VertexQuadraticEnergy> avm)
{
    ArrayHandle<Dscalar2> h_moduli(avm->Moduli, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_areaPeri(avm->AreaPeri, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_pref(avm->AreaPeriPreferences, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_vertexPos(avm->vertexPositions, access_location::host, access_mode::read);
    ArrayHandle<int> h_vn(avm->vertexNeighbors, access_location::host, access_mode::read);
    ArrayHandle<int> h_vcn(avm->vertexCellNeighbors, access_location::host, access_mode::read);
    ArrayHandle<Dscalar> h_passiveTension(avm->EdgeTensionPassive, access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar> h_edgeLen(avm->EdgeLen, access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar> h_edgeAngle(avm->EdgeAngle, access_location::host, access_mode::overwrite);

    for (int v1 = 0; v1 < avm->Nvertices; ++v1)
    {
        for (int dir = 0; dir < 3; ++dir)
        {
            int v2 = h_vn.data[3 * v1 + dir];

            int commonCells[2] = {-1, -1};
            int count = 0;
            for (int i = 0; i < 3; ++i)
            {
                int cellA = h_vcn.data[3 * v1 + i];
                for (int j = 0; j < 3; ++j)
                {
                    int cellB = h_vcn.data[3 * v2 + j];
                    if ((cellA >= 0) && (cellA == cellB) && (count < 2))
                    {
                        commonCells[count] = cellA;
                        ++count;
                    }
                }
            }

            Dscalar passiveTension = 0.0;
            if (count == 2)
            {
                int cellA = commonCells[0];
                int cellB = commonCells[1];
                passiveTension = 2.0 * h_moduli.data[cellA].y * (h_areaPeri.data[cellA].y - h_pref.data[cellA].y)
                               + 2.0 * h_moduli.data[cellB].y * (h_areaPeri.data[cellB].y - h_pref.data[cellB].y);
            }

            Dscalar2 edgeVector;
            avm->Box->minDist(h_vertexPos.data[v2], h_vertexPos.data[v1], edgeVector);

            h_passiveTension.data[3 * v1 + dir] = passiveTension;
            h_edgeLen.data[3 * v1 + dir] = sqrt(edgeVector.x * edgeVector.x + edgeVector.y * edgeVector.y);
            h_edgeAngle.data[3 * v1 + dir] = atan2(edgeVector.y, edgeVector.x);
        }
    }
}

void InitializeEdgeTensions(shared_ptr<VertexQuadraticEnergy> avm)
{
    ArrayHandle<Dscalar2> h_moduli(avm->Moduli, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_areaPeri(avm->AreaPeri, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_pref(avm->AreaPeriPreferences, access_location::host, access_mode::read);
    ArrayHandle<int> h_vn(avm->vertexNeighbors, access_location::host, access_mode::read);
    ArrayHandle<int> h_vcn(avm->vertexCellNeighbors, access_location::host, access_mode::read);
    ArrayHandle<Dscalar> h_edgeTension(avm->EdgeTension, access_location::host, access_mode::overwrite);

    for (int v1 = 0; v1 < avm->Nvertices; ++v1)
    {
        for (int dir = 0; dir < 3; ++dir)
        {
            int v2 = h_vn.data[3 * v1 + dir];

            int commonCells[2] = {-1, -1};
            int count = 0;
            for (int i = 0; i < 3; ++i)
            {
                int cellA = h_vcn.data[3 * v1 + i];
                for (int j = 0; j < 3; ++j)
                {
                    int cellB = h_vcn.data[3 * v2 + j];
                    if ((cellA >= 0) && (cellA == cellB) && (count < 2))
                    {
                        commonCells[count] = cellA;
                        ++count;
                    }
                }
            }

            Dscalar tension = 0.0;
            if (count == 2)
            {
                int cellA = commonCells[0];
                int cellB = commonCells[1];
                tension = 2.0 * h_moduli.data[cellA].y * (h_areaPeri.data[cellA].y - h_pref.data[cellA].y)
                        + 2.0 * h_moduli.data[cellB].y * (h_areaPeri.data[cellB].y - h_pref.data[cellB].y);
            }

            h_edgeTension.data[3 * v1 + dir] = tension;
        }
    }
}

void InitializeEdgeTensionsNormalRand(shared_ptr<VertexQuadraticEnergy> avm,
                                      Dscalar mean,
                                      Dscalar std,
                                      int randSeed)
{
    ArrayHandle<int> h_vn(avm->vertexNeighbors, access_location::host, access_mode::read);
    ArrayHandle<Dscalar> h_edgeTension(avm->EdgeTension, access_location::host, access_mode::overwrite);

    noiseSource randGen;
    randGen.setReproducible(true);
    randGen.setReproducibleSeed(randSeed);

    std::vector<std::vector<int>> seenEdge(avm->Nvertices, std::vector<int>(avm->Nvertices, -1));

    for (int v1 = 0; v1 < avm->Nvertices; ++v1)
    {
        for (int dir = 0; dir < 3; ++dir)
        {
            int v2 = h_vn.data[3 * v1 + dir];
            if ((seenEdge[v1][v2] != -1) || (seenEdge[v2][v1] != -1))
                continue;

            Dscalar sampledTension = randGen.getRealNormal(mean, std);
            while (sampledTension < 0.0)
                sampledTension = randGen.getRealNormal(mean, std);

            h_edgeTension.data[3 * v1 + dir] = sampledTension;

            for (int reverseDir = 0; reverseDir < 3; ++reverseDir)
            {
                if (h_vn.data[3 * v2 + reverseDir] == v1)
                {
                    h_edgeTension.data[3 * v2 + reverseDir] = sampledTension;
                    break;
                }
            }

            seenEdge[v1][v2] = 1;
            seenEdge[v2][v1] = 1;
        }
    }
}

Dscalar ComputeFractionVerticalEdges(shared_ptr<VertexQuadraticEnergy> avm,
                                     Dscalar theta_c_rad)
{
    ComputeEdgeCharacteristics(avm);

    ArrayHandle<Dscalar> h_edgeAngle(avm->EdgeAngle, access_location::host, access_mode::read);

    Dscalar countVertical = 0.0;
    Dscalar countAll = 0.0;
    for (int edge = 0; edge < 3 * avm->Nvertices; ++edge)
    {
        Dscalar angle = h_edgeAngle.data[edge];
        if (angle < 0.0)
            angle += M_PI;

        if (fabs(angle - M_PI / 2.0) <= theta_c_rad)
            countVertical += 1.0;

        countAll += 1.0;
    }

    return (countAll > 0.0) ? (countVertical / countAll) : 0.0;
}

void AffinePureShearStrainBoxReshapeStepwise(shared_ptr<VertexQuadraticEnergy> avm,
                                             Dscalar PureShearStrainPre,
                                             Dscalar PureShearStrain)
{
    ArrayHandle<Dscalar2> h_vertexPos(avm->vertexPositions, access_location::host, access_mode::overwrite);

    Dscalar x11, x12, x21, x22;
    avm->Box->getBoxDims(x11, x12, x21, x22);
    avm->Box->setGeneral(sqrt(avm->Ncells) * (1.0 + PureShearStrain),
                         x12,
                         x21,
                         sqrt(avm->Ncells) / (1.0 + PureShearStrain));

    Dscalar strainRatio = (1.0 + PureShearStrain) / (1.0 + PureShearStrainPre);
    for (int vertex = 0; vertex < avm->Nvertices; ++vertex)
    {
        h_vertexPos.data[vertex].x *= strainRatio;
        h_vertexPos.data[vertex].y /= strainRatio;
        avm->Box->putInBoxReal(h_vertexPos.data[vertex]);
    }
}
