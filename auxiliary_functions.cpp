/**********************************************************************************
 * auxiliary_functions.cpp
 *
 * Simplified reference implementations of the model-level routines that the
 * three example drivers (external_shear.cpp, local_rule.cpp, gd.cpp) call on
 * the vertex-model object.
 *
 * In the project code base these routines are members of the vertex-model
 * classes that ship with cellGPU (Simple2DCell / vertexModelBase /
 * VertexQuadraticEnergy), which is why the drivers invoke them as
 * `avm->SomeRoutine(...)`. They are shortened so that the
 * algorithmic content can be read at a glance.
 *
 * ---------------------------------------------------------------------------
 * Data layout conventions used throughout
 * ---------------------------------------------------------------------------
 * The tissue is a trivalent vertex network: every vertex has exactly three
 * neighbours, and a tissue of Ncells cells has
 *
 *      Nvertices = 2 * Ncells,      Nedges = 3 * Nvertices.
 *
 * Per-edge quantities (EdgeTension, EdgeTensionPassive, EdgeLen, EdgeAngle,
 * EdgeTensionGrad, ...) are stored in *directed* form, indexed by
 *
 *      3 * v + dir,     dir = 0, 1, 2,
 *
 * where `vertexNeighbors[3*v + dir]` is the neighbour vertex reached along that
 * direction. Every undirected junction therefore appears twice, once as
 * v1 -> v2 and once as v2 -> v1, and the two copies must always be kept
 * consistent. To visit each junction exactly once we use the `v1 < v2`;
 * the project code uses an equivalent "have I seen this edge" table.
 *
 * `EdgeTension` holds the *active* tension Lambda_ij.
 * `EdgeTensionPassive` holds the passive contribution
 *
 *      Lambda^P_ij = 2 K_P [ (P_m - P_0m) + (P_n - P_0n) ]                
 *
 * coming from the perimeter elasticity of the two cells m, n adjacent to the
 * junction. The total tension is T_ij = Lambda_ij + Lambda^P_ij, and the local
 * feedback rules of the paper act on T_ij.
 *
 * Dependencies:
 *   cellGPU (Daniel Sussman group) plus the project-specific extensions to the
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

// =====================================================================
// Small index helpers shared by the routines below
// =====================================================================

/*!
 * Return the directed-edge index of the junction v1 -> v2, i.e. the value
 * 3*v1 + dir for which vertexNeighbors[3*v1 + dir] == v2. Returns -1 when v2
 * is not a neighbour of v1, which should never happen on a consistent network
 * and is therefore only a defensive check.
 */
static int directedEdgeIndex(const ArrayHandle<int> &h_vn, int v1, int v2)
{
    for (int dir = 0; dir < 3; ++dir)
    {
        if (h_vn.data[3 * v1 + dir] == v2)
            return 3 * v1 + dir;
    }
    return -1;
}

/*!
 * Find the (at most two) cells that both v1 and v2 belong to; these are the two
 * cells that share the junction v1 -> v2. Returns the number of cells found,
 * which is 2 for every junction of a confluent periodic tissue.
 */
static int sharedCellsOfEdge(const ArrayHandle<int> &h_vcn, int v1, int v2, int sharedCells[2])
{
    int count = 0;
    sharedCells[0] = -1;
    sharedCells[1] = -1;

    for (int i = 0; i < 3; ++i)
    {
        int cellA = h_vcn.data[3 * v1 + i];
        if (cellA < 0)
            continue;
        for (int j = 0; j < 3; ++j)
        {
            if ((cellA == h_vcn.data[3 * v2 + j]) && (count < 2))
            {
                sharedCells[count] = cellA;
                ++count;
            }
        }
    }
    return count;
}

/*!
 * Passive tension of the junction shared by cells m and n:
 *      Lambda^P = 2 K_P [ (P_m - P_0m) + (P_n - P_0n) ].
 * Uses the cell geometry as of the last force/geometry computation.
 */
static Dscalar passiveTensionOfEdge(const ArrayHandle<Dscalar2> &h_moduli,
                                    const ArrayHandle<Dscalar2> &h_areaPeri,
                                    const ArrayHandle<Dscalar2> &h_pref,
                                    int cellM,
                                    int cellN)
{
    return 2.0 * h_moduli.data[cellM].y * (h_areaPeri.data[cellM].y - h_pref.data[cellM].y)
         + 2.0 * h_moduli.data[cellN].y * (h_areaPeri.data[cellN].y - h_pref.data[cellN].y);
}

// =====================================================================
// Cell-level preferences and moduli
// =====================================================================

/*!
 * Give every cell the same target area A0 and target perimeter P0.
 */
void Simple2DCell::setCellPreferencesUniform(Dscalar A0, Dscalar P0)
{
    AreaPeriPreferences.resize(Ncells);
    ArrayHandle<Dscalar2> h_pref(AreaPeriPreferences, access_location::host, access_mode::overwrite);

    for (int cell = 0; cell < Ncells; ++cell)
    {
        h_pref.data[cell].x = A0;
        h_pref.data[cell].y = P0;
    }
}

/*!
 * Draw each cell's target perimeter from a normal distribution N(mean_P0, std_P0)
 * while keeping the target area fixed at A0. Negative draws are rejected and
 * redrawn, so the realised distribution is a truncated Gaussian; for the
 * (mean, std) ranges scanned the truncation is negligible.
 *
 * The width of this distribution is one of the two knobs (the other being the
 * edge-tension distribution below) that the drivers scan over in order to match
 * the experimentally measured cell shape and junction orientation statistics at
 * the onset of convergent extension.
 */
void Simple2DCell::setCellPreferencesNormalRand(Dscalar A0, Dscalar mean_P0, Dscalar std_P0, int randSeed)
{
    AreaPeriPreferences.resize(Ncells);
    ArrayHandle<Dscalar2> h_pref(AreaPeriPreferences, access_location::host, access_mode::overwrite);

    noiseSource randGen;
    randGen.setReproducible(true);
    randGen.setReproducibleSeed(randSeed);

    for (int cell = 0; cell < Ncells; ++cell)
    {
        Dscalar sampledP0 = randGen.getRealNormal(mean_P0, std_P0);
        while (sampledP0 < 0.0)
            sampledP0 = randGen.getRealNormal(mean_P0, std_P0);

        h_pref.data[cell].x = A0;
        h_pref.data[cell].y = sampledP0;
    }
}

/*!
 * Set the area and perimeter stiffnesses K_A, K_P for every cell.
 * The scalar members KA, KP are kept in sync with the per-cell array because
 * some routines read the scalars directly.
 */
void Simple2DCell::setModuliUniform(Dscalar newKA, Dscalar newKP)
{
    KA = newKA;
    KP = newKP;
    Moduli.resize(Ncells);

    ArrayHandle<Dscalar2> h_moduli(Moduli, access_location::host, access_mode::overwrite);
    for (int cell = 0; cell < Ncells; ++cell)
    {
        h_moduli.data[cell].x = newKA;
        h_moduli.data[cell].y = newKP;
    }
}

// =====================================================================
// Per-edge geometry and passive tension
// =====================================================================

/*!
 * Fill the per-edge arrays EdgeTensionPassive, EdgeLen and EdgeAngle for every
 * directed edge of the network.
 *
 *   EdgeTensionPassive[e] : Lambda^P
 *   EdgeLen[e]            : l_ij, the minimum-image edge length
 *   EdgeAngle[e]          : theta_ij = atan2(dy, dx), in (-pi, pi], measured
 *                           with respect to the x-axis
 *
 * This routine must be called whenever the vertex positions or the cell
 * geometry have changed and an up-to-date passive tension is needed -- in
 * particular before every update of the local rules below.
 */
void VertexQuadraticEnergy::ComputeEdgeCharacteristics()
{
    ArrayHandle<Dscalar2> h_moduli(Moduli, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_areaPeri(AreaPeri, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_pref(AreaPeriPreferences, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_vertexPos(vertexPositions, access_location::host, access_mode::read);
    ArrayHandle<int> h_vn(vertexNeighbors, access_location::host, access_mode::read);
    ArrayHandle<int> h_vcn(vertexCellNeighbors, access_location::host, access_mode::read);

    ArrayHandle<Dscalar> h_passiveTension(EdgeTensionPassive, access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar> h_edgeLen(EdgeLen, access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar> h_edgeAngle(EdgeAngle, access_location::host, access_mode::overwrite);

    for (int v1 = 0; v1 < Nvertices; ++v1)
    {
        for (int dir = 0; dir < 3; ++dir)
        {
            int v2 = h_vn.data[3 * v1 + dir];

            int sharedCells[2];
            int numShared = sharedCellsOfEdge(h_vcn, v1, v2, sharedCells);

            Dscalar passiveTension = 0.0;
            if (numShared == 2)
                passiveTension = passiveTensionOfEdge(h_moduli, h_areaPeri, h_pref,
                                                      sharedCells[0], sharedCells[1]);

            // Minimum-image edge vector v2 - v1, so that periodic wrapping is handled.
            Dscalar2 edgeVector;
            Box->minDist(h_vertexPos.data[v2], h_vertexPos.data[v1], edgeVector);

            h_passiveTension.data[3 * v1 + dir] = passiveTension;
            h_edgeLen.data[3 * v1 + dir] = sqrt(edgeVector.x * edgeVector.x + edgeVector.y * edgeVector.y);
            h_edgeAngle.data[3 * v1 + dir] = atan2(edgeVector.y, edgeVector.x);
        }
    }
}

// =====================================================================
// Initialization of the active edge tensions
// =====================================================================

/*!
 * Initialize the active tensions to the current passive tensions,
 * Lambda_ij = Lambda^P_ij. Applied to a force-balanced tissue this doubles the
 * total tension on every junction; the drivers use it only as the starting
 * point of the initial-condition search, which then overwrites the tensions
 * with random draws.
 */
void VertexQuadraticEnergy::InitializeEdgeTensions()
{
    ArrayHandle<Dscalar2> h_moduli(Moduli, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_areaPeri(AreaPeri, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_pref(AreaPeriPreferences, access_location::host, access_mode::read);
    ArrayHandle<int> h_vn(vertexNeighbors, access_location::host, access_mode::read);
    ArrayHandle<int> h_vcn(vertexCellNeighbors, access_location::host, access_mode::read);
    ArrayHandle<Dscalar> h_edgeTension(EdgeTension, access_location::host, access_mode::overwrite);

    for (int v1 = 0; v1 < Nvertices; ++v1)
    {
        for (int dir = 0; dir < 3; ++dir)
        {
            int v2 = h_vn.data[3 * v1 + dir];

            int sharedCells[2];
            int numShared = sharedCellsOfEdge(h_vcn, v1, v2, sharedCells);

            Dscalar tension = 0.0;
            if (numShared == 2)
                tension = passiveTensionOfEdge(h_moduli, h_areaPeri, h_pref,
                                               sharedCells[0], sharedCells[1]);

            h_edgeTension.data[3 * v1 + dir] = tension;
        }
    }
}

/*!
 * Draw one active tension per *junction* from N(mean, std), rejecting negative
 * values, and write it to both directed copies of that junction.
 */
void VertexQuadraticEnergy::InitializeEdgeTensionsNormalRand(Dscalar mean, Dscalar std, int randSeed)
{
    ArrayHandle<int> h_vn(vertexNeighbors, access_location::host, access_mode::read);
    ArrayHandle<Dscalar> h_edgeTension(EdgeTension, access_location::host, access_mode::overwrite);

    noiseSource randGen;
    randGen.setReproducible(true);
    randGen.setReproducibleSeed(randSeed);

    for (int v1 = 0; v1 < Nvertices; ++v1)
    {
        for (int dir = 0; dir < 3; ++dir)
        {
            int v2 = h_vn.data[3 * v1 + dir];

            // Visit each undirected junction once. Because v1 runs in ascending
            // order this reaches every junction exactly once, at its lower-index
            // endpoint.
            if (v2 <= v1)
                continue;

            Dscalar sampledTension = randGen.getRealNormal(mean, std);
            while (sampledTension < 0.0)
                sampledTension = randGen.getRealNormal(mean, std);

            int reverseIndex = directedEdgeIndex(h_vn, v2, v1);
            if (reverseIndex < 0)
                continue;

            h_edgeTension.data[3 * v1 + dir] = sampledTension;
            h_edgeTension.data[reverseIndex] = sampledTension;
        }
    }
}

/*!
 * The drivers call this every tuning step and feed the result to
 * SetT1EdgeTension(): after a T1 transition the newly created junction has its
 * tension reset to the current network average <T>, which prevents the
 * repeated back-and-forth flipping of a single short edge.
 */
Dscalar VertexQuadraticEnergy::ReturnAverageEdgeTensions()
{
    ArrayHandle<Dscalar> h_edgeTension(EdgeTension, access_location::host, access_mode::read);

    Dscalar tensionSum = 0.0;
    int edgeCount = 0;
    for (int edge = 0; edge < 3 * Nvertices; ++edge)
    {
        tensionSum += h_edgeTension.data[edge];
        ++edgeCount;
    }

    return (edgeCount > 0) ? (tensionSum / (1.0 * edgeCount)) : 0.0;
}

// =====================================================================
// Observables
// =====================================================================

/*!
 * Fraction f_v of junctions whose orientation lies within theta_c_rad of the
 * y-axis (the DV axis of the embryo).
 *
 * Edge angles are first folded into [0, pi) so that the two directed copies of
 * a junction, whose angles differ by pi, are counted identically; the
 * normalisation over all 3*Nvertices directed edges is therefore the same as a
 * normalisation over junctions.
 */
Dscalar VertexQuadraticEnergy::ComputeFractionVerticalEdges(Dscalar theta_c_rad)
{
    // Refresh EdgeAngle for the current vertex positions.
    ComputeEdgeCharacteristics();

    ArrayHandle<Dscalar> h_edgeAngle(EdgeAngle, access_location::host, access_mode::read);

    Dscalar countVertical = 0.0;
    Dscalar countAll = 0.0;
    for (int edge = 0; edge < 3 * Nvertices; ++edge)
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

// =====================================================================
// Externally imposed deformation
// =====================================================================

/*!
 * Apply one increment of affine pure shear, taking the tissue from a state at
 * strain PureShearStrainPre to a state at strain PureShearStrain.
 *
 * The periodic box is reshaped to
 *
 *      Lx = sqrt(Ncells) * (1 + eps),   Ly = sqrt(Ncells) / (1 + eps)
 */
void VertexQuadraticEnergy::AffinePureShearStrainBoxReshapeStepwise(Dscalar PureShearStrainPre,
                                                                    Dscalar PureShearStrain)
{
    ArrayHandle<Dscalar2> h_vertexPos(vertexPositions, access_location::host, access_mode::readwrite);

    // Keep the off-diagonal box components; only the diagonal is rescaled.
    Dscalar x11, x12, x21, x22;
    Box->getBoxDims(x11, x12, x21, x22);
    Box->setGeneral(sqrt(Ncells) * (1.0 + PureShearStrain),
                    x12,
                    x21,
                    sqrt(Ncells) / (1.0 + PureShearStrain));

    Dscalar strainRatio = (1.0 + PureShearStrain) / (1.0 + PureShearStrainPre);
    for (int vertex = 0; vertex < Nvertices; ++vertex)
    {
        h_vertexPos.data[vertex].x *= strainRatio;
        h_vertexPos.data[vertex].y /= strainRatio;
        Box->putInBoxReal(h_vertexPos.data[vertex]);
    }
}
