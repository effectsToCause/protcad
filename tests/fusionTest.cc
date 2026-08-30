// fusionTest -- is the energy monotonically repulsive as two atoms fuse?
//
// The clash pre-filter that protMinCU used to carry was not a speed
// optimisation.  Its purpose was physical: to veto conformations that cannot
// occur in solution because closed-shell electron clouds repel, a constraint
// that a smooth r^-12 term only approximates.
//
// That approximation can fail in a specific way, and this energy function has
// exactly the machinery to make it fail.  The Lennard-Jones penalty is large
// but finite and smooth, so a Metropolis test at finite temperature will accept
// an overlapped pair whenever some other term pays for it.  Crowding raises
// local occupancy, which raises the effective dielectric, which screens
// electrostatics and shifts the Born/Gill solvation terms.  If that shift is
// favourable and can outrun r^-12 over any interval, the minimiser will find
// it and drive atoms together for solvation credit -- an artefact, not physics.
//
// So the question is not whether clashed structures score badly.  It is
// whether the total is *monotonically* repulsive all the way in.  Any interval
// where dE/dr > 0 at short range is an attractive channel into fusion, and a
// walker that reaches it is captured.  A finite barrier is nearly as bad: a
// finite-temperature walk tunnels through it eventually.
//
// Two things this scan must not do, both of which it originally did.
//
// It must not flag the attractive branch.  Down to the minimum of the total the
// energy is *supposed* to fall on approach; that is what makes it a bound pair,
// not a fusion channel.  Note that r = R_i + R_j is the LJ minimum itself, not
// the onset of repulsion, and Coulomb attraction shifts the minimum of the
// total further in still, so "inside contact" is the wrong gate and a scan grid
// that straddles the minimum will trip it.  The property that actually matters
// is single-basin: from the minimum of the total, inward, the energy must rise
// all the way.  That is what is checked.
//
// It must not scan inside energyParams::minSeparation.  The kernel pins the
// pair distance at that floor to keep 1/r finite, so within it the scanned
// pair contributes a constant while the moving atom still recedes from its
// other neighbours.  The total therefore falls for a purely geometric reason
// and monotonicity in r is not even defined there.  The floor is where the
// model stops making a statement, so it is where the scan stops.
//
// Finally, a drop is only interesting if a walker could reach it.  Metropolis
// acceptance is exp(-dE/kT); a channel sitting at 10^6 kcal/mol is not a
// sampling problem at any temperature this code will ever run.  The barrier
// that must be crossed to enter a channel is reported alongside it.
//
// Scanned here for the three cases that differ in sign structure:
//   opposite charges  -- electrostatics attracts, so this is the worst case
//   like charges      -- electrostatics repels, screening *helps* on approach
//   neutral           -- isolates vdW plus the nonpolar/entropy solvation terms
//
// Build:
//   nvcc -O3 -arch=sm_61 -Isrc/ensemble -x cu tests/fusionTest.cc \
//        src/ensemble/energy.cu -o /tmp/fusionTest

#include "energy.h"
#include <cstdio>
#include <vector>
#include <cmath>

static int failures = 0;

// Shared verdict for both scans.  Locates the minimum of the total over the
// samples, then requires that the energy rise monotonically from there inward
// to the floor.  A single basin followed by an unbroken climb is exactly the
// statement "there is no way into fusion"; anything else is a channel, and the
// energy at which it opens says whether a walker could ever reach it.
static void verdict(const std::vector<double>& r, const std::vector<double>& e)
{
    if (r.size() < 3) { printf("  too few samples to judge\n"); ++failures; return; }

    size_t imin = 0;
    for (size_t i = 1; i < e.size(); ++i) if (e[i] < e[imin]) imin = i;

    printf("  minimum of the total at r = %.2f A, E = %.4g kcal/mol\n",
           r[imin], e[imin]);

    // The minimum landing on the innermost sample is the worst case, not a
    // pass: it means the energy never turned over and the pair is downhill all
    // the way to the floor.  Checking only for falls *after* the minimum would
    // silently wave this through, since there is nothing after it.
    if (imin == e.size() - 1) {
        printf("  ENERGY FALLS ALL THE WAY TO THE FLOOR: the total never turns"
               " over, so there is no repulsive wall at all\n");
        ++failures;
        return;
    }

    int falls = 0; double firstR = 0, eAtFirst = 0, peak = 0, drop = 0;
    for (size_t i = imin + 1; i < e.size(); ++i) {
        if (e[i] < e[i - 1] - 1e-9) {
            if (!falls) { firstR = r[i]; eAtFirst = e[i]; peak = e[i - 1]; }
            drop += e[i - 1] - e[i];
            ++falls;
        }
    }

    if (falls) {
        printf("  ENERGY FALLS ON APPROACH at %d separations inside the minimum,"
               " first at r = %.2f A\n", falls, firstR);
        printf("    E there %.3g kcal/mol, barrier to reach it %.3g kcal/mol,"
               " cumulative drop %.4g\n", eAtFirst, peak, drop);
        ++failures;
    } else {
        printf("  single basin, then monotonically repulsive to the floor:"
               " no fusion channel\n");
    }
}

static void scan(const char* label, double q1, double q2)
{
    const int N = 2;
    double rad[2] = { 1.908, 1.6612 };          // AMBER C and O
    double eps[2] = { 0.1094, 0.2100 };
    double chg[2] = { q1, q2 };
    int    res[2] = { 0, 1 };                   // separate residues: not excluded
    unsigned char sil[2] = { 0, 0 };
    int    ec[2]  = { 0, 0 };
    std::vector<int> el(2, 0);

    energyTopology t;
    t.numAtoms = N; t.radius = rad; t.epsilon = eps; t.charge = chg;
    t.residueIndex = res; t.silent = sil;
    t.exclusionCount = ec; t.exclusionList = &el[0]; t.exclusionStride = 1;

    energyParams p = defaultEnergyParams();
    energyContext* ctx = energyCreate(t, p);
    if (!ctx) { printf("  energyCreate FAILED\n"); ++failures; return; }

    const double contact = rad[0] + rad[1];
    const double floorR  = p.minSeparation;

    printf("\n%s  (q1=%+.2f q2=%+.2f, contact = %.2f A, floor = %.2f A)\n",
           label, q1, q2, contact, floorR);
    printf("  %6s %12s %12s %12s %12s %12s %14s\n",
           "r(A)", "vdw", "elec", "solvPolar", "solvNonp", "solvEntr", "total");

    std::vector<double> rs, es;
    // Walk inward, stopping at the floor: inside it the kernel pins the pair
    // distance, so the curve there is an artefact rather than a statement.
    for (double r = 4.00; r >= floorR; r -= 0.05)
    {
        double x[2] = { 0.0, r }, y[2] = { 0, 0 }, z[2] = { 0, 0 };
        double total = 0; energyBreakdown b;
        if (energyCompute(ctx, x, y, z, &total, &b) != 0) {
            printf("  energyCompute FAILED: %s\n", energyLastError(ctx));
            ++failures; energyDestroy(ctx); return;
        }
        bool show = (r > 3.99) || (r < 2.55 && r > 2.45) || (r < 2.05 && r > 1.95)
                 || (r < 1.55 && r > 1.45) || (r < 1.05 && r > 0.95)
                 || (r < 0.55 && r > 0.45) || (r < 0.25);
        if (show)
            printf("  %6.2f %12.4g %12.4g %12.4g %12.4g %12.4g %14.6g\n",
                   r, b.vdw, b.electrostatic, b.solvationPolar,
                   b.solvationNonpolar, b.solvationEntropy, total);
        rs.push_back(r); es.push_back(total);
    }

    verdict(rs, es);
    energyDestroy(ctx);
}


// ---------------------------------------------------------------------------
// The two-atom scan above cannot see the mechanism of interest: with only two
// atoms the shell occupancy is already saturated at 4 A, so the dielectric
// never responds and the solvation terms are constant.  The compensation
// channel only exists in a crowded environment, where pushing atom 0 into its
// neighbour changes the occupancy -- and therefore the effective dielectric,
// the screening, and the Born term -- of every atom in the neighbourhood.
//
// This scan builds a dense jittered lattice, then marches atom 0 along the
// vector toward atom 1 while everything else stays fixed.  If the solvation
// terms move at all, the channel is live and the question of whether it can
// outrun r^-12 is meaningful.  If the *total* ever falls on approach inside
// contact, the minimiser has a way to buy overlap with solvation credit and
// the clash pre-filter was doing real physical work.
static void scanDense(const char* label, double qScale)
{
    const int side = 4, N = side * side * side;      // 64 atoms, ~3.3 A spacing
    std::vector<double> x(N), y(N), z(N), rad(N), eps(N), chg(N);
    std::vector<int> res(N), ec(N, 0), el(N, 1);
    std::vector<unsigned char> sil(N, 0);
    unsigned seed = 12345;
    for (int i = 0; i < N; ++i) {
        int ix = i % side, iy = (i / side) % side, iz = i / (side * side);
        seed = seed * 1103515245u + 12345u;
        double j1 = ((seed >> 16) & 0x7fff) / 32767.0 - 0.5;
        seed = seed * 1103515245u + 12345u;
        double j2 = ((seed >> 16) & 0x7fff) / 32767.0 - 0.5;
        seed = seed * 1103515245u + 12345u;
        double j3 = ((seed >> 16) & 0x7fff) / 32767.0 - 0.5;
        x[i] = ix * 3.3 + 0.3 * j1;
        y[i] = iy * 3.3 + 0.3 * j2;
        z[i] = iz * 3.3 + 0.3 * j3;
        rad[i] = 1.8; eps[i] = 0.12;
        chg[i] = qScale * ((i % 2) ? 0.5 : -0.5);
        res[i] = i / 4;
    }
    energyTopology t;
    t.numAtoms = N; t.radius = &rad[0]; t.epsilon = &eps[0]; t.charge = &chg[0];
    t.residueIndex = &res[0]; t.silent = &sil[0];
    t.exclusionCount = &ec[0]; t.exclusionList = &el[0]; t.exclusionStride = 1;

    energyParams p = defaultEnergyParams();
    energyContext* ctx = energyCreate(t, p);
    if (!ctx) { printf("  energyCreate FAILED\n"); ++failures; return; }

    const double floorR = p.minSeparation;

    // Target is the lattice neighbour of atom 0 along +x.
    double x0 = x[0], y0 = y[0], z0 = z[0];
    double dx = x[1] - x0, dy = y[1] - y0, dz = z[1] - z0;
    double d0 = sqrt(dx*dx + dy*dy + dz*dz);
    dx /= d0; dy /= d0; dz /= d0;

    printf("\n%s  (dense lattice, N=%d, start separation %.2f A, contact %.2f A,"
           " floor %.2f A)\n", label, N, d0, 2 * rad[0], floorR);
    printf("  %6s %12s %12s %12s %12s %12s %14s\n",
           "r(A)", "vdw", "elec", "solvPolar", "solvNonp", "solvEntr", "total");

    double sp0 = 0, snp0 = 0, se0 = 0; bool havePrev = false;
    std::vector<double> rs, es;
    for (double r = d0; r >= floorR; r -= 0.05)
    {
        x[0] = x[1] - dx * r; y[0] = y[1] - dy * r; z[0] = z[1] - dz * r;
        double total = 0; energyBreakdown b;
        if (energyCompute(ctx, &x[0], &y[0], &z[0], &total, &b) != 0) {
            printf("  energyCompute FAILED: %s\n", energyLastError(ctx));
            ++failures; energyDestroy(ctx); return;
        }
        if (!havePrev) { sp0 = b.solvationPolar; snp0 = b.solvationNonpolar; se0 = b.solvationEntropy; }
        if (r > d0 - 0.01 || fabs(r - 3.0) < 0.025 || fabs(r - 2.5) < 0.025 ||
            fabs(r - 2.0) < 0.025 || fabs(r - 1.5) < 0.025 ||
            fabs(r - 1.0) < 0.025 || fabs(r - 0.5) < 0.025 || r < 0.35)
            printf("  %6.2f %12.4g %12.4g %12.4g %12.4g %12.4g %14.6g\n",
                   r, b.vdw, b.electrostatic, b.solvationPolar,
                   b.solvationNonpolar, b.solvationEntropy, total);
        rs.push_back(r); es.push_back(total);
        havePrev = true;
    }
    // Did the dielectric machinery respond at all?
    double total = 0; energyBreakdown b;
    x[0] = x[1] - dx * floorR; y[0] = y[1] - dy * floorR; z[0] = z[1] - dz * floorR;
    energyCompute(ctx, &x[0], &y[0], &z[0], &total, &b);
    printf("  solvation response over the scan: polar %+.4g  nonpolar %+.4g  entropy %+.4g\n",
           b.solvationPolar - sp0, b.solvationNonpolar - snp0, b.solvationEntropy - se0);
    if (fabs(b.solvationPolar - sp0) + fabs(b.solvationNonpolar - snp0)
      + fabs(b.solvationEntropy - se0) < 1e-9)
        printf("  WARNING: solvation still frozen -- this scan did not probe the channel\n");

    verdict(rs, es);
    energyDestroy(ctx);
}

int main()
{
    printf("fusionTest: short-range monotonicity of the total energy\n");
    scan("opposite charges", +0.85, -0.85);
    scan("like charges",     +0.85, +0.85);
    scan("neutral",           0.00,  0.00);

    scanDense("dense, charged",   1.0);
    scanDense("dense, neutral",   0.0);

    printf("\nRESULT: %s (%d failure%s)\n",
           failures ? "FUSION CHANNEL PRESENT" : "no fusion channel",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
