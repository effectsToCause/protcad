// filename: amberParamsTest.cc
//
// Checks the native Amber parameter reader against the file it replaces and
// against counts taken independently from the distributed tables.
//
// The point of reading upstream files directly is to make transcription errors
// impossible, so the interesting output here is not "pass" but the diff against
// data/amberVDW.frc: every disagreement is either a bug in the reader or an
// error in the table protcad has been using.

#include "amberParams.h"
#include "amberVDW.h"
#include "generalio.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <map>

using namespace std;

static int failures = 0;

static void check(bool ok, const string& what)
{
    cout << "  " << left << setw(52) << what << (ok ? "ok" : "FAIL") << endl;
    if (!ok) failures++;
}

int main()
{
    amberParams ff;
    ff.loadFF14SB();

    cout << "\n[table sizes]" << endl;
    cout << "  vdW types " << ff.vdwTypes().size()
         << ", bonds " << ff.bondCount() / 2
         << ", angles " << ff.angleCount() / 2
         << ", torsions " << ff.torsionCount()
         << ", impropers " << ff.improperCount() << endl;

    check(ff.vdwTypes().size() > 55, "vdW table populated");
    check(ff.torsionCount() > 200, "torsion table populated");
    check(ff.improperCount() > 30, "improper table populated");

    cout << "\n[spot checks against the distributed files]" << endl;

    // parm10 MOD4: "  N           1.8240  0.1700"
    check(fabs(ff.vdw("N").radius - 1.8240) < 1e-6 &&
          fabs(ff.vdw("N").epsilon - 0.1700) < 1e-9, "N vdW from MOD4");

    // frcmod.ff14SB NONB overrides / adds: "  2C  1.9080  0.1094"
    check(fabs(ff.vdw("2C").radius - 1.9080) < 1e-6 &&
          fabs(ff.vdw("2C").epsilon - 0.1094) < 1e-9, "2C vdW from frcmod NONB");

    // Equivalencing: NA is not in MOD4 and must inherit N.
    check(fabs(ff.vdw("NA").radius - ff.vdw("N").radius) < 1e-9 &&
          ff.vdw("NA").defined, "NA inherits N by equivalencing");

    // HO carries no vdW at all in any Amber release.  This is the single most
    // consequential number in the file: a nonzero radius here puts a repulsive
    // wall in the middle of every hydroxyl hydrogen bond.
    check(ff.vdw("HO").radius == 0.0 && ff.vdw("HO").epsilon == 0.0,
          "HO has zero vdW");

    // Polarizability comes from the MASS section.
    check(fabs(ff.vdw("C").polarizability - 0.616) < 1e-9,
          "C polarizability from MASS");

    // A four-term backbone torsion from frcmod.ff14SB:
    //   CT-CX-N -C  1 0.000 0.0 -4. / 0.800 0.0 -3. / 1.800 0.0 -2. / 2.000 0.0 1.
    {
        const vector<amberTorsionTerm>& t = ff.torsion("CT", "CX", "N", "C");
        bool ok = (t.size() == 4);
        if (ok) ok = fabs(t[3].barrier - 2.000) < 1e-9 &&
                     fabs(t[3].periodicity - 1.0) < 1e-9 &&
                     fabs(t[2].barrier - 1.800) < 1e-9 &&
                     fabs(t[2].periodicity - 2.0) < 1e-9 &&
                     fabs(t[0].periodicity - 4.0) < 1e-9;
        check(ok, "CT-CX-N-C is a 4-term series, signs stripped");
    }

    // Reverse lookup must find the same series.
    {
        const vector<amberTorsionTerm>& f = ff.torsion("CT", "CX", "N", "C");
        const vector<amberTorsionTerm>& r = ff.torsion("C", "N", "CX", "CT");
        check(f.size() == r.size() && f.size() == 4 &&
              fabs(f[3].barrier - r[3].barrier) < 1e-12,
              "torsion lookup is direction independent");
    }

    // Wildcard fallback: X-C-CT-X exists in parm10, the specific quartet does not.
    {
        const vector<amberTorsionTerm>& t = ff.torsion("O", "C", "CT", "HC");
        check(!t.empty(), "wildcard X-C-CT-X reached by fallback");
    }

    // A torsion that genuinely has no parameters must come back empty rather
    // than silently returning something adjacent.
    check(ff.torsion("ZZ", "ZZ", "ZZ", "ZZ").empty(), "unknown torsion is empty");

    // The disulfide torsion.  protein.cc retypes a bonded SG from the thiol
    // type SH to S before looking bonded parameters up, because protcad finds
    // disulfides geometrically and leaves the residue typed as reduced CYS.
    // If this quartet ever stopped resolving, every disulfide in every
    // structure would silently lose its torsional restraint.
    {
        const vector<amberTorsionTerm>& t = ff.torsion("CT", "S", "S", "CT");
        check(!t.empty(), "disulfide torsion CT-S-S-CT found");
        check(ff.torsion("SH", "SH", "SH", "SH").empty(),
              "thiol-typed disulfide quartet has no parameters");
    }

    // IDIVF is carried, not folded in: X-C-CT-X has IDIVF 6.
    {
        const vector<amberTorsionTerm>& t = ff.torsion("X", "C", "CT", "X");
        check(!t.empty() && fabs(t[0].divisor - 6.0) < 1e-9, "IDIVF preserved");
    }

    // Impropers are indexed on the third position and are otherwise symmetric.
    {
        const vector<amberTorsionTerm>& a = ff.improper("X", "X", "C", "O");
        check(!a.empty(), "improper X-X-C-O found");
    }

    cout << "\n[1-4 scaling]" << endl;
    check(fabs(amberParams::scee() - 1.2) < 1e-12 &&
          fabs(amberParams::scnb() - 2.0) < 1e-12, "SCEE 1.2 / SCNB 2.0");

    // ---------------------------------------------------------------
    // Diff against the hand-maintained table this replaces.
    // ---------------------------------------------------------------
    cout << "\n[diff vs data/amberVDW.frc]" << endl;
    string path = getEnvironmentVariable(string("PROTCADDIR")) + "/data/amberVDW.frc";
    ifstream in(path.c_str());
    if (!in) { cout << "  (not found, skipped)" << endl; }
    else
    {
        int same = 0, diff = 0, absent = 0;
        cout << "  " << left << setw(6) << "type"
             << right << setw(10) << "file R*" << setw(10) << "file eps"
             << setw(12) << "ff14SB R*" << setw(10) << "ff14SB eps" << endl;
        string line;
        while (getline(in, line))
        {
            if (line.empty()) continue;
            char c = line[0];
            if (c == '#' || c == '@' || c == '!' || c == '>') continue;
            istringstream ss(line);
            double flag, zero, r, e, pol, vol; string type;
            if (!(ss >> flag >> zero >> type >> r >> e >> pol >> vol)) continue;
            if (!ff.hasVdw(type)) { absent++; continue; }
            const amberVdwParam& p = ff.vdw(type);
            if (fabs(p.radius - r) < 1e-4 && fabs(p.epsilon - e) < 1e-6) same++;
            else
            {
                diff++;
                cout << "  " << left << setw(6) << type << right
                     << setw(10) << fixed << setprecision(4) << r
                     << setw(10) << setprecision(5) << e
                     << setw(12) << setprecision(4) << p.radius
                     << setw(10) << setprecision(5) << p.epsilon << endl;
            }
        }
        cout << "  agree " << same << ", differ " << diff
             << ", not in ff14SB " << absent << endl;
    }

    // ---------------------------------------------------------------
    // amberVDW now overlays ff14SB onto its own table.  The overlay must fix
    // the polar hydrogens without disturbing row order, because indices are
    // cached across the ensemble code and water is indexed by a literal.
    // ---------------------------------------------------------------
    cout << "\n[amberVDW overlay]" << endl;
    {
        amberVDW v;
        int iHO = v.getIndexFromNameString("HO");
        int iH  = v.getIndexFromNameString("H");
        int iN  = v.getIndexFromNameString("N");
        check(iHO >= 0 && iH >= 0 && iN >= 0, "polar H types still resolve");
        check(v.getRadius(iHO) == 0.0 && v.getEpsilon(iHO) == 0.0,
              "HO radius and epsilon now zero");
        check(fabs(v.getRadius(iH) - 0.6) < 1e-6, "H radius now 0.6");
        check(v.getVolume(iHO) == 0.0, "HO displaces no solvent");
        check(fabs(v.getRadius(iN) - 1.8240) < 1e-6, "N unchanged by overlay");

        // getWaterEnergy() indexes water as literal 54.  If the overlay ever
        // reorders rows this is the check that catches it.
        check(v.getIndexFromNameString("OW") == 54, "water still at index 54");

        // Volume must satisfy the file's own 4.18*R^3 relation everywhere.
        bool volOk = true;
        for (int i = 0; i < 54; ++i)
        {
            double r = v.getRadius(i);
            if (fabs(v.getVolume(i) - 4.18 * r * r * r) > 1e-3) volOk = false;
        }
        check(volOk, "volume consistent with 4.18*R^3");
    }

    const vector<string>& w = ff.warnings();    if (!w.empty())
    {
        cout << "\n[warnings]" << endl;
        for (size_t i = 0; i < w.size(); ++i) cout << "  " << w[i] << endl;
    }

    cout << "\nRESULT: " << failures << " failures" << endl;
    return failures ? 1 : 0;
}
