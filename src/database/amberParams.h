// filename: amberParams.h
//
// Reader for Amber's native parameter formats: the main parm*.dat tables and
// frcmod.* patch files.
//
// protcad previously carried hand-transcribed copies of the parts of the force
// field it used -- data/amberVDW.frc for the 12-6 terms, data/amber.prep for
// types and connectivity.  A transcription cannot be diffed against upstream,
// and this one had drifted: the polar hydrogens H, HO and HS all carried radii
// that no Amber release has ever specified, which put several kcal/mol of
// spurious repulsion on every hydrogen bond in the structure.  Reading the
// distributed files directly removes the class of error rather than the
// instance.
//
// One file is loaded as the base and any number of frcmod files are layered
// over it, later definitions replacing earlier ones, which is what tleap does
// with loadamberparams.  Sections understood: MASS, BOND, ANGLE/ANGL, DIHE,
// IMPROPER/IMPR, the 10-12 hydrogen-bond block (skipped), the vdW
// equivalencing block, and MOD4/NONB.
//
// The dihedral tables are the reason this exists at all: protcad has no
// torsional term, so rotation about a sidechain chi is opposed only by
// long-range sterics and the rotamer wells that define a sidechain's accessible
// states are simply absent from the potential.

#include <string>
#include <vector>
#include <map>

#ifndef _AMBER_PARAMS_H
#define _AMBER_PARAMS_H

using namespace std;

// One term of a Fourier torsion series.
//
//   E = (barrier / divisor) * (1 + cos(periodicity * phi - phase))
//
// Amber writes a multi-term torsion as consecutive records with a negative
// periodicity on every record but the last; the sign is a continuation marker
// and is stripped here, so periodicity is always positive.
struct amberTorsionTerm
{
    double barrier;       // PK, kcal/mol, already the full barrier height
    double phase;         // radians
    double periodicity;   // PN, positive
    double divisor;       // IDIVF; 1.0 for impropers, which have no such field
    amberTorsionTerm() : barrier(0), phase(0), periodicity(0), divisor(1) {}
};

struct amberVdwParam
{
    double radius;         // R*, the position of the minimum for a like pair
    double epsilon;        // well depth
    double polarizability; // from MASS, used by the solvation model
    bool   defined;
    amberVdwParam() : radius(0), epsilon(0), polarizability(0), defined(false) {}
};

struct amberBondParam  { double k, r0; };
struct amberAngleParam { double k, theta0; };  // theta0 in radians

class amberParams
{
public:
    amberParams();
    ~amberParams();

    // Load data/amber/parm10.dat plus data/amber/frcmod.ff14SB.
    void loadFF14SB();

    void loadParm(const string& _fileName);     // base table
    void loadFrcmod(const string& _fileName);   // patch layered on top

    bool   hasVdw(const string& _type) const;
    const  amberVdwParam& vdw(const string& _type) const;
    vector<string> vdwTypes() const;

    // Torsions.  Lookup tries the exact quartet in both directions, then the
    // wildcard form X-B-C-X, again in both directions, which is the order tleap
    // uses.  Returns an empty vector when the quartet has no parameters.
    const vector<amberTorsionTerm>& torsion(const string& a, const string& b,
                                            const string& c, const string& d) const;
    const vector<amberTorsionTerm>& improper(const string& a, const string& b,
                                             const string& c, const string& d) const;

    bool bond(const string& a, const string& b, amberBondParam& out) const;
    bool angle(const string& a, const string& b, const string& c,
               amberAngleParam& out) const;

    size_t torsionCount() const { return itsTorsions.size(); }
    size_t improperCount() const { return itsImpropers.size(); }
    size_t bondCount() const { return itsBonds.size(); }
    size_t angleCount() const { return itsAngles.size(); }

    // Amber's 1-4 scale factors for this force field.  ff14SB inherits the
    // Cornell values.  These belong with the dihedral parameters and not in a
    // separate header, because the torsion barriers were fitted with the 1-4
    // terms scaled by exactly these numbers -- using one without the other
    // means using the parameters outside the fit that produced them.
    static double scee() { return 1.2; }   // divides 1-4 electrostatics
    static double scnb() { return 2.0; }   // divides 1-4 van der Waals

    const vector<string>& warnings() const { return itsWarnings; }

private:
    void   parseFile(const string& _path, bool _isFrcmod);
    void   applyEquivalences();
    string dataPath(const string& _fileName) const;

    // Section handlers.  Each takes one already-trimmed line.
    void   readMass(const string& line);
    void   readBond(const string& line);
    void   readAngle(const string& line);
    void   readTorsion(const string& line, bool improper);
    void   readNonbon(const string& line);
    void   readEquivalence(const string& line);

    static string trimType(const string& s);
    static string key2(const string& a, const string& b);
    static string key3(const string& a, const string& b, const string& c);
    static string key4(const string& a, const string& b, const string& c,
                       const string& d);

    map<string, amberVdwParam>  itsVdw;
    map<string, amberBondParam> itsBonds;
    map<string, amberAngleParam> itsAngles;
    map<string, vector<amberTorsionTerm> > itsTorsions;
    map<string, vector<amberTorsionTerm> > itsImpropers;

    // Deferred vdW equivalencing: parm files may declare that a set of types
    // shares another type's 12-6 parameters.  Recorded during parsing and
    // resolved afterwards, since the referenced type may appear later.
    vector<pair<string, string> > itsEquivalences;   // (alias, source)

    // Accumulator for a multi-term torsion.  A record with negative
    // periodicity means the series continues on the next record.
    string                   itsPendingKey;
    vector<amberTorsionTerm> itsPendingTerms;
    bool                     itsPendingImproper;

    vector<string> itsWarnings;
    vector<amberTorsionTerm> itsEmpty;
};

#endif
