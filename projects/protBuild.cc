// protBuild -- build a chain de novo from a sequence, using mol.lib geometry.
//
//   protBuild <sequence> <out.pdb> [phi] [psi]
//
// sequence is one-letter codes, e.g. "GLG".  phi/psi default to the extended
// beta region (-139, 135), which is where an unfolded chain spends most of its
// time and so is the right place to start a reference-state sampler.
//
// Written for the ddG reference state.  A mutation changes the atom count, so
// E(mut) - E(wt) carries the self-energy of the deleted atoms -- tens of
// kcal/mol that have nothing to do with stability.  It only cancels if the
// difference is taken against an unfolded state,
//
//     ddG = [G_f(mut) - G_u(mut)] - [G_f(wt) - G_u(wt)]
//
// so the self-energy cancels inside each bracket rather than between them.
// That reference has to come from this same forcefield and sampler, because
// what makes it work is common-mode cancellation of model error; a tabulated
// value computed under someone else's forcefield cancels nothing.
//
// Gly-X-Gly is the immediate use, but nothing here is tripeptide-specific.
//
// Construction: every residue is instantiated from its mol.lib template, so
// all bond lengths, bond angles and sidechain geometry are protcad's own and
// this file never invents any.  Residue i+1 is then placed as a rigid body,
// which cannot distort what the template supplied -- only the three torsions
// between residues are imposed here.
//
// Note that residue::setPhi and the atom tree are both residue-local (linkage
// is built per residue in residue.cc), so neither can place a *following*
// residue.  That is why placement is done by explicit frame construction
// rather than by building the chain and dialling in torsions afterwards.

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "ensemble.h"
#include "PDBInterface.h"

// Inter-residue geometry (Engh & Huber).  Only the peptide bond itself is
// hardcoded; everything inside a residue is read back from its template.
static const double kBondCN   = 1.329;   // C(i) - N(i+1)
static const double kAngCACN  = 116.2;   // CA(i) - C(i) - N(i+1)
static const double kAngCNCA  = 121.7;   // C(i) - N(i+1) - CA(i+1)
static const double kBondNH   = 1.010;   // N - H, amide
static const double kAngCNH   = 119.8;   // C(i) - N(i+1) - H
static const double kOmega    = 180.0;   // trans

typedef std::vector<double> vec3;

static vec3 mk(const dblVec& v) {vec3 r(3); for (int i = 0; i < 3; i++) {r[i] = v[i];} return r;}
static vec3 sub(const vec3& a, const vec3& b) {vec3 r(3); for (int i = 0; i < 3; i++) {r[i] = a[i] - b[i];} return r;}
static double dot(const vec3& a, const vec3& b) {return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];}
static vec3 cross(const vec3& a, const vec3& b)
{	vec3 r(3);
	r[0] = a[1]*b[2] - a[2]*b[1];
	r[1] = a[2]*b[0] - a[0]*b[2];
	r[2] = a[0]*b[1] - a[1]*b[0];
	return r;
}
static vec3 norm(const vec3& a)
{	double n = sqrt(dot(a, a));
	vec3 r(3); for (int i = 0; i < 3; i++) {r[i] = a[i] / n;}
	return r;
}
static double dist(const vec3& a, const vec3& b) {return sqrt(dot(sub(a, b), sub(a, b)));}

// Angle a-b-c in degrees.
static double angleOf(const vec3& a, const vec3& b, const vec3& c)
{	vec3 u = norm(sub(a, b)), v = norm(sub(c, b));
	double d = dot(u, v);
	if (d > 1.0) {d = 1.0;} if (d < -1.0) {d = -1.0;}
	return acos(d) * 180.0 / M_PI;
}

// Place a fourth atom from three known ones, given bond/angle/torsion.
static vec3 nerf(const vec3& a, const vec3& b, const vec3& c,
                 double bond, double angleDeg, double torsionDeg)
{	double ang = angleDeg * M_PI / 180.0, tor = torsionDeg * M_PI / 180.0;
	vec3 bc = norm(sub(c, b));
	vec3 n  = norm(cross(sub(b, a), bc));
	vec3 nbc = cross(n, bc);
	double d0 = -bond * cos(ang);
	double d1 =  bond * sin(ang) * cos(tor);
	double d2 =  bond * sin(ang) * sin(tor);
	vec3 r(3);
	for (int i = 0; i < 3; i++) {r[i] = c[i] + bc[i]*d0 + nbc[i]*d1 + n[i]*d2;}
	return r;
}

// Torsion p0-p1-p2-p3 in degrees.  b0 runs p0->p1: reversing it negates n1,
// which shifts the result by 180 in a way that is self-consistent and so will
// not show up in any check written with the same convention.
static double dihedralOf(const vec3& p0, const vec3& p1, const vec3& p2, const vec3& p3)
{	vec3 b0 = sub(p1, p0), b1 = sub(p2, p1), b2 = sub(p3, p2);
	vec3 n1 = cross(b0, b1), n2 = cross(b1, b2), m = cross(n1, b1);
	double lb1 = sqrt(dot(b1, b1));
	return -atan2(dot(m, n2) / lb1, dot(n1, n2)) * 180.0 / M_PI;
}

// Rigidly move a residue so its N/CA/C land on the supplied targets.
//
// Only three points are involved, so the frame is built directly instead of
// running a general superposition: e1 along N->CA, e2 the part of CA->C
// perpendicular to it, e3 their cross product.  The map between two such
// frames is exactly rigid, which is the property that matters -- it guarantees
// the template's internal geometry survives placement untouched.
static void placeResidue(residue* pRes, UInt iN, UInt iCA, UInt iC,
                         const vec3& tN, const vec3& tCA, const vec3& tC)
{
	vec3 sN = mk(pRes->getCoords(iN)), sCA = mk(pRes->getCoords(iCA)), sC = mk(pRes->getCoords(iC));

	vec3 se1 = norm(sub(sCA, sN));
	vec3 sv  = sub(sC, sCA);
	double sp = dot(sv, se1);
	vec3 se2(3); for (int i = 0; i < 3; i++) {se2[i] = sv[i] - sp*se1[i];}
	se2 = norm(se2);
	vec3 se3 = cross(se1, se2);

	vec3 te1 = norm(sub(tCA, tN));
	vec3 tv  = sub(tC, tCA);
	double tp = dot(tv, te1);
	vec3 te2(3); for (int i = 0; i < 3; i++) {te2[i] = tv[i] - tp*te1[i];}
	te2 = norm(te2);
	vec3 te3 = cross(te1, te2);

	for (UInt a = 0; a < pRes->getNumAtoms(); a++)
	{	vec3 p = sub(mk(pRes->getCoords(a)), sCA);
		double c1 = dot(p, se1), c2 = dot(p, se2), c3 = dot(p, se3);
		dblVec out(3);
		for (int i = 0; i < 3; i++) {out[i] = tCA[i] + c1*te1[i] + c2*te2[i] + c3*te3[i];}
		pRes->setCoords(a, out);
	}
}

static int findAtom(residue* pRes, const string& name)
{	for (UInt a = 0; a < pRes->getNumAtoms(); a++)
	{	if (pRes->getAtomName(a) == name) {return (int)a;}
	}
	return -1;
}

// One-letter code to mol.lib name.  Looked up by name rather than by a
// hardcoded index so that reordering mol.lib cannot silently build the wrong
// sequence.  HIS maps to HIE, the neutral tautomer.
static string threeLetter(char c)
{	switch (toupper(c))
	{	case 'A': return "ALA"; case 'R': return "ARG"; case 'N': return "ASN";
		case 'D': return "ASP"; case 'C': return "CYS"; case 'Q': return "GLN";
		case 'E': return "GLU"; case 'H': return "HIE"; case 'I': return "ILE";
		case 'L': return "LEU"; case 'K': return "LYS"; case 'M': return "MET";
		case 'F': return "PHE"; case 'P': return "PRO"; case 'S': return "SER";
		case 'T': return "THR"; case 'W': return "TRP"; case 'Y': return "TYR";
		case 'V': return "VAL"; case 'G': return "GLY";
		default: return "";
	}
}

static int typeIndexOf(const string& name)
{	for (UInt i = 0; i < residue::dataBase.size(); i++)
	{	if (residue::dataBase[i].getName() == name) {return (int)i;}
	}
	return -1;
}

int main(int argc, char* argv[])
{
	if (argc < 3)
	{	cout << "protBuild <sequence> <out.pdb> [phi] [psi]" << endl;
		cout << "  sequence is one-letter codes, e.g. GLG" << endl;
		cout << "  phi/psi default to -139/135 (extended)" << endl;
		return 1;
	}

	string seq = argv[1];
	string outfile = argv[2];
	double phi = (argc > 3) ? atof(argv[3]) : -139.0;
	double psi = (argc > 4) ? atof(argv[4]) :  135.0;

	residue::setupDataBase(true);

	protein* prot = new protein("built");
	chain* pChain = new chain('A');

	vec3 prevN(3), prevCA(3), prevC(3);
	vector<residue*> placed;

	for (UInt r = 0; r < seq.size(); r++)
	{
		string name = threeLetter(seq[r]);
		if (name == "")
		{	cout << "protBuild: unknown one-letter code '" << seq[r] << "'" << endl;
			return 1;
		}
		int type = typeIndexOf(name);
		if (type < 0)
		{	cout << "protBuild: " << name << " not present in mol.lib" << endl;
			return 1;
		}

		residue* pRes = new residue((UInt)type, true);
		int iN = findAtom(pRes, "N"), iCA = findAtom(pRes, "CA"), iC = findAtom(pRes, "C");
		if (iN < 0 || iCA < 0 || iC < 0)
		{	cout << "protBuild: " << name << " template is missing a backbone atom" << endl;
			return 1;
		}

		vec3 tN, tCA, tC;
		if (r == 0)
		{	// First residue defines the frame; leave the template where it is.
			tN  = mk(pRes->getCoords(iN));
			tCA = mk(pRes->getCoords(iCA));
			tC  = mk(pRes->getCoords(iC));
		}
		else
		{	// Read this template's own N-CA, CA-C and N-CA-C rather than
			// assuming ideal values, so the rigid placement below has nothing
			// left to distort.
			double bNCA = dist(mk(pRes->getCoords(iN)),  mk(pRes->getCoords(iCA)));
			double bCAC = dist(mk(pRes->getCoords(iCA)), mk(pRes->getCoords(iC)));
			double aNCAC = angleOf(mk(pRes->getCoords(iN)), mk(pRes->getCoords(iCA)), mk(pRes->getCoords(iC)));

			tN  = nerf(prevN,  prevCA, prevC, kBondCN, kAngCACN, psi);
			tCA = nerf(prevCA, prevC,  tN,    bNCA,    kAngCNCA, kOmega);
			tC  = nerf(prevC,  tN,     tCA,   bCAC,    aNCAC,    phi);
		}

		placeResidue(pRes, (UInt)iN, (UInt)iCA, (UInt)iC, tN, tCA, tC);
		pRes->setResNum(r + 1);

		// The carbonyl O and the amide H are the two atoms whose position
		// depends on a torsion being imposed here, so the template's copies
		// are stale and have to be rebuilt.  Everything else is either
		// backbone (which defines the frame) or sidechain (chi-dependent,
		// carried rigidly, and explored by the sampler afterwards).
		if (r > 0)
		{	int prevO = findAtom(placed[r-1], "O");
			int iH = findAtom(pRes, "H");
			if (prevO >= 0 && iH >= 0)
			{	// Amide H sits anti to the preceding carbonyl O.
				vec3 h = nerf(mk(placed[r-1]->getCoords((UInt)prevO)), prevC, tN,
				              kBondNH, kAngCNH, 180.0);
				dblVec hv(3); for (int i = 0; i < 3; i++) {hv[i] = h[i];}
				pRes->setCoords((UInt)iH, hv);
			}
		}

		// chain::add, not chain::addResidue: add is the public entry point and
		// also registers the chainPosition and secondaryStructure that the
		// rest of the ensemble expects every residue to have.
		pChain->add(pRes);
		placed.push_back(pRes);
		prevN = tN; prevCA = tCA; prevC = tC;
	}

	// O depends on psi, which is only defined once the next residue exists.
	for (UInt r = 0; r + 1 < placed.size(); r++)
	{	int iO = findAtom(placed[r], "O");
		int iN = findAtom(placed[r], "N"), iCA = findAtom(placed[r], "CA"), iC = findAtom(placed[r], "C");
		if (iO < 0) {continue;}
		// Anti to the following N, i.e. psi + 180 about CA->C.
		vec3 o = nerf(mk(placed[r]->getCoords((UInt)iN)), mk(placed[r]->getCoords((UInt)iCA)),
		              mk(placed[r]->getCoords((UInt)iC)), 1.231, 120.8, psi + 180.0);
		dblVec ov(3); for (int i = 0; i < 3; i++) {ov[i] = o[i];}
		placed[r]->setCoords((UInt)iO, ov);
	}

	prot->add(pChain);
	pdbWriter(prot, outfile);

	// Report the torsions actually built.  Cheap, and it turns a silent
	// geometry error into a visible one at the point of construction.
	for (UInt r = 0; r < placed.size(); r++)
	{	int iN = findAtom(placed[r], "N"), iCA = findAtom(placed[r], "CA"), iC = findAtom(placed[r], "C");
		double gotPhi = 0.0, gotPsi = 0.0, gotOmega = 0.0;
		bool hasPhi = false, hasPsi = false;
		if (r > 0)
		{	int pC = findAtom(placed[r-1], "C");
			gotPhi = dihedralOf(mk(placed[r-1]->getCoords((UInt)pC)), mk(placed[r]->getCoords((UInt)iN)),
			                    mk(placed[r]->getCoords((UInt)iCA)), mk(placed[r]->getCoords((UInt)iC)));
			int pCA = findAtom(placed[r-1], "CA");
			gotOmega = dihedralOf(mk(placed[r-1]->getCoords((UInt)pCA)), mk(placed[r-1]->getCoords((UInt)pC)),
			                      mk(placed[r]->getCoords((UInt)iN)), mk(placed[r]->getCoords((UInt)iCA)));
			hasPhi = true;
		}
		if (r + 1 < placed.size())
		{	int nN = findAtom(placed[r+1], "N");
			gotPsi = dihedralOf(mk(placed[r]->getCoords((UInt)iN)), mk(placed[r]->getCoords((UInt)iCA)),
			                    mk(placed[r]->getCoords((UInt)iC)), mk(placed[r+1]->getCoords((UInt)nN)));
			hasPsi = true;
		}
		cout << "res " << r+1 << " " << residue::dataBase[placed[r]->getTypeIndex()].getName();
		if (hasPhi) {cout << "  phi " << gotPhi << "  omega " << gotOmega;}
		if (hasPsi) {cout << "  psi " << gotPsi;}
		cout << endl;
	}

	cout << "protBuild: wrote " << outfile << " (" << seq.size() << " residues)" << endl;
	return 0;
}
