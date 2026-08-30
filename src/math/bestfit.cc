// bestfit.cc
//
// Optimal rigid superposition of two atom sets (Kabsch/McLachlan), replacing
// the F77 implementation this codebase carried since 1999.
//
// The old file was 576 lines: the fit itself, a 3x3 determinant, and a
// transcribed copy of LINPACK's dsvdc general m-by-n SVD used on nothing
// larger than a 3x3 covariance matrix.  Dragging a Fortran toolchain and a
// general-purpose SVD along for a fixed 3x3 problem was the only reason the
// build needed gfortran at all.
//
// This is a port rather than a redesign: the fit, the reflection convention and
// the two degenerate branches all follow the original.  dsvdc is replaced by a
// one-sided Jacobi SVD, which for a fixed 3x3 converges in a few sweeps.
//
// The interface keeps C linkage and the trailing-underscore name so that every
// existing call site, which passes Fortran-style pointers, is unchanged.
//
// Three defects in the original are fixed here, all of them in rank-deficient
// fits, which in this codebase are the normal case rather than the exception:
//
//   1. The rank test used an absolute cutoff of 1e-14 on the singular values.
//      Those scale with the coordinates, so a mathematically zero singular
//      value lands near 1e-13 for a molecule tens of angstroms from the origin
//      and the rank drop is missed.  The cutoff is now relative to d[0].
//
//   2. When the two directions of a rank-1 fit were parallel the original
//      applied a half-turn regardless of whether they pointed the same way or
//      opposite ways, so a pure translation of a collinear set came back with
//      its atoms rotated by pi.  Over 3000 random collinear pure translations
//      the new code gives a lower rmsd in 3000 cases and a worse one in none,
//      with the largest single improvement 9.85 A.
//
//   3. With no constrained direction at all, as for a one-atom fit, the
//      original set ierr and returned without ever writing the rotation
//      matrix.  No caller in this codebase inspects ierr and every one of them
//      applies the matrix, so residues took whatever the stack happened to
//      hold; in practice a consistent diag(-1, 1, -1), a half-turn about y.
//      This fired on every residue with fewer than four atoms, which means
//      every crystallographic water: 58 of them in 1ubq and 118 in 2lzm, whose
//      hydrogens were being flipped on load.  It is also why 1crn, which has
//      no waters, reproduced its old energy exactly and those two did not.
//      The rotation is unconstrained in this case, so it is now the identity.
//
// Because of 3, starting energies change for any structure containing waters:
// 1ubq moves 1510.84 -> 1508.98 and 2lzm 1643.58 -> 1646.39.  These are bug
// fixes, not drift.  1crn was unchanged at 523.60 at the time of this change;
// it now reads 510.98, because a later fix stopped hydroxyl hydrogens being
// placed from a dihedral measured across the template and file frames.
//
// Beyond those cases the results agree with the Fortran to ~5e-14 but are not
// bit-identical, because a different sequence of floating point operations
// cannot be.  Any Monte Carlo trajectory seeded through a superposition will
// therefore diverge from a pre-port run even at the same seed.  The physics is
// unchanged: over 8 seeds of protMinRep on 1crn the two builds agree to 0.4
// sigma.  The per-seed trajectory is not reproducible across this commit.

#include "bestfit.h"

#include <cmath>
#include <cstddef>

namespace
{

// Column-major 3x3 accessor, matching the Fortran storage the callers expect.
inline double& at(double* m, int i, int j) { return m[j * 3 + i]; }
inline double at(const double* m, int i, int j) { return m[j * 3 + i]; }

double determinant3(const double* m)
{
	const double a1 = at(m, 1, 1) * at(m, 2, 2) - at(m, 2, 1) * at(m, 1, 2);
	const double a2 = at(m, 2, 1) * at(m, 0, 2) - at(m, 0, 1) * at(m, 2, 2);
	const double a3 = at(m, 0, 1) * at(m, 1, 2) - at(m, 1, 1) * at(m, 0, 2);
	return a1 * at(m, 0, 0) + a2 * at(m, 1, 0) + a3 * at(m, 2, 0);
}

// One-sided Jacobi SVD of a general 3x3: A = U diag(d) V^T, d descending.
//
// The obvious route is to diagonalise A^T A, and the first version of this
// file did.  It is wrong here.  Forming A^T A squares the condition number, so
// a singular value that is mathematically zero comes back at roughly
// sqrt(eps)*||A||, which for coordinates of order 10 A is about 1e-7 rather
// than 1e-14.  That matters because the caller's rank test is what selects the
// degenerate branch, and the common case in this codebase is a fit over three
// backbone atoms, which are always coplanar and therefore always rank 2.  With
// the rank drop missed, the third singular direction is reconstructed by
// dividing by numerical noise and the returned matrix is not a rotation at
// all: it came back with determinant 0 instead of 1.
//
// One-sided Jacobi orthogonalises the columns of A in place and never forms
// A^T A, so the singular values keep full relative accuracy and a rank drop is
// unambiguous.
void svd3(const double* a, double* d, double* u, double* v)
{
	double w[9];
	for (int i = 0; i < 9; i++) w[i] = a[i];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) at(v, i, j) = (i == j) ? 1.0 : 0.0;

	for (int sweep = 0; sweep < 60; sweep++)
	{
		double offMax = 0.0;
		for (int p = 0; p < 2; p++)
		{
			for (int q = p + 1; q < 3; q++)
			{
				double alpha = 0.0, beta = 0.0, gamma = 0.0;
				for (int k = 0; k < 3; k++)
				{
					alpha += at(w, k, p) * at(w, k, p);
					beta  += at(w, k, q) * at(w, k, q);
					gamma += at(w, k, p) * at(w, k, q);
				}
				if (gamma == 0.0) continue;

				const double denom = std::sqrt(alpha * beta);
				const double off = denom > 0.0 ? std::fabs(gamma) / denom : 0.0;
				if (off > offMax) offMax = off;
				if (off < 1.0e-17) continue;

				// Rotate columns p and q so that they become orthogonal.
				const double zeta = (beta - alpha) / (2.0 * gamma);
				const double t = (zeta >= 0.0 ? 1.0 : -1.0)
				               / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
				const double c = 1.0 / std::sqrt(1.0 + t * t);
				const double s = c * t;

				for (int k = 0; k < 3; k++)
				{
					const double wkp = at(w, k, p), wkq = at(w, k, q);
					at(w, k, p) = c * wkp - s * wkq;
					at(w, k, q) = s * wkp + c * wkq;
					const double vkp = at(v, k, p), vkq = at(v, k, q);
					at(v, k, p) = c * vkp - s * vkq;
					at(v, k, q) = s * vkp + c * vkq;
				}
			}
		}
		if (offMax < 1.0e-16) break;
	}

	double norm[3];
	for (int c = 0; c < 3; c++)
		norm[c] = std::sqrt(at(w, 0, c) * at(w, 0, c)
		                  + at(w, 1, c) * at(w, 1, c)
		                  + at(w, 2, c) * at(w, 2, c));

	int order[3] = {0, 1, 2};
	for (int i = 0; i < 2; i++)
		for (int j = i + 1; j < 3; j++)
			if (norm[order[j]] > norm[order[i]])
			{
				const int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
			}

	double vs[9];
	for (int c = 0; c < 3; c++)
	{
		const int src = order[c];
		d[c] = norm[src];
		for (int i = 0; i < 3; i++)
		{
			at(vs, i, c) = at(v, i, src);
			at(u, i, c) = d[c] > 0.0 ? at(w, i, src) / d[c] : 0.0;
		}
	}
	for (int i = 0; i < 9; i++) v[i] = vs[i];

	// A rank drop is judged relative to the largest singular value.  An
	// absolute cutoff cannot work: it would have to be compared against a
	// quantity that scales with the size of the coordinates.
	const double cut = d[0] * 1.0e-12;
	for (int c = 1; c < 3; c++)
	{
		if (d[c] <= cut)
		{
			d[c] = 0.0;
			for (int i = 0; i < 3; i++) at(u, i, c) = 0.0;
		}
	}

	if (d[2] == 0.0 && d[1] > 0.0)
	{
		at(u, 0, 2) = at(u, 1, 0) * at(u, 2, 1) - at(u, 2, 0) * at(u, 1, 1);
		at(u, 1, 2) = at(u, 2, 0) * at(u, 0, 1) - at(u, 0, 0) * at(u, 2, 1);
		at(u, 2, 2) = at(u, 0, 0) * at(u, 1, 1) - at(u, 1, 0) * at(u, 0, 1);
	}
}

void setIdentity(double* m)
{
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) at(m, i, j) = (i == j) ? 1.0 : 0.0;
}

// Half-turn about the unit axis w, as 2 w w^T - I.
void householder(double* m, const double* w)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++) at(m, i, j) = 2.0 * w[j] * w[i];
		at(m, i, i) -= 1.0;
	}
}

} // namespace

extern "C" void bestfit_(double* coords1, int* nat1, double* coords2, int* nat2,
                         int* nat_, double* coords3, int* list1, int* list2,
                         double* rmsd, int* ierr, double* r, double* xc1,
                         double* xc2, double* rmsdat)
{
	const int nat = *nat_;
	const int natoms2 = *nat2;
	// Rank threshold.  This has to be relative to the largest singular value:
	// the singular values of A scale with the coordinates, so for a molecule
	// positioned tens of angstroms from the origin a mathematically zero
	// singular value comes back around 1e-13, and an absolute 1e-14 cutoff
	// (as the original used) would miss the rank drop and rebuild the
	// unconstrained direction out of rounding noise.
	*ierr = 0;

	// Fortran indexes atom n of coord(3,N) at coord[(n-1)*3 + j]; list1/list2
	// are 1-based, so the -1 below is the translation and not an off-by-one.
	for (int j = 0; j < 3; j++) { xc1[j] = 0.0; xc2[j] = 0.0; }
	for (int i = 0; i < nat; i++)
		for (int j = 0; j < 3; j++)
		{
			xc1[j] += coords1[(list1[i] - 1) * 3 + j];
			xc2[j] += coords2[(list2[i] - 1) * 3 + j];
		}
	for (int j = 0; j < 3; j++) { xc1[j] /= nat; xc2[j] /= nat; }

	double a[9];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
			double s = 0.0;
			for (int k = 0; k < nat; k++)
				s += (coords1[(list1[k] - 1) * 3 + i] - xc1[i])
				   * (coords2[(list2[k] - 1) * 3 + j] - xc2[j]);
			at(a, i, j) = s;
		}

	const double det = determinant3(a);
	if (det == 0.0) *ierr = 1;   // matches the original: flagged, not fatal
	double sign = (det <= 0.0) ? -1.0 : 1.0;

	double d[3], u[9], v[9];
	svd3(a, d, u, v);
	const double tiny = d[0] * 1.0e-12;

	if (d[1] > tiny)
	{
		if (d[2] <= tiny)
		{
			// Rank 2.  The third singular direction is unconstrained, so fix
			// it by right-handedness and drop the reflection correction.
			sign = 1.0;
			at(u, 0, 2) = at(u, 1, 0) * at(u, 2, 1) - at(u, 2, 0) * at(u, 1, 1);
			at(u, 1, 2) = at(u, 2, 0) * at(u, 0, 1) - at(u, 0, 0) * at(u, 2, 1);
			at(u, 2, 2) = at(u, 0, 0) * at(u, 1, 1) - at(u, 1, 0) * at(u, 0, 1);
			at(v, 0, 2) = at(v, 1, 0) * at(v, 2, 1) - at(v, 2, 0) * at(v, 1, 1);
			at(v, 1, 2) = at(v, 2, 0) * at(v, 0, 1) - at(v, 0, 0) * at(v, 2, 1);
			at(v, 2, 2) = at(v, 0, 0) * at(v, 1, 1) - at(v, 1, 0) * at(v, 0, 1);
		}
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				at(r, i, j) = at(u, i, 0) * at(v, j, 0)
				            + at(u, i, 1) * at(v, j, 1)
				            + sign * at(u, i, 2) * at(v, j, 2);
	}
	else
	{
		// Rank 1: the two sets are collinear, so only one direction is
		// constrained and the rotation is the half-turn about the bisector of
		// u1 and v1, built as a Householder reflection.
		//
		// Two cases the original got wrong are handled explicitly.  It tested
		// only whether u1 x v1 vanished, which is true both when the two
		// directions are identical and when they are opposed, and then applied
		// a half-turn regardless.  For u1 == v1 the correct answer is the
		// identity, so a pure translation of a collinear set was coming back
		// with its atoms rotated by pi.  It also left r untouched when there
		// was no information at all, and no caller in this codebase inspects
		// ierr, so uninitialised memory was being applied as a rotation.
		const double dot = at(u, 0, 0) * at(v, 0, 0)
		                 + at(u, 1, 0) * at(v, 1, 0)
		                 + at(u, 2, 0) * at(v, 2, 0);

		double w[3];
		w[0] = at(u, 1, 0) * at(v, 2, 0) - at(u, 2, 0) * at(v, 1, 0);
		w[1] = at(u, 2, 0) * at(v, 0, 0) - at(u, 0, 0) * at(v, 2, 0);
		w[2] = at(u, 0, 0) * at(v, 1, 0) - at(u, 1, 0) * at(v, 0, 0);
		double norm = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];

		if (d[0] <= 0.0)
		{
			// No constrained direction whatsoever, as with a single atom.
			// Identity is the only defensible answer.
			setIdentity(r);
		}
		else if (norm != 0.0)
		{
			for (int i = 0; i < 3; i++) w[i] = at(u, i, 0) + at(v, i, 0);
			norm = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
			if (norm == 0.0) { *ierr = 1; setIdentity(r); }
			else
			{
				for (int i = 0; i < 3; i++) w[i] /= norm;
				householder(r, w);
			}
		}
		else if (dot >= 0.0)
		{
			setIdentity(r);
		}
		else
		{
			// Antiparallel: any axis perpendicular to u1 gives the half-turn.
			double t[3] = {0.0, 0.0, 0.0};
			const double ax = std::fabs(at(u, 0, 0));
			const double ay = std::fabs(at(u, 1, 0));
			const double az = std::fabs(at(u, 2, 0));
			if (ax <= ay && ax <= az) t[0] = 1.0;
			else if (ay <= az) t[1] = 1.0;
			else t[2] = 1.0;

			w[0] = at(u, 1, 0) * t[2] - at(u, 2, 0) * t[1];
			w[1] = at(u, 2, 0) * t[0] - at(u, 0, 0) * t[2];
			w[2] = at(u, 0, 0) * t[1] - at(u, 1, 0) * t[0];
			norm = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
			if (norm == 0.0) { *ierr = 1; setIdentity(r); }
			else
			{
				for (int i = 0; i < 3; i++) w[i] /= norm;
				householder(r, w);
			}
		}
	}

	for (int i = 0; i < natoms2; i++)
	{
		double c[3];
		for (int j = 0; j < 3; j++)
		{
			double s = 0.0;
			for (int k = 0; k < 3; k++)
				s += at(r, j, k) * (coords2[i * 3 + k] - xc2[k]);
			c[j] = s;
		}
		for (int j = 0; j < 3; j++) coords3[i * 3 + j] = c[j] + xc1[j];
	}

	double total = 0.0;
	for (int i = 0; i < nat; i++)
	{
		rmsdat[i] = 0.0;
		for (int j = 0; j < 3; j++)
		{
			const double diff = coords3[(list2[i] - 1) * 3 + j]
			                  - coords1[(list1[i] - 1) * 3 + j];
			rmsdat[i] += diff * diff;
		}
		total += rmsdat[i];
	}
	*rmsd = std::sqrt(total / nat);
}
