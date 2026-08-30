// bestfitTest.cc
//
// Guards the C++ superposition that replaced the F77 bestfit.  The Fortran is
// gone, so this cannot be a differential test; it checks the properties the
// callers actually depend on.
//
// The property that matters most is that the returned matrix is a proper
// rotation.  residue::mutateNew fits three backbone atoms and then applies the
// result to every atom of a newly built residue, so an improper matrix would
// silently mirror a sidechain and invert its chirality.  Three points are
// always coplanar, which makes the rank-2 branch the normal case here rather
// than an edge case, and it was exactly that branch that the first port got
// wrong.

#include "bestfit.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what)
{
	if (!ok) { printf("  FAIL: %s\n", what); failures++; }
}

static double det3(const double* m)
{
	return m[0] * (m[4] * m[8] - m[7] * m[5])
	     - m[3] * (m[1] * m[8] - m[7] * m[2])
	     + m[6] * (m[1] * m[5] - m[4] * m[2]);
}

// Largest deviation of R^T R from the identity.
static double orthoError(const double* m)
{
	double worst = 0.0;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
			double s = 0.0;
			for (int k = 0; k < 3; k++) s += m[i * 3 + k] * m[j * 3 + k];
			const double d = std::fabs(s - (i == j ? 1.0 : 0.0));
			if (d > worst) worst = d;
		}
	return worst;
}

static void rotationFromAxisAngle(double* R, double ax, double ay, double az, double th)
{
	const double n = std::sqrt(ax * ax + ay * ay + az * az);
	ax /= n; ay /= n; az /= n;
	const double c = std::cos(th), s = std::sin(th), C = 1.0 - c;
	R[0] = c + ax * ax * C;    R[3] = ax * ay * C - az * s; R[6] = ax * az * C + ay * s;
	R[1] = ay * ax * C + az * s; R[4] = c + ay * ay * C;    R[7] = ay * az * C - ax * s;
	R[2] = az * ax * C - ay * s; R[5] = az * ay * C + ax * s; R[8] = c + az * az * C;
}

int main()
{
	// Idealised N/CA/C triad with real bond lengths and angles.
	const double triad[9] = { 0.000, 0.000, 0.000,
	                          1.458, 0.000, 0.000,
	                          2.009, 1.420, 0.000 };

	printf("known-answer recovery of a proper rotation from a 3-atom backbone fit\n");
	{
		std::srand(11);
		int improper = 0, nonortho = 0, badrmsd = 0;
		double worstRmsd = 0.0, worstOrtho = 0.0;
		const int trials = 4000;
		for (int t = 0; t < trials; t++)
		{
			double R[9];
			rotationFromAxisAngle(R,
				std::rand() / (double)RAND_MAX - 0.5,
				std::rand() / (double)RAND_MAX - 0.5,
				std::rand() / (double)RAND_MAX - 0.5,
				(std::rand() / (double)RAND_MAX) * 6.283185307);

			double p1[9], p2[9];
			for (int i = 0; i < 9; i++) p1[i] = triad[i];
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
				{
					double v = 0.0;
					for (int k = 0; k < 3; k++) v += R[k * 3 + j] * triad[i * 3 + k];
					p2[i * 3 + j] = v;
				}

			int l1[3] = {1, 2, 3}, l2[3] = {1, 2, 3}, n1 = 3, n2 = 3, n = 3, ierr = 0;
			double c3[9], m[9], x1[3], x2[3], da[3], rmsd = 0.0;
			bestfit_(p1, &n1, p2, &n2, &n, c3, l1, l2, &rmsd, &ierr, m, x1, x2, da);

			if (std::fabs(det3(m) - 1.0) > 1e-8) improper++;
			const double oe = orthoError(m);
			if (oe > 1e-10) nonortho++;
			if (rmsd > 1e-9) badrmsd++;
			if (rmsd > worstRmsd) worstRmsd = rmsd;
			if (oe > worstOrtho) worstOrtho = oe;
		}
		printf("  trials %d, improper %d, non-orthonormal %d, nonzero rmsd %d\n",
		       trials, improper, nonortho, badrmsd);
		printf("  worst rmsd %.3e, worst orthonormality error %.3e\n",
		       worstRmsd, worstOrtho);
		check(improper == 0, "3-atom fit must return a proper rotation (det +1)");
		check(nonortho == 0, "3-atom fit must return an orthonormal matrix");
		check(badrmsd == 0, "an exact superposition must give rmsd 0");
	}

	printf("proper rotation and orthonormality over random atom counts\n");
	{
		std::srand(23);
		int improper = 0, nonortho = 0;
		double worstOrtho = 0.0;
		const int trials = 4000;
		for (int t = 0; t < trials; t++)
		{
			// Includes nat = 3, which is always rank 2, and larger sets which
			// are generically rank 3.
			const int nat = 3 + std::rand() % 10;
			std::vector<double> a(nat * 3), b(nat * 3), c3(nat * 3), da(nat);
			std::vector<int> l1(nat), l2(nat);
			for (int i = 0; i < nat; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					a[i * 3 + j] = (std::rand() / (double)RAND_MAX) * 30.0 - 15.0;
					// A perturbed copy, so the fit is well posed rather than a
					// match between two unrelated shapes.
					b[i * 3 + j] = a[i * 3 + j]
					             + (std::rand() / (double)RAND_MAX) * 0.4 - 0.2;
				}
				l1[i] = i + 1; l2[i] = i + 1;
			}
			int n1 = nat, n2 = nat, n = nat, ierr = 0;
			double m[9], x1[3], x2[3], rmsd = 0.0;
			bestfit_(a.data(), &n1, b.data(), &n2, &n, c3.data(),
			         l1.data(), l2.data(), &rmsd, &ierr, m, x1, x2, da.data());
			if (std::fabs(det3(m) - 1.0) > 1e-8) improper++;
			const double oe = orthoError(m);
			if (oe > 1e-10) nonortho++;
			if (oe > worstOrtho) worstOrtho = oe;
		}
		printf("  trials %d, improper %d, non-orthonormal %d, worst ortho error %.3e\n",
		       trials, improper, nonortho, worstOrtho);
		check(improper == 0, "every fit must return a proper rotation");
		check(nonortho == 0, "every fit must return an orthonormal matrix");
	}

	printf("centroids and rmsd\n");
	{
		// Two atoms 2 A apart along x, translated by a known vector.
		double a[6] = {0, 0, 0, 2, 0, 0};
		double b[6] = {10, 5, -3, 12, 5, -3};
		int l1[2] = {1, 2}, l2[2] = {1, 2}, n1 = 2, n2 = 2, n = 2, ierr = 0;
		double c3[6], m[9], x1[3], x2[3], da[2], rmsd = 0.0;
		bestfit_(a, &n1, b, &n2, &n, c3, l1, l2, &rmsd, &ierr, m, x1, x2, da);
		check(std::fabs(x1[0] - 1.0) < 1e-12 && std::fabs(x1[1]) < 1e-12
		      && std::fabs(x1[2]) < 1e-12, "reference centroid");
		check(std::fabs(x2[0] - 11.0) < 1e-12 && std::fabs(x2[1] - 5.0) < 1e-12
		      && std::fabs(x2[2] + 3.0) < 1e-12, "test centroid");
		check(rmsd < 1e-12, "pure translation must superpose exactly");
		for (int i = 0; i < 6; i++)
			check(std::fabs(c3[i] - a[i]) < 1e-12, "translated coordinates land on the reference");
	}

	printf("rmsd is invariant to the order of the two structures\n");
	{
		std::srand(31);
		const int nat = 7;
		std::vector<double> a(nat * 3), b(nat * 3), c3(nat * 3), da(nat);
		std::vector<int> l1(nat), l2(nat);
		for (int i = 0; i < nat; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				a[i * 3 + j] = (std::rand() / (double)RAND_MAX) * 20.0 - 10.0;
				b[i * 3 + j] = (std::rand() / (double)RAND_MAX) * 20.0 - 10.0;
			}
			l1[i] = i + 1; l2[i] = i + 1;
		}
		int n1 = nat, n2 = nat, n = nat, ierr = 0;
		double m[9], x1[3], x2[3], r1 = 0.0, r2 = 0.0;
		bestfit_(a.data(), &n1, b.data(), &n2, &n, c3.data(), l1.data(), l2.data(),
		         &r1, &ierr, m, x1, x2, da.data());
		bestfit_(b.data(), &n1, a.data(), &n2, &n, c3.data(), l2.data(), l1.data(),
		         &r2, &ierr, m, x1, x2, da.data());
		printf("  rmsd %.12f vs %.12f\n", r1, r2);
		check(std::fabs(r1 - r2) < 1e-10, "rmsd must not depend on argument order");
	}

	printf("RESULT: %d failures\n", failures);
	return failures ? 1 : 0;
}
