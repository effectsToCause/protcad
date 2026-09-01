// rotamerRank -- does a cheap energy term rank rotamers the way the full model does?
//
// The acceleration strategy in docs/packing-search.md depends on holding the
// local dielectric fixed across an outer iteration, because the many-body
// dielectric is what blocks the rotamer pair table, the incremental evaluation
// and parallel move classes alike.  That is only sound if the terms it affects
// are not the ones deciding which rotamer wins.
//
// This measures that directly.  For each flexible residue it samples rotamers,
// evaluates the full per-term breakdown for each, and asks how well a reduced
// scoring function reproduces the full model's ordering:
//
//   Spearman   rank correlation over the whole sampled set
//   top-1      does the reduced score pick the same best rotamer
//   top-5      is the full model's best rotamer inside the reduced score's top 5
//
// Two reduced scores are reported.  "vdw" is van der Waals alone.  "vdw+tors"
// adds the bonded term, which is free to evaluate and independent of solvent.
//
// Note what this does and does not bound.  Freezing the dielectric is strictly
// milder than dropping solvation: a frozen eps still evaluates every solvation
// and electrostatic term, it merely lags eps by one outer iteration.  So the
// ranking damage from freezing eps is bounded above by the damage from
// discarding those terms entirely, which is what is measured here.  A good
// result is therefore conclusive in the safe direction, and a bad result is not
// necessarily fatal.
//
// The claim under test is also specifically that agreement should be best where
// the energy is high, since a clash is an overwhelming vdW signal that no
// solvation term can offset.  Results are therefore split by whether the
// residue's sampled range includes a clashing conformation.
//
//   rotamerRank <in.pdb> [samplesPerResidue] [seed]

#include "ensemble.h"
#include "PDBInterface.h"
#include "protein.h"
#include "energy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace std;

// Spearman rank correlation.  Ties are averaged, which matters because
// symmetric rotamers can produce genuinely equal energies.
static double spearman(const vector<double>& a, const vector<double>& b)
{
	const size_t n = a.size();
	if (n < 3) {return 0.0;}

	auto ranks = [n](const vector<double>& v)
	{
		vector<size_t> idx(n);
		iota(idx.begin(), idx.end(), 0);
		sort(idx.begin(), idx.end(), [&v](size_t p, size_t q){return v[p] < v[q];});
		vector<double> r(n);
		size_t i = 0;
		while (i < n)
		{
			size_t j = i;
			while (j + 1 < n && v[idx[j + 1]] == v[idx[i]]) {j++;}
			const double mean = 0.5 * (double)(i + j) + 1.0;
			for (size_t k = i; k <= j; k++) {r[idx[k]] = mean;}
			i = j + 1;
		}
		return r;
	};

	const vector<double> ra = ranks(a), rb = ranks(b);
	const double m = 0.5 * (double)(n + 1);
	double num = 0.0, da = 0.0, db = 0.0;
	for (size_t i = 0; i < n; i++)
	{
		const double x = ra[i] - m, y = rb[i] - m;
		num += x * y; da += x * x; db += y * y;
	}
	if (da <= 0.0 || db <= 0.0) {return 0.0;}
	return num / sqrt(da * db);
}

struct tally
{
	int residues = 0, top1 = 0, top5 = 0;
	double rho = 0.0;
	void add(double r, bool t1, bool t5) {residues++; rho += r; top1 += t1; top5 += t5;}
	void report(const char* label) const
	{
		if (!residues) {printf("  %-10s (none)\n", label); return;}
		printf("  %-10s residues %4d   mean rho %+6.3f   top-1 %5.1f%%   top-5 %5.1f%%\n",
		       label, residues, rho / residues,
		       100.0 * top1 / residues, 100.0 * top5 / residues);
	}
};

int main(int argc, char** argv)
{
	if (argc < 2)
	{	printf("rotamerRank <in.pdb> [samplesPerResidue] [seed]\n");
		return 1; }

	const int nSample = (argc > 2) ? atoi(argv[2]) : 48;
	const unsigned seed = (argc > 3) ? (unsigned)atoi(argv[3]) : 1u;
	srand(seed);

	PDBInterface* thePDB = new PDBInterface(argv[1]);
	ensemble* theEnsemble = thePDB->getEnsemblePointer();
	molecule* pMol = theEnsemble->getMoleculePointer(0);
	protein* prot = static_cast<protein*>(pMol);

	residue::setElectroSolvationScaleFactor(1.0);
	residue::setHydroSolvationScaleFactor(1.0);
	residue::setPolarizableElec(false);
	amberElec::setScaleFactor(1.0);
	amberVDW::setScaleFactor(1.0);
	residue::setTemperature(300);
	residue::setEntropyFactor(0.0);

#ifndef __CUDA__
	printf("rotamerRank requires a CUDA build\n");
	return 1;
#else
	prot->loadDeviceMemAll();

	printf("\n%s  (%d atoms)  %d samples/residue  seed %u\n",
	       argv[1], prot->getNumAtoms(), nSample, seed);

	// Uniform chi sampling is overwhelmingly clashes, and ranking one clash
	// against another is not the question -- every score agrees that a 10^6
	// kcal/mol conformation is bad, which inflates rank correlation without
	// meaning anything.  The decision that actually settles a packing is made
	// among the conformations that are already clash-free, so the samples are
	// also analysed restricted to a band above the best vdW found.
	const double band = 10.0;   // kcal/mol above the best sampled vdW

	tally vdwAll, vtAll, vdwNear, vtNear;
	double sumKept = 0.0;
	int nearResidues = 0;

	// Reference dielectric field at the input conformation.  Every residue is
	// perturbed away from this same structure, so one snapshot serves as the
	// frozen field for all of them.  atomKey packs chain/residue/atom so the
	// moved residue's own atoms can be excluded: nobody proposes freezing eps
	// on the atoms that just moved.
	prot->updateDielectricsCU();
	vector< vector< vector<double> > > refEps(prot->getNumChains());
	for (UInt c = 0; c < prot->getNumChains(); c++)
	{
		refEps[c].resize(prot->getNumResidues(c));
		for (UInt r = 0; r < prot->getNumResidues(c); r++)
		{
			const UInt na = prot->getNumAtoms(c, r);
			refEps[c][r].resize(na);
			for (UInt a = 0; a < na; a++) {refEps[c][r][a] = prot->getDielectric(c, r, a);}
		}
	}

	double epsRelSum = 0.0, epsRelMaxSum = 0.0;
	double epsRelSumSelf = 0.0;
	double epsOver1 = 0.0, epsOver5 = 0.0;
	int epsSamples = 0;
	double sumSpreadVdw = 0.0, sumSpreadPolar = 0.0, sumSpreadNonpolar = 0.0;
	double sumSpreadElec = 0.0, sumSpreadTors = 0.0;
	int counted = 0;

	for (UInt c = 0; c < prot->getNumChains(); c++)
	{
		for (UInt r = 0; r < prot->getNumResidues(c); r++)
		{
			vector< vector<double> > base = prot->getSidechainDihedrals(c, r);
			if (base.empty() || base[0].empty()) {continue;}
			const size_t nChi = base[0].size();

			vector<double> full, vdw, vdwTors, polar, nonpolar, elec, tors;
			vector< vector<double> > sampled;
			bool ok = true;

			for (int s = 0; s < nSample && ok; s++)
			{
				vector< vector<double> > angles = base;
				for (size_t k = 0; k < nChi; k++)
				{
					// Uniform over the circle: a rotamer library would be
					// tighter, but uniform sampling is the harder test and
					// needs no assumption about which wells matter.
					angles[0][k] = 360.0 * (rand() / (double)RAND_MAX) - 180.0;
				}
				prot->setSidechainDihedralAngles(c, r, angles);

				energyBreakdown b;
				if (!prot->protEnergyBreakdownCU(b)) {ok = false; break;}

				sampled.push_back(angles[0]);
				full.push_back(b.total);
				vdw.push_back(b.vdw);
				vdwTors.push_back(b.vdw + b.torsion);
				polar.push_back(b.solvationPolar);
				nonpolar.push_back(b.solvationNonpolar);
				elec.push_back(b.electrostatic);
				tors.push_back(b.torsion);
			}

			prot->setSidechainDihedralAngles(c, r, base);
			if (!ok || full.size() < 3u) {continue;}

			auto spread = [](const vector<double>& v)
			{	return *max_element(v.begin(), v.end()) - *min_element(v.begin(), v.end()); };

			// agree(): does the reduced score's top-N contain the full model's
			// best?  Operates on a caller-supplied subset of the samples.
			auto agree = [](const vector<double>& score, const vector<double>& ref,
			                const vector<size_t>& sub, int topN)
			{
				size_t bestFull = sub[0];
				for (size_t i : sub) {if (ref[i] < ref[bestFull]) {bestFull = i;}}
				vector<size_t> idx = sub;
				const int n = min<int>(topN, (int)idx.size());
				partial_sort(idx.begin(), idx.begin() + n, idx.end(),
				             [&score](size_t p, size_t q){return score[p] < score[q];});
				for (int i = 0; i < n; i++) {if (idx[i] == bestFull) {return true;}}
				return false;
			};

			auto subset = [](const vector<double>& v, const vector<size_t>& sub)
			{
				vector<double> out;
				out.reserve(sub.size());
				for (size_t i : sub) {out.push_back(v[i]);}
				return out;
			};

			vector<size_t> all(full.size());
			iota(all.begin(), all.end(), 0);

			vdwAll.add(spearman(vdw, full), agree(vdw, full, all, 1), agree(vdw, full, all, 5));
			vtAll.add(spearman(vdwTors, full), agree(vdwTors, full, all, 1), agree(vdwTors, full, all, 5));

			// The near-native band.
			const double vdwFloor = *min_element(vdw.begin(), vdw.end());
			vector<size_t> near;
			for (size_t i = 0; i < vdw.size(); i++)
			{	if (vdw[i] <= vdwFloor + band) {near.push_back(i);} }

			if (near.size() >= 5u)
			{
				const vector<double> nf = subset(full, near), nv = subset(vdw, near),
				                     nt = subset(vdwTors, near);
				vdwNear.add(spearman(nv, nf), agree(vdw, full, near, 1), agree(vdw, full, near, 5));
				vtNear.add(spearman(nt, nf), agree(vdwTors, full, near, 1), agree(vdwTors, full, near, 5));
				sumKept += near.size();
				nearResidues++;

				// How far does the dielectric field actually move under a
				// near-native rotamer change?  This is the error a frozen eps
				// commits, and it is a much weaker requirement than discarding
				// solvation outright.  Capped at a few samples per residue --
				// the drift is systematic, not noisy, so more samples buy
				// nothing but wall clock.
				const size_t nEpsSample = min<size_t>(near.size(), 6);
				for (size_t s = 0; s < nEpsSample; s++)
				{
					vector< vector<double> > angles = base;
					angles[0] = sampled[near[s]];
					prot->setSidechainDihedralAngles(c, r, angles);
					prot->updateDielectricsCU();

					double relSum = 0.0, relMax = 0.0, relSelf = 0.0;
					int nOther = 0, nSelf = 0, nOver1 = 0, nOver5 = 0;
					for (UInt cc = 0; cc < prot->getNumChains(); cc++)
					{
						for (UInt rr = 0; rr < prot->getNumResidues(cc); rr++)
						{
							for (UInt aa = 0; aa < prot->getNumAtoms(cc, rr); aa++)
							{
								const double e0 = refEps[cc][rr][aa];
								if (e0 <= 0.0) {continue;}
								const double rel =
									fabs(prot->getDielectric(cc, rr, aa) - e0) / e0;
								if (cc == c && rr == r) {relSelf += rel; nSelf++;}
								else
								{	relSum += rel; nOther++;
									if (rel > relMax) {relMax = rel;}
									if (rel > 0.01) {nOver1++;}
									if (rel > 0.05) {nOver5++;} }
							}
						}
					}
					if (nOther)
					{
						epsRelSum += relSum / nOther;
						epsRelMaxSum += relMax;
						epsRelSumSelf += nSelf ? relSelf / nSelf : 0.0;
						epsOver1 += nOver1;
						epsOver5 += nOver5;
						epsSamples++;
					}
				}
				prot->setSidechainDihedralAngles(c, r, base);
			}

			sumSpreadVdw += spread(vdw);
			sumSpreadPolar += spread(polar);
			sumSpreadNonpolar += spread(nonpolar);
			sumSpreadElec += spread(elec);
			sumSpreadTors += spread(tors);
			counted++;
		}
	}

	if (!counted) {printf("no flexible residues evaluated\n"); return 1;}

	printf("\nmean spread across sampled rotamers (kcal/mol), the signal available to ranking\n");
	printf("  vdW %.1f   elec %.1f   torsion %.1f   solv-polar %.1f   solv-nonpolar %.1f\n",
	       sumSpreadVdw / counted, sumSpreadElec / counted, sumSpreadTors / counted,
	       sumSpreadPolar / counted, sumSpreadNonpolar / counted);

	printf("\nagreement of reduced score with the full model\n");
	printf("\n all sampled rotamers (dominated by clashes, ranking here is easy)\n");
	vdwAll.report("vdw");
	vtAll.report("vdw+tors");
	printf("\n within %.0f kcal/mol of the best sampled vdW (near-native, decides the packing)\n", band);
	printf("  %d of %d residues had >= 5 samples in band, mean %.1f kept\n",
	       nearResidues, counted, nearResidues ? sumKept / nearResidues : 0.0);
	vdwNear.report("vdw");
	vtNear.report("vdw+tors");

	printf("\ndielectric drift under a near-native rotamer change (%d samples)\n", epsSamples);
	if (epsSamples)
	{
		printf("  atoms outside the moved residue:  mean |d eps|/eps %.4f%%   worst atom %.3f%%\n",
		       100.0 * epsRelSum / epsSamples, 100.0 * epsRelMaxSum / epsSamples);
		printf("  atoms of the moved residue:       mean |d eps|/eps %.4f%%\n",
		       100.0 * epsRelSumSelf / epsSamples);
		printf("  environment atoms moving >1%%: %.1f    >5%%: %.1f   (of %d)\n",
		       epsOver1 / epsSamples, epsOver5 / epsSamples, prot->getNumAtoms());
	}
	printf("\n");
	return 0;
#endif
}
