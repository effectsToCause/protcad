#!/usr/bin/env python3
# Analysis for the paired dE cancellation measurement.
#
# Reports, per mutation site, the spread of the energy difference under four
# pairings that differ only in how much the two halves share:
#
#   unpaired      independent WT and mutant runs, all (s,t) combinations
#   seed-paired   same seed, but the mutant started from the mutated crystal
#                 rather than from the WT minimum -- shares the stream, not the
#                 basin
#   basin-matched mutant grown into the seed-s WT minimum, differenced against
#                 that same WT run
#   basin-crossed the same mutant energies differenced against a WT run from a
#                 different seed.  This is the control that separates genuine
#                 cancellation from the mutant simply being warm-started: only
#                 matched should tighten if the pairing is doing the work.
import sys, math, statistics as st
from itertools import product

path = sys.argv[1] if len(sys.argv) > 1 else 'ddg_1crn.txt'
E = {}
for line in open(path):
    kind, site, seed, val = line.split()
    E.setdefault((kind, site), {})[int(seed)] = float(val)

wt = E[('wt', '-')]
seeds = sorted(wt)
sd = lambda v: st.stdev(v) if len(v) > 1 else float('nan')

print("seeds %d   sd(E_wt) = %.2f kcal/mol   mean %.2f"
      % (len(seeds), sd(list(wt.values())), st.mean(list(wt.values()))))
print()
hdr = ("site", "sd(E_mut)", "unpaired", "seed", "matched", "crossed", "cancel%", "mean dE")
print("%-6s %9s %9s %9s %9s %9s %8s %9s" % hdr)

sites = sorted({s for (k, s) in E if k == 'mutP'})
for site in sites:
    mi, mp = E[('mutI', site)], E[('mutP', site)]
    unp = [mi[s] - wt[t] for s, t in product(seeds, seeds)]
    seedp = [mi[s] - wt[s] for s in seeds]
    matched = [mp[s] - wt[s] for s in seeds]
    crossed = [mp[s] - wt[t] for s, t in product(seeds, seeds) if s != t]
    cancel = 100.0 * (1.0 - sd(matched) / sd(unp))
    print("%-6s %9.2f %9.2f %9.2f %9.2f %9.2f %7.0f%% %9.2f"
          % (site, sd(list(mi.values())), sd(unp), sd(seedp),
             sd(matched), sd(crossed), cancel, st.mean(matched)))

print()
print("Pre-registered reading of sd(matched), as the noise on one dE estimate")
print("against a ddG signal of 1-2 kcal/mol:")
print("  <= 0.5   single-seed ddG is resolvable")
print("  0.5-1.5  marginal; needs seed averaging, se = sd/sqrt(n)")
print("  > 1.5    basin pairing does not rescue it")
print()
print("crossed must stay near unpaired.  If crossed is also tight then the")
print("mutant runs are simply less variable and no cancellation is happening.")
