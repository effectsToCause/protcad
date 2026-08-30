#!/bin/bash
# Paired dE cancellation measurement.
#
# The question this answers is the one that decides whether a ddG programme is
# viable here at all.  A single minimisation carries 5-6 kcal/mol of seed noise
# against a ddG signal of 1-2 kcal/mol, so an unpaired difference is hopeless.
# ddG is a difference, though, and if the two halves share their conformational
# basin then most of that noise is common-mode and subtracts out.  The
# cancellation fraction is the number that has never been measured.
#
# Three energies per (site, seed):
#
#   E_wt(s)     WT minimised from the crystal with seed s
#   E_mutI(s)   mutant minimised from the mutated crystal with seed s
#               -- shares only the seed, not the basin
#   E_mutP(s)   mutant grown into the seed-s WT minimum and re-minimised
#               -- shares the basin
#
# Warm-starting alone would shrink the spread of E_mutP without any pairing
# being involved, which would look exactly like cancellation.  The crossed
# control separates them: E_mutP(s) - E_wt(t) for s != t reuses the same
# numbers but breaks the pairing.  If matched is tight and crossed is not, the
# cancellation is real.
set -u
export CUDA_VISIBLE_DEVICES=1
BIN=$HOME/protcad/bin
PROT=${1:-1crn}
SWEEPS=${2:-8000}
SEEDS=${3:-8}
OUT=ddg_${PROT}.txt
: > $OUT

# site label, internal residue index, target aaIndex (mol.lib order, ALA=0)
SITES="I6A:6:0 L17A:17:0 Y28A:28:0"

for s in $(seq 1 $SEEDS); do
  PROTCAD_MC_SEED=$s $BIN/protMinRep $SWEEPS 1 $PROT.pdb wt_$s.pdb 2>/dev/null \
    | awk -v s=$s '/Ending Energy/{print "wt", "-", s, $(NF-1)}' >> $OUT
done

for site in $SITES; do
  lab=${site%%:*}; rest=${site#*:}; res=${rest%%:*}; aa=${rest##*:}
  $BIN/protMutate $PROT.pdb 0 $res $aa mutcrys_$lab.pdb >/dev/null 2>&1
  for s in $(seq 1 $SEEDS); do
    PROTCAD_MC_SEED=$s $BIN/protMinRep $SWEEPS 1 mutcrys_$lab.pdb mi_${lab}_$s.pdb 2>/dev/null \
      | awk -v l=$lab -v s=$s '/Ending Energy/{print "mutI", l, s, $(NF-1)}' >> $OUT
    $BIN/protMutate wt_$s.pdb 0 $res $aa mp_${lab}_$s.pdb >/dev/null 2>&1
    PROTCAD_MC_SEED=$s $BIN/protMinRep $SWEEPS 1 mp_${lab}_$s.pdb mpo_${lab}_$s.pdb 2>/dev/null \
      | awk -v l=$lab -v s=$s '/Ending Energy/{print "mutP", l, s, $(NF-1)}' >> $OUT
  done
done
echo "done -> $OUT"
