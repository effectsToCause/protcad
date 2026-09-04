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
# GPU 0 is the RTX 3090.  GPU 1 is the P2200 driving the display and must
# never be used for training or long minimisation runs.
export CUDA_VISIBLE_DEVICES=0

# Pinned rather than inherited.  The parameter set is a pre-registered arm of
# this experiment, and letting it come in from the caller's environment means a
# result cannot be attributed to one.
export PROTCAD_ENERGY_LEGACY=${PROTCAD_ENERGY_LEGACY:-0}

# The folded state is sampled with a RIGID backbone, and this is pinned rather
# than inherited because it is a correctness condition, not a speed knob.
#
# Mobile-backbone folded sampling does not converge.  1crn at 32 replicas shows
# <E> still falling 9.5 and T*S still rising 4.2 per tripling of sweeps out to
# 108k, with no plateau, and the WT-mutant difference is precise but biased:
# dA reads -9.68/-10.91/-8.96 at 12k/36k/108k while two seeds agree to 0.2-0.7
# at each rung.  Sidechain-only sampling converges by 12k (<E> 449.4/446.2/
# 445.7, T*S 42.6/41.6/42.0) and gives dA -10.04 +/- 0.24 over three seeds.
#
# Freezing the backbone leaves R*T*ln24 = 1.89 kcal/mol per backbone torsion
# unaccounted in S_f relative to S_u.  That offset cancels here because ddG is
# a DOUBLE difference and WT and mutant have the same backbone torsion count
# for any non-Gly, non-Pro substitution.  It would NOT cancel in a single
# folding free energy, so do not read the per-structure A as one.
export PROTCAD_MC_BACKBONE=0
BIN=$HOME/protcad/bin
PROT=${1:-1crn}
# 36000 rather than 8000: the sidechain ensemble is flat by 12k, but the
# seed spread of dA tightens from 0.72 to 0.24 between 12k and 36k.
SWEEPS=${2:-36000}
SEEDS=${3:-8}
# nReplicas was 1, which cannot estimate an ensemble: consecutive sweeps of a
# single chain are strongly correlated, so S_conf undersamples badly at any
# sweep count.  Measured on Gly-Leu-Gly, T*S read 9.44/10.63 kcal/mol at 1
# replica (3000/12000 sweeps) versus a converged 11.17/11.02 at 32.
REPLICAS=${4:-32}
OUTFULL=ddg_${PROT}_full.txt
OUT=ddg_${PROT}.txt
: > $OUT
: > $OUTFULL

# The reported value is A = <E> - T*S_conf, not the best conformation seen.
# Best-seen is biased by an amount that grows with accessible torsion count,
# which is precisely what a ddG resolves; see protein::protMinReplicaCU.
emit() {   # kind label seed
  awk -v k=$1 -v l=$2 -v s=$3 -v out=$OUT -v full=$OUTFULL '
    /^minE/{mn=$2} /^<E>/{m=$2; sd=$4} /^T\*S_conf/{ts=$2} /^A =/{a=$(NF-1)} END{
      if (a == "") {exit 1}
      print k, l, s, a >> out
      print k, l, s, mn, m, sd, ts, a >> full }'
}

# site label, internal residue index, target aaIndex (mol.lib order, ALA=0)
SITES="I6A:6:0 L17A:17:0 Y28A:28:0"

for s in $(seq 1 $SEEDS); do
  PROTCAD_MC_SEED=$s $BIN/protMinRep $SWEEPS $REPLICAS $PROT.pdb wt_$s.pdb 2>/dev/null \
    | emit wt - $s
done

for site in $SITES; do
  lab=${site%%:*}; rest=${site#*:}; res=${rest%%:*}; aa=${rest##*:}
  $BIN/protMutate $PROT.pdb 0 $res $aa mutcrys_$lab.pdb >/dev/null 2>&1
  for s in $(seq 1 $SEEDS); do
    PROTCAD_MC_SEED=$s $BIN/protMinRep $SWEEPS $REPLICAS mutcrys_$lab.pdb mi_${lab}_$s.pdb 2>/dev/null \
      | emit mutI $lab $s
    $BIN/protMutate wt_$s.pdb 0 $res $aa mp_${lab}_$s.pdb >/dev/null 2>&1
    PROTCAD_MC_SEED=$s $BIN/protMinRep $SWEEPS $REPLICAS mp_${lab}_$s.pdb mpo_${lab}_$s.pdb 2>/dev/null \
      | emit mutP $lab $s
  done
done
echo "done -> $OUT (A = <E> - T*S_conf); breakdown -> $OUTFULL"
