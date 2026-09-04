#!/bin/bash
# Unfolded reference leg for ddG.
#
#     ddG = [G_f(mut) - G_u(mut)] - [G_f(wt) - G_u(wt)]
#
# A mutation changes the atom count, so E(mut) - E(wt) carries the self-energy
# of the deleted atoms -- tens of kcal/mol having nothing to do with stability.
# It cancels only against an unfolded state computed by the SAME sampler and
# the SAME energy function, because what makes it cancel is common-mode model
# error, not the number itself.  A tabulated Doig/Sternberg value computed
# under someone else's forcefield cancels nothing.
#
# Two things are deliberately matched to ddg_cancel.sh rather than done the
# convenient way:
#
#   - The sidechain is placed by protMutate on a triglycine backbone, not by
#     protBuild directly.  Both draw from mol.lib, but the folded leg reaches
#     its sidechain through protMutate, and the cancellation argument only
#     holds if the two legs share the path.
#   - Sweeps, replicas, seeds and the parameter set are taken from the same
#     defaults, so a ddG is a difference of two like-sampled ensembles.
#
# The one thing that is deliberately NOT matched is the backbone.  The folded
# leg pins PROTCAD_MC_BACKBONE=0 because rigid-backbone folded sampling is the
# only converged regime; the unfolded reference is meaningless with a frozen
# backbone, since an unfolded state IS its backbone ensemble.  Extended
# phi -139 / psi 135 is the starting point, not the answer.
set -u
export CUDA_VISIBLE_DEVICES=0
export PROTCAD_ENERGY_LEGACY=${PROTCAD_ENERGY_LEGACY:-0}
export PROTCAD_MC_BACKBONE=1

BIN=$HOME/protcad/bin
SWEEPS=${1:-36000}
SEEDS=${2:-8}
REPLICAS=${3:-32}
WALL=${PROTCAD_VDW_WALL:-lj}

OUT=unfolded_${WALL}.txt
OUTFULL=unfolded_${WALL}_full.txt
: > $OUT
: > $OUTFULL

emit() {   # label seed
  awk -v l=$1 -v s=$2 -v out=$OUT -v full=$OUTFULL '
    /^minE/{mn=$2} /^<E>/{m=$2; sd=$4} /^T\*S_conf/{ts=$2} /^A =/{a=$(NF-1)} END{
      if (a == "") {exit 1}
      print l, s, a >> out
      print l, s, mn, m, sd, ts, a >> full }'
}

# mol.lib order: ALA 0, ILE 14, LEU 15, TYR 24.  Middle residue of the
# tripeptide is internal index 1 (protMutate indexes from 0; the folded leg's
# I6A/L17A/Y28A at 6/17/28 are the 7th/18th/29th residues of 1crn, which fixes
# the convention).
RES="ALA:0 ILE:14 LEU:15 TYR:24"

$BIN/protBuild GGG GGG.pdb >/dev/null 2>&1
if [ ! -s GGG.pdb ]; then echo "protBuild failed to write GGG.pdb" >&2; exit 1; fi

for r in $RES; do
  lab=${r%%:*}; aa=${r##*:}
  $BIN/protMutate GGG.pdb 0 1 $aa g${lab}g.pdb >/dev/null 2>&1
  if [ ! -s g${lab}g.pdb ]; then echo "protMutate failed for $lab" >&2; exit 1; fi
  for s in $(seq 1 $SEEDS); do
    PROTCAD_MC_SEED=$s $BIN/protMinRep $SWEEPS $REPLICAS g${lab}g.pdb g${lab}g_min_$s.pdb 2>/dev/null \
      | emit $lab $s
  done
done
echo "done -> $OUT (A = <E> - T*S_conf); breakdown -> $OUTFULL"
