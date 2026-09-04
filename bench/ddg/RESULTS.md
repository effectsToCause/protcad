# ddG on 1crn: what is settled and what is not

Configuration: 36000 sweeps, 32 replicas, 4 seeds, folded state with a rigid
backbone, unfolded reference Gly-X-Gly with a mobile backbone, reported
quantity `A = <E> - T*S_conf` rather than the best conformation seen.

## The pre-registered parameter-set arm resolves

| arm | sd(E_wt) | sd(matched) | verdict |
|---|---|---|---|
| default (`PROTCAD_ENERGY_LEGACY=0`) | 0.28 | 0.33 - 0.67 | passes |
| legacy (`PROTCAD_ENERGY_LEGACY=1`)  | 2.93 | 8.47 - 9.83 | fails, > 1.5 |

Legacy is out. Everything below is the default set.

## Basin pairing turned out to be unnecessary rather than wrong

The harness exists because a single minimisation carried 5-6 kcal/mol of seed
noise against a 1-2 kcal/mol signal, and the hope was that pairing WT and
mutant into a shared basin would cancel it. Measured unpaired sd is now
0.33-0.37. The noise it was built to fight was an artefact of reporting the
best-seen conformation from one replica; an ensemble average over 32 replicas
removes it at the source. Reported cancellation percentages are meaningless
in this regime (they come out negative) because the denominator is already at
the noise floor. The crossed control is retained but no longer discriminates.

## ddG, uncalibrated

`ddG = dA_folded - dA_unfolded`, positive meaning the mutant is less stable.

| site | dA_folded | dA_unfolded | ddG | sd(dA_folded) |
|---|---|---|---|---|
| I6A  | -10.25 | -17.00 |  6.75 | 0.32 |
| L17A |   3.56 | -13.78 | 17.34 | 0.32 |
| Y28A |   6.68 | -10.57 | 17.25 | 0.24 |

Signs are right: all three are buried and all three destabilise. Precision is
0.2-0.7 kcal/mol, comfortably under the signal. **The magnitudes are wrong**,
by roughly 3-4x against the 2-5 kcal/mol typical of a buried hydrophobic
deletion. These are not quotable as ddG values.

The failure is not sampling. Sidechain ensembles are flat by 12k sweeps, seeds
agree to 0.24, and the numbers reproduce. It is the energy model: there is no
solvation term, so deleting a buried hydrophobic pays the full vacuum vdW
contact cost with no compensating desolvation credit in the unfolded state.
That asymmetry inflates every X->Ala deletion and inflates it most for the
largest sidechains, which is the observed ordering (Leu and Tyr at 17, Ile at
7).

The Ile outlier is worth a separate look: dA_folded is *negative*, meaning the
mutant is more stable folded, which should not happen for a buried deletion.
The likely cause is that Ile6's template rotamer starts strained under this
force field, so removing it releases spurious clash energy. This is the same
suspicion recorded in the vdW blowup work and is still unproven.

## Not established

- No absolute ddG is calibrated. Do not quote one.
- `S_conf` is a first-order occupancy entropy over independent torsions and so
  an upper bound. It is only used here inside differences.
- The frozen-backbone offset of R*T*ln24 = 1.89 kcal/mol per backbone torsion
  cancels in this double difference because WT and mutant share backbone
  torsion count. It does not cancel in a single folding free energy.

## Correction: solvation is present, and was never missing

The section above attributes the 3-4x magnitude error to "no solvation term".
That is wrong. All three solvation terms are live at factor 1.0 and all three
are summed into the total (energy.cu:2958). On 1crn, default parameters:

    vdw -85.94  elec -98.62  solvPolar -389.14  solvNonpolar -171.86
    solvEntropy +716.64  total 492.79

Solvation is the largest thing in the energy, not an absent one. The -10 to -17
kcal/mol of dA_unfolded *is* desolvation credit being paid.

Two traps that produced the misreading:

  - Name collision. `residue::EntropyFactor`, which protMinRep sets to 0.0, is
    the per-residue ROTAMERIC entropy (residue.cc:4140+). The shell-water
    ordering entropy is `energyParams::entropyFactor`, hard-coded to 1.0 at
    energy.cu:311 and reachable from no project file. Reading protMinRep.cc
    suggests entropy is off while the dominant entropy term is unconditionally
    on.

  - solvationNonpolar and solvationEntropy are geometrically degenerate. Both
    are strictly proportional to the same per-atom shell-water count w:
    -eps_i*sqrt(eps_water)*w and +kT*ln2*w. Only the net coefficient is
    observable; no conformational change separates them. Adding a SASA term
    would be a third copy of the same coordinate. Note kT*ln2 is atom-type
    blind -- the sole type dependence in the whole surface term is the weak
    sqrt(eps_i), 0.10-0.18 across C/N/O/S.

## Harness bug: PDBs are not per-arm

run_both.sh renames only ddg_*.txt between the LG=0 and LG=1 passes. Every
wt_*/mi_*/mp_*/mpo_* PDB on disk is therefore overwritten by the second
(legacy) pass. Scoring those structures under default parameters shows a
spurious +94.89 vdW; scored under legacy, their own arm, wt_1 reads -129.73.
Rename the PDBs per arm before drawing any conclusion from them.

## Negative result: sidechain step size is not the bottleneck

The generic span rule `180/(chis distal within residue)` floors distal at 2, so
the terminal chi always steps +/-90 deg and the finest move anywhere is +/-45.
That looked like the sidechain analogue of the backbone bbSpan defect. It is
not. `PROTCAD_MC_SCSPAN` (added, default off, trajectories bit-identical when
unset) sweeps it on 1crn at 12k sweeps / 32 replicas / seed 1:

    span     accept%   <E>       minE      T*S
    legacy   13.5      449.36    433.94    42.55
    60       17.6      447.35    434.25    40.72
    30       29.0      447.96    433.72    41.77
    15       45.0      446.78    433.07    40.30
    8        58.7      448.78    433.14    40.69
    4        69.2      449.33    434.31    38.67

Acceptance rises 13.5 -> 69% and <E> does not move: 2.6 kcal/mol of scatter
against a 4.65 sample sd, minE flat at 433-434. Sidechain sampling was already
converged. Small spans are mildly harmful: T*S falls monotonically because
fewer torsion bins are visited per sweep, the same undersampling-from-below
seen with nReplicas=1. Keep the knob for diagnostics; do not set it.

## The energy surface is too stiff at thermal amplitude

Isotropic Gaussian jitter on the relaxed default structure, 3 draws per sigma,
baseline vdW -93.97:

    sigma  dVdW        dTotal      dVdW/sigma^2
    0.01     0.58        2.46         5833
    0.02     3.54       11.84         8842
    0.05    27.65       82.38        11059
    0.10   130.92      346.84        13092
    0.15   379.07      845.52        16847
    0.20  7974.16     8724.38       199354
    0.30 134633.27   135696.77      1495925

Harmonic response would hold dE/sigma^2 constant. It drifts 2.9x out to 0.15 A,
then detonates between 0.15 and 0.20. Harmonic extrapolation from the sigma->0
limit predicts 525 kcal/mol at sigma=0.3; the measured value is 134633, a
factor of 256.

Scale: equipartition gives 3N/2*kT = 579 kcal/mol at 300 K for 648 atoms, which
the model reaches at sigma ~ 0.16-0.17 A. 1crn's own B-factors (mean B = 6.92
A^2) give RMSF = 0.296 A. The amplitude this protein is experimentally observed
to sample sits on the far side of the cliff, where the force field charges
~1e5 kT.

Caveat: independent per-atom jitter is not the physical ensemble. Real motion is
collective and the torsional MC never changes bond lengths; 1-2/1-3 pairs are
excluded so they contribute nothing, but uncorrelated noise still drives more
pairs into close approach than a collective mode would. Treat 0.17 A as a lower
bound on the model's thermal amplitude. The 256x anharmonicity is a property of
the curve and is unaffected.

Mechanism: the LJ 12-6 has no shoulder near its zero crossing. For rm = ri + rj
the minimum is at rm, the zero crossing at 0.891*rm, and the only inflection at
1.109*rm -- on the attractive side, beyond the minimum. The slope at the zero
crossing is already -0.768 kcal/mol/A and more than doubles 0.16 A further in.
r^-12 was chosen because it is the square of r^-6, not for physics; real
exchange repulsion is exponential (Born-Mayer / Buckingham exp-6) and softer.

Next lever, replacing "add a nonpolar SASA term": soften the repulsive wall
(exp-6, or a soft-core). This is also the standard fix for the endpoint
singularity in alchemical free energy, which is exactly what X->Ala is, and it
is the same wall behind the 3e5 kcal/mol blowups from 0.001 A pairs.

## Settled negative: the repulsive wall is not what inflates ddG

The section above nominates softening the wall as the next lever. It was
implemented (exp-6 and soft-core, bf200af) and measured. **It does not work.**

Two arms, identical binaries and identical budgets (36000 sweeps, 8 seeds, 32
replicas, rigid folded backbone), differing only in PROTCAD_VDW_WALL. Each arm
got its own unfolded reference, because dA_unfolded is wall-dependent and
carrying the LJ values across would have manufactured the answer.

| site | lj dA_f | exp6 dA_f | lj dA_u | exp6 dA_u | lj ddG | exp6 ddG |
|---|---|---|---|---|---|---|
| I6A  | -11.54 | -11.77 | -16.91 | -16.42 |  5.37 |  4.65 |
| L17A |   2.96 |  -0.92 | -13.71 | -13.28 | 16.67 | 12.36 |
| Y28A |   6.40 |   8.32 | -12.88 | -11.59 | 19.28 | 19.91 |

Target is 2-5 kcal/mol for a buried hydrophobic deletion. exp-6 brings L17A
down 4.3 but leaves it 2.5x high, moves Y28A not at all, and only preserves
I6A, which was already in range. The pre-registered readout was "L17A and Y28A
come down toward 2-5 with all three signs positive". Signs held; magnitudes did
not. The arm fails.

Autopsy. exp-6 did what it was built to do -- anharmonicity over the thermal
amplitude falls from 135x above harmonic to 5.0x -- and ddG barely responded.
That is the informative part. If r^-12 were setting the magnitude, all three
sites would have moved down together, roughly in proportion to buried contact
area. Instead the responses have different signs and no common scale. The 3-4x
error is therefore not stored in the shape of the repulsive wall.

It also costs precision: sd(dA_folded) goes from 0.53-0.60 under lj to
0.76-1.13 under exp-6, as a flatter landscape admits more basin diversity.
Nothing bought, precision spent. **Default stays lj.** exp-6 and soft-core are
retained as lambda-path devices, which is what soft-core was already reduced to
(see bf200af for why it is disqualified as an endpoint Hamiltonian).

Where the error must live instead. The unfolded leg is precise to sd 0.02-0.06
-- a tripeptide converges hard -- so both the noise and the inflation sit
entirely on the folded side. The residual ordering is Tyr > Leu >> Ile, which
tracks sidechain size and burial rather than wall stiffness. Combined with the
solvation correction above (all three solvation terms are live, and large),
that points at the balance between the solvation terms, not at vdW. The
type-specific solvent-entropy coefficient is the next arm.

Also settled, secondarily: I6A's negative dA_folded is not a wall artefact. The
standing explanation was a strained Ile6 template rotamer releasing spurious
clash energy, which predicts the anomaly softens as the wall softens. It does
not move at all (-11.54 -> -11.77). That explanation is dead; the anomaly is
still unexplained.

### Provenance note on the unfolded leg

The dA_unfolded values in the uncalibrated-ddG table further up were produced by
a build_gxg.py that was never committed and is now gone from disk; only its
.pyc survived, in git. The leg has been rebuilt on protBuild as ddg_unfolded.sh.
It reproduces the lost script to better than 0.1 kcal/mol for ALA, ILE and LEU
(-16.91 vs -17.00, -13.71 vs -13.78).

TYR does not reproduce: -12.88 here against -10.57 committed. sd is 0.03, so
this is a real difference in method, not noise, and TYR was likewise the outlier
in a crude smoke test. The exp6-vs-lj comparison above is unaffected, since both
arms use this same leg, but the absolute Y28A ddG should be held loosely until
the discrepancy is explained.
