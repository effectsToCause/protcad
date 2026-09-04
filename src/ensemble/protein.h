//*****************************************************************************************************
//*****************************************************************************************************
//******************************  ___  ____ ____ ___ ____ _ _  _  _  _   ******************************
//******************************  |__] |__/ |  |  |  |___ | |\ |  |__|   ******************************
//******************************  |    |  \ |__|  |  |___ | | \| o|  |   ******************************
//******************************                                         ******************************
//******************************      class protein is defined           ******************************
//*****************************************************************************************************
//*****************************************************************************************************

//--Class Setup----------------------------------------------------------------------------------------
#include "pcAssert.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <random>
#include "typedef.h"
#include "ran.h"
#include "molecule.h"
#include "chain.h"
#include "chainModBuffer.h"
#include "residueTemplate.h"
#ifndef PDBWRITER_H
#include "pdbWriter.h"
#endif
#ifndef ATOMITERATOR_H
class atomIterator;
#endif
#ifndef RESIDUEITERATOR_H
class residueIterator;
#endif
#ifndef ENERGY_H
#include "energy.h"
#endif
#ifndef PROTEIN_H
#define PROTEIN_H
class protein : public molecule
{
public:
	friend class atomIterator;
    friend class residueIterator;

//--Functions--------------------------------------------------------------------------------------------	
	
	//--Constructor and destructor declaration
	protein();
	protein(const string& _name);
	protein(const protein& _rhs);
	~protein();

	//--Coordinate and atom functions
	static UInt getHowMany() {return howMany; }
	void add(chain* _pChain);
	chain* getChain(UInt _chainIndex) {return itsChains[_chainIndex]; }
	dblVec getCoords(const UInt _chainIndex, const UInt _resIndex, const string _atomName)
		{ return itsChains[_chainIndex]->getCoords(_resIndex, _atomName);}
	dblVec getCoords(const UInt _chainIndex, const UInt _resIndex, const UInt _atomIndex) 
		{ return itsChains[_chainIndex]->getCoords(_resIndex, _atomIndex);}
	int getNumAtoms();
	UInt getNumAtoms(const UInt _chainIndex, const UInt _resIndex) 
		{ return itsChains[_chainIndex]->getNumAtoms(_resIndex); }
	void setCoords(UInt _chainIndex, UInt _resIndex, UInt _atomIndex, dblVec _coords)
		{ return itsChains[_chainIndex]->setCoords(_resIndex, _atomIndex, _coords);}
	void setAllCoords( UInt chainIndex, UInt resIndex, vector<dblVec> allCoords);
	void makeAtomSilent(const UInt _chainIndex, const UInt _residueIndex, const UInt _atomIndex);
	void makeResidueSilent(const UInt _chainIndex, const UInt _residueIndex);
	vector <chainPosition*> getChainPositionVector(const UInt _chain);
	UIntVec getItsIndependentChainsMap() { return itsIndependentChainsMap; }
	vector <vector <int> > getItsChainLinkageMap() { return itsChainLinkageMap; }
	vector <dblVec> saveCoords( UInt chainIndex, UInt resIndex);
	void finishProteinBuild();
	void listSecondaryStructure();
	void listDihedrals();
	double getRadius(UInt chainIndex, UInt resIndex, UInt atomIndex) {return itsChains[chainIndex]->getRadius(resIndex, atomIndex);}
	char getChainID(UInt chainIndex) {return itsChains[chainIndex]->getChainID();}
	void listConnectivity(UInt _chainIndex, UInt _resIndex) {return itsChains[_chainIndex]->listConnectivity(_resIndex);}

	
	//--Organization functions
	void updateTotalNumResidues();
	void initializeModificationMethods();

	void resetAllBuffers();
	static void silenceMessages() {messagesActive = false; }
	void accessChainZeroResZero();
	void symmetryLinkChainAtoB(UInt _aIndex, UInt _bIndex);
	void printAllLinkageInfo();
	int getIndexFromResNum(UInt _chainIndex, UInt _resnum);
	UInt getNumChains() const {return itsChains.size();}
	UInt getNumBpt(UInt restype) {return residue::getNumBpt(restype);}
	int getResNum(UInt _chainIndex, UInt _resIndex) {return itsChains[_chainIndex]->getResNum(_resIndex);}
	void setResNum(UInt _chainIndex, UInt _resIndex, const UInt _num) {itsChains[_chainIndex]->setResNum(_resIndex, _num);}
	UInt getNumResidues(UInt _chainIndex) const;
	UInt getTypeFromResNum(UInt _chainIndex, UInt _resNum) { return itsChains[_chainIndex]->getTypeFromResNum(_resNum);}
	string getTypeStringFromAtomNum(UInt _chainIndex, UInt _resNum, UInt _atomNum) { return itsChains[_chainIndex]->getTypeStringFromAtomNum(_resNum, _atomNum);}
	string getTypeStringFromResNum(UInt _chainIndex, UInt _resNum) {return itsChains[_chainIndex]->getTypeStringFromResNum(_resNum);}
	void removeResidue(UInt _chainIndex, UInt _resNum) {return itsChains[_chainIndex]->removeResidue(_resNum);}
	void removeChain(UInt _chainIndex);


	//--Mutations functions
	void stripToGlycine();
	void activateForRepacking(const UInt _chainIndex, const UInt _residueIndex);
	void activateForRepacking(const UInt _chainIndex, const UInt _start, const UInt _end);
	void activateAllForRepacking(const UInt _chainIndex);
	void setResNotAllowed(const UInt _chainIndex, const UInt _residueIndex, const UInt _resType);
	void setListNotAllowed(const UInt _chainIndex, const UInt _residueIndex, const vector <UInt> _typeIndexVector);
	void setResAllowed(const UInt _chainIndex, const UInt _residueIndex, const UInt _resType);
	UIntVec getResAllowed (const UInt _chainIndex, const UInt _residueIndex);
	void setOnlyHydrophilic(const UInt _chainIndex, const UInt _residueIndex);
	void setOnlyROCHydrophobic(const UInt _chainIndex, const UInt _residueIndex);
	void setOnlyCharged(const UInt _chainIndex, const UInt _residueIndex);
	void setOnlyNativeIdentity(const UInt _chainIndex, const UInt _residueIndex);
	void setAllHydrogensOn(const bool _hydrogensOn);
	void setAllPolarHydrogensOn(const bool _polarHydrogensOn);
	void mutate(const UInt _chainIndex, const UInt _resIndex, const UInt _aaIndex);
	void mutateWBC(const UInt _chainIndex, const UInt _resIndex, const UInt _aaIndex);
	int mutate(vector <int> _position, UInt _resType);
	bool performRandomMutation(ran& _ran);
	bool performRandomMutation(ran& _ran, vector <int> _position);	
	residue* superimposeGLY(const UInt _chain, const UInt _residue);
	void commitLastMutation();
	void undoLastMutation();
	void saveCurrentState();
	void undoState();
	void commitState();	
	void saveState(string& _filename);
	void saveState(string _filename);
	void setupSystem(ran& _ran);
	vector <int> getLastModification();
	vector <int> chooseNextTargetPosition(ran& _ran);
	UInt chooseNextMutationIdentity(ran& _ran, vector <int> _position);
	int modify(ran& _ran);
	int modify(ran& _ran, vector <int> _position);
	void acceptModification();
	void rejectModification();
	double netCharge();
	
	//--Optimization functions
	void protSampling(UInt iterations);
	void protRelax(UInt _sweeps, bool _backbone);
	void protRelax(UIntVec _frozenResidues, UIntVec _activeChains);
	void cofactorRelax(UInt _plateau);
	void protMin(bool _backbone);
	void protMin(bool _backbone, UIntVec _frozenResidues, UIntVec _activeChains);
	vector <vector < double > > getRotationEnergySurface(vector < UIntVec > _active, UInt _steps, double _stepSize, UInt _activePos, vector <vector< double > > _bestChiArray, double &_lowestEnergy);
	vector < double >  getRotationEnergySurface(UIntVec _active, UInt _steps, double _stepSize, UInt _chiPos, vector <double>  _bestChiArray, double &_lowestEnergy);
	bool isCofactor(UInt chainIndex, UInt resIndex){return itsChains[chainIndex]->isCofactor(resIndex);}   

	//--Energy functions
	void setMoved (UInt chainIndex, UInt resIndex, bool _moved, UInt _EorC) {itsChains[chainIndex]->setMoved(resIndex, _moved, _EorC);}
	void setMoved(bool _moved, UInt EorC);
	bool getMoved(UInt chainIndex, UInt resIndex, UInt EorC) {return itsChains[chainIndex]->getMoved(resIndex, EorC);}
	double protEnergy();
	double protEnergy(UInt chainIndex);
	void updateEnergy();
	void updateEnergy(UInt chainIndex);
	double protEnergy(UInt chainIndex, UInt resIndex);
	double getMedianResidueEnergy();
	double getMedianResidueEnergy(UIntVec _activeChains);
	double getMedianResidueEnergy(UIntVec _activeChains, UIntVec _activeResidues);
	bool boltzmannEnergyCriteria(double _deltaEnergy);
	double boltzmannProbabilityToEnergy(double Pi, double Pj);
	UInt getNumChis(const UInt _chainIndex, const UInt _resIndex, const UInt _bpt) {return itsChains[_chainIndex]->getNumChis(_resIndex,0); }
	UInt getNumHardClashes(UInt chainIndex, UInt resIndex);
	UInt getNumHardClashes();
	void updateClashes();
	UInt getNumHardBackboneClashes();
	void updateBackboneClashes();
	UInt getMedianResidueNumHardClashes();
	double getSolvationEnergy(UInt _chainIndex, UInt _residueIndex) {return itsChains[_chainIndex]->getSolvationEnergy(_residueIndex); }
	double getAtomCharge(UInt _chainNum, UInt _resNum, UInt _atomNum) { return itsChains[_chainNum]->getAtomCharge(_resNum, _atomNum); }
	double calculateHCA_O_hBondEnergy();
	double getHBondEnergy(const UInt _chain1, const UInt _res1, const UInt _chain2, const UInt _res2);
	double getResPairEnergy(const UInt _chain1, const UInt _res1, const UInt _chain2, const UInt _res2);
	double getIntraEnergy(const UInt _chainIndex1, const UInt _resIndex1, const UInt _atomIndex1, const UInt _chainIndex2, const UInt _resIndex2, const UInt _atomIndex2);
	double getPairwiseResidueEnergy(const UInt _chain1, const UInt _res1, const UInt _chain2, const UInt _res2);
	double getDielectric(UInt _chainIndex, UInt _residueIndex) {return itsChains[_chainIndex]->getDielectric(_residueIndex); }
	double getDielectric(UInt _chainIndex, UInt _resIndex, UInt _atomIndex) {return itsChains[_chainIndex]->itsResidues[_resIndex]->itsAtoms[_atomIndex]->getDielectric();}
	double intraEnergy();
	double intraSoluteEnergy(bool _updateDielectrics, UInt _activeChain);
	double interSoluteEnergy(bool _updateDielectrics, UInt _chain1, UInt _chain2);
	double getSoluteEnergy(UInt chainIndex, UInt resIndex, UInt atomIndex, UInt otherChainIndex, UInt otherResIndex, UInt otherAtomIndex);
	double getBackboneHBondEnergy(UInt donorChainIndex, UInt donorResIndex, UInt acceptorChainIndex, UInt acceptorResIndex);
	vector <double> chainBindingEnergy();
	//vector <double> calculateChainIndependentDielectric(chain* _chain, residue* _residue, atom* _atom);
	//vector <double> calculateResidueIndependentDielectric(residue* _residue, atom* _atom);
	void updateDielectrics();
	void updateDielectrics(UInt chainIndex);
	void updateMovedDependence(UInt _EorC);
	//void updatePositionDielectrics(UInt _chainIndex, UInt _residueIndex);
	//void updateChainIndependentDielectrics(UInt _chainIndex);
	//void updateResidueIndependentDielectrics(UInt _chainIndex, UInt _resIndex);
	double intraEnergy(const UInt _chain);
	double intraEnergy(UInt _chain1, UInt _chain2);
	double getRotamerEnergy(UInt _chain, UInt _residue) { return itsChains[_chain]->rotamerEnergy(_residue); }
	double getSelfEnergy(UInt _chainIndex, UInt _residueIndex);
	vector <double> protLigandBindingEnergy(UInt ligChainIndex, UInt ligResIndex);
	static void setTemperature( const double _temp ) { residue::setTemperature(_temp); }
	static double Temperature() { return residue::getTemperature(); }

	// --CUDA related functions
	void loadDeviceMemEnergy();
	void freeDeviceMemEnergy();
	void loadDeviceMemClash();
	void freeDeviceMemClash();
	void loadDeviceMemAll();
	void freeDeviceMemAll();
	double protEnergyCU();
	double protEnergyCU(UInt _chainIndex);
	void updateDielectricsCU();

	// Frozen dielectric.  See energy.h.  Atom indices are in atomIterator
	// order, the same ordering buildEnergyContext uses.
	bool protFreezeDielectricCU();
	bool protThawDielectricAllCU();
	bool protThawDielectricNoneCU();
	// Thaw every atom within _radius of any atom of the given residue.  This is
	// the near-shell exemption the packing work is built on: the far field is
	// held, the neighbourhood of the moved sidechain stays exact.  Returns the
	// number of atoms thawed, or -1 on failure.
	int protThawDielectricNearCU(UInt _chainIndex, UInt _resIndex, double _radius,
	                             bool _accumulate = false);
	void protReleaseDielectricCU();
	// Snapshot the device-order coordinates, so a caller can diff them against
	// the post-move state and thaw around what actually moved.
	// Smallest exemption radius for which the frozen dielectric is provably
	// exact, derived from the occupancy kernel's own neighbour test.
	double protDielectricInfluenceRadiusCU();
	// Evaluate only what the last move changed.  Requires a frozen dielectric
	// and a thaw set built by protThawDielectricForMoveCU.  Returns the changed
	// part and the full torsion energy; see energyComputeDelta.
	int protEnergyDeltaCU(double& _part, double& _torsion);
	// Change the nonbonded cutoff at runtime, for measuring what it buys.
	// The switching window is kept 2 A wide below the cutoff.
	void protSetCutoffCU(double _cutoff);
	double protGetCutoffCU();
	void getDeviceCoordsCU(std::vector<double>& _x, std::vector<double>& _y,
	                       std::vector<double>& _z);
	// Thaw the exact set a move perturbs: every atom within _radius of an atom
	// that actually changed position, taken over both the before and after
	// geometry.  Thawing around the whole residue instead is correct but
	// wasteful, since a chi rotation leaves the backbone and the proximal
	// sidechain atoms where they were.
	int protThawDielectricForMoveCU(const std::vector<double>& _oldX,
	                                const std::vector<double>& _oldY,
	                                const std::vector<double>& _oldZ,
	                                double _radius = 0.0, int* _movedOut = 0,
	                                double _tol = 1e-9);
	// Refresh the held dielectric from the field the last delta left resident,
	// instead of recomputing it.  Exact for the same reason the exemption
	// radius is; see energyRefreezeDielectric.
	bool protRefreezeDielectricCU();
	// How many atoms the current thaw set holds, for reporting how much of the
	// structure a delta actually has to touch.
	int protDielectricThawCountCU();
	// Device-order indices of one residue's atoms.  A sidechain move touches
	// only these, so a candidate batch never needs a full coordinate refresh.
	void residueAtomIndicesCU(UInt _chainIndex, UInt _resIndex,
	                          std::vector<int>& _out);
	// Thaw the union of what a whole batch of candidate conformations perturbs.
	// _cand* are _nCand * numAtoms coordinate arrays in device-order.  Exact:
	// the set is a superset of every individual candidate's, built from a
	// per-moved-atom bounding sphere over the candidates rather than from an
	// O(N |A| K) pairwise scan, which would cost more on the host than the
	// batch saves on the device.
	int protThawDielectricForBatchCU(const std::vector<double>& _oldX,
	                                 const std::vector<double>& _oldY,
	                                 const std::vector<double>& _oldZ,
	                                 const std::vector<double>& _candX,
	                                 const std::vector<double>& _candY,
	                                 const std::vector<double>& _candZ,
	                                 int _nCand, double _radius = 0.0,
	                                 int* _movedOut = 0, double _tol = 1e-9,
	                                 const std::vector<int>* _support = 0,
	                                 bool _coordsCurrent = false);
	// Steepest descent over a candidate batch, evaluated as deltas against a
	// held dielectric rather than as full energies.  _nbCurrent is the
	// nonbonded part of the current total; the winner's nonbonded part and
	// torsion come back in _nbBest and _torBest so the caller can carry the
	// anchor forward.  Returns the winner's total energy, or 1E30 on failure,
	// in which case nothing has been committed.
	double bestSidechainCandidateDeltaCU(UInt _chainIndex, UInt _resIndex,
	                                     UInt _numCandidates,
	                                     std::vector < std::vector <double> > &_bestConf,
	                                     double _nbCurrent, double& _nbBest,
	                                     double& _torBest,
	                                     std::vector<double>* _allEnergies = 0,
	                                     std::vector < std::vector < std::vector <double> > >* _allConfs = 0);
	// Minimiser delta chain.  minDeltaAnchorCU pays for one coupled evaluation
	// and a fresh snapshot; minDeltaCommitCU carries an accepted move forward
	// without one.  Both return false if the chain cannot be maintained, and
	// the caller must then fall back to full evaluations.
	bool minDeltaAnchorCU(double& _nbCurrent, double& _total);
	bool minDeltaCommitCU(double _nbBest, double& _nbCurrent);
	void updateResidueEnergiesCU();
	void updateResidueClashesCU();
	UInt getNumHardBackboneClashesCU();
	int getNumClashesCU();
	void buildEnergyContext();
	int itsDisulfideCount;
	// Override the model used by the next buildEnergyContext().  Intended for
	// validation and ablation; call freeDeviceMemAll() first to force a rebuild.
	void setEnergyParamsOverride(const energyParams& _p);
	int updateDeviceCoords();
	void refreshDeviceCoords(const std::vector<int>& _atoms);
	void snapshotConformationCU(std::vector<double>& _x, std::vector<double>& _y,
	                            std::vector<double>& _z);
	void restoreConformationCU(const std::vector<double>& _x, const std::vector<double>& _y,
	                           const std::vector<double>& _z);
	// Register sidechain rotation groups with the energy context so candidate
	// conformations are generated on the device. Returns the number of groups.
	int buildRotationGroups();
	// Push current atom coordinates to the device and make them resident.
	int syncDeviceCoords();
	int getBatchCoords(int _k, double* _x, double* _y, double* _z);
	// Generate and evaluate nCand candidates entirely on the device.
	int energyRotamerBatch(int _nCand, int _groupBegin, int _nGroups,
	                       const double* _anglesDeg, double* _totals);
	// First group index and chi count for a residue, valid after
	// buildRotationGroups(). Returns false if the residue has no chis.
	bool getRotationGroupRange(UInt _chainIndex, UInt _resIndex, int &_begin, int &_count) const;
	// Evaluate K random sidechain conformations for one residue in a single
	// batched GPU launch and return the best. bestConf receives the winning
	// dihedral set; the protein is left in its original conformation.
	double bestSidechainCandidateCU(UInt _chainIndex, UInt _resIndex, UInt _numCandidates,
	                                vector < vector <double> > &_bestConf);
	bool protEnergyBreakdownCU(energyBreakdown& _out);
	// --- Population Monte Carlo -------------------------------------------
	// Seed nRepl independent Metropolis walkers from the current conformation.
	int setDeviceReplicas(int _nRepl);
	// Propose one move per replica, applied to that replica's own state, and
	// evaluate the whole population in a single launch. Angles are deltas in
	// degrees, row-major [nRepl][angleStride].
	int energyReplicaBatch(int _nRepl, const int* _groupBegin, const int* _nGroups,
	                       const double* _anglesDeg, int _angleStride, double* _totals);
	// Keep the proposals of replicas with accept[k] != 0; the rest are unchanged.
	int commitReplicas(int _nRepl, const int* _accept);
	int getReplicaCoords(int _k, double* _x, double* _y, double* _z);
	// Overwrite every atom position from a flat array in atomIterator order.
	void setCoordsFromArray(const double* _x, const double* _y, const double* _z);
	// Largest chi count over all residues, i.e. the angle stride a replica
	// sweep needs so a mixed set of residue types fits one rectangular array.
	int maxRotationGroupCount() const;
	// Fixed-budget population Monte Carlo over sidechain torsions.
	// Ensemble statistics accumulated over the Metropolis trajectory.
	//
	// The MC loop is a correct canonical chain -- proposals are accepted with
	// the Boltzmann criterion -- so the states it visits are distributed as
	// exp(-E/kT).  Only the single lowest-energy conformation was ever
	// reported, which discards that ensemble.  For a folded protein the
	// minimum is a defensible summary; for an unfolded reference state it is
	// exactly the wrong one, because the unfolded state is defined by the
	// breadth of the distribution rather than its floor.  Worse, the size of
	// the error scales with the number of accessible torsions, which is the
	// very quantity a ddG is trying to resolve.
	struct ensembleStats
	{
		bool   valid;            // false until a run has accumulated samples
		UInt   samples;          // number of (sweep, replica) states counted
		UInt   torsions;         // rotation groups histogrammed
		double meanEnergy;       // <E> over the sampled trajectory
		double sdEnergy;         // sqrt(<E^2> - <E>^2), the canonical fluctuation
		double minEnergy;        // best conformation seen, the legacy estimator
		double conformEntropy;   // S_conf in kcal/(mol K), from torsion occupancy
		double freeEnergy;       // <E> - T*S_conf
		double acceptRate;       // accepted proposals / total, the MC health check
		ensembleStats() : valid(false), samples(0), torsions(0), meanEnergy(0.0),
		                  sdEnergy(0.0), minEnergy(0.0), conformEntropy(0.0),
		                  freeEnergy(0.0), acceptRate(0.0) {}
	};
	const ensembleStats& getEnsembleStats() const {return itsEnsembleStats;}

	void protMinReplicaCU(UInt _sweeps, UInt _nReplicas);
	// Number of disulfide cross-links removed from the nonbonded sum.
	int getDisulfideCount() const {return itsDisulfideCount;}
	// Randomized-order sweeps of energy-based steepest descent over sidechain
	// torsions. Same objective and same GPU path as protMinCU, but bounded by a
	// sweep count instead of a Metropolis plateau, so it is cheap enough to run
	// after every mutation. Returns the final energy.
	double relaxSidechainsCU(UIntVec _frozenResidues, UIntVec _activeChains, UInt _maxSweeps);
	void protRelaxCU(UInt _sweeps, bool _backbone);
	void protRelaxCU(UIntVec _frozenResidues, UIntVec _activeChains);
	// Termination is by diminishing returns per doubling of the budget, not by
	// the consecutive-failure count these routines used to run on -- that
	// counter does not survive a batched move and effectively never fires.  See
	// minStopCU in protein.cc.  `_plateau` is retained only for the legacy path,
	// reachable with PROTCAD_MIN_PLATEAU=1; the live knobs are PROTCAD_MIN_GAIN
	// (kcal/mol bought per doubling, below which the run is spent),
	// PROTCAD_MIN_STALE (consecutive such checkpoints required) and
	// PROTCAD_MIN_MAXSWEEPS (hard bound).
	void protMinCU(bool _backbone, UIntVec _frozenResidues, UIntVec _activeChains, UInt _plateau = 1000);
	void protMinCU(bool _backbone, UInt _plateau = 1000);

	//--Transformation functions
	double getBetaChi(UInt _chainIndex, UInt _residueIndex) {return itsChains[_chainIndex]->getBetaChi(_residueIndex); }
	void setBetaChi(UInt _chainIndex, UInt _residueIndex, double _chi) {return itsChains[_chainIndex]->setBetaChi(_residueIndex, _chi); }
	int setPhi(const UInt _chain, const UInt _res, double _angle);
	int setPsi(const UInt _chain, const UInt _res, double _angle);
	int setDihedral(const UInt _chainIndex, const UInt _resIndex, double _dihedral, UInt _angleType, UInt _direction);
	void updateResiduesPerTurnType();
	UInt getBackboneSequenceType(double RPT, double phi){return itsChains[0]->getBackboneSequenceType(RPT,phi);}
	UInt getBackboneSequenceType(UInt _chainIndex, UInt _resIndex) {return itsChains[_chainIndex]->getBackboneSequenceType(_resIndex);}
	vector <double> getRandConformationFromBackboneType(double _phi, double _psi);
	double getResiduesPerTurn(double phi, double psi) {return itsChains[0]->getResiduesPerTurn(phi,psi);}
	double getResiduesPerTurn(UInt _chainIndex, UInt _resIndex) {return itsChains[_chainIndex]->getResiduesPerTurn(_resIndex);}
	double getPhi(UInt _chain, UInt _res) {return itsChains[_chain]->getPhi(_res);}
	double getPsi(UInt _chain, UInt _res) {return itsChains[_chain]->getPsi(_res);}
	double getAngle(UInt _chain, UInt _res, UInt angleType) {return itsChains[_chain]->getAngle(_res, angleType);}
	double getRMSD(protein* _other);
	void translate(const dblVec& _dblVec);
	void translate(const UInt _index, const dblVec& _dblVec);
	void translate(const UInt _index, const double _x,const double _y,const double _z);
	void translate(const double _x, const double _y, const double _z);
	void translateChain(UInt _chain, const double _x, const double _y, const double _z);
	void translateChainR(UInt _chain, const double _x, const double _y, const double _z);
	void transform(const UInt _index, const dblMat& _dblMat);
	void rotate(const UInt _index, const axis _axis, const double _theta);
	void rotate(const UInt _index, const point& _point, const dblVec& _R_axis, const double _theta);
	void rotate(const axis _axis, const double _theta);
	dblVec getBackBoneCentroid();
	dblVec getBackBoneCentroid(UInt _chain) { return itsChains[_chain]->getBackBoneCentroid(); }
	void coilcoil(const double _pitch);
	void coilcoil(UInt _chain, double _pitch);
	void eulerRotate(const double _phi, const double _theta, const double _psi);
	void undoEulerRotate(const double _phi, const double _theta, const double _psi);
	void eulerRotate(UInt _chain, const double _phi, const double _theta, const double _psi);
	void undoEulerRotate(UInt _chain, const double _phi, const double _theta, const double _psi);
	void rotateChain(UInt _chain, const axis _axis, const double _theta);
	void rotateChainRelative(UInt _chain, const axis _axis, const double _theta);
	void alignToAxis(const axis _axis);

	//--Rotamer functions
	void setRotamerNotAllowed(const UInt _chainIndex, const UInt _resIndex, const UInt _resType, const UInt _bpt, const UInt _rotamer);
	void listAllowedRotamers(UInt _chain, UInt _resIndex);
	void setRotamer(const UInt _chainIndex, const UInt _resIndex, const UInt _bpt, const UInt _rotamer);
	void setRotamerWBC(const UInt _chainIndex, const UInt _resIndex, const UInt _bpt, const UInt _rotamer);
	void setPolarHRotamer(const UInt _chainIndex, const UInt _resIndex, const UInt _rotamer);
	UIntVec getCurrentRotamer(UInt _chainPos, UInt _resIndex) { return itsChains[_chainPos]->getCurrentRotamer(_resIndex); }
	void setCanonicalHelixRotamersOnly(const UInt _chainIndex, const UInt _resIndex, const UInt _resType);
	bool performRandomRotamerChange(ran& _ran);
	bool performRandomRotamerRotation(ran& _ran);
	bool performRandomRotamerChange(ran& _ran, vector <int> _position);
	bool performRandomRotamerRotation(ran& _ran, vector <int> _position);
	void undoLastRotamerChange();
	void undoLastRotamerRotation();
	void commitLastRotamerChange();
	void commitLastRotamerRotation();
	void setCanonicalHelixRotamersOnly(const UInt _chainIndex, const UInt _resIndex);
	void setCanonicalHelixRotamersOnly(const UInt _chainIndex);
	UIntVec getAllowedRotamers(UInt _chainIndex, UInt _resIndex, UInt _resType, UInt _bpt) { return itsChains[_chainIndex]->getAllowedRotamers(_resIndex, _resType, _bpt); }
	vector <UIntVec> getAllowedRotamers(UInt _chainIndex, UInt _resIndex, UInt _resType) { return itsChains[_chainIndex]->getAllowedRotamers(_resIndex, _resType); }
	void setRelativeChi(const UInt _chainIndex, const UInt _resIndex, const UInt _bpt, const UInt _chi, const double _angle);
	void setChi (const UInt _chainIndex, const UInt _resIndex, const UInt _bpt, const UInt _chi, const double _angle);
	double getChi (const UInt _chainIndex, const UInt _resIndex, const UInt _bpt, const UInt _chi) { return itsChains[_chainIndex]->getChi(_resIndex, _bpt, _chi); }
	vector < vector <double> >  getSidechainDihedrals(UInt _chainIndex, UInt _indexInChain) {return itsChains[_chainIndex]->getSidechainDihedralAngles(_indexInChain);}
	void chiRotationTopologyCU(UInt _chainIndex, UInt _indexInChain,
	                           vector<int>& _bpt, vector<int>& _idx,
	                           vector<int>& _axis1, vector<int>& _axis2,
	                           vector<double>& _entry,
	                           vector<int>& _movedFlat, vector<int>& _movedOff)
	{	itsChains[_chainIndex]->chiRotationTopology(_indexInChain, _bpt, _idx, _axis1,
		                                            _axis2, _entry, _movedFlat, _movedOff); }
	void setSidechainDihedralAngles(UInt _chainIndex, UInt _indexInChain, vector< vector<double> > Angles);
	vector< vector< double > > randContinuousSidechainConformation(UInt _chainIndex, UInt _resIndex) {return itsChains[_chainIndex]->randContinuousSidechainConformation(_resIndex);}

	
	//--Surface area functions
	double getVolume(UInt _method);
	void initializeSpherePoints();
	void initializeSpherePoints(UInt _chain);
	void initializeSpherePoints(UInt _chain, UInt _residue);
	void removeSpherePoints();
	void removeSpherePoints(UInt _chain);
	void removeSpherePoints(UInt _chain, UInt _residue);
	double tabulateSurfaceArea();
	double tabulateSurfaceArea(UInt _chain);
	double tabulateSurfaceArea(UInt _chain, UInt _residue);
	double tabulateSurfaceArea(UInt _chainIndex, UInt _residueIndex, UInt _atomIndex);
	double tabulateSolvationEnergy();
	double tabulateSolvationEnergy(UInt _chain);
	double tabulateSolvationEnergy(UInt _chain, UInt _residue);
	double getItsSolvationParam();
	void setItsSolvationParam(UInt _param);

	//Sequence analysis
	double getHammingDistance(vector<string>seq1,vector<string>seq2);
//--Defined functions------------------------------------------------------------------------------------
private:
	bool isValidHelixRotamer ( UInt _resType, UInt _bpt, UInt _allowedRotamer ); // contains our definitions for canonical helix rotamers
	int chooseTargetChain(ran& _ran);
	int chooseModificationMethod(ran& _ran);
	
	//--Variable declarations
	static bool messagesActive;
	static bool calcSelfEnergy;
	static UInt howMany;
	UInt itsNumResidues;
	static UInt itsSolvationParam;
	vector <chain*> itsChains;
	vector <UInt> itsIndependentChainsMap;
	vector<vector<int> > itsChainLinkageMap;
	bool (protein::*itsModificationMethods[5])(ran& _ran);

	//--CUDA variables
	// One opaque handle replaces the eight raw device pointers.  All static
	// per-atom data and the bonded-exclusion lists live inside it, so energy
	// and clash queries share a single allocation instead of each rebuilding
	// their own copy of the topology.
// Declared unconditionally.  Guarding data members with #ifdef __CUDA__ makes
// sizeof(protein) depend on a compile flag, so any translation unit built
// without it disagrees about the object layout -- an ODR violation that shows
// up as heap corruption far from the actual cause.
	energyContext* itsEnergyContext = 0;
	energyParams itsEnergyParams;
	bool itsEnergyParamsSet = false;
	std::vector<double> itsCoordX, itsCoordY, itsCoordZ;
	// atomIterator walks the residue tree through virtual calls, which is a
	// large fixed cost per delta.  Index it once and refresh by atom instead.
	std::vector<atom*> itsAtomPtrs;
	void buildAtomIndex();
	std::vector<int> itsThawList;   // atoms the last move thawed, ascending
	// Per-trial candidate coordinate scratch, addressed k * N + atom because
	// that is the layout both the thaw builder and the batch delta read.  Held
	// across trials rather than rebuilt: only the moved residue and the thaw
	// set are ever read out of these, and both are written in full every trial,
	// so anything left over from the previous trial is unreachable.
	std::vector<double> itsBatchX, itsBatchY, itsBatchZ;
	std::vector<int> itsRotGroupFirst, itsRotGroupCount;   // per residue, walk order
	// Backbone group index per residue, -1 when absent (proline has no free
	// phi).  Needed by the crankshaft proposal, which pairs psi of one residue
	// with phi of the next and so cannot work from a per-residue range alone.
	std::vector<int> itsPhiGroup, itsPsiGroup, itsRotResChain;
	std::vector< std::vector<int> > itsResRotIndex;        // [chain][resInChain] -> walk index
	double E;
	int clash;
	bool deviceMemLoadedEnergy = false;
	bool deviceMemLoadedClash = false;
	bool deviceMemLoadedAll = false;
	ensembleStats itsEnsembleStats;
	
	//--Buffering
	int itsLastModifiedChain;
	int itsLastModificationMethod;
	ran itsRan;

};
#endif
