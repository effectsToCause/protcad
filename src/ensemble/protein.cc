// filename: protein.cc
// contents: class protein implementation

#include "protein.h"
#include <cstring>
#include <chrono>
#include "amberParams.h"
#include <set>
bool protein::messagesActive = false;
typedef vector<UInt>::iterator iterUINT;
typedef vector<int>::iterator iterINT;
typedef vector<vector<int> >:: iterator iterINTVEC;

bool protein::calcSelfEnergy = true;
UInt protein::howMany = 0;
UInt protein::itsSolvationParam = 0;

protein::protein() : molecule()
{
#ifdef PROTEIN_DEBUG
	cout<< "default protein constructor called" << endl;
	cout << itsName << endl;
#endif
	itsChains.resize(0);
	setMoleculeType(1);
	resetAllBuffers();
	itsLastModifiedChain = -1;
	initializeModificationMethods();
	howMany++;
}

protein::protein(const string& _name) : molecule(_name)
{
#ifdef PROTEIN_DEBUG
	cout<< "protein constructor for " << _name << " called" << endl;
#endif
	itsChains.resize(0);
    itsIndependentChainsMap.resize(0);
	itsChainLinkageMap.resize(0);
	itsLastModifiedChain = -1;
	setMoleculeType(1);
	resetAllBuffers();
	initializeModificationMethods();
	howMany++;
}

protein::protein(const protein& _rhs)
{	for (UInt i=0; i<_rhs.itsChains.size(); i++)
	{	add(new chain( *(_rhs.itsChains[i])));
	}
	itsName = _rhs.itsName;
	itsLastModifiedChain = _rhs.itsLastModifiedChain;
	itsLastModificationMethod = _rhs.itsLastModificationMethod;
	initializeModificationMethods();
	setMoleculeType(1);
	itsChainLinkageMap = _rhs.itsChainLinkageMap;
	itsIndependentChainsMap = _rhs.itsIndependentChainsMap;
}

protein::~protein()
{
#ifdef PROTEIN_DEBUG
	cout << "protein destructor called " << endl;
#endif
	for(UInt i=0; i<itsChains.size(); i++)
	{	delete itsChains[i];
	}
    howMany--;
}

void protein::resetAllBuffers()
{
	itsLastModifiedChain = -1;
	itsLastModificationMethod = -1;
}

/*chain& protein::getChain(UInt _chainIndex)
{
	if (_chainIndex < itsChains.size())
		return itsChains(_chainIndex);
	else
	{
		cout << "ERROR:  chain " << _chainIndex << " not found." << endl;
		return itsChains(0);
	}
}*/

void protein::add(chain* _pChain)
{
	itsChains.push_back(_pChain);
	UInt index = itsChains.size() - 1;
	itsIndependentChainsMap.push_back(index);
	vector<int> tempIntVec;
	tempIntVec.resize(0);
	tempIntVec.push_back(-1);
	itsChainLinkageMap.push_back(tempIntVec);
}

void protein::initializeModificationMethods()
{
	bool (protein::* pFunc)(ran&);
	pFunc = &protein::performRandomMutation;
	itsModificationMethods[0] = pFunc;
	pFunc = &protein::performRandomRotamerChange;
	itsModificationMethods[1] = pFunc;
	pFunc = &protein::performRandomRotamerRotation;
	itsModificationMethods[2] = pFunc;
	itsModificationMethods[3] = 0;
	itsModificationMethods[4] = 0;
}
void protein::removeChain(UInt _chainIndex)
{
	
	for (UInt i = 0; i < itsChains.size(); i++)
	{
		if (i > _chainIndex)
		{
			cout << "map " << itsIndependentChainsMap[i] << endl;
			itsIndependentChainsMap[i] = itsIndependentChainsMap[i]-1;
		}
	}
	delete itsChains[_chainIndex];
	itsChains.resize(itsChains.size()-1);
	
}

void protein::symmetryLinkChainAtoB(UInt _aIndex, UInt _bIndex)
{
	//cout << "Before Linkage" << endl;
	//printAllLinkageInfo();
	// first, remove index of chain A from itsIndependentChainsMap
	UInt Aindex=0;
	UInt Bindex=0;
	bool AfoundInIndependentList = false;
	bool BfoundInIndependentList = false;
	// find chain A in ChainsMap
	for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
	{	if ( _aIndex == itsIndependentChainsMap[i])
		{	Aindex = i;
			AfoundInIndependentList = true;
		}
	}

	if (AfoundInIndependentList)
	{
		// remove the cell from itsIndependentChainsMap
		iterUINT firstUINT;
		firstUINT = itsIndependentChainsMap.begin();
		if ( Aindex < itsIndependentChainsMap.size())
		{       itsIndependentChainsMap.erase(firstUINT + Aindex);
		}

		// also remove the corresponding cell from itsChainLinkageMap, if it exists,
		// and retain the data
		vector<int> subLinkages = itsChainLinkageMap[Aindex];
		iterINTVEC firstINTVEC;
		firstINTVEC = itsChainLinkageMap.begin();
		if ( Aindex < itsChainLinkageMap.size())
		{	itsChainLinkageMap.erase(firstINTVEC + Aindex);
		}

		if (subLinkages.size() == 1 && subLinkages[0] == -1)
		{	// its not linked to anthing -- ignore
		}
		else
		{
		// we need to so something about sublinkages here.... recursion?
			for (UInt i=0; i<subLinkages.size(); i++)
			{
				symmetryLinkChainAtoB(subLinkages[i], _bIndex);
			}
		}
	}

	// now, find out where chain B resides in the ChainsMap
	for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
	{	if ( _bIndex == itsIndependentChainsMap[i])
		{	Bindex = i;
			BfoundInIndependentList = true;
		}
	}
	if (BfoundInIndependentList)
	{
		// is this the first one?
		if ( itsChainLinkageMap[Bindex].size() == 1 && itsChainLinkageMap[Bindex][0] == -1)
		{	itsChainLinkageMap[Bindex][0] = _aIndex;
		}
		else
		{	itsChainLinkageMap[Bindex].push_back(_aIndex);
		}
	}
	else
	{	cout << "Error in  protein::symmetryLinkChainAtoB" << endl;
		cout << "Cannot find the independent chain " << _bIndex << " specified" << endl;
	}
	// now we need to copy active residue information from chain B to chain A

	//cout << "After Linkage" << endl;
	//cout << endl;
	if (messagesActive)	cout << "Chain " << _aIndex << " linked to chain " << _bIndex << endl;
	printAllLinkageInfo();
}

void protein::printAllLinkageInfo()
{
	if (messagesActive) cout << "Linkage Info:" << endl << "--------------" << endl;
	UInt numIndependentChains = itsIndependentChainsMap.size();
	if (messagesActive) cout << "Independent Chains" << endl;
	for (UInt i=0; i<numIndependentChains; i++)
	{	if (messagesActive) cout << itsIndependentChainsMap[i] << " " ;
	}
	if (messagesActive) cout << endl;

	UInt sizeOfChainLinkageMap = itsChainLinkageMap.size();

	if (messagesActive) cout << "Linked Chains" << endl;
	UInt largest = 0;
	for (UInt i=0; i<sizeOfChainLinkageMap; i++)
	{	if (itsChainLinkageMap[i].size() > largest)
		{	largest = itsChainLinkageMap[i].size();
		}
	}

	UInt counter = 0;
	while (counter <= largest)
	{
		for (UInt i=0; i<sizeOfChainLinkageMap; i++)
		{
			if (counter < itsChainLinkageMap[i].size())
			{	if (itsChainLinkageMap[i][counter] != -1)
				{	if (messagesActive) cout << itsChainLinkageMap[i][counter];
					if (messagesActive) cout << " ";
				}
				else
				{	if (messagesActive) cout << "- ";
				}
			}
		}
		if (messagesActive) cout << endl;
		counter++;
	}
	if (messagesActive) cout << "Done with linkage info" << endl;
}

void protein::activateForRepacking(const UInt _chainIndex, const UInt _residueIndex)
{
	if (_chainIndex < itsChains.size())
	{	if( !(itsChains[_chainIndex]->activateForRepacking(_residueIndex)))
		{	//cout << "Activation Failure on chain " << _chainIndex;
			//cout << ", residue " << _residueIndex << endl;
		}
	}
	else
	{	cout << "Error from protein::activateForRepacking" << endl;
		cout << "chain index out of bounds :" << _chainIndex << endl;
	}

	// find chain in independent chain list
	UInt indChainIndex = 0;
	for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
	{	if (itsIndependentChainsMap[i] == _chainIndex)
		{	indChainIndex = i;
		}
	}

	UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
	if ( /*numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
	{       for (UInt i=0; i<numSymLinkedChains;i++)
		{
			itsChains[itsChainLinkageMap[indChainIndex][i]]->activateForRepacking(_residueIndex);
		}
	}
}

void protein::activateForRepacking(const UInt _chainIndex, const UInt _start, const UInt _end)
{
	for (UInt i=_start; i<=_end; i++)
	{	activateForRepacking(_chainIndex,i);
	}
}

void protein::activateAllForRepacking(const UInt _chainIndex)
{
	if (_chainIndex < itsChains.size())
	{	UInt tempint = itsChains[_chainIndex]->getNumResidues();
		for (UInt i=0; i<tempint; i++)
		{	activateForRepacking(_chainIndex,i);
		}
	}
	else
	{	cout << "Error from protein::activateAllForRepacking" << endl;
		cout << "chain index out of bounds :" << _chainIndex << endl;
	}
}

UIntVec protein::getResAllowed (const UInt _chainIndex, const UInt _residueIndex)
{
	UIntVec allowedResidues;
	if (_chainIndex < itsChains.size() && _chainIndex >= 0)
	{
		allowedResidues = itsChains[_chainIndex]->getResAllowed(_residueIndex);
		return allowedResidues;
	}
	else
	{
		cout << "Error from protein::getResAllowed ... chain value passed is " << _chainIndex << endl;
		allowedResidues.resize(0);
		return allowedResidues; // return null list
	}
}


void protein::setResNotAllowed(const UInt _chainIndex, const UInt _residueIndex, const UInt _residueType)
{
	if (_chainIndex < itsChains.size())
	{	itsChains[_chainIndex]->setResNotAllowed(_residueIndex, _residueType);
	}
	else
	{	cout << "Error from protein::setResNotAllowed" << endl;
		cout << "chain index out of bounds :" << _chainIndex << endl;
	}

	// find chain in independent chain list
	UInt indChainIndex = 0;
	for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
	{	if (itsIndependentChainsMap[i] == _chainIndex)
		{	indChainIndex = i;
		}
	}

	UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
	if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
	{       for (UInt i=0; i<numSymLinkedChains;i++)
		{
			itsChains[itsChainLinkageMap[indChainIndex][i]]->setResNotAllowed(_residueIndex, _residueType);
		}
	}
}

void protein::setResAllowed(const UInt _chainIndex, const UInt _residueIndex, const UInt _residueType)
{
	if (_chainIndex < itsChains.size())
	{	itsChains[_chainIndex]->setResAllowed(_residueIndex, _residueType);
	}
	else
	{	cout << "Error from protein::setResAllowed" << endl;
		cout << "chain index out of bounds :" << _chainIndex << endl;
	}
	// find chain in independent chain list
	UInt indChainIndex = 0;
	for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
	{	if (itsIndependentChainsMap[i] == _chainIndex)
		{	indChainIndex = i;
		}
	}

	UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
	if ( /*numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
	{       for (UInt i=0; i<numSymLinkedChains;i++)
		{
			itsChains[itsChainLinkageMap[indChainIndex][i]]->setResAllowed(_residueIndex, _residueType);
		}
	}
}

void protein::setOnlyCharged(const UInt _chainIndex, const UInt _residueIndex)
{
	// only allow LYS, ARG, GLU, HIS and ASP

        setResNotAllowed(_chainIndex,_residueIndex,0); //ALA
        setResNotAllowed(_chainIndex,_residueIndex,2); //ASN
        setResNotAllowed(_chainIndex,_residueIndex,4); //CYS
        setResNotAllowed(_chainIndex,_residueIndex,5); //GLN
        setResNotAllowed(_chainIndex,_residueIndex,7); //GLY
        setResNotAllowed(_chainIndex,_residueIndex,9); //ILE
        setResNotAllowed(_chainIndex,_residueIndex,10); //LEU
        setResNotAllowed(_chainIndex,_residueIndex,12); //MET
        setResNotAllowed(_chainIndex,_residueIndex,13); //PHE
        setResNotAllowed(_chainIndex,_residueIndex,14); //PRO
        setResNotAllowed(_chainIndex,_residueIndex,15); //SER
        setResNotAllowed(_chainIndex,_residueIndex,16); //THR
        setResNotAllowed(_chainIndex,_residueIndex,17); //TRP
        setResNotAllowed(_chainIndex,_residueIndex,18); //TYR
        setResNotAllowed(_chainIndex,_residueIndex,19); //VAL
		return;
}

void protein::setListNotAllowed(const UInt _chainIndex, const UInt _residueIndex, const vector <UInt> _typeIndexVector)
{
	for (UInt i = 0; i < _typeIndexVector.size(); i++)
		setResNotAllowed(_chainIndex,_residueIndex,_typeIndexVector[i]);
	return;
}

void protein::setOnlyHydrophilic(const UInt _chainIndex, const UInt _residueIndex)
{
		setResNotAllowed(_chainIndex,_residueIndex,0); //ALA
		setResNotAllowed(_chainIndex,_residueIndex,4); //CYS
		setResNotAllowed(_chainIndex,_residueIndex,7); //GLY
		setResNotAllowed(_chainIndex,_residueIndex,9); //ILE
		setResNotAllowed(_chainIndex,_residueIndex,10); //LEU
		setResNotAllowed(_chainIndex,_residueIndex,12); //MET
		setResNotAllowed(_chainIndex,_residueIndex,13); //PHE
		setResNotAllowed(_chainIndex,_residueIndex,14); //PRO
		setResNotAllowed(_chainIndex,_residueIndex,15); //SER
		setResNotAllowed(_chainIndex,_residueIndex,16); //THR
		setResNotAllowed(_chainIndex,_residueIndex,17); //TRP
		setResNotAllowed(_chainIndex,_residueIndex,18); //TYR
		setResNotAllowed(_chainIndex,_residueIndex,19); //VAL
}

void protein::setOnlyROCHydrophobic(const UInt _chainIndex, const UInt _residueIndex)
{
		setResNotAllowed(_chainIndex,_residueIndex,0); //ALA
		setResNotAllowed(_chainIndex,_residueIndex,1); //ARG
		setResNotAllowed(_chainIndex,_residueIndex,2); //ASN
		setResNotAllowed(_chainIndex,_residueIndex,3); //ASP
		setResNotAllowed(_chainIndex,_residueIndex,4); //CYS
		setResNotAllowed(_chainIndex,_residueIndex,5); //GLN
		setResNotAllowed(_chainIndex,_residueIndex,6); //GLU
		setResNotAllowed(_chainIndex,_residueIndex,7); //GLY
		setResNotAllowed(_chainIndex,_residueIndex,8); //HIS
		setResNotAllowed(_chainIndex,_residueIndex,11); //LYS
		setResNotAllowed(_chainIndex,_residueIndex,12); //MET
		setResNotAllowed(_chainIndex,_residueIndex,14); //PRO
		setResNotAllowed(_chainIndex,_residueIndex,15); //SER
		setResNotAllowed(_chainIndex,_residueIndex,16); //THR
		setResNotAllowed(_chainIndex,_residueIndex,17); //TRP
		setResNotAllowed(_chainIndex,_residueIndex,18); //TYR
}

void protein::setOnlyNativeIdentity(const UInt _chainIndex, const UInt _residueIndex)
{
	itsChains[_chainIndex]->setOnlyNativeIdentity(_residueIndex);
	// find chain in independent chain list
	UInt indChainIndex = 0;
	for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
	{	if (itsIndependentChainsMap[i] == _chainIndex)
		{	indChainIndex = i;
		}
	}

	UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
	if ( /*numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
	{       for (UInt i=0; i<numSymLinkedChains;i++)
		{
			itsChains[itsChainLinkageMap[indChainIndex][i]]->setOnlyNativeIdentity(_residueIndex);
		}
	}
}

void protein::setAllHydrogensOn(const bool _hydrogensOn)
{
	for(UInt i=0; i<itsChains.size();i++)
	{
		itsChains[i]->setAllHydrogensOn(_hydrogensOn);
	}
}

void protein::setAllPolarHydrogensOn(const bool _polarHydrogensOn)
{
	// note: if true, (all) hydrogensOn should be false
	for(UInt i=0;i<itsChains.size();i++)
	{
		itsChains[i]->setAllPolarHydrogensOn(_polarHydrogensOn);
	}
}

double protein::netCharge()
{
    double nCharge = 0.0;
    for(UInt i=0;i<itsChains.size();i++)
    {
        nCharge += itsChains[i]->netCharge();
    }
    return nCharge;
}


void protein::stripToGlycine()
{
	for (UInt theChain =0; theChain<itsChains.size(); theChain++)
	{	for (UInt res=0; res< itsChains[theChain]->getNumResidues(); res++)
		{	mutateWBC(theChain,res,7);
		}
	}
}

int protein::mutate(vector <int> _position, UInt _resType)
{
	if (_position[1] >=0 && _position[1] < (int)itsChains.size())
	{
		mutateWBC(_position[1], _position[2], _resType);
		return 1;
	}
	else cout << "ERROR in protein::mutate(vector<int> _position, UInt _resType) ... chain position passed to function is out of range." << endl;
	return -1;
}

void protein::mutateWBC(const UInt _chainIndex, const UInt _resIndex, const UInt _aaIndex)
{	if(_chainIndex < itsChains.size())
	{	itsChains[_chainIndex]->mutate(_resIndex,_aaIndex);
		itsChains[_chainIndex]->commitLastMutation();
		// find chain in independent chain list
		UInt indChainIndex = 0;
		for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
		{	if (itsIndependentChainsMap[i] == _chainIndex)
			{	indChainIndex = i;
			}
		}

		UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
		if ( /*numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
		{       for (UInt i=0; i<numSymLinkedChains;i++)
			{
				itsChains[itsChainLinkageMap[indChainIndex][i]]->mutate(_resIndex,_aaIndex);
				itsChains[itsChainLinkageMap[indChainIndex][i]]->commitLastMutation();
			}
		}
	}
}

void protein::mutate(const UInt _chainIndex, const UInt _resIndex, const UInt _aaIndex)
{	itsChains[_chainIndex]->mutate(_resIndex,_aaIndex);
	itsChains[_chainIndex]->commitLastMutation();
	// find chain in independent chain list
	UInt indChainIndex = 0;
	for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
	{	if (itsIndependentChainsMap[i] == _chainIndex)
		{	indChainIndex = i;
		}
	}

	UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
	if (itsChainLinkageMap[indChainIndex][0] != -1)
	{       for (UInt i=0; i<numSymLinkedChains;i++)
		{
			itsChains[itsChainLinkageMap[indChainIndex][i]]->mutate(_resIndex,_aaIndex);
			itsChains[itsChainLinkageMap[indChainIndex][i]]->commitLastMutation();
		}
	}
}

void protein::makeAtomSilent(const UInt _chainIndex, const UInt _resIndex, const UInt _atomIndex)
{
	if ( _chainIndex >=0 && _chainIndex < itsChains.size())
	{
		itsChains[_chainIndex]->makeAtomSilent(_resIndex, _atomIndex);
	}
	else
		cout << "ERROR in protein::makeAtomSilent(...) ...\n\t" << _chainIndex << " value for chain index is out of range." << endl;
	return;
}

void protein::makeResidueSilent(const UInt _chainIndex, const UInt _resIndex)
{
	if ( _chainIndex >=0 && _chainIndex < itsChains.size())
	{
		itsChains[_chainIndex]->makeResidueSilent(_resIndex);
	}
	else
		cout << "ERROR in protein::makeAtomSilent(...) ...\n\t" << _chainIndex << " value for chain index is out of range." << endl;
	return;
}

vector <int> protein::getLastModification()
{
	vector <int> position;
	position.resize(0);
	position.push_back(itsLastModifiedChain);
	position.push_back(itsChains[itsLastModifiedChain]->getLastModificationPosition());
	return position;
}

int protein::modify(ran& _ran, vector <int> _position)
{
    if (chainPosition::getHowMany() == 0)
    {
        cout << "Error reported from protein::modify()" << endl;
        cout << "No chain positions have been activated for modification!" << endl;
        return -2;
    }

    int modificationMethod = chooseModificationMethod(_ran);
    if (modificationMethod >= 0)
    {
		itsLastModificationMethod = modificationMethod;
		switch (modificationMethod)
		{
			case 0:
				if(performRandomMutation(_ran, _position)) return 1;
				break;
			case 1:
				if(performRandomRotamerChange(_ran, _position)) return 1;
				break;
			case 2:
				if(performRandomRotamerRotation(_ran, _position)) return 1;
				break;
			default:
				cout << "ERROR or ABORT in protein::modify(_ran, _position)" << endl;
				return -1;
		}
    }
    return -1;
}

int protein::modify(ran& _ran)
{
    if (chainPosition::getHowMany() == 0)
    {
		cout << "Error reported from protein::modify()" << endl;
		cout << "No chain positions have been activated for modification!" << endl;
        return -2;
    }
	int modificationMethod = chooseModificationMethod(_ran);
	if (modificationMethod >= 0)
	{
        if( (this->*itsModificationMethods[modificationMethod])(_ran) )
		{
			itsLastModificationMethod = modificationMethod;
			return 1;
		}
	}
	return -1;
}

bool protein::performRandomMutation(ran& _ran)
{	int chainToModify = chooseTargetChain(_ran);
	if (messagesActive) cout << "CHAIN " << chainToModify << " ";
	if (messagesActive) cout << "MUT ";
	if (chainToModify >= 0)
	{
		vector<chainModBuffer> theBuffers;
		theBuffers = itsChains[chainToModify]->performRandomMutation(_ran);
		/*
		cout << "ChainModBuffers for chain " << chainToModify << endl;
		theBuffers[0].printAll();
		theBuffers[1].printAll();
		*/
		if ( theBuffers[0].containsData() )
		{	itsLastModifiedChain = chainToModify;
			itsLastModificationMethod = 0;
			// find chain in independent chain list
			UInt indChainIndex = 0;
			for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
			{	if (int(itsIndependentChainsMap[i]) == chainToModify)
				{	indChainIndex = i;
				}
			}

			UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
			if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
			{       for (UInt i=0; i<numSymLinkedChains;i++)
				{
					itsChains[itsChainLinkageMap[indChainIndex][i]]->repeatModification(theBuffers[1]);
				}
			}
			return true;
		}
		else
		{
		//	cout << "Protein level has detected abort..." << endl;
		}

	}
	else
	{	cout << "Error from protein::performRandomMutation" << endl;
		cout << "Protein contains no chains" << endl;
	}
	resetAllBuffers();
	return false;
}

bool protein::performRandomMutation(ran& _ran, vector <int> _position)
{
	int chainToModify = _position[1];
	if (messagesActive) cout << "CHAIN " << chainToModify << " ";
	if (messagesActive) cout << "MUT ";
	if (chainToModify >= 0)
	{
		vector<chainModBuffer> theBuffers;
		theBuffers = itsChains[chainToModify]->performRandomMutation(_ran, _position);
		if (theBuffers[0].containsData())
		{
			itsLastModifiedChain = chainToModify;
			itsLastModificationMethod = 0;
			UInt indChainIndex = 0;
            for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
            {   if (int(itsIndependentChainsMap[i]) == chainToModify)
                {   indChainIndex = i;
                }
            }

            UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
            if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
            {       for (UInt i=0; i<numSymLinkedChains;i++)
                {
                    itsChains[itsChainLinkageMap[indChainIndex][i]]->repeatModification(theBuffers[1]);
                }
            }
            return true;
        }
        else
        {
        //  cout << "Protein level has detected abort..." << endl;
        }

    }
    else
    {   cout << "Error from protein::performRandomMutation" << endl;
        cout << "Protein contains no chains" << endl;
    }
    resetAllBuffers();
    return false;
}

void protein::setupSystem(ran& _ran)
{
	// try mutation first
	// if mutation fails, try rotamer changes
	vector<UInt> targetID;
	for (UInt i=0; i < itsIndependentChainsMap.size(); i++)
	{
        UInt iChain = itsIndependentChainsMap[i];
        // get number of active residues in chain
        vector<chainPosition*> cpVector;
        vector<UInt> activeMap;
        UIntVec allowedRes;
        cpVector = itsChains[iChain]->getChainPositionVector();
        activeMap = itsChains[iChain]->getRepackActivePositionMap();
        UInt numActive = activeMap.size();
        for (UInt j=0; j< numActive; j++)
        {   if (cpVector[activeMap[j]])
            {   allowedRes = cpVector[activeMap[j]]->getResAllowed();
                UInt tempTargetID = allowedRes[int(_ran.getNext() * (allowedRes.size()-1))];
                mutate(iChain,activeMap[j],tempTargetID);
            }
        }
	}
/*
           UInt numSymLinkedChains = itsChainLinkageMap[iChain].size();
           if ( numSymLinkedChains != 1 && itsChainLinkageMap[iChain][0] != -1)
           {       for (UInt i=0; i<numSymLinkedChains;i++)
               {
                   itsChains[itsChainLinkageMap[iChain][i]]->repeatModification(theBuffers[1]);
               }
           }
*/
}

int protein::chooseModificationMethod(ran& _ran)
{
	// in this 'bare bones' version, every modification
	// method is given equal weight.  Not necessarily
	// the best implementation, but useful for testing
	int methodSize = 3;
	if (methodSize != 0)
	{
		int i =  int(_ran.getNext() * methodSize);
		return i;
	}
	return -1;
}

void protein::acceptModification()
{	if (itsLastModifiedChain >=0)
	{	switch (itsLastModificationMethod)
		{
			case 0:	commitLastMutation();
					break;
			case 1:	commitLastRotamerChange();
					break;
			case 2:	commitLastRotamerRotation();
					break;
			default:	cout << "oops! hit default value in protein::acceptModification";
					cout << endl << "something is definitely wrong here." << endl;
					break;
		}
	}
}

void protein::rejectModification()
{	if (itsLastModifiedChain >=0)
	{	switch (itsLastModificationMethod)
		{
			case 0:	undoLastMutation();
					break;
			case 1:	undoLastRotamerChange();
					break;
			case 2:	undoLastRotamerRotation();
					break;
			default:	cout <<"oops! hit default value in protein::rejectModification";
					cout << endl << "something is definitely wrong here." << endl;
					break;
		}
	}
}

void protein::finishProteinBuild()
{
	for (unsigned int i=0; i< itsChains.size(); i++)
	{
		itsChains[i]->rebuildResiduesInChain();
		itsChains[i]->finishChainBuild();
	}
/*
	vector < UIntVec > artificialPos;
	artificialPos.resize(0);
	for (UInt i = 0; i < itsChains.size(); i++)
	{
		UIntVec artificialChainAndRes;
		for (UInt j = 0; j < itsChains[i]->getNumResidues(); j++)
		{
			if (itsChains[i]->isArtificiallyBuilt(j))
			{
				artificialChainAndRes.push_back(i);
				artificialChainAndRes.push_back(j);
				artificialPos.push_back(artificialChainAndRes);
				artificialChainAndRes.resize(0);
				//cout << "Flag squirted " << i << " "<< j<< endl;
			}
		}
	}
    if (artificialPos.size() <= 10 && artificialPos.size() != 0)
    {
        optimizeRotamers(artificialPos);
    }
    else
    {
        if (artificialPos.size() != 0) cout << "Too many residues to optimize with this algorithm.\n OPTIMIZATION ABORTED." << endl;
    }
 */
    return;
}

void protein::listSecondaryStructure()
{
	if (messagesActive) cout << "SECONDARY STRUCTURE" << endl;
	for (unsigned int i=0; i< itsChains.size(); i++)
	{	if (messagesActive) cout << "CHAIN " << i << endl;
		itsChains[i]->listSecondaryStructure();
	}
}

void protein::listDihedrals()
{
	if (messagesActive) cout << "DIHEDRAL ANGLES" << endl;
	for (unsigned int i=0; i< itsChains.size(); i++)
	{	if (messagesActive) cout << "CHAIN " << i << endl;
		itsChains[i]->listDihedrals();
	}
}

vector <int> protein::chooseNextTargetPosition(ran& _ran)
{
	vector <int> position;
	position.resize(0);

	int chainPos = chooseTargetChain(_ran);
	int resPos = itsChains[chainPos]->chooseNextTargetPosition(_ran);

	position.push_back(chainPos);
	position.push_back(resPos);

	return position;
}

UInt protein::chooseNextMutationIdentity(ran& _ran, vector <int> _position)
{
	if (_position[1] >=0 && _position[1] < (int)itsChains.size())
	{
		return itsChains[_position[1]]->chooseNextMutationIdentity(_ran, _position);
	}
	else cout << "ERROR in chooseNextMutationIdentity at protein level ... chain ID is out of range." << endl;
	return 0;
}

int protein::chooseTargetChain(ran& _ran)
{	int chainSize = itsIndependentChainsMap.size();
	if (chainSize != 0)
	{	return int(itsIndependentChainsMap[int(_ran.getNext() * chainSize)]);
	}
	return -1;
}

void protein::commitLastMutation()
{	if (itsLastModifiedChain >=0 && itsLastModificationMethod == 0)
	{	itsChains[itsLastModifiedChain]->commitLastMutation();
		// find chain in independent chain list
		UInt indChainIndex = 0;
		for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
		{	if (int(itsIndependentChainsMap[i]) == itsLastModifiedChain)
			{	indChainIndex = i;
			}
		}

		UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
		if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
		{       for (UInt i=0; i<numSymLinkedChains;i++)
			{
				itsChains[itsChainLinkageMap[indChainIndex][i]]->commitLastMutation();
			}
		}
		resetAllBuffers();
	}
	else
	{	cout << "Error reported by protein::commitLastMutation()";
		cout << endl << "No last modified chain" << endl;
	}
	if (messagesActive) cout << "ACCEPTED " << endl;
}

void protein::undoLastMutation()
{	if (itsLastModifiedChain >=0 && itsLastModificationMethod == 0)
	{	itsChains[itsLastModifiedChain]->undoLastMutation();
		// find chain in independent chain list
		UInt indChainIndex = 0;
		for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
		{	if (int(itsIndependentChainsMap[i]) == itsLastModifiedChain)
			{	indChainIndex = i;
			}
		}

		UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
		if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
		{       for (UInt i=0; i<numSymLinkedChains;i++)
			{
				itsChains[itsChainLinkageMap[indChainIndex][i]]->undoLastMutation();
			}
		}
		resetAllBuffers();
	}
	else
	{	cout << "Error reported by protein::undoLastMutation()";
		cout << endl << "No last modified chain" << endl;
	}
	if (messagesActive) cout << "REJECTED " << endl;
}

vector <chainPosition*> protein::getChainPositionVector(const UInt _chain)
{	if (itsChains[_chain])
	{
		return itsChains[_chain]->getChainPositionVector();
	}
	else
	{	cout << "Error reported by protein::getChainPositionVector()";
		cout << endl << "chain " << _chain << " is invalid specifier" << endl;
		cout << "Returning null vector - further behavior unpredictable!!!" << endl;
	}
	chainPosition* nullCP = 0;
	vector <chainPosition*> nullVec;
	nullVec.push_back(nullCP);
	return nullVec;
}

double protein::getVolume(UInt _method)
{
	double itsVolume = 0.0;
	for (UInt i = 0; i < itsChains.size(); i++)
	{
		itsVolume += itsChains[i]->getVolume(_method);
	}
	return itsVolume;
}

// get the interaction energy of a particular modified position with the rest of the system, taking into account positions
// symmetry linked to the passed position
int protein::setPhi(const UInt _chain, const UInt _res, double _phi)
{
	if (_chain < itsChains.size())
	{
		return itsChains[_chain]->setPhi(_res, _phi);
	}
	else
	{
		cout << "chain index out of range" << endl;
		return -1;
	}
}

int protein::setPsi(const UInt _chain, const UInt _res, double _psi)
{
	if (_chain < itsChains.size())
	{
		return itsChains[_chain]->setPsi(_res, _psi);
	}
	else
	{
		cout << "chain index out of range" << endl;
		return -1;
	}
}

int protein::setDihedral(const UInt _chainIndex, const UInt _resIndex, double _dihedral, UInt _angleType, UInt _direction)
{
	if (_chainIndex < itsChains.size())
	{
		return itsChains[_chainIndex]->setDihedral(_resIndex, _dihedral, _angleType, _direction);
	}
	else
	{
		cout << "chain index out of range" << endl;
		return -1;
	}
}

//**************CUDA related functions***************************************

#ifdef __CUDA__

// Build the persistent GPU context: static per-atom properties plus the
// bonded-exclusion lists.
//
// The original built a dense N*(N-1)/2 bonding matrix with a host loop over
// every atom pair.  For a 3000-atom protein that is 4.5M isSeparatedBy...
// calls and 54 MB of device memory, and it grew quadratically.  Bonded
// exclusions never reach beyond an adjacent residue, so restricting the search
// to a residue window makes this O(N) with a small constant while producing an
// identical exclusion set.
void protein::setEnergyParamsOverride(const energyParams& _p)
{
	itsEnergyParams = _p; itsEnergyParamsSet = true;
}

void protein::buildEnergyContext()
{
	if (itsEnergyContext) {return;}
	itsDisulfideCount = 0;

	// Enumerate atoms with the same atomIterator used to upload coordinates.
	// Deriving the topology from a separate nested walk over chains/residues
	// risks a different order or count than the coordinate pass, which would
	// silently pair every atom with another atom's radius and exclusions.
	vector<residue*> resPtr; vector<int> resChain, resFirstAtom, resNumAtoms;
	vector<double> hrad, heps, hchg;
	vector<int> hres, atomLocalIndex;
	vector<unsigned char> hsilent;

	int lastChain = -1, lastRes = -1;
	for (atomIterator aIter(this); !(aIter.last()); aIter++)
	{
		residue* pRes = aIter.getResiduePointer();
		int ci = int(aIter.getChainIndex()), rin = int(aIter.getResidueIndex());
		UInt ai = aIter.getAtomIndex();
		if (ci != lastChain || rin != lastRes)
		{
			resPtr.push_back(pRes); resChain.push_back(ci);
			resFirstAtom.push_back(int(hrad.size())); resNumAtoms.push_back(0);
			lastChain = ci; lastRes = rin;
		}
		resNumAtoms.back()++;
		// Atom radius source.  This single array is the radius for the vdW,
		// solvation and clash terms alike, so it has to be the AMBER
		// energy-type table: getVDWEpsilon() on the next line is an AMBER well
		// depth, and pairing it with a radius from a different table would mix
		// two force fields in one Lennard-Jones pair.
		//
		// The alternative, getRadius(), is the element-level table in the
		// atom.cc dataBase.  It agrees with AMBER to within 0.01 A on heavy
		// atoms but assigns every hydrogen a flat 1.090 A, which is a C-H bond
		// length rather than a van der Waals radius, where AMBER resolves H by
		// chemical environment over 1.00-1.49 A.  The CPU clash test uses that
		// element table, so GPU and CPU clash counts differ by roughly 1.8x;
		// that is the CPU being wrong, not a GPU calibration error, and it goes
		// away when the CPU path is retired.  energyParams::clashTolerance is a
		// cube-to-sphere geometry factor and is independent of this choice.
		hrad.push_back(pRes->getVDWRadius(ai));
		heps.push_back(pRes->getVDWEpsilon(ai));
		hchg.push_back(pRes->getCharge(ai));
		hres.push_back(int(resPtr.size()) - 1);
		atomLocalIndex.push_back(int(ai));
		hsilent.push_back(pRes->getAtom(ai)->getSilentStatus() ? 1 : 0);
	}

	const int N = int(hrad.size());
	if (N <= 0) {return;}
	const int numRes = int(resPtr.size());

	// ---------------------------------------------------------------
	// Covalent bond graph.
	//
	// Exclusions used to be derived from residue::isSeparatedByFewBonds and a
	// +/-1 residue window, which answers "are these within three bonds" without
	// ever materialising the graph.  That was enough while every pair inside
	// three bonds was simply deleted.  It is not enough now: Amber does not
	// delete 1-4 pairs, it damps them, so the code has to be able to tell three
	// bonds from two.  It also has to enumerate dihedrals, which is a question
	// about the graph and not about any pair.
	//
	// Building the graph explicitly answers all three, and it subsumes the
	// cross-link special case rather than bolting onto it: a disulfide is just
	// another edge, and 1-4 pairs across it fall out of the same traversal that
	// produces them everywhere else.  The old code handled 1-2 and 1-3 across a
	// disulfide by hand and silently missed 1-4.
	//
	// Cost is linear.  Degree is at most four, so a depth-3 walk from an atom
	// reaches at most 1 + 4 + 12 + 36 nodes regardless of protein size.
	// ---------------------------------------------------------------
	vector< vector<int> > bonded(N);
	vector<int> disulfideS;
	{
		vector<int> sgAtom;
		// Intra-residue, straight from the amber.prep connectivity.
		for (int ri = 0; ri < numRes; ri++)
		{
			for (int ai = 0; ai < resNumAtoms[ri]; ai++)
			{
				int i = resFirstAtom[ri] + ai;
				vector<UInt> nb = resPtr[ri]->getBondedAtoms(UInt(atomLocalIndex[i]));
				for (UInt k = 0; k < nb.size(); k++)
				{
					// The bonding pattern is in residue-local indices; map back
					// through the same enumeration used for coordinates.
					for (int aj = 0; aj < resNumAtoms[ri]; aj++)
					{
						int j = resFirstAtom[ri] + aj;
						if (atomLocalIndex[j] == int(nb[k]) && j != i)
						{ bonded[i].push_back(j); break; }
					}
				}
			}
		}

		// Peptide bonds.  Consecutive residues of the same chain only, and only
		// when the C-N distance is actually a bond: a PDB with a chain break
		// leaves consecutive residue indices metres apart in space, and joining
		// them would invent a dihedral across the gap.
		for (int ri = 0; ri + 1 < numRes; ri++)
		{
			if (resChain[ri] != resChain[ri + 1]) {continue;}
			int cAtom = -1, nAtom = -1;
			for (int k = 0; k < resNumAtoms[ri]; k++)
			{
				int i = resFirstAtom[ri] + k;
				if (resPtr[ri]->getAtom(atomLocalIndex[i])->getName() == "C") {cAtom = i; break;}
			}
			for (int k = 0; k < resNumAtoms[ri + 1]; k++)
			{
				int i = resFirstAtom[ri + 1] + k;
				if (resPtr[ri + 1]->getAtom(atomLocalIndex[i])->getName() == "N") {nAtom = i; break;}
			}
			if (cAtom < 0 || nAtom < 0)
			{
				if (getenv("PROTCAD_TOPO_DEBUG"))
					cout << "  no C/N for peptide bond " << ri << "->" << ri+1
					     << " (C=" << cAtom << " N=" << nAtom << ")" << endl;
				continue;
			}
			dblVec ci = resPtr[hres[cAtom]]->getAtom(atomLocalIndex[cAtom])->getCoords();
			dblVec cj = resPtr[hres[nAtom]]->getAtom(atomLocalIndex[nAtom])->getCoords();
			double dx = ci[0]-cj[0], dy = ci[1]-cj[1], dz = ci[2]-cj[2];
			if (dx*dx + dy*dy + dz*dz > 2.0 * 2.0)
			{
				if (getenv("PROTCAD_TOPO_DEBUG"))
					cout << "  peptide bond " << ri << "->" << ri+1 << " too long: "
					     << sqrt(dx*dx+dy*dy+dz*dz) << " A" << endl;
				continue;
			}
			bonded[cAtom].push_back(nAtom);
			bonded[nAtom].push_back(cAtom);
		}

		// Covalent cross-links (disulfides), found geometrically.
		//
		// residue::isSeparatedByThreeBackboneBonds does contain a disulfide
		// rule, but it only fires for residues typed CYX/CXD.  Structures
		// loaded straight from a PDB commonly keep bonded cysteines typed as
		// reduced CYS -- crambin's three disulfides do, hydrogens and all -- so
		// that guard never catches them.  Testing the SG-SG distance is
		// independent of residue typing and of sequence separation, which is
		// what makes it reliable.
		//
		// A real disulfide is ~2.05 A; 2.5 A is comfortably above that and well
		// below the ~3.5 A of a non-bonded sulfur contact.  Left in the
		// nonbonded sum an unexcluded S-S pair at 2.0 A contributes of order
		// 1e3 kcal/mol through the r^-12 term.
		for (int i = 0; i < N; i++)
		{
			if (resPtr[hres[i]]->getAtom(atomLocalIndex[i])->getName() == "SG")
			{ sgAtom.push_back(i); }
		}
		int numLinks = 0;
		for (UInt a = 0; a < sgAtom.size(); a++)
		{
			for (UInt b = a + 1; b < sgAtom.size(); b++)
			{
				int i = sgAtom[a], j = sgAtom[b];
				if (hres[i] == hres[j]) {continue;}
				dblVec ci = resPtr[hres[i]]->getAtom(atomLocalIndex[i])->getCoords();
				dblVec cj = resPtr[hres[j]]->getAtom(atomLocalIndex[j])->getCoords();
				double dx = ci[0]-cj[0], dy = ci[1]-cj[1], dz = ci[2]-cj[2];
				if (dx*dx + dy*dy + dz*dz > 2.5 * 2.5) {continue;}
				numLinks++;
				bonded[i].push_back(j);
				bonded[j].push_back(i);
				disulfideS.push_back(i);
				disulfideS.push_back(j);
			}
		}
		itsDisulfideCount = numLinks;
	}

	// ---------------------------------------------------------------
	// Exclusions by graph distance.  1-2 and 1-3 are removed outright; 1-4 is
	// stored as -(j+1) and damped in the kernel by SCEE/SCNB.
	// ---------------------------------------------------------------
	vector< vector<int> > excl(N);
	{
		vector<int> dist(N, -1), touched;
		for (int i = 0; i < N; i++)
		{
			touched.clear();
			dist[i] = 0; touched.push_back(i);
			vector<int> frontier(1, i);
			for (int depth = 1; depth <= 3; depth++)
			{
				vector<int> next;
				for (UInt f = 0; f < frontier.size(); f++)
				{
					const vector<int>& nb = bonded[frontier[f]];
					for (UInt k = 0; k < nb.size(); k++)
					{
						int j = nb[k];
						if (dist[j] >= 0) {continue;}
						dist[j] = depth; touched.push_back(j); next.push_back(j);
						excl[i].push_back(depth == 3 ? -(j + 1) : j);
					}
				}
				frontier.swap(next);
			}
			for (UInt k = 0; k < touched.size(); k++) {dist[touched[k]] = -1;}
		}
	}

	// One-off validation of the exclusion rewrite: recompute the old
	// window-based set and report the difference.  Kept because the two schemes
	// answer the same question by different means, so a diff is the cheapest
	// check that the graph is wired correctly.
	if (getenv("PROTCAD_TOPO_DEBUG"))
	{
		vector< set<int> > oldExcl(N), newExcl(N);
		const int WINDOW = 1;
		for (int ri = 0; ri < numRes; ri++)
		{
			residue* res1 = resPtr[ri];
			int lo = (ri - WINDOW > 0) ? ri - WINDOW : 0;
			int hi = (ri + WINDOW < numRes - 1) ? ri + WINDOW : numRes - 1;
			for (int ai = 0; ai < resNumAtoms[ri]; ai++)
			{
				int i = resFirstAtom[ri] + ai;
				int li = atomLocalIndex[i];
				for (int rj = lo; rj <= hi; rj++)
				{
					residue* res2 = resPtr[rj];
					for (int aj = 0; aj < resNumAtoms[rj]; aj++)
					{
						int j = resFirstAtom[rj] + aj;
						if (j == i) {continue;}
						int lj = atomLocalIndex[j];
						bool bnd;
						if (ri == rj) {bnd = res1->isSeparatedByFewBonds(li, lj);}
						else if (resChain[ri] == resChain[rj]) {bnd = res1->isSeparatedByThreeBackboneBonds(li, res2, lj);}
						else {bnd = false;}
						if (bnd) {oldExcl[i].insert(j);}
					}
				}
			}
		}
		for (int i = 0; i < N; i++)
			for (UInt k = 0; k < excl[i].size(); k++)
				newExcl[i].insert(excl[i][k] < 0 ? -excl[i][k] - 1 : excl[i][k]);

		int onlyOld = 0, onlyNew = 0, both = 0;
		for (int i = 0; i < N; i++)
		{
			for (set<int>::iterator it = oldExcl[i].begin(); it != oldExcl[i].end(); ++it)
			{
				if (newExcl[i].count(*it)) {both++;} else {onlyOld++;}
			}
			for (set<int>::iterator it = newExcl[i].begin(); it != newExcl[i].end(); ++it)
			{
				if (!oldExcl[i].count(*it)) {onlyNew++;}
			}
		}
		cout << "[excl diff] shared " << both / 2
		     << "  only-old " << onlyOld / 2
		     << "  only-new " << onlyNew / 2 << endl;
	}

	// ---------------------------------------------------------------
	// Dihedrals.
	//
	// One quartet per central bond per pair of substituents.  Emitted once,
	// keyed on b < c, so a-b-c-d and d-c-b-a are not both generated: they are
	// the same angle and would double the barrier.
	//
	// Impropers are added at every atom with exactly three neighbours that has
	// a matching parameter.  That is the trigonal-centre rule, and on protein
	// topology it selects the planar groups the impropers exist to keep planar
	// -- carbonyl carbons, amide nitrogens, aromatic ring carbons -- and
	// nothing else, because ff14SB simply has no improper for an sp3 centre.
	// ---------------------------------------------------------------
	vector<int> torAtoms;
	vector<double> torParams;
	{
		amberParams ff;
		ff.loadFF14SB();

		vector<string> aType(N);
		for (int i = 0; i < N; i++)
			aType[i] = resPtr[hres[i]]->getAmberTypeName(UInt(atomLocalIndex[i]));

		// A sulfur in a disulfide is Amber type S, not the thiol SH.  protcad
		// finds disulfides geometrically and leaves the residue typed as
		// reduced CYS, so the type read off the template is the thiol one and
		// no CT-S-S-CT parameter would ever match.  Retyping here covers the
		// bonded terms; the nonbonded type is left alone, which is the smaller
		// error and is not this commit's to fix.
		//
		// The same typing failure leaves a thiol hydrogen bonded to a sulfur
		// that already has two heavy neighbours.  That hydrogen should not
		// exist.  Torsions through it are dropped rather than invented, since
		// there is no ff14SB parameter for a three-coordinate sulfur and
		// guessing one would put a barrier on a bond that is itself spurious.
		vector<bool> isSS(N, false), spuriousH(N, false);
		for (UInt k = 0; k < disulfideS.size(); k++)
		{
			int sg = disulfideS[k];
			isSS[sg] = true;
			aType[sg] = "S";
			for (UInt m = 0; m < bonded[sg].size(); m++)
			{
				int h = bonded[sg][m];
				if (resPtr[hres[h]]->getAtom(atomLocalIndex[h])->getName()[0] == 'H')
					{ spuriousH[h] = true; }
			}
		}
		if (!disulfideS.empty() && getenv("PROTCAD_TOPO_DEBUG"))
		{
			int nh = 0;
			for (int i = 0; i < N; i++) {if (spuriousH[i]) {nh++;}}
			cout << "[topo] retyped " << disulfideS.size() << " disulfide S"
			     << ", ignoring " << nh << " leftover thiol H in bonded terms" << endl;
		}

		int missing = 0;
		for (int b = 0; b < N; b++)
		{
			for (UInt kc = 0; kc < bonded[b].size(); kc++)
			{
				int c = bonded[b][kc];
				if (c < b) {continue;}
				for (UInt ka = 0; ka < bonded[b].size(); ka++)
				{
					int a = bonded[b][ka];
					if (a == c) {continue;}
					for (UInt kd = 0; kd < bonded[c].size(); kd++)
					{
						int d = bonded[c][kd];
						if (d == b || d == a) {continue;}
						if (spuriousH[a] || spuriousH[d]) {continue;}
						const vector<amberTorsionTerm>& t =
							ff.torsion(aType[a], aType[b], aType[c], aType[d]);
						if (t.empty())
						{
							missing++;
							if (getenv("PROTCAD_TOPO_DEBUG"))
							{
								cout << "  no torsion for " << aType[a] << "-" << aType[b]
								     << "-" << aType[c] << "-" << aType[d] << endl;
							}
							continue;
						}
						for (UInt m = 0; m < t.size(); m++)
						{
							if (t[m].barrier == 0.0) {continue;}
							torAtoms.push_back(a); torAtoms.push_back(b);
							torAtoms.push_back(c); torAtoms.push_back(d);
							torParams.push_back(t[m].barrier / t[m].divisor);
							torParams.push_back(t[m].phase);
							torParams.push_back(t[m].periodicity);
						}
					}
				}
			}
		}

		for (int c = 0; c < N; c++)
		{
			if (bonded[c].size() != 3) {continue;}
			if (isSS[c]) {continue;}
			int a = bonded[c][0], b = bonded[c][1], d = bonded[c][2];
			const vector<amberTorsionTerm>& t =
				ff.improper(aType[a], aType[b], aType[c], aType[d]);
			if (t.empty()) {continue;}
			for (UInt m = 0; m < t.size(); m++)
			{
				if (t[m].barrier == 0.0) {continue;}
				torAtoms.push_back(a); torAtoms.push_back(b);
				torAtoms.push_back(c); torAtoms.push_back(d);
				torParams.push_back(t[m].barrier / t[m].divisor);
				torParams.push_back(t[m].phase);
				torParams.push_back(t[m].periodicity);
			}
		}

		if (getenv("PROTCAD_TOPO_DEBUG"))
		{
			int n12 = 0, n14 = 0, nb2 = 0;
			for (int i = 0; i < N; i++)
			{
				nb2 += int(bonded[i].size());
				for (UInt k = 0; k < excl[i].size(); k++)
				{
					if (excl[i][k] < 0) {n14++;} else {n12++;}
				}
			}
			cout << "[topo] atoms " << N << "  bonds " << nb2 / 2
			     << "  disulfides " << itsDisulfideCount
			     << "  excl(1-2,1-3) " << n12 / 2 << "  excl(1-4) " << n14 / 2
			     << "  torsion terms " << torParams.size() / 3 << endl;
		}

		if (getenv("PROTCAD_TOPO_DUMP"))
		{
			ofstream f(getenv("PROTCAD_TOPO_DUMP"));
			f.precision(12);
			for (int i = 0; i < N; i++)
			{
				dblVec c = resPtr[hres[i]]->getAtom(atomLocalIndex[i])->getCoords();
				f << "A " << i << " " << aType[i] << " " << c[0] << " " << c[1] << " " << c[2] << endl;
			}
			for (UInt t = 0; t * 4 < torAtoms.size(); t++)
			{
				f << "T " << torAtoms[4*t] << " " << torAtoms[4*t+1] << " "
				  << torAtoms[4*t+2] << " " << torAtoms[4*t+3] << " "
				  << torParams[3*t] << " " << torParams[3*t+1] << " " << torParams[3*t+2] << endl;
			}
			f.close();
		}

		if (missing > 0)
		{
			cout << "protein::buildEnergyContext: " << missing
			     << " dihedrals had no ff14SB parameters and were dropped" << endl;
		}
	}

	// Pack into the fixed-stride form the kernel reads.
	int stride = 1;
	for (int i = 0; i < N; i++) {if (int(excl[i].size()) > stride) {stride = int(excl[i].size());}}
	vector<int> hcount(N), hlist((size_t)N * stride, 0);
	for (int i = 0; i < N; i++)
	{
		hcount[i] = int(excl[i].size());
		for (int k = 0; k < hcount[i]; k++) {hlist[(size_t)i * stride + k] = excl[i][k];}
	}

	energyTopology topo;
	topo.numAtoms = N;
	topo.radius = &hrad[0]; topo.epsilon = &heps[0]; topo.charge = &hchg[0];
	topo.residueIndex = &hres[0]; topo.silent = &hsilent[0];
	topo.exclusionCount = &hcount[0]; topo.exclusionList = &hlist[0];
	topo.exclusionStride = stride;
	topo.torsionCount = int(torParams.size() / 3);
	topo.torsionAtoms  = torAtoms.empty()  ? 0 : &torAtoms[0];
	topo.torsionParams = torParams.empty() ? 0 : &torParams[0];

	// Pull the scale factors from the same accessors the CPU path uses rather
	// than hardcoding them, which is what let the two paths drift apart.
	// Model selection.  The default enables the corrected physics (occupancy
	// dielectric, inscribed-cube clashes, exact 4pi/3); PROTCAD_ENERGY_LEGACY=1
	// reproduces the original kernel's model bit-for-bit, which is what the
	// existing energy calibrations were tuned against.  Kept as a runtime
	// switch so old and new can be compared without a rebuild.
	energyParams par;
	if (itsEnergyParamsSet) {par = itsEnergyParams;}
	else
	{
		const char* legacyEnv = getenv("PROTCAD_ENERGY_LEGACY");
		bool useLegacy = (legacyEnv && legacyEnv[0] == '1');
		par = useLegacy ? legacyEnergyParams() : defaultEnergyParams();
	}
	par.eSolvationFactor = residue::getElectroSolvationScaleFactor();
	par.hSolvationFactor = residue::getHydroSolvationScaleFactor();
	par.vdwScale = amberVDW::getScaleFactor();
	par.elecScale = amberElec::getScaleFactor();

	itsEnergyContext = energyCreate(topo, par);
	if (!itsEnergyContext)
	{
		cout << "protein::buildEnergyContext failed: " << energyLastError(itsEnergyContext) << endl;
		return;
	}
	itsCoordX.resize(N); itsCoordY.resize(N); itsCoordZ.resize(N);
	deviceMemLoadedEnergy = true; deviceMemLoadedClash = true; deviceMemLoadedAll = true;
}

// Retained so existing callers keep working; the context now serves energy and
// clash from one allocation, so all three requests are the same operation.
void protein::loadDeviceMemEnergy() {buildEnergyContext();}
void protein::loadDeviceMemClash()  {buildEnergyContext();}
void protein::loadDeviceMemAll()    {buildEnergyContext();}

void protein::freeDeviceMemEnergy() {freeDeviceMemAll();}
void protein::freeDeviceMemClash()  {freeDeviceMemAll();}

void protein::freeDeviceMemAll()
{
	if (itsEnergyContext) {energyDestroy(itsEnergyContext); itsEnergyContext = 0;}
	itsCoordX.clear(); itsCoordY.clear(); itsCoordZ.clear(); itsAtomPtrs.clear();
	deviceMemLoadedEnergy = false; deviceMemLoadedClash = false; deviceMemLoadedAll = false;
}

// Refresh coordinates and, if the topology changed underneath us, rebuild.
// The original silently reused a stale context after a mutation changed the
// atom count, which read past the end of the device arrays.
int protein::updateDeviceCoords()
{
	buildEnergyContext();
	if (!itsEnergyContext) {return 0;}

	const int N = getNumAtoms();
	if (N != int(itsCoordX.size()))
	{
		freeDeviceMemAll(); buildEnergyContext();
		if (!itsEnergyContext) {return 0;}
	}

	buildAtomIndex();
	const int n = (int)itsAtomPtrs.size();
	// getCoords returns a dblVec by value, so pulling N atoms through it was N
	// heap allocations a trial -- 286 us on 1ake -- to read three doubles each.
	// The component accessors read the same stored values with no copy.
	for (int i = 0; i < n; i++)
	{
		const atom* a = itsAtomPtrs[i];
		itsCoordX[i] = a->getX(); itsCoordY[i] = a->getY(); itsCoordZ[i] = a->getZ();
	}
	return n;
}

void protein::buildAtomIndex()
{
	const int N = getNumAtoms();
	if ((int)itsAtomPtrs.size() == N) {return;}
	itsAtomPtrs.clear(); itsAtomPtrs.reserve(N);
	atomIterator aIter(this); int i = 0;
	for (; !(aIter.last()) && i < N; aIter++, i++)
	{	itsAtomPtrs.push_back(aIter.getAtomPointer()); }
}

// A move only displaces atoms inside the changed set, so a delta has no reason
// to re-read N coordinates out of the residue tree.
void protein::refreshDeviceCoords(const std::vector<int>& _atoms)
{
	for (size_t k = 0; k < _atoms.size(); k++)
	{
		const int i = _atoms[k];
		if (i < 0 || i >= (int)itsAtomPtrs.size()) {continue;}
		const atom* a = itsAtomPtrs[i];
		itsCoordX[i] = a->getX(); itsCoordY[i] = a->getY(); itsCoordZ[i] = a->getZ();
	}
}

// Generate K random sidechain conformations for one residue, evaluate them all
// in a single batched launch, and report the best.
//
// This is the point of batching. Per-candidate cost falls steeply with K and
// then saturates: measured at N=1404 on a P2200, a single evaluation costs
// 8772 us while K=16 costs 855 us/candidate, K=64 costs 596 and K=256 costs
// 535. The saturation matters as much as the speedup -- beyond K ~ 64 the GPU
// is fully occupied and further candidates are paid for linearly, so a rotamer
// cross-product over two long sidechains is not affordable at any batch size.
// It also converts the move from first-improvement -- accept the first trial
// that helps -- into steepest-descent over K trials.
//
// Note that steepest descent is a biased proposal: it no longer samples the
// Boltzmann distribution, and it breaks the plateau termination rule that
// protMinCU inherited. See protMinReplicaCU for the population formulation
// that spends the same K evaluations without either defect.
//
// Candidate coordinates are still generated on the CPU, one setSidechain call
// per candidate, which is now the dominant per-candidate cost. Moving the
// dihedral transform onto the device removes it.
//
// The protein is restored to its entry conformation before returning.
// Collect every atom strictly distal to _root in the bonded tree.
static void collectDistal(treeNode* _root, vector<atom*> &_out)
{
	for (treeNode* t = _root->getChild(); t; t = t->getNextSib())
	{
		_out.push_back(static_cast<atom*>(t));
		collectDistal(t, _out);
	}
}

// Register the sidechain rotation groups with the energy context.
//
// A rotation group is one chi: an axis, and the set of atoms distal to it that
// the rotation carries. This is the same decomposition residue::rotate performs
// on the host -- axis atoms from the residue's chi definition, moved set from
// the child subtree of the second axis atom -- lifted into global atom indices
// so the device can apply it without touching the atom tree.
//
// Building this once is what lets a move send K*nChi angles to the GPU instead
// of K*3N coordinates, and removes the host dihedral transform from the inner
// loop entirely.
int protein::buildRotationGroups()
{
	if (!itsEnergyContext) {buildEnergyContext();}
	if (!itsEnergyContext) {return 0;}

	// Local atom index -> global index, per residue, in the same atomIterator
	// order buildEnergyContext used. Any other walk risks a different order.
	vector< vector<int> > localToGlobal;
	vector<residue*> resPtr; vector<int> resChain, resIndexInChain;
	int lastChain = -1, lastRes = -1, g = 0;
	for (atomIterator aIter(this); !(aIter.last()); aIter++, g++)
	{
		int ci = int(aIter.getChainIndex()), rin = int(aIter.getResidueIndex());
		if (ci != lastChain || rin != lastRes)
		{
			localToGlobal.push_back(vector<int>());
			resPtr.push_back(aIter.getResiduePointer());
			resChain.push_back(ci); resIndexInChain.push_back(rin);
			lastChain = ci; lastRes = rin;
		}
		UInt ai = aIter.getAtomIndex();
		if (localToGlobal.back().size() <= ai) {localToGlobal.back().resize(ai + 1, -1);}
		localToGlobal.back()[ai] = g;
	}

	vector<int> axisA, axisB, memberStart, members;
	memberStart.push_back(0);
	itsRotGroupFirst.assign(resPtr.size(), -1);
	itsRotGroupCount.assign(resPtr.size(), 0);
	itsResRotIndex.assign(getNumChains(), vector<int>());

	// Backbone phi/psi groups are opt-in, because they are only ever wanted for
	// one thing. A phi rotation carries the whole downstream chain as a rigid
	// body: in a folded protein that swings half the structure through the
	// other half, so every proposal is rejected and the groups are dead weight
	// on every sweep. In an unfolded reference peptide the downstream segment
	// is a residue or two, and the move is precisely the backbone freedom the
	// reference state is supposed to have and the folded state is not.
	const char* bbEnv = getenv("PROTCAD_MC_BACKBONE");
	const bool wantBackbone = (bbEnv && bbEnv[0] == '1');

	// Global atom span of each residue, and of the chain it belongs to. The
	// atomIterator walk above assigned global indices in (chain, residue, atom)
	// order, so a residue's atoms are contiguous and "everything downstream in
	// this chain" is an index range. It has to be built this way: the atom tree
	// is linked only within a residue (see residue.cc), so collectDistal cannot
	// reach past the backbone no matter where it starts.
	vector<int> resAtomBegin(resPtr.size(), -1), resAtomEnd(resPtr.size(), -1);
	for (UInt r = 0; r < resPtr.size(); r++)
	{
		for (UInt a = 0; a < localToGlobal[r].size(); a++)
		{
			int gi = localToGlobal[r][a];
			if (gi < 0) {continue;}
			if (resAtomBegin[r] < 0 || gi < resAtomBegin[r]) {resAtomBegin[r] = gi;}
			if (gi + 1 > resAtomEnd[r]) {resAtomEnd[r] = gi + 1;}
		}
	}
	vector<int> chainAtomEnd(resPtr.size(), -1);
	for (int r = (int)resPtr.size() - 1, endOfChain = -1, cur = -2; r >= 0; r--)
	{
		if (resChain[r] != cur) {cur = resChain[r]; endOfChain = resAtomEnd[r];}
		chainAtomEnd[r] = endOfChain;
	}

	for (UInt r = 0; r < resPtr.size(); r++)
	{
		UInt c = (UInt)resChain[r], rin = (UInt)resIndexInChain[r];
		if (itsResRotIndex[c].size() <= rin) {itsResRotIndex[c].resize(rin + 1, -1);}
		itsResRotIndex[c][rin] = (int)r;

		residue* pRes = resPtr[r];
		UInt type = pRes->getTypeIndex();
		int nChi = residue::dataBase[type].getNumberOfChis(0);
		if (nChi <= 0 && !wantBackbone) {continue;}

		itsRotGroupFirst[r] = (int)axisA.size();
		for (int i = 0; i < nChi; i++)
		{
			UIntVec def = residue::dataBase[type].getAtomsOfChi(0, (UInt)i);
			if (def.size() < 4) {break;}
			UInt l1 = def[1], l2 = def[2];
			if (l1 >= localToGlobal[r].size() || l2 >= localToGlobal[r].size()) {break;}
			int ga = localToGlobal[r][l1], gb = localToGlobal[r][l2];
			if (ga < 0 || gb < 0) {break;}

			vector<atom*> distal;
			collectDistal(pRes->getAtom(l2), distal);

			// Map moved atoms back to global indices. An atom outside this
			// residue's own list would mean the chi reaches through the
			// backbone into the next residue, which a sidechain chi never does.
			vector<int> gm; bool ok = true;
			for (UInt d = 0; d < distal.size(); d++)
			{
				int gi = -1;
				for (UInt a = 0; a < localToGlobal[r].size(); a++)
				{
					if (pRes->getAtom(a) == distal[d]) {gi = localToGlobal[r][a]; break;}
				}
				if (gi < 0) {ok = false; break;}
				gm.push_back(gi);
			}
			if (!ok) {break;}

			axisA.push_back(ga); axisB.push_back(gb);
			for (UInt m = 0; m < gm.size(); m++) {members.push_back(gm[m]);}
			memberStart.push_back((int)members.size());
			itsRotGroupCount[r]++;
		}

		if (wantBackbone)
		{
			// phi (N->CA) and psi (CA->C). The device applies a rotation group
			// as "spin these member atoms about this axis", with no notion of
			// what the axis means, so the backbone needs no kernel support --
			// only a member set that legitimately leaves the residue.
			//
			// The in-residue half still comes from the atom tree, which roots
			// at N: distal-from-CA is the sidechain plus the carbonyl,
			// distal-from-C is the carbonyl oxygen. The rest of the chain is
			// appended as a range. Rotating about N-CA therefore carries the
			// sidechain too, which is correct -- the sidechain is rigidly
			// attached to CA, and holding N fixed is just a choice of which
			// half of the molecule stays put.
			int lN = -1, lCA = -1, lC = -1;
			for (UInt a = 0; a < localToGlobal[r].size(); a++)
			{
				if (localToGlobal[r][a] < 0) {continue;}
				string an = pRes->getAtomName(a);
				if (an == "N") {lN = (int)a;} else if (an == "CA") {lCA = (int)a;}
				else if (an == "C") {lC = (int)a;}
			}

			const int dsBegin = resAtomEnd[r], dsEnd = chainAtomEnd[r];
			const bool isPro = (residue::dataBase[type].getName() == "PRO");

			for (int which = 0; which < 2; which++)
			{
				// Proline's CD bonds back to N, so the ring closes across the
				// phi axis. Rotating it would tear the ring open rather than
				// change a torsion; phi is not a free coordinate in proline.
				if (which == 0 && isPro) {continue;}
				int la = (which == 0) ? lN : lCA;
				int lb = (which == 0) ? lCA : lC;
				if (la < 0 || lb < 0) {continue;}
				int ga = localToGlobal[r][la], gb = localToGlobal[r][lb];
				if (ga < 0 || gb < 0) {continue;}

				vector<atom*> distal;
				collectDistal(pRes->getAtom(lb), distal);

				vector<int> gm; bool ok = true;
				for (UInt d = 0; d < distal.size(); d++)
				{
					int gi = -1;
					for (UInt a = 0; a < localToGlobal[r].size(); a++)
					{
						if (pRes->getAtom(a) == distal[d]) {gi = localToGlobal[r][a]; break;}
					}
					if (gi < 0) {ok = false; break;}
					gm.push_back(gi);
				}
				// The axis atoms must not be carried by their own rotation, and
				// the tree must have handed back the half we expected. If the
				// residue is linked differently than assumed, skip the group
				// rather than silently rotate the wrong set of atoms.
				for (UInt m = 0; ok && m < gm.size(); m++)
				{
					if (gm[m] == ga || gm[m] == gb) {ok = false;}
				}
				if (which == 0 && ok)
				{
					bool carriesC = false;
					for (UInt m = 0; m < gm.size(); m++) {if (gm[m] == localToGlobal[r][lC]) {carriesC = true;}}
					if (lC >= 0 && !carriesC) {ok = false;}
				}
				if (!ok) {continue;}

				for (int gi = dsBegin; gi < dsEnd; gi++) {gm.push_back(gi);}
				if (gm.empty()) {continue;}

				axisA.push_back(ga); axisB.push_back(gb);
				for (UInt m = 0; m < gm.size(); m++) {members.push_back(gm[m]);}
				memberStart.push_back((int)members.size());
				itsRotGroupCount[r]++;
			}
		}

		if (itsRotGroupCount[r] == 0) {itsRotGroupFirst[r] = -1;}
	}

	int nG = (int)axisA.size();
	if (nG == 0) {return 0;}
	if (energySetRotationGroups(itsEnergyContext, nG, &axisA[0], &axisB[0],
	                            &memberStart[0], members.empty() ? &nG : &members[0]) != 0)
	{
		cout << "protein::buildRotationGroups failed: " << energyLastError(itsEnergyContext) << endl;
		return 0;
	}
	return nG;
}

int protein::energyRotamerBatch(int _nCand, int _groupBegin, int _nGroups,
                                const double* _anglesDeg, double* _totals)
{
	if (!itsEnergyContext) {return -1;}
	return energyComputeRotamerBatch(itsEnergyContext, _nCand, _groupBegin,
	                                 _nGroups, _anglesDeg, _totals);
}

int protein::syncDeviceCoords()
{
	int N = updateDeviceCoords();
	if (N == 0) {return 0;}
	if (energySetCoords(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0]) != 0) {return 0;}
	return N;
}

int protein::getBatchCoords(int _k, double* _x, double* _y, double* _z)
{
	if (!itsEnergyContext) {return -1;}
	return energyGetBatchCoords(itsEnergyContext, _k, _x, _y, _z);
}

bool protein::getRotationGroupRange(UInt _chainIndex, UInt _resIndex, int &_begin, int &_count) const
{
	if (_chainIndex >= itsResRotIndex.size()) {return false;}
	if (_resIndex >= itsResRotIndex[_chainIndex].size()) {return false;}
	int r = itsResRotIndex[_chainIndex][_resIndex];
	if (r < 0 || r >= (int)itsRotGroupCount.size()) {return false;}
	if (itsRotGroupCount[r] <= 0) {return false;}
	_begin = itsRotGroupFirst[r]; _count = itsRotGroupCount[r];
	return true;
}

double protein::bestSidechainCandidateCU(UInt _chainIndex, UInt _resIndex, UInt _numCandidates,
                                         vector < vector <double> > &_bestConf)
{
	int N = updateDeviceCoords();
	if (N == 0 || _numCandidates == 0) {return 1E30;}

	vector < vector <double> > entryConf = getSidechainDihedrals(_chainIndex, _resIndex);
	vector < vector < vector <double> > > confs(_numCandidates);
	vector <double> bx((size_t)_numCandidates * N), by((size_t)_numCandidates * N), bz((size_t)_numCandidates * N);

	// A sidechain move can only displace its own residue, so seed every slice
	// from the entry geometry once and refresh just that residue per candidate.
	// Rebuilding all N coordinates from the atom pointers once per candidate is
	// host bookkeeping that grows with the whole structure while the move does
	// not, and on larger proteins it outweighs the evaluation it feeds.
	vector<int> resAtoms;
	residueAtomIndicesCU(_chainIndex, _resIndex, resAtoms);
	if (resAtoms.empty()) {return 1E30;}
	for (UInt k = 0; k < _numCandidates; k++)
	{
		const size_t off = (size_t)k * N;
		memcpy(&bx[off], &itsCoordX[0], (size_t)N * sizeof(double));
		memcpy(&by[off], &itsCoordY[0], (size_t)N * sizeof(double));
		memcpy(&bz[off], &itsCoordZ[0], (size_t)N * sizeof(double));
	}

	for (UInt k = 0; k < _numCandidates; k++)
	{
		confs[k] = randContinuousSidechainConformation(_chainIndex, _resIndex);
		setSidechainDihedralAngles(_chainIndex, _resIndex, confs[k]);
		refreshDeviceCoords(resAtoms);
		size_t off = (size_t)k * N;
		for (size_t m = 0; m < resAtoms.size(); m++)
		{
			const int i = resAtoms[m];
			bx[off + i] = itsCoordX[i]; by[off + i] = itsCoordY[i]; bz[off + i] = itsCoordZ[i];
		}
	}

	// Restore, then seed the spatial order from the entry conformation so every
	// candidate is culled and ranked against the same order.
	setSidechainDihedralAngles(_chainIndex, _resIndex, entryConf);
	refreshDeviceCoords(resAtoms);
	energySetCoords(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0]);

	vector <double> energies(_numCandidates, 1E30);
	if (energyComputeBatch(itsEnergyContext, (int)_numCandidates, &bx[0], &by[0], &bz[0], &energies[0]))
	{
		cout << "protein::bestSidechainCandidateCU failed: " << energyLastError(itsEnergyContext) << endl;
		return 1E30;
	}

	UInt best = 0;
	for (UInt k = 1; k < _numCandidates; k++) {if (energies[k] < energies[best]) {best = k;}}
	_bestConf = confs[best];
	return energies[best];
}

bool protein::protRefreezeDielectricCU()
{
	if (!itsEnergyContext) {return false;}
	if (energyRefreezeDielectric(itsEnergyContext))
	{
		cout << "protein::protRefreezeDielectricCU failed: "
		     << energyLastError(itsEnergyContext) << endl;
		return false;
	}
	itsThawList.clear();
	return true;
}

void protein::residueAtomIndicesCU(UInt _chainIndex, UInt _resIndex,
                                   std::vector<int>& _out)
{
	_out.clear();
	const int N = (int)itsCoordX.size();
	int i = 0;
	for (atomIterator aIter(this); !(aIter.last()) && i < N; aIter++, i++)
	{
		if (aIter.getChainIndex() == (int)_chainIndex &&
		    aIter.getResidueIndex() == (int)_resIndex) {_out.push_back(i);}
	}
}

int protein::protDielectricThawCountCU()
{
	if (!itsEnergyContext) {return -1;}
	return energyDielectricThawCount(itsEnergyContext);
}

int protein::protThawDielectricForBatchCU(const vector<double>& _oldX,
                                          const vector<double>& _oldY,
                                          const vector<double>& _oldZ,
                                          const vector<double>& _candX,
                                          const vector<double>& _candY,
                                          const vector<double>& _candZ,
                                          int _nCand, double _radius,
                                          int* _movedOut, double _tol,
                                          const vector<int>* _support,
                                          bool _coordsCurrent)
{
	// The batch caller refreshed every coordinate immediately before building
	// its candidates, and in the flat-transform path the residue object is
	// never moved at all, so re-reading all N atoms out of the object graph
	// here was a second full-N pass per trial to reproduce what itsCoord*
	// already held.
	int N = _coordsCurrent ? (int)itsCoordX.size() : updateDeviceCoords();
	if (_coordsCurrent && (N == 0 || !itsEnergyContext)) {N = updateDeviceCoords();}
	if (N == 0 || (int)_oldX.size() < N || _nCand <= 0) {return -1;}
	if ((int)_candX.size() < (size_t)_nCand * N) {return -1;}
	if (_radius <= 0.0)
	{	_radius = energyDielectricInfluenceRadius(itsEnergyContext); }

	// An atom is moved if any candidate moves it.  The set is shared by the
	// whole batch, which is what lets one changed set and one tile list serve
	// every candidate.
	// _support, when given, is the set of atoms the caller could have moved;
	// the candidate arrays are only meaningful there.  Scanning all N instead
	// would force the caller to materialise every candidate in full, which for
	// a batch is megabytes of copying to discover that nothing outside one
	// residue changed.
	vector<int> movedAtoms;
	const int nScan = _support ? (int)_support->size() : N;
	for (int s = 0; s < nScan; s++)
	{
		const int a = _support ? (*_support)[s] : s;
		bool moved = false;
		for (int k = 0; k < _nCand && !moved; k++)
		{
			const size_t o = (size_t)k * N + a;
			const double dx = _candX[o] - _oldX[a];
			const double dy = _candY[o] - _oldY[a];
			const double dz = _candZ[o] - _oldZ[a];
			if (dx * dx + dy * dy + dz * dz > _tol * _tol) {moved = true;}
		}
		if (moved) {movedAtoms.push_back(a);}
	}
	sort(movedAtoms.begin(), movedAtoms.end());
	if (_movedOut) {*_movedOut = (int)movedAtoms.size();}
	if (movedAtoms.empty())
	{	energySetDielectricThaw(itsEnergyContext, 0, 0, 0);
		itsThawList.clear();
		return 0; }

	// One bounding sphere per moved atom, over its position before the move and
	// in every candidate.  Thawing everything within (sphere + radius) of any
	// of them is a superset of the exact per-candidate union, so it is exact in
	// the sense that matters, and it costs O(N |A|) rather than O(N |A| K).
	const int nMoved = (int)movedAtoms.size();
	vector<double> cx(nMoved), cy(nMoved), cz(nMoved), cr(nMoved);
	for (int m = 0; m < nMoved; m++)
	{
		const int b = movedAtoms[m];
		double sx = _oldX[b], sy = _oldY[b], sz = _oldZ[b];
		double mnx = sx, mxx = sx, mny = sy, mxy = sy, mnz = sz, mxz = sz;
		for (int k = 0; k < _nCand; k++)
		{
			const size_t o = (size_t)k * N + b;
			if (_candX[o] < mnx) {mnx = _candX[o];} if (_candX[o] > mxx) {mxx = _candX[o];}
			if (_candY[o] < mny) {mny = _candY[o];} if (_candY[o] > mxy) {mxy = _candY[o];}
			if (_candZ[o] < mnz) {mnz = _candZ[o];} if (_candZ[o] > mxz) {mxz = _candZ[o];}
		}
		cx[m] = 0.5 * (mnx + mxx); cy[m] = 0.5 * (mny + mxy); cz[m] = 0.5 * (mnz + mxz);
		double r2 = 0.0;
		{	const double hx = 0.5 * (mxx - mnx), hy = 0.5 * (mxy - mny), hz = 0.5 * (mxz - mnz);
			r2 = hx * hx + hy * hy + hz * hz; }
		cr[m] = sqrt(r2) + _radius;
	}

	// Every one of those spheres sits inside one residue, so almost every atom
	// in the structure misses all of them.  One sphere enclosing the lot turns
	// that verdict into a single test instead of nMoved of them.  It only ever
	// rejects -- anything it admits is still checked against each sphere -- so
	// the thaw set is exactly the set the plain scan produces.
	double ex = 0.0, ey = 0.0, ez = 0.0;
	for (int m = 0; m < nMoved; m++) {ex += cx[m]; ey += cy[m]; ez += cz[m];}
	ex /= nMoved; ey /= nMoved; ez /= nMoved;
	double er = 0.0;
	for (int m = 0; m < nMoved; m++)
	{	const double dx = cx[m] - ex, dy = cy[m] - ey, dz = cz[m] - ez;
		const double d = sqrt(dx*dx + dy*dy + dz*dz) + cr[m];
		if (d > er) {er = d;} }
	const double er2 = er * er;

	vector<int> thaw;
	for (int a = 0; a < N; a++)
	{
		{	const double dx = itsCoordX[a] - ex;
			const double dy = itsCoordY[a] - ey;
			const double dz = itsCoordZ[a] - ez;
			if (dx*dx + dy*dy + dz*dz > er2) {continue;} }
		bool hit = false;
		for (int m = 0; m < nMoved && !hit; m++)
		{
			const double dx = itsCoordX[a] - cx[m];
			const double dy = itsCoordY[a] - cy[m];
			const double dz = itsCoordZ[a] - cz[m];
			if (dx * dx + dy * dy + dz * dz <= cr[m] * cr[m]) {hit = true;}
		}
		if (hit) {thaw.push_back(a);}
	}
	if (energySetDielectricThaw(itsEnergyContext, thaw.empty() ? 0 : &thaw[0],
	                            (int)thaw.size(), 0))
	{	return -1; }
	itsThawList = thaw;
	return energyDielectricThawCount(itsEnergyContext);
}


// Phase timing for the delta candidate path, enabled with PROTCAD_PROFILE=1.
// Added because the accounting did not close: on 1crn a trial costs about
// 2.4 ms, of which kernels are 0.45 ms and host move generation 0.23 ms, and
// guessing at the remaining 1.7 ms is how one ends up rewriting the wrong
// thing.
struct deltaProf
{
	static const int NP = 9;
	double t[NP]; long n;
	deltaProf() : n(0) {for (int i = 0; i < NP; i++) {t[i] = 0.0;}}
	~deltaProf()
	{
		if (!n) {return;}
		static const char* nm[NP] = {"setup (updateDeviceCoords + N copies)",
		                             "candidate generation (K x transform)",
		                             "thaw set construction",
		                             "  cand: random draw",
		                             "energySetCoords",
		                             "  cand: chi transform",
		                             "energyComputeBatchDelta",
		                             "ranking",
		                             "  cand: refresh + copy"};
		double tot = 0.0; for (int i = 0; i < NP; i++) {tot += t[i];}
		cout << "\ndeltaProf: " << n << " calls" << endl;
		for (int i = 0; i < NP; i++)
		{	cout << "  " << nm[i] << "  " << t[i] << " s  "
			     << (t[i] * 1e6 / n) << " us/trial  "
			     << (tot > 0 ? 100.0 * t[i] / tot : 0.0) << " %" << endl; }
		cout << "  TOTAL  " << tot << " s  " << (tot * 1e6 / n) << " us/trial" << endl;
	}
};
static deltaProf g_deltaProf;
static const bool g_hostChi = (getenv("PROTCAD_HOSTCHI") != 0);

// Apply one chi rotation in place over a flat residue-local coordinate block.
// This is residue::rotate with the object graph taken out of it: the same
// translate-to-origin, rotate-children, translate-back, and the same rotation
// matrix as CMath::rotationMatrix, including its truncated degree conversion,
// so the geometry it produces is the geometry the residue would have produced.
// What it does not do is walk a pointer tree three times or allocate a dblVec
// per atom and a dblMat per call, which is where the time was going.
static void chiRotateFlat(double* wx, double* wy, double* wz,
                          const int a1, const int a2, const double thetaDeg,
                          const int* moved, const int nMoved)
{
	const double ox = wx[a1], oy = wy[a1], oz = wz[a1];
	const double vx = wx[a2] - ox, vy = wy[a2] - oy, vz = wz[a2] - oz;
	const double norm = sqrt(vx*vx + vy*vy + vz*vz);
	if (norm == 0.0) {return;}
	const double n1 = vx / norm, n2 = vy / norm, n3 = vz / norm;
	const double degToRad = 0.017453293;
	const double theta = degToRad * thetaDeg;
	const double s = sin(theta), c = cos(theta);
	const double n11 = n1*n1, n12 = n1*n2, n13 = n1*n3;
	const double n22 = n2*n2, n23 = n2*n3, n33 = n3*n3;
	const double r00 = n11 + (1-n11)*c, r01 = n12*(1-c) - n3*s, r02 = n13*(1-c) + n2*s;
	const double r10 = n12*(1-c) + n3*s, r11 = n22 + (1-n22)*c, r12 = n23*(1-c) - n1*s;
	const double r20 = n13*(1-c) - n2*s, r21 = n23*(1-c) + n1*s, r22 = n33 + (1-n33)*c;
	for (int i = 0; i < nMoved; i++)
	{
		const int m = moved[i];
		const double px = wx[m] - ox, py = wy[m] - oy, pz = wz[m] - oz;
		wx[m] = r00*px + r01*py + r02*pz + ox;
		wy[m] = r10*px + r11*py + r12*pz + oy;
		wz[m] = r20*px + r21*py + r22*pz + oz;
	}
}
static const bool g_deltaProfOn = (getenv("PROTCAD_PROFILE") != 0);
static double profNow()
{	return std::chrono::duration<double>(
	         std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define PROF_T(i, expr) do { if (g_deltaProfOn) { const double _a = profNow(); expr; \
	g_deltaProf.t[i] += profNow() - _a; } else { expr; } } while (0)

double protein::bestSidechainCandidateDeltaCU(UInt _chainIndex, UInt _resIndex,
                                              UInt _numCandidates,
                                              vector < vector <double> > &_bestConf,
                                              double _nbCurrent, double& _nbBest,
                                              double& _torBest,
                                              vector<double>* _allEnergies,
                                              vector < vector < vector <double> > >* _allConfs)
{
	const double _p0 = g_deltaProfOn ? profNow() : 0.0;
	int N = updateDeviceCoords();
	if (N == 0 || _numCandidates == 0 || !itsEnergyContext) {return 1E30;}

	vector < vector <double> > entryConf = getSidechainDihedrals(_chainIndex, _resIndex);
	if (entryConf.empty()) {return 1E30;}

	const vector<double> oldX = itsCoordX, oldY = itsCoordY, oldZ = itsCoordZ;
	if (g_deltaProfOn) {g_deltaProf.t[0] += profNow() - _p0; g_deltaProf.n++;}

	// Only the moved residue's atoms can differ between candidates, so seed
	// every slice from the entry geometry once and then overwrite that residue.
	// Refreshing all N coordinates from the atom pointers 32 times over is pure
	// host bookkeeping, and on a structure of any size it costs more than the
	// device work the delta is trying to save.
	vector<int> resAtoms;
	residueAtomIndicesCU(_chainIndex, _resIndex, resAtoms);
	if (resAtoms.empty()) {return 1E30;}

	// The candidate arrays are addressed as k * N + atom, but only the thaw
	// set is ever read out of them, so they are left unwritten outside the
	// moved residue until the thaw set is known and the few entries that
	// matter can be filled directly.  Seeding all of them from the entry
	// geometry instead is several megabytes of memcpy per batch to supply
	// values that are, by construction, unchanged.
	// Zeroing them was costing 1.7 ms a trial on 1ake -- 6 MB of stores per
	// trial to initialise values that are never read -- and the allocation
	// itself recurred for a size that never changes.  Every entry the
	// consumers touch (the moved residue for the thaw builder, the thaw set
	// for the batch delta) is written below before either runs, so the buffer
	// only needs to be large enough, not clean.
	vector < vector < vector <double> > > confs(_numCandidates);
	// One slot past the candidates holds the entry conformation, so the
	// pre-move part is evaluated by the same call, in the same batch, over the
	// same changed set as everything it will be subtracted from.
	const size_t bneed = (size_t)(_numCandidates + 1) * N;
	if (itsBatchX.size() < bneed)
	{	itsBatchX.resize(bneed); itsBatchY.resize(bneed); itsBatchZ.resize(bneed); }
	vector<double>& bx = itsBatchX;
	vector<double>& by = itsBatchY;
	vector<double>& bz = itsBatchZ;
	const double _p1 = g_deltaProfOn ? profNow() : 0.0;
	// Every chi is a rigid rotation of a fixed subset of the residue about a
	// fixed bond, and both the subset and the entry angle are properties of the
	// residue we are about to leave untouched.  Reading them once and driving a
	// flat coordinate block is the same geometry the residue object produces,
	// arrived at without measuring a dihedral we already know or walking a
	// pointer tree three times per chi per candidate.
	//
	// It also generates every candidate from the *entry* conformation rather
	// than from its predecessor's.  That is the arithmetic the caller uses when
	// it adopts the winner -- it applies the chosen chis to the entry residue --
	// so the evaluated geometry and the adopted geometry are now reached the
	// same way instead of by two different chains of rotations.
	const int nRes = (int)resAtoms.size();
	if (!g_hostChi)
	{
		vector<int> cBpt, cIdx, cA1, cA2, cMoved, cOff; vector<double> cEntry;
		chiRotationTopologyCU(_chainIndex, _resIndex, cBpt, cIdx, cA1, cA2,
		                      cEntry, cMoved, cOff);
		vector<double> wx((size_t)nRes), wy((size_t)nRes), wz((size_t)nRes);
		const int nChi = (int)cBpt.size();
		for (UInt k = 0; k < _numCandidates; k++)
		{
			const double _c0 = g_deltaProfOn ? profNow() : 0.0;
			confs[k] = randContinuousSidechainConformation(_chainIndex, _resIndex);
			const double _c1 = g_deltaProfOn ? profNow() : 0.0;
			for (int m = 0; m < nRes; m++)
			{	const int i = resAtoms[m];
				wx[m] = oldX[i]; wy[m] = oldY[i]; wz[m] = oldZ[i]; }
			for (int c = 0; c < nChi; c++)
			{
				const int b = cBpt[c], d = cIdx[c];
				if (b >= (int)confs[k].size() || d >= (int)confs[k][b].size()) {continue;}
				const double target = confs[k][b][d];
				// 1000.0 is the "no measurable dihedral" sentinel, the same one
				// chain::setSidechainDihedralAngles skips over.
				if (target > 999.0) {continue;}
				if (cA1[c] >= nRes || cA2[c] >= nRes) {continue;}
				chiRotateFlat(&wx[0], &wy[0], &wz[0], cA1[c], cA2[c],
				              target - cEntry[c], &cMoved[cOff[c]], cOff[c+1] - cOff[c]);
			}
			const double _c2 = g_deltaProfOn ? profNow() : 0.0;
			const size_t off = (size_t)k * N;
			for (int m = 0; m < nRes; m++)
			{	const int i = resAtoms[m];
				bx[off + i] = wx[m]; by[off + i] = wy[m]; bz[off + i] = wz[m]; }
			if (g_deltaProfOn)
			{	g_deltaProf.t[3] += _c1 - _c0;
				g_deltaProf.t[5] += _c2 - _c1;
				g_deltaProf.t[8] += profNow() - _c2; }
		}
		// The residue was never moved, so there is nothing to restore.
	}
	else
	{
	for (UInt k = 0; k < _numCandidates; k++)
	{
		const double _c0 = g_deltaProfOn ? profNow() : 0.0;
		confs[k] = randContinuousSidechainConformation(_chainIndex, _resIndex);
		const double _c1 = g_deltaProfOn ? profNow() : 0.0;
		setSidechainDihedralAngles(_chainIndex, _resIndex, confs[k]);
		const double _c2 = g_deltaProfOn ? profNow() : 0.0;
		refreshDeviceCoords(resAtoms);
		if (g_deltaProfOn)
		{	g_deltaProf.t[3] += _c1 - _c0;
			g_deltaProf.t[5] += _c2 - _c1;
			g_deltaProf.t[8] += profNow() - _c2; }
		const size_t off = (size_t)k * N;
		for (size_t m = 0; m < resAtoms.size(); m++)
		{
			const int i = resAtoms[m];
			bx[off + i] = itsCoordX[i]; by[off + i] = itsCoordY[i]; bz[off + i] = itsCoordZ[i];
		}
	}
	setSidechainDihedralAngles(_chainIndex, _resIndex, entryConf);
	refreshDeviceCoords(resAtoms);
	}
	{
		const size_t off = (size_t)_numCandidates * N;
		for (size_t m = 0; m < resAtoms.size(); m++)
		{
			const int i = resAtoms[m];
			bx[off + i] = oldX[i]; by[off + i] = oldY[i]; bz[off + i] = oldZ[i];
		}
	}
	if (g_deltaProfOn) {g_deltaProf.t[1] += profNow() - _p1;}
	// Seed the spatial order from the entry conformation, so every candidate is
	// culled and ranked against the same order and the delta's tile list means
	// the same thing for all of them.
	const double _p4 = g_deltaProfOn ? profNow() : 0.0;
	energySetCoords(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0]);
	if (g_deltaProfOn) {g_deltaProf.t[4] += profNow() - _p4;}

	// The thaw set is the batch's, not any one candidate's, and it has to be
	// installed before the pre-move part is measured: P(old) and P(new) must be
	// taken over the same set or the difference is not a difference.
	int nMoved = 0;
	const double _p2 = g_deltaProfOn ? profNow() : 0.0;
	const int nThaw = protThawDielectricForBatchCU(oldX, oldY, oldZ, bx, by, bz,
	                                               (int)_numCandidates, 0.0, &nMoved,
	                                               1e-9, &resAtoms, true);
	if (g_deltaProfOn) {g_deltaProf.t[2] += profNow() - _p2;}
	if (nThaw <= 0) {return 1E30;}

	// Fill in the thawed atoms outside the moved residue.  They are unchanged
	// by definition, but the batch reads coordinates for every atom of the
	// changed set and will not infer that.
	// The thawed atoms outside the moved residue used to be copied into every
	// candidate slice here, because the batch staged the whole changed set and
	// would otherwise have shipped uninitialised values.  It now stages only
	// the moved residue and takes the rest from the resident conformation it
	// already seeds each candidate from, so writing them was supplying the
	// device with coordinates it had, at O(K * |thaw|) per trial.
	const double _p3 = g_deltaProfOn ? profNow() : 0.0;

	if (g_deltaProfOn) {g_deltaProf.t[3] += profNow() - _p3;}
	// The pre-move part used to be its own energyComputeDelta call.  It was
	// evaluating one conformation with the machinery built for thirty-two, so
	// it paid a full set of launches and a staging round trip for a
	// thirty-second of the work -- 12% of a trial on 1ake against 57% for the
	// batch that does the other thirty-two.  Widening the batch by one is a
	// few percent of a call that already exists.
	//
	// It also removes a real asymmetry rather than only cost.  P(old) and
	// P(new) have to be taken over the same changed set for their difference
	// to mean anything, and they were -- but by two different code paths with
	// two different reduction orders, so the difference carried the gap
	// between the paths as well as the move.  Now both come out of one call.
	const double _p6 = g_deltaProfOn ? profNow() : 0.0;
	const int nEval = (int)_numCandidates + 1;
	vector<double> parts(nEval, 0.0), tors(nEval, 0.0);
	if (energyComputeBatchDelta(itsEnergyContext, nEval,
	                            &bx[0], &by[0], &bz[0], &itsThawList[0],
	                            (int)itsThawList.size(), &parts[0], &tors[0],
	                            &resAtoms[0], (int)resAtoms.size()))
	{
		cout << "protein::bestSidechainCandidateDeltaCU failed: "
		     << energyLastError(itsEnergyContext) << endl;
		return 1E30;
	}

	if (g_deltaProfOn) {g_deltaProf.t[6] += profNow() - _p6;}
	const double pOld = parts[_numCandidates];
	const double _p7 = g_deltaProfOn ? profNow() : 0.0;
	UInt best = 0; double bestE = 1E30;
	if (_allEnergies) {_allEnergies->assign(_numCandidates, 0.0);}
	for (UInt k = 0; k < _numCandidates; k++)
	{
		const double e = _nbCurrent - pOld + parts[k] + tors[k];
		if (_allEnergies) {(*_allEnergies)[k] = e;}
		if (k == 0 || e < bestE) {bestE = e; best = k;}
	}
	if (_allConfs) {*_allConfs = confs;}
	_bestConf = confs[best];
	_nbBest = _nbCurrent - pOld + parts[best];
	_torBest = tors[best];
	if (g_deltaProfOn) {g_deltaProf.t[7] += profNow() - _p7;}
	return bestE;
}

int protein::setDeviceReplicas(int _nRepl)
{
	if (syncDeviceCoords() == 0) {return -1;}
	return energySetReplicas(itsEnergyContext, _nRepl);
}

int protein::energyReplicaBatch(int _nRepl, const int* _groupBegin, const int* _nGroups,
                                const double* _anglesDeg, int _angleStride, double* _totals)
{
	if (!itsEnergyContext) {return -1;}
	return energyComputeReplicaBatch(itsEnergyContext, _nRepl, _groupBegin, _nGroups,
	                                 _anglesDeg, _angleStride, _totals);
}

int protein::commitReplicas(int _nRepl, const int* _accept)
{
	if (!itsEnergyContext) {return -1;}
	return energyCommitReplicas(itsEnergyContext, _nRepl, _accept);
}

int protein::getReplicaCoords(int _k, double* _x, double* _y, double* _z)
{
	if (!itsEnergyContext) {return -1;}
	return energyGetReplicaCoords(itsEnergyContext, _k, _x, _y, _z);
}

void protein::setCoordsFromArray(const double* _x, const double* _y, const double* _z)
{
	// Inverse of updateDeviceCoords(): same atomIterator traversal, so the
	// index mapping is identical by construction rather than by assumption.
	const int N = getNumAtoms();
	atomIterator aIter(this); int i = 0;
	for (; !(aIter.last()) && i < N; aIter++, i++)
	{
		aIter.getAtomPointer()->setCoords(_x[i], _y[i], _z[i]);
	}
}

int protein::maxRotationGroupCount() const
{
	int m = 0;
	for (UInt i = 0; i < itsRotGroupCount.size(); i++)
	{
		if (itsRotGroupCount[i] > m) {m = itsRotGroupCount[i];}
	}
	return m;
}

// Fixed-budget population Monte Carlo over sidechain torsions.
//
// This replaces the plateau counter used by protMinCU, which does not survive
// contact with a batched move. That loop terminates after `plateau` consecutive
// trials with no improvement better than KT, a criterion written for
// first-improvement on a single candidate. Steepest descent over a batch finds
// some improvement on nearly every trial, so the counter resets almost every
// iteration and the required run of consecutive failures effectively never
// occurs -- the expected iteration count grows like (1-q)^-plateau in the reset
// probability q. A 1crn minimisation that took 13 s under the old move ran for
// 83 minutes without terminating. A sweep budget is not a workaround for that;
// it is the correct control for a sampler, whose cost should be chosen rather
// than discovered.
//
// The population is the point. Best-of-K spends K energy evaluations to advance
// one chain by one step, and the resulting move is a biased proposal with no
// compensating acceptance term, so the chain no longer samples the Boltzmann
// distribution. K replicas spend the same K evaluations to advance K chains by
// one step each, and every proposal is drawn from a symmetric kernel and tested
// on its own.
//
// What that argument does NOT buy is speed of minimisation, and an early
// version of this comment claimed it did, on the grounds that per-candidate
// cost is flat out to K ~ 64. Flat per-candidate cost is not zero marginal wall
// cost. On 1ubq a single evaluation is 8.8 ms while a 64-batch is 38 ms, so the
// batch delivers 64 candidates for 4.3x the wall time -- excellent throughput,
// but it also means one chain advances 4.3x more slowly in wall-clock terms
// than it would alone. Breadth has to be worth more than 4.3x to pay for
// itself, and for driving a single structure downhill it is not.
//
// Measured on 1ubq at a fixed ~50 s budget, replicas against sweeps-per-chain:
//
//     P=1  -1223 | P=2  -1213 | P=4  -1215 | P=8  -1191
//     P=16 -1180 | P=32 -1158 | P=64 -1090 | P=128 -1019
//
// Monotone in depth, with no interior optimum. That sweep set the budget by
// calibration and the realised wall times drifted by a factor of nearly two,
// so it could rank the large-P settings but could not separate P=1, 2 and 4.
// Read on its face it made P=4 look like an efficiency point.
//
// A direct matched-wall test settles it. Six seeds, ~25 s each, paired by
// seed, on the j-split kernels:
//
//     P=1  -1234.7 (25.8 s) | P=2  -1212.0 (25.5 s) | P=4  -1210.2 (23.8 s)
//     paired vs P=1:   P=2  +22.6 +- 7.9 (t 2.9)   P=4  +24.4 +- 6.4 (t 3.8)
//
// P=1 wins, and it wins while the comparison is tilted against it -- it took
// 8% more wall than P=4, worth about 5 kcal/mol at the measured slope of 59
// kcal per e-fold of budget, so the corrected penalty for P=4 is still around
// 20 kcal/mol. So the population is not merely unnecessary for minimisation,
// it is mildly harmful: every candidate beyond the first buys breadth the
// accept/reject rule cannot use, and pays for it in depth.
//
// The default is therefore one replica. Large P remains the right choice for
// sampling an ensemble, which is a different objective and is not what this
// routine is usually asked for. For reference the pre-batching single-
// candidate path using randContinuousSidechainConformation reaches -1158 in
// 47 s, which P=1 now beats by ~77 kcal/mol in about half the time.
//
// Two deliberate departures from randContinuousSidechainConformation:
//
//   One chi per trial. That generator perturbs every chi of a residue at once,
//   so acceptance falls off roughly exponentially in the chi count and the long
//   sidechains -- Lys, Arg, Met, Glu, Gln -- are penalised hardest. That is
//   backwards: those are the ones that most need to move. Moving a single
//   randomly chosen chi keeps the acceptance rate comparable across residue
//   types. The lever-arm scaling of the original is kept, since measurement
//   showed it does what it was designed to do: span 180/(distal chis) holds the
//   RMS Cartesian displacement of the moved atoms near 2 A regardless of which
//   chi is chosen.
//
//   A rotamer-hop component. An incremental walk of at most +/-90 degrees
//   cannot cross the ~120 degree barriers between sp3 rotamer wells, because
//   the eclipsed intermediate is high enough in energy that Metropolis rejects
//   the path. Chi1 therefore stays in whatever well the input structure
//   supplied, and no amount of sampling fixes it. Twenty percent of proposals
//   are instead a discrete +/-120 degree jump, which lands directly in an
//   adjacent well without traversing the barrier. Both components are symmetric
//   proposals, so Metropolis remains valid with no Hastings correction.
//
//   Measured, this component does not currently earn its keep: at a matched
//   budget, hop fractions of 0 and 0.20 are indistinguishable on 1crn (-489.2
//   vs -490.8) and 0.20 is slightly worse on 1ubq. The reasoning above is
//   sound but 20% is untuned, and it may simply be spending too many proposals
//   on jumps that land in occupied space. PROTCAD_MC_HOP exists to test it.
void protein::protMinReplicaCU(UInt _sweeps, UInt _nReplicas)
{
	saveCurrentState();
	if (!deviceMemLoadedAll) {loadDeviceMemAll();}
	if (_sweeps == 0 || _nReplicas == 0) {return;}

	const int N = syncDeviceCoords();
	if (N == 0) {return;}
	if (buildRotationGroups() <= 0) {return;}

	const int P = (int)_nReplicas;
	if (energySetReplicas(itsEnergyContext, P) != 0)
	{
		cout << "protein::protMinReplicaCU failed: " << energyLastError(itsEnergyContext) << endl;
		return;
	}

	const int stride = maxRotationGroupCount();
	if (stride <= 0) {return;}

	// Note on throughput: per-candidate cost rises over a long run (116 us at
	// 2000 sweeps to 187 us at 20000 on 1crn). This is not a defect. It was
	// first suspected to be stale spatial ordering -- the tiling is rebuilt
	// every launch but from the resident coordinates, which the sweep loop
	// never touches -- but periodically re-seeding the order from the best
	// conformation recovered only ~1%. The actual cause is physical: a 2000
	// sweep run started from an already-minimized 1crn costs 164 us/candidate
	// versus 116 us from the crystal. Minimization compacts the structure, so
	// more pairs genuinely survive the 12 A cutoff. The extra time is real
	// work, and it is not worth optimizing away.

	const double KT = KB * Temperature();
	const UInt chainNum = getNumChains();

	double startEnergy = protEnergyCU();
	vector <double> current(P, startEnergy), trial(P, 0.0);
	vector <int> groupBegin(P, 0), nGroups(P, 0), accept(P, 0);
	vector <double> angles((size_t)P * stride, 0.0);

	// The best conformation seen must be captured when it is seen: the replica
	// that found it keeps walking and will usually leave it behind.
	double bestEnergy = startEnergy; bool haveBest = false;
	vector <double> bestX(N), bestY(N), bestZ(N);

	// Proposal knobs, exposed so the proposal itself can be A/B tested at a
	// fixed sweep budget. Defaults reproduce the shipped behavior exactly.
	//
	//   PROTCAD_MC_HOP       fraction of trials that are discrete +/-120 deg
	//                        rotamer hops. Set to 0 to isolate the incremental
	//                        proposal's quality with no barrier crossing.
	//   PROTCAD_MC_PROPOSAL  "onechi" perturbs a single randomly chosen chi;
	//                        "allchi" perturbs every chi of the residue at once,
	//                        which is what randContinuousSidechainConformation
	//                        does and makes acceptance decay exponentially in
	//                        chi count.
	//   PROTCAD_MC_SEED      fixes the RNG seed. Needed because time(NULL) makes
	//                        replicates launched in the same second identical.
	double hopFraction = 0.20;
	if (const char* h = getenv("PROTCAD_MC_HOP")) {hopFraction = atof(h);}
	bool allChi = false;
	if (const char* pm = getenv("PROTCAD_MC_PROPOSAL")) {allChi = (string(pm) == "allchi");}
	if (const char* sd = getenv("PROTCAD_MC_SEED")) {srand((unsigned)atoi(sd));}
	else {srand(time(NULL));}

	for (UInt sweep = 0; sweep < _sweeps; sweep++)
	{
		for (int k = 0; k < P; k++)
		{
			for (int g = 0; g < stride; g++) {angles[(size_t)k * stride + g] = 0.0;}
			groupBegin[k] = 0; nGroups[k] = 0;

			// Pick a residue that actually has rotatable chis. Glycine and
			// alanine are common enough that rejection sampling needs a bound.
			int begin = 0, count = 0;
			for (int attempt = 0; attempt < 32; attempt++)
			{
				UInt randchain = rand() % chainNum;
				UInt randres = rand() % getNumResidues(randchain);
				if (getRotationGroupRange(randchain, randres, begin, count)) {break;}
				count = 0;
			}
			if (count <= 0) {continue;}   // null move: legal, costs one slot

			groupBegin[k] = begin; nGroups[k] = count;

			const int jFixed = rand() % count;
			for (int j = 0; j < count; j++)
			{
				if (!allChi && j != jFixed) {continue;}

				int distal = count - j; if (distal < 2) {distal = 2;}
				const double span = 180.0 / distal;

				double delta;
				if ((rand() / (double)RAND_MAX) < hopFraction)
				{
					delta = (rand() % 2) ? 120.0 : -120.0;
				}
				else
				{
					delta = 2.0 * span * (rand() / (double)RAND_MAX - 0.5);
				}
				angles[(size_t)k * stride + j] = delta;
			}
		}

		if (energyReplicaBatch(P, &groupBegin[0], &nGroups[0], &angles[0], stride, &trial[0]) != 0)
		{
			cout << "protein::protMinReplicaCU failed: " << energyLastError(itsEnergyContext) << endl;
			break;
		}

		for (int k = 0; k < P; k++)
		{
			const double deltaEnergy = trial[k] - current[k];
			bool boltzmannAcceptance = (deltaEnergy <= 0.0) ||
			                           ((rand() / (double)RAND_MAX) < exp(-deltaEnergy / KT));
			accept[k] = boltzmannAcceptance ? 1 : 0;
			if (boltzmannAcceptance)
			{
				current[k] = trial[k];
				if (trial[k] < bestEnergy)
				{
					// Still in the batch buffer at this point, before commit.
					if (getBatchCoords(k, &bestX[0], &bestY[0], &bestZ[0]) == 0)
					{
						bestEnergy = trial[k]; haveBest = true;
					}
				}
			}
		}

		if (commitReplicas(P, &accept[0]) != 0)
		{
			cout << "protein::protMinReplicaCU commit failed: "
			     << energyLastError(itsEnergyContext) << endl;
			break;
		}
	}

	if (haveBest)
	{
		setCoordsFromArray(&bestX[0], &bestY[0], &bestZ[0]);
		syncDeviceCoords();
	}
	E = bestEnergy;
	return;
}

double protein::protEnergyCU()
{
	int N = updateDeviceCoords();
	if (N == 0) {return 0.0;}
	double e = 0.0;
	if (energyCompute(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0], &e, 0))
	{
		cout << "protein::protEnergyCU failed: " << energyLastError(itsEnergyContext) << endl;
		return 0.0;
	}
	E = e;
	return e;
}

// Energy of one chain in isolation.
//
// This is deliberately not a sum of that chain's per-atom contributions in the
// complex.  Those contributions include the interchain terms, which is exactly
// what a binding energy is defined to remove, so summing them would return the
// complex energy partitioned rather than the isolated-chain energy and drive
// the binding energy to zero.  Masking the other chains instead reproduces the
// CPU convention: evaluate the chain as if nothing else were present.
//
// The mask goes through energySetSilent rather than a context rebuild, so the
// topology, exclusion lists and device allocations are all reused.  The prior
// silent flags are restored on both the success and failure paths; leaving a
// chain masked would silently corrupt every later evaluation.
double protein::protEnergyCU(UInt _chainIndex)
{
	int N = updateDeviceCoords();
	if (N == 0) {return 0.0;}
	vector<unsigned char> base(N, 0), mask(N, 1);
	int i = 0;
	for (atomIterator aIter(this); !(aIter.last()) && i < N; aIter++, i++)
	{
		residue* pRes = aIter.getResiduePointer();
		unsigned char s = pRes->getAtom(aIter.getAtomIndex())->getSilentStatus() ? 1 : 0;
		base[i] = s;
		mask[i] = (aIter.getChainIndex() == _chainIndex) ? s : 1;
	}
	energySetSilent(itsEnergyContext, &mask[0]);
	double e = 0.0;
	int rc = energyCompute(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0], &e, 0);
	energySetSilent(itsEnergyContext, &base[0]);
	if (rc)
	{
		cout << "protein::protEnergyCU(chain) failed: " << energyLastError(itsEnergyContext) << endl;
		return 0.0;
	}
	return e;
}

// Refresh every atom's stored local dielectric from the kernel.
//
// The CPU path derived these from a pairwise polarizability pass followed by a
// per-chain reduction.  The kernel already computes the same field -- the shell
// occupancy that scales the electrostatic term -- so exporting it is both
// cheaper and guaranteed consistent: a dielectric reported here is by
// construction the one the energy was evaluated with, which the two separate
// implementations could not promise.
//
// Values land in the atoms in atomIterator order, the same ordering contract
// buildEnergyContext uses, so residue::getDielectric keeps working unchanged.
void protein::updateDielectricsCU()
{
	int N = updateDeviceCoords();
	if (N == 0) {return;}
	vector<double> diel(N, 0.0);
	if (shellCompute(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0],
	                 &diel[0], 0))
	{
		cout << "protein::updateDielectricsCU failed: "
		     << energyLastError(itsEnergyContext) << endl;
		return;
	}
	int i = 0;
	for (atomIterator aIter(this); !(aIter.last()) && i < N; aIter++, i++)
	{
		aIter.getResiduePointer()->getAtom(aIter.getAtomIndex())->setDielectric(diel[i]);
	}
}

bool protein::protFreezeDielectricCU()
{
	int N = updateDeviceCoords();
	if (N == 0) {return false;}
	if (energyFreezeDielectric(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0]))
	{
		cout << "protein::protFreezeDielectricCU failed: "
		     << energyLastError(itsEnergyContext) << endl;
		return false;
	}
	return true;
}

bool protein::protThawDielectricAllCU()
{
	if (!itsEnergyContext) {return false;}
	return energySetDielectricThaw(itsEnergyContext, 0, -1, 0) == 0;
}

bool protein::protThawDielectricNoneCU()
{
	if (!itsEnergyContext) {return false;}
	return energySetDielectricThaw(itsEnergyContext, 0, 0, 0) == 0;
}

int protein::protThawDielectricNearCU(UInt _chainIndex, UInt _resIndex, double _radius,
                                      bool _accumulate)
{
	int N = updateDeviceCoords();
	if (N == 0) {return -1;}

	// Collect the moved residue's atoms first, then sweep once.  Done on the
	// host because this is set-up, not the inner loop; a per-move version
	// belongs on the device.
	vector<UInt> targets;
	int i = 0;
	for (atomIterator aIter(this); !(aIter.last()) && i < N; aIter++, i++)
	{
		if (aIter.getChainIndex() == (int)_chainIndex &&
		    aIter.getResidueIndex() == (int)_resIndex) {targets.push_back(i);}
	}
	if (targets.empty()) {return -1;}

	const double r2 = _radius * _radius;
	vector<int> thaw;
	for (int a = 0; a < N; a++)
	{
		for (UInt t = 0; t < targets.size(); t++)
		{
			const UInt b = targets[t];
			const double dx = itsCoordX[a] - itsCoordX[b];
			const double dy = itsCoordY[a] - itsCoordY[b];
			const double dz = itsCoordZ[a] - itsCoordZ[b];
			if (dx * dx + dy * dy + dz * dz <= r2) {thaw.push_back(a); break;}
		}
	}
	if (energySetDielectricThaw(itsEnergyContext, thaw.empty() ? 0 : &thaw[0],
	                            (int)thaw.size(), _accumulate ? 1 : 0))
	{
		cout << "protein::protThawDielectricNearCU failed: "
		     << energyLastError(itsEnergyContext) << endl;
		return -1;
	}
	return energyDielectricThawCount(itsEnergyContext);
}

void protein::protSetCutoffCU(double _cutoff)
{
	if (!itsEnergyContext) {return;}
	energyParams p = energyGetParams(itsEnergyContext);
	p.cutoff = _cutoff;
	p.switchStart = _cutoff - 2.0;
	energySetParams(itsEnergyContext, p);
}

double protein::protGetCutoffCU()
{
	if (!itsEnergyContext) {return 0.0;}
	return (double)energyGetParams(itsEnergyContext).cutoff;
}

int protein::protEnergyDeltaCU(double& _part, double& _torsion)
{
	if (itsThawList.empty()) {return -1;}
	buildEnergyContext();
	if (!itsEnergyContext) {return -1;}
	const int N = getNumAtoms();
	if (N != (int)itsCoordX.size() || (int)itsAtomPtrs.size() != N)
	{
		if (updateDeviceCoords() == 0) {return -1;}
	}
	// The delta must cover exactly the atoms whose occupancy is allowed to
	// differ from the snapshot, which is the thaw set the move installed, and
	// nothing outside it can have moved.
	refreshDeviceCoords(itsThawList);
	return energyComputeDelta(itsEnergyContext, &itsCoordX[0], &itsCoordY[0],
	                          &itsCoordZ[0], &itsThawList[0],
	                          (int)itsThawList.size(), &_part, &_torsion);
}

double protein::protDielectricInfluenceRadiusCU()
{
	return energyDielectricInfluenceRadius(itsEnergyContext);
}

void protein::getDeviceCoordsCU(vector<double>& _x, vector<double>& _y,
                                vector<double>& _z)
{
	updateDeviceCoords();
	_x = itsCoordX; _y = itsCoordY; _z = itsCoordZ;
}

int protein::protThawDielectricForMoveCU(const vector<double>& _oldX,
                                         const vector<double>& _oldY,
                                         const vector<double>& _oldZ,
                                         double _radius, int* _movedOut,
                                         double _tol)
{
	int N = updateDeviceCoords();
	if (N == 0 || (int)_oldX.size() < N) {return -1;}
	if (_radius <= 0.0)
	{	_radius = energyDielectricInfluenceRadius(itsEnergyContext); }

	vector<UInt> movedAtoms;
	for (int a = 0; a < N; a++)
	{
		const double dx = itsCoordX[a] - _oldX[a];
		const double dy = itsCoordY[a] - _oldY[a];
		const double dz = itsCoordZ[a] - _oldZ[a];
		if (dx * dx + dy * dy + dz * dz > _tol * _tol) {movedAtoms.push_back(a);}
	}
	if (_movedOut) {*_movedOut = (int)movedAtoms.size();}
	if (movedAtoms.empty())
	{	energySetDielectricThaw(itsEnergyContext, 0, 0, 0);
		itsThawList.clear();
		return 0; }

	const double r2 = _radius * _radius;
	vector<int> thaw;
	for (int a = 0; a < N; a++)
	{
		bool hit = false;
		for (UInt m = 0; m < movedAtoms.size() && !hit; m++)
		{
			const UInt b = movedAtoms[m];
			// Both endpoints: the move perturbs an atom's shell whether it
			// brings a neighbour in or takes one out.
			double dx = itsCoordX[a] - itsCoordX[b];
			double dy = itsCoordY[a] - itsCoordY[b];
			double dz = itsCoordZ[a] - itsCoordZ[b];
			if (dx * dx + dy * dy + dz * dz <= r2) {hit = true; break;}
			dx = itsCoordX[a] - _oldX[b];
			dy = itsCoordY[a] - _oldY[b];
			dz = itsCoordZ[a] - _oldZ[b];
			if (dx * dx + dy * dy + dz * dz <= r2) {hit = true;}
		}
		if (hit) {thaw.push_back(a);}
	}
	if (energySetDielectricThaw(itsEnergyContext, thaw.empty() ? 0 : &thaw[0],
	                            (int)thaw.size(), 0))
	{	return -1; }
	itsThawList = thaw;
	return energyDielectricThawCount(itsEnergyContext);
}

void protein::protReleaseDielectricCU()
{
	if (itsEnergyContext) {energyReleaseDielectric(itsEnergyContext);}
}

// The CPU minimisers and relaxers are gone; these names now route to the
// kernel-backed implementations so callers keep working.
void protein::protMin(bool _backbone) {protMinCU(_backbone);}

void protein::protMin(bool _backbone, UIntVec _frozenResidues, UIntVec _activeChains)
{
	protMinCU(_backbone, _frozenResidues, _activeChains);
}

void protein::protRelax(UInt _sweeps, bool _backbone) {protRelaxCU(_sweeps, _backbone);}

void protein::protRelax(UIntVec _frozenResidues, UIntVec _activeChains)
{
	protRelaxCU(_frozenResidues, _activeChains);
}

// Backbone-only clash count.  The CPU pass defined the backbone as a residue's
// first four atoms (N, CA, C, O) plus CB where present; that definition is
// preserved here by silencing every other atom and reusing the same clash
// kernel the whole-protein count goes through, so the two agree on what a
// clash is.  The old pair loop used the pre-ff14SB radii and did not.
UInt protein::getNumHardBackboneClashesCU()
{
	int N = updateDeviceCoords();
	if (N == 0) {return 0;}
	vector<unsigned char> base(N, 0), mask(N, 1);
	int i = 0;
	for (atomIterator aIter(this); !(aIter.last()) && i < N; aIter++, i++)
	{
		residue* pRes = aIter.getResiduePointer();
		UInt a = aIter.getAtomIndex();
		unsigned char s = pRes->getAtom(a)->getSilentStatus() ? 1 : 0;
		base[i] = s;
		bool backbone = (a < 4) || (a == 4 && pRes->getAtom(4)->getName() == "CB");
		mask[i] = backbone ? s : 1;
	}
	energySetSilent(itsEnergyContext, &mask[0]);
	int c = 0;
	int rc = clashCompute(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0], &c);
	energySetSilent(itsEnergyContext, &base[0]);
	if (rc)
	{
		cout << "protein::getNumHardBackboneClashesCU failed: "
		     << energyLastError(itsEnergyContext) << endl;
		return 0;
	}
	return c < 0 ? 0 : (UInt)c;
}

UInt protein::getNumHardBackboneClashes()
{
	return getNumHardBackboneClashesCU();
}

// Median residue clash count, now read off the per-residue GPU pass.
UInt protein::getMedianResidueNumHardClashes()
{
	updateResidueClashesCU();
	vector<UInt> counts;
	for (UInt i = 0; i < itsChains.size(); i++)
	{
		for (UInt j = 0; j < getNumResidues(i); j++)
		{
			counts.push_back(itsChains[i]->getClashes(j));
		}
	}
	if (counts.empty()) {return 0;}
	sort(counts.begin(), counts.end());
	return counts[counts.size()/2];
}

// Sum of the isolated-chain energies, i.e. the total with every interchain
// contribution removed.
double protein::intraEnergy()
{
	double e = 0.0;
	for (UInt i = 0; i < itsChains.size(); i++) {e += protEnergyCU(i);}
	return e;
}

// Per-residue energies and clashes, derived from the kernel's per-atom exports
// rather than from a second implementation of the model.  energyComputeAtoms
// splits every pair interaction evenly between its two atoms, so summing a
// residue's atoms gives that residue's share of the total and the residue
// values sum back to the protein energy.
void protein::updateResidueEnergiesCU()
{
	int N = updateDeviceCoords();
	if (N == 0) {return;}
	vector<double> perAtom(N, 0.0);
	double total = 0.0;
	if (energyComputeAtoms(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0],
	                       &total, 0, &perAtom[0]))
	{
		cout << "protein::updateResidueEnergiesCU failed: "
		     << energyLastError(itsEnergyContext) << endl;
		return;
	}
	E = total;
	residue* current = 0;
	double sum = 0.0;
	int i = 0;
	for (atomIterator aIter(this); !(aIter.last()) && i < N; aIter++, i++)
	{
		residue* pRes = aIter.getResiduePointer();
		if (pRes != current)
		{
			if (current) {current->setEnergy(sum);}
			current = pRes;
			sum = 0.0;
		}
		sum += perAtom[i];
	}
	if (current) {current->setEnergy(sum);}
	setMoved(false,0);
}

// perAtomOut records each clashing pair once for each of its two atoms, so a
// residue's count is the number of clashing contacts its atoms take part in.
void protein::updateResidueClashesCU()
{
	int N = updateDeviceCoords();
	if (N == 0) {return;}
	vector<int> perAtom(N, 0);
	int total = 0;
	if (clashComputeAtoms(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0],
	                      &total, &perAtom[0]))
	{
		cout << "protein::updateResidueClashesCU failed: "
		     << energyLastError(itsEnergyContext) << endl;
		return;
	}
	clash = total;
	residue* current = 0;
	UInt sum = 0;
	int i = 0;
	for (atomIterator aIter(this); !(aIter.last()) && i < N; aIter++, i++)
	{
		residue* pRes = aIter.getResiduePointer();
		if (pRes != current)
		{
			if (current) {current->setClashes(sum);}
			current = pRes;
			sum = 0;
		}
		sum += (UInt)perAtom[i];
	}
	if (current) {current->setClashes(sum);}
	setMoved(false,1);
}

// Per-term decomposition, which the original could not provide: it reduced
// everything into a single accumulator inside the kernel.
bool protein::protEnergyBreakdownCU(energyBreakdown& _out)
{
	int N = updateDeviceCoords();
	if (N == 0) {return false;}
	double e = 0.0;
	if (energyCompute(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0], &e, &_out)) {return false;}
	E = e;
	return true;
}

int protein::getNumClashesCU()
{
	int N = updateDeviceCoords();
	if (N == 0) {return 0;}
	int c = 0;
	if (clashCompute(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0], &c))
	{
		cout << "protein::getNumClashesCU failed: " << energyLastError(itsEnergyContext) << endl;
		return 0;
	}
	clash = c;
	return c;
}

double protein::relaxSidechainsCU(UIntVec _frozenResidues, UIntVec _activeChains, UInt _maxSweeps)
{
	// Energy-based steepest descent, not clash-based. The old relax accepted a
	// rotamer whenever it lowered a hard-sphere clash count, an objective the
	// selection criterion downstream never reads. Measured on 1crn that left a
	// ~2600 kcal/mol spread between runs at an essentially fixed clash count,
	// which swamps a Boltzmann test resolving a few kcal/mol. Clash and vdW now
	// share the same AMBER radii, so the clash count is only a step-function
	// shadow of a term the energy already computes more sharply. Let energy decide.
	if (!deviceMemLoadedAll){loadDeviceMemAll();}
	double pastEnergy = protEnergyCU();
	if (_activeChains.size() == 0) {return pastEnergy;}

	static const UInt SIDECHAIN_CANDIDATES =
		getenv("PROTCAD_CANDIDATES") ? (UInt)atoi(getenv("PROTCAD_CANDIDATES")) : 32;
	vector < vector <double> > currentConf, newConf;
	double trialEnergy;

	// Build the list of movable residues once, then visit it in a fresh random
	// order each sweep so no residue is systematically optimized last.
	vector < pair <UInt, UInt> > movable;
	for (UInt c = 0; c < _activeChains.size(); c++)
	{
		UInt chainIndex = _activeChains[c];
		if (chainIndex >= getNumChains()) {continue;}
		for (UInt r = 0; r < getNumResidues(chainIndex); r++)
		{
			bool skip = false;
			for (UInt i = 0; i < _frozenResidues.size(); i++)
			{
				if (r == _frozenResidues[i]) {skip = true; break;}
			}
			if (skip) {continue;}
			// Residues with no rotatable chi have nothing to sample.
			currentConf = getSidechainDihedrals(chainIndex, r);
			bool hasChi = false;
			for (UInt b = 0; b < currentConf.size(); b++)
			{
				if (currentConf[b].size() > 0) {hasChi = true; break;}
			}
			if (!hasChi) {continue;}
			movable.push_back(make_pair(chainIndex, r));
		}
	}
	if (movable.size() == 0) {return pastEnergy;}

	for (UInt sweep = 0; sweep < _maxSweeps; sweep++)
	{
		for (UInt i = movable.size(); i > 1; i--)
		{
			UInt j = rand() % i;
			swap(movable[i-1], movable[j]);
		}
		bool improved = false;
		for (UInt m = 0; m < movable.size(); m++)
		{
			UInt chainIndex = movable[m].first, resIndex = movable[m].second;
			currentConf = getSidechainDihedrals(chainIndex, resIndex);
			trialEnergy = bestSidechainCandidateCU(chainIndex, resIndex, SIDECHAIN_CANDIDATES, newConf);
			if (trialEnergy < pastEnergy)
			{
				setSidechainDihedralAngles(chainIndex, resIndex, newConf);
				pastEnergy = trialEnergy;
				improved = true;
			}
		}
		// A sweep that improves nothing will not improve on the next pass either,
		// beyond the sampling noise of the candidate draw.
		if (!improved) {break;}
	}
	E = pastEnergy;
	return pastEnergy;
}

void protein::protRelaxCU(UInt _sweeps, bool _backbone)
{
	saveCurrentState();
	UIntVec frozenResidues, activeChains;
	for (UInt i = 0; i < getNumChains(); i++) {activeChains.push_back(i);}
	relaxSidechainsCU(frozenResidues, activeChains, _sweeps);
	return;
}

void protein::protRelaxCU(UIntVec _frozenResidues, UIntVec _activeChains)
{
	saveCurrentState();
	relaxSidechainsCU(_frozenResidues, _activeChains, 5);
	return;
}

bool protein::minDeltaAnchorCU(double& _nbCurrent, double& _total)
{
	// The anchor is a coupled evaluation, not a frozen one: the chain's whole
	// error budget is referenced to it, so it must not itself be approximate.
	// Splitting off torsion lets the delta replace that term wholesale rather
	// than difference it, and priming the per-torsion cache is a side effect
	// the delta depends on.
	protReleaseDielectricCU();
	energyBreakdown b;
	if (!protEnergyBreakdownCU(b)) {return false;}
	_total = b.total;
	_nbCurrent = b.total - b.torsion;
	return protFreezeDielectricCU();
}

bool protein::minDeltaCommitCU(double _nbBest, double& _nbCurrent)
{
	// The batch left the device at the entry geometry and its occupancy with
	// it.  One single-candidate delta at the winner brings the resident field
	// to the accepted conformation -- a thirty-second of the batch, over the
	// same thaw set -- and the snapshot then follows from it directly.
	double part = 0.0, tor = 0.0;
	if (protEnergyDeltaCU(part, tor) != 0) {return false;}
	if (!protRefreezeDielectricCU()) {return false;}
	_nbCurrent = _nbBest;
	return true;
}


// Termination for protMinCU.
//
// The criterion this replaces counted consecutive trials with no improvement
// better than KT and stopped once that count reached `plateau`.  That is a
// hitting time, and it measures a property of the proposal mechanism rather
// than of the trajectory.  It worked for a first-improvement move on a single
// candidate, whose success probability really does fall towards zero as the
// structure converges.  It does not survive steepest descent over a batch of
// 32: some candidate clears the KT bar on nearly every trial no matter how well
// optimised the structure already is, so the counter resets almost every
// iteration and the required run of consecutive failures effectively never
// occurs.  A 1crn minimisation that took 13 s under the old move ran for 83
// minutes without terminating.
//
// protMinReplicaCU answered that by refusing to measure anything and taking a
// fixed sweep budget instead, on the argument that a sampler's cost should be
// chosen rather than discovered.  That is right for sampling an ensemble and
// wrong here, because a fixed budget spends the same wall clock on a structure
// that is already good as on one that needs deep optimisation, which is the
// entire distinction a minimiser is asked to make.
//
// What makes a trajectory criterion possible is the shape of the descent, and
// the shape is roughly log-linear -- protMinReplicaCU measures about 59 kcal/mol
// per e-fold of budget on 1ubq.  Improvement per *fixed* window therefore
// decays like 1/B by construction and has no plateau to detect, which is the
// deeper reason the old counter had nothing to converge on.  Improvement per
// *doubling* of the budget is the quantity that is flat under that law, so it
// is the one whose decay actually signals diminishing returns.
//
// Hence: checkpoint at geometrically spaced trial counts, and at each
// checkpoint compare the best energy seen against the best energy seen one
// doubling ago.  Stop when a doubling of the entire budget spent so far buys
// less than `gainTol`.  Measured over 1 -> 256 sweeps, gain per doubling:
//
//     1crn  13.1  3.7  10.0  12.7  5.9  3.0  0.8  1.4  0.0
//     1ubq 103   495   50.3  49.0 30.0 14.3 38.2
//
// 1crn is spent by 64 sweeps having recovered 49.1 of 50.5 available kcal/mol;
// 1ubq is still buying 38 kcal/mol per doubling at 64 sweeps and should keep
// going.  A single absolute threshold separates them, and this is exactly the
// adaptivity a budget cannot express.
//
// The threshold is deliberately absolute rather than a fraction of the gain so
// far.  At the checkpoint where 1crn has 1.4 kcal/mol left to win and 1ubq has
// 58, both have last-doubling gains near 1.8% of their cumulative gain, so a
// relative rule cannot tell them apart and would stop 1ubq far too early.  What
// matters is whether more work buys a meaningful amount of energy, not a
// meaningful fraction of an extensive quantity.
//
// Two properties fall out for free.  Overshoot is bounded: the criterion can
// never spend more than twice the work that was actually productive, because
// checkpoints double.  And the unit of work is a sweep -- one trial per
// residue -- so the budget scales with the structure without any constant
// needing to be retuned per size.
//
// `staleNeeded` guards against the noise visible above, where 1crn dips to 0.8
// and recovers to 1.4 and 1ubq dips to 14.3 and recovers to 38.2.  It is 1 by
// default because insurance costs a doubling and the recovered amounts are
// small next to the threshold; raise it when final energy matters more than
// wall clock.
struct minStopCU
{
	double gainTol, best, bestAtHalf;
	UInt staleNeeded, maxTrials, trial, nextCheck, stale, sweep;
	bool debug;

	static double envD(const char* n, double d)
	{	const char* v = getenv(n); return v ? atof(v) : d; }
	static UInt envU(const char* n, UInt d)
	{	const char* v = getenv(n); return v ? (UInt)atol(v) : d; }

	minStopCU(UInt _sweep, double _startEnergy, bool _debug)
	{
		sweep = _sweep < 1 ? 1 : _sweep;
		gainTol     = envD("PROTCAD_MIN_GAIN", 1.0);
		staleNeeded = envU("PROTCAD_MIN_STALE", 2);
		maxTrials   = envU("PROTCAD_MIN_MAXSWEEPS", 256) * sweep;
		best = bestAtHalf = _startEnergy;
		trial = 0; nextCheck = sweep; stale = 0; debug = _debug;
	}

	// Called once per trial with the energy of the accepted conformation.
	void observe(double _energy)
	{
		trial++;
		if (_energy < best) {best = _energy;}
		if (trial < nextCheck) {return;}
		const double gain = bestAtHalf - best;
		if (gain <= gainTol) {stale++;} else {stale = 0;}
		if (debug)
		{	cout << "protMinCU: " << (double)trial / sweep << " sweeps, best "
			     << best << ", gain/doubling " << gain << ", stale " << stale << endl; }
		bestAtHalf = best; nextCheck *= 2;
	}

	bool done() const {return stale >= staleNeeded || trial >= maxTrials;}
};

void protein::protMinCU(bool _backbone, UIntVec _frozenResidues, UIntVec _activeChains, UInt _plateau)
{
	// Sidechain and backslide optimization with a local dielectric scaling of electrostatics and corresponding Born/Gill implicit solvation energy
	saveCurrentState();
	
	// load data on GPU
	if (!deviceMemLoadedAll){loadDeviceMemAll();}

	//--Initialize variables for loop, calculate starting energy and build energy vectors-----
	UInt randchain, randres, resnum, backboneOrSidechain = 1;
	UInt chainNum = _activeChains.size(), plateau = _plateau;
	static const UInt SIDECHAIN_CANDIDATES =
		getenv("PROTCAD_CANDIDATES") ? (UInt)atoi(getenv("PROTCAD_CANDIDATES")) : 32;
	double trialEnergy = 0.0; bool haveTrialEnergy = false;
	double Energy, pastEnergy = protEnergyCU(), deltaEnergy, sPhi, sPsi,nobetter = 0.0, KT = KB*Temperature();
	//double rotX, rotY, rotZ, transX, transY, transZ;
	vector < DouVec > currentSidechainConf, newSidechainConf; srand (time(NULL)); vector <double> backboneAngles(2);
	bool sidechainTest, backboneTest, revert, cofactorTest, energyTest, skip, boltzmannAcceptance;
	vector <dblVec> currentCoords;
	// Delta chain.  Sidechain candidate batches are evaluated against a held
	// dielectric instead of from scratch, which is where nearly all of this
	// loop's time goes.  The chain is re-anchored periodically because the
	// identity accumulates rounding over thousands of accepted moves, and it is
	// abandoned entirely on a backbone move, which perturbs the whole structure
	// and invalidates the snapshot.
	static const bool deltaMin = (getenv("PROTCAD_MIN_NO_DELTA") == 0);
	static const bool deltaDebug = (getenv("PROTCAD_MIN_DEBUG") != 0);
	const int ANCHOR_EVERY = 64;
	bool anchored = false, havePending = false;
	double nbCurrent = 0.0, pendingNb = 0.0, worstDrift = 0.0;
	int sinceAnchor = 0;
	// Stop when a doubling of the budget stops paying; see minStopCU above.
	static const bool legacyStop = (getenv("PROTCAD_MIN_PLATEAU") != 0);
	UInt sweepTrials = 0;
	for (UInt ci = 0; ci < _activeChains.size(); ci++)
	{	sweepTrials += getNumResidues(_activeChains[ci]); }
	if (sweepTrials > _frozenResidues.size()) {sweepTrials -= _frozenResidues.size();}
	minStopCU stop(sweepTrials, pastEnergy, deltaDebug);
	// Best-so-far.  Boltzmann acceptance takes uphill moves, so the last
	// conformation is not the best one; keep a copy and hand that back.
	vector<double> bestX, bestY, bestZ;
	double bestEnergy = pastEnergy;
	snapshotConformationCU(bestX, bestY, bestZ);
	//--Run optimizaiton loop to local minima defined by an RT plateau------------------------
	do{
		//--choose random residue not frozen of active chains
			randchain = _activeChains[rand() % chainNum];
			do{
				skip = false;
				randres = rand() % getNumResidues(randchain);
				for (UInt i = 0; i < _frozenResidues.size(); i++)
				{
					if (randres == _frozenResidues[i]) {skip = true; break;}
				}
			} while (skip);
		resnum = getNumResidues(randchain); nobetter++;
		// No clash pre-filter.  The discrete clash count was a CPU-era device for
		// skipping an expensive energy evaluation after an obviously bad move.  On
		// the GPU it does not pay for itself, and worse, "clashes <= clashesStart"
		// is a hard veto that rejects any move adding a single clash regardless of
		// energy -- overriding the Boltzmann criterion that is supposed to own that
		// decision.  Since energy radii became AMBER, clash and vdW read the same
		// radius array, so the gate is a step-function shadow of the r^-12 term the
		// energy already computes more sharply and smoothly.  Let energy decide.
		backboneTest = false, sidechainTest = false, cofactorTest = false, energyTest = false, revert = true;
		haveTrialEnergy = false; trialEnergy = 0.0;
		
		/*if (isCofactor(randchain, randres))
		{
			//--Rock and Roll cofactor in site
			cofactorTest = true;
			currentCoords.clear();
			currentCoords = saveCoords(randchain, randres);
			rotX = rand() % 2, rotY = rand() % 2, rotZ = rand() % 2;
			transX = (rand() % 30)/100, transY = (rand() % 30)/100, transZ = (rand() % 30)/100;
			rotateChainRelative(randchain,X_axis,rotX), rotateChainRelative(randchain,Y_axis,rotY), rotateChainRelative(randchain,Z_axis,rotZ);
			translateChain(randchain, transX, transY, transZ);
			clashes = getNumHardClashes();
			if (clashes <= clashesStart){
					energyTest = true; revert = false;
			}
		}
		else{*/
			//--Backbone conformation trial--------------------------------------------------------
			if (_backbone) {backboneOrSidechain = rand() % 2;}
			if (randres > 0 && randres < resnum-2 && backboneOrSidechain == 0){
				// A backbone move displaces everything downstream of it, so the
				// held field is no longer the structure's and the chain ends here.
				if (anchored) {protReleaseDielectricCU(); anchored = false;}
				backboneTest = true;
				sPhi = getPhi(randchain,randres), sPsi = getPsi(randchain,randres);
				backboneAngles = getRandConformationFromBackboneType(sPhi, sPsi);
				setDihedral(randchain,randres,backboneAngles[0],0,0); setDihedral(randchain,randres,backboneAngles[1],1,0);
				energyTest = true; revert = false;
			}
			//--Sidechain conformation trial--------------------------------------------------------
			else{
				// Steepest descent over a batch of candidates rather than
				// first-improvement on a single one. The batch costs little more
				// than one candidate because the GPU is otherwise idle, and it
				// returns the winner's energy, so no separate evaluation follows.
				sidechainTest = true;
				currentSidechainConf = getSidechainDihedrals(randchain, randres);
				havePending = false; trialEnergy = 1E30;
				if (deltaMin)
				{
					if (!anchored || sinceAnchor >= ANCHOR_EVERY)
					{
						double fresh = 0.0;
						const bool wasAnchored = anchored;
						if (minDeltaAnchorCU(nbCurrent, fresh))
						{
							// The gap between the carried energy and a fresh
							// coupled one is the chain's accumulated error, and
							// it is the only thing here that can go wrong
							// silently, so it is measured rather than assumed.
							if (wasAnchored)
							{
								const double drift = fabs(fresh - pastEnergy);
								if (drift > worstDrift) {worstDrift = drift;}
								if (deltaDebug)
								{	cout << "protMinCU: re-anchor drift " << drift
									     << " over " << sinceAnchor << " moves" << endl; }
							}
							pastEnergy = fresh;
							anchored = true; sinceAnchor = 0;
						}
						else {protReleaseDielectricCU(); anchored = false;}
					}
					if (anchored)
					{
						double nbBest = 0.0, torBest = 0.0;
						trialEnergy = bestSidechainCandidateDeltaCU(randchain, randres,
						                                            SIDECHAIN_CANDIDATES,
						                                            newSidechainConf,
						                                            nbCurrent, nbBest, torBest);
						if (trialEnergy < 1E29) {pendingNb = nbBest; havePending = true;}
						else {protReleaseDielectricCU(); anchored = false;}
					}
				}
				if (!havePending)
				{
					if (anchored) {protReleaseDielectricCU(); anchored = false;}
					trialEnergy = bestSidechainCandidateCU(randchain, randres, SIDECHAIN_CANDIDATES, newSidechainConf);
				}
				if (trialEnergy < 1E29)
				{
					setSidechainDihedralAngles(randchain, randres, newSidechainConf);
					haveTrialEnergy = true; energyTest = true; revert = false;
				}
			}
		//}
		//--Energy-Test-------------------------------------------------------------------------
		if (energyTest){
			Energy = haveTrialEnergy ? trialEnergy : protEnergyCU();
			deltaEnergy = Energy - pastEnergy;
			boltzmannAcceptance = boltzmannEnergyCriteria(deltaEnergy);
			if (boltzmannAcceptance){
				pastEnergy = Energy;
				if (pastEnergy < bestEnergy)
				{	bestEnergy = pastEnergy;
					snapshotConformationCU(bestX, bestY, bestZ); }
				if (havePending)
				{
					if (minDeltaCommitCU(pendingNb, nbCurrent)) {sinceAnchor++;}
					else {protReleaseDielectricCU(); anchored = false;}
				}
				if (deltaEnergy < -KT){nobetter = 0;}
			}
			else{revert = true;}
		}
		//--Revert conformation-----------------------------------------------------------------
		if (revert){
			if(cofactorTest)
			{
				setAllCoords(randchain, randres, currentCoords);
			}
			if(backboneTest){
				setDihedral(randchain,randres,sPhi,0,0);
				setDihedral(randchain,randres,sPsi,1,0);
			}
			if(sidechainTest){
				setSidechainDihedralAngles(randchain, randres, currentSidechainConf);
			}
		}
		stop.observe(pastEnergy);
	} while (legacyStop ? (nobetter < plateau) : !stop.done());
	// A rejected move leaves the device at the entry geometry, so the chain
	// survives it untouched; only the exit has to put the coupled model back.
	if (anchored)
	{
		protReleaseDielectricCU();
		if (deltaDebug)
		{	cout << "protMinCU: worst re-anchor drift " << worstDrift << endl; }
	}
	// Hand back the best conformation seen, not the last one wandered into.
	if (bestEnergy < pastEnergy) {restoreConformationCU(bestX, bestY, bestZ);}
	return;
}

void protein::protMinCU(bool _backbone, UInt _plateau)
{
	// Sidechain and backslide optimization with a local dielectric scaling of electrostatics and corresponding Born/Gill implicit solvation energy
	saveCurrentState();
	
	// load data on GPU
	if (!deviceMemLoadedAll){loadDeviceMemAll();}

	//--Initialize variables for loop, calculate starting energy and build energy vectors-----
	UInt randchain, randres, resnum, backboneOrSidechain = 1;
	UInt chainNum = getNumChains(), plateau = _plateau;
	static const UInt SIDECHAIN_CANDIDATES =
		getenv("PROTCAD_CANDIDATES") ? (UInt)atoi(getenv("PROTCAD_CANDIDATES")) : 32;
	double trialEnergy = 0.0; bool haveTrialEnergy = false;
	double Energy, pastEnergy = protEnergyCU(), deltaEnergy, sPhi, sPsi,nobetter = 0.0, KT = KB*Temperature();
	//double rotX, rotY, rotZ, transX, transY, transZ;
	vector < DouVec > currentSidechainConf, newSidechainConf; srand (time(NULL)); vector <double> backboneAngles(2);
	bool sidechainTest, backboneTest, revert, cofactorTest, energyTest, boltzmannAcceptance;
	vector <dblVec> currentCoords;
	// Delta chain.  Sidechain candidate batches are evaluated against a held
	// dielectric instead of from scratch, which is where nearly all of this
	// loop's time goes.  The chain is re-anchored periodically because the
	// identity accumulates rounding over thousands of accepted moves, and it is
	// abandoned entirely on a backbone move, which perturbs the whole structure
	// and invalidates the snapshot.
	static const bool deltaMin = (getenv("PROTCAD_MIN_NO_DELTA") == 0);
	static const bool deltaDebug = (getenv("PROTCAD_MIN_DEBUG") != 0);
	const int ANCHOR_EVERY = 64;
	bool anchored = false, havePending = false;
	double nbCurrent = 0.0, pendingNb = 0.0, worstDrift = 0.0;
	int sinceAnchor = 0;
	// Stop when a doubling of the budget stops paying; see minStopCU above.
	static const bool legacyStop = (getenv("PROTCAD_MIN_PLATEAU") != 0);
	UInt sweepTrials = 0;
	for (UInt ci = 0; ci < getNumChains(); ci++)
	{	sweepTrials += getNumResidues(ci); }
	minStopCU stop(sweepTrials, pastEnergy, deltaDebug);
	// Best-so-far.  Boltzmann acceptance takes uphill moves, so the last
	// conformation is not the best one; keep a copy and hand that back.
	vector<double> bestX, bestY, bestZ;
	double bestEnergy = pastEnergy;
	snapshotConformationCU(bestX, bestY, bestZ);
	//--Run optimizaiton loop to local minima defined by an RT plateau------------------------
	do{
		//--choose random residue not frozen of active chains
		randchain = rand() % chainNum;
		randres = rand() % getNumResidues(randchain);
		resnum = getNumResidues(randchain); nobetter++;
		// No clash pre-filter.  The discrete clash count was a CPU-era device for
		// skipping an expensive energy evaluation after an obviously bad move.  On
		// the GPU it does not pay for itself, and worse, "clashes <= clashesStart"
		// is a hard veto that rejects any move adding a single clash regardless of
		// energy -- overriding the Boltzmann criterion that is supposed to own that
		// decision.  Since energy radii became AMBER, clash and vdW read the same
		// radius array, so the gate is a step-function shadow of the r^-12 term the
		// energy already computes more sharply and smoothly.  Let energy decide.
		backboneTest = false, sidechainTest = false, cofactorTest = false, energyTest = false, revert = true;
		haveTrialEnergy = false; trialEnergy = 0.0;
		
		/*if (isCofactor(randchain, randres))
		{
			//--Rock and Roll cofactor in site
			cofactorTest = true;
			currentCoords.clear();
			currentCoords = saveCoords(randchain, randres);
			rotX = rand() % 2, rotY = rand() % 2, rotZ = rand() % 2;
			transX = (rand() % 30)/100, transY = (rand() % 30)/100, transZ = (rand() % 30)/100;
			rotateChainRelative(randchain,X_axis,rotX), rotateChainRelative(randchain,Y_axis,rotY), rotateChainRelative(randchain,Z_axis,rotZ);
			translateChain(randchain, transX, transY, transZ);
			clashes = getNumHardClashes();
			if (clashes <= clashesStart){
					energyTest = true; revert = false;
			}
		}
		else{*/
			//--Backbone conformation trial--------------------------------------------------------
			if (_backbone) {backboneOrSidechain = rand() % 2;}
			if (randres > 0 && randres < resnum-2 && backboneOrSidechain == 0){
				// A backbone move displaces everything downstream of it, so the
				// held field is no longer the structure's and the chain ends here.
				if (anchored) {protReleaseDielectricCU(); anchored = false;}
				backboneTest = true;
				sPhi = getPhi(randchain,randres), sPsi = getPsi(randchain,randres);
				backboneAngles = getRandConformationFromBackboneType(sPhi, sPsi);
				setDihedral(randchain,randres,backboneAngles[0],0,0); setDihedral(randchain,randres,backboneAngles[1],1,0);
				energyTest = true; revert = false;
			}
			//--Sidechain conformation trial--------------------------------------------------------
			else{
				// Steepest descent over a batch of candidates rather than
				// first-improvement on a single one. The batch costs little more
				// than one candidate because the GPU is otherwise idle, and it
				// returns the winner's energy, so no separate evaluation follows.
				sidechainTest = true;
				currentSidechainConf = getSidechainDihedrals(randchain, randres);
				havePending = false; trialEnergy = 1E30;
				if (deltaMin)
				{
					if (!anchored || sinceAnchor >= ANCHOR_EVERY)
					{
						double fresh = 0.0;
						const bool wasAnchored = anchored;
						if (minDeltaAnchorCU(nbCurrent, fresh))
						{
							// The gap between the carried energy and a fresh
							// coupled one is the chain's accumulated error, and
							// it is the only thing here that can go wrong
							// silently, so it is measured rather than assumed.
							if (wasAnchored)
							{
								const double drift = fabs(fresh - pastEnergy);
								if (drift > worstDrift) {worstDrift = drift;}
								if (deltaDebug)
								{	cout << "protMinCU: re-anchor drift " << drift
									     << " over " << sinceAnchor << " moves" << endl; }
							}
							pastEnergy = fresh;
							anchored = true; sinceAnchor = 0;
						}
						else {protReleaseDielectricCU(); anchored = false;}
					}
					if (anchored)
					{
						double nbBest = 0.0, torBest = 0.0;
						trialEnergy = bestSidechainCandidateDeltaCU(randchain, randres,
						                                            SIDECHAIN_CANDIDATES,
						                                            newSidechainConf,
						                                            nbCurrent, nbBest, torBest);
						if (trialEnergy < 1E29) {pendingNb = nbBest; havePending = true;}
						else {protReleaseDielectricCU(); anchored = false;}
					}
				}
				if (!havePending)
				{
					if (anchored) {protReleaseDielectricCU(); anchored = false;}
					trialEnergy = bestSidechainCandidateCU(randchain, randres, SIDECHAIN_CANDIDATES, newSidechainConf);
				}
				if (trialEnergy < 1E29)
				{
					setSidechainDihedralAngles(randchain, randres, newSidechainConf);
					haveTrialEnergy = true; energyTest = true; revert = false;
				}
			}
		//}
		//--Energy-Test-------------------------------------------------------------------------
		if (energyTest){
			Energy = haveTrialEnergy ? trialEnergy : protEnergyCU();
			deltaEnergy = Energy - pastEnergy;
			boltzmannAcceptance = boltzmannEnergyCriteria(deltaEnergy);
			if (boltzmannAcceptance){
				pastEnergy = Energy;
				if (pastEnergy < bestEnergy)
				{	bestEnergy = pastEnergy;
					snapshotConformationCU(bestX, bestY, bestZ); }
				if (havePending)
				{
					if (minDeltaCommitCU(pendingNb, nbCurrent)) {sinceAnchor++;}
					else {protReleaseDielectricCU(); anchored = false;}
				}
				if (deltaEnergy < -KT){nobetter = 0;}
			}
			else{revert = true;}
		}
		//--Revert conformation-----------------------------------------------------------------
		if (revert){
			if(cofactorTest)
			{
				setAllCoords(randchain, randres, currentCoords);
			}
			if(backboneTest){
				setDihedral(randchain,randres,sPhi,0,0);
				setDihedral(randchain,randres,sPsi,1,0);
			}
			if(sidechainTest){
				setSidechainDihedralAngles(randchain, randres, currentSidechainConf);
			}
		}
		stop.observe(pastEnergy);
	} while (legacyStop ? (nobetter < plateau) : !stop.done());
	// A rejected move leaves the device at the entry geometry, so the chain
	// survives it untouched; only the exit has to put the coupled model back.
	if (anchored)
	{
		protReleaseDielectricCU();
		if (deltaDebug)
		{	cout << "protMinCU: worst re-anchor drift " << worstDrift << endl; }
	}
	// Hand back the best conformation seen, not the last one wandered into.
	if (bestEnergy < pastEnergy) {restoreConformationCU(bestX, bestY, bestZ);}
	return;
}

#endif

//**************CUDA related functions end***************************************

int protein::getNumAtoms()
{
	int N = 0;
	for(UInt i=0; i<itsChains.size(); i++)
	{N += itsChains[i]->getNumAtoms();}
	return N;
}

//**************Default non-Redundant optimized Energy Function***************************************
// protEnergy is now a thin alias for the CUDA evaluation.
//
// The CPU pair-loop that used to live here was not a second opinion on the
// same model, it was a stale one: it predated the ff14SB radii, the torsion
// term and the dielectric fix, and on 1crn it returned 4869.34 against the
// kernel's 510.98.  Keeping both meant every caller silently got whichever
// physics its author happened to reach for.  The name is kept so the project
// call sites read unchanged.
double protein::protEnergy()
{
	double e = protEnergyCU();
	E = e;
	return e;
}

double protein::protEnergy(UInt chainIndex) //Energy of chain alone
{
	return protEnergyCU(chainIndex);
}

void protein::updateDielectrics()
{
	updateDielectricsCU();
}

void protein::updateDielectrics(UInt chainIndex)
{
	updateDielectricsCU();
}

void protein::updateMovedDependence(UInt _EorC)
{
	for(UInt i=0; i<itsChains.size(); i++)
	{
		itsChains[i]->updateMovedDependence(_EorC);
		for(UInt j=i+1; j<itsChains.size(); j++)
		{
			itsChains[i]->updateMovedDependence(itsChains[j], _EorC);
		}
	}
}

void protein::setMoved(bool _moved, UInt _EorC)
{
	for(UInt i=0; i<itsChains.size(); i++)
	{
		itsChains[i]->setMoved(_moved,_EorC);
	}
}

double protein::protEnergy(UInt chainIndex, UInt resIndex)
{
	if (getMoved(chainIndex, resIndex, 0)){
		updateResidueEnergiesCU();
	}
	return itsChains[chainIndex]->getEnergy(resIndex);
}

bool protein::boltzmannEnergyCriteria(double _deltaEnergy) //calculate boltzmann probability of an energy to determine acceptance criteria
{
	// The draw happens first and unconditionally: the position of the RNG
	// stream must not depend on the sign of the move.  u is uniform on
	// (0, 1] in steps of 1e-6.
	const double u = ((rand() % 1000000) + 1) / 1000000.0;
	if (_deltaEnergy <= 0.0) {return true;}
	return exp(-_deltaEnergy / residue::getKT()) >= u;

	// Was: Entropy = 1000000/((rand() % 1000000)+1), compared against
	// pow(EU, +dE/KT).  The two sign inversions cancel and the tail is right,
	// but 1000000/(...) is integer division, so Entropy was floor(1e6/U) --
	// exactly 1 for half of all draws, and never less than 1.  Against a
	// strict <, that made the acceptance probability 1/(floor(e^(dE/KT))+1)
	// rather than e^(-dE/KT): a factor of two too cold at dE = 0, and wrong
	// for every move up to dE = KT*ln2, which is most of what a Metropolis
	// walk near a minimum actually proposes.  Downhill moves were unaffected,
	// so the bias was invisible in the energies and showed up only as a
	// search that was greedier than the temperature claimed.
}

double protein::boltzmannProbabilityToEnergy(double Pi, double Pj) //calculate boltzmann Energy from a probability (Pi) compared to a reference (Pj) probability to determine acceptance criteria
{
	double Energy = -residue::getKT()*log(Pi/Pj);
	return Energy;
}

// As protEnergy: the clash count now comes from the kernel, which shares the
// AMBER radius array with the vdW term.  The CPU counter used the older
// radii, so the two disagreed on which contacts were clashes.
UInt protein::getNumHardClashes()
{
	int c = getNumClashesCU();
	return c < 0 ? 0 : (UInt)c;
}

UInt protein::getNumHardClashes(UInt chainIndex, UInt resIndex)
{
	if (getMoved(chainIndex, resIndex,1)){
		updateResidueClashesCU();
	}
	return itsChains[chainIndex]->getClashes(resIndex);
}

void protein::updateResiduesPerTurnType()
{
	for(UInt i=0; i<itsChains.size(); i++)
	{
		itsChains[i]->updateResiduesPerTurnType();
	}
}

void protein::updateTotalNumResidues()
{
	UInt numResidues = 0;
	for(UInt i=0; i<itsChains.size(); i++)
	{
		numResidues += getNumResidues(i);
	}
	itsNumResidues = numResidues;
}

UInt protein::getNumResidues(UInt _chainIndex) const
{	if (_chainIndex < itsChains.size())
	{	return itsChains[_chainIndex]->getNumResidues();
	}
	else
	{	cout << "Error reparoted from protein::getNumResidues" << endl;
		cout << "chainIndex out of bounds" << endl;
	}
	return 0;
}

int protein::getIndexFromResNum(UInt _chainIndex, UInt _resnum)
{
	// is chain valid?
	if (_chainIndex < itsChains.size())
	{
		// first, take a guess and find direction to search
		int tempint = itsChains[_chainIndex]->mapResNumToChainPosition(_resnum);
		return tempint;
	}
	else
	{
		cout << "Invalid Chain Specifier: " << _chainIndex << endl;
	}
	return -1;
}

void protein::setChi(const UInt _chainIndex, const UInt _resIndex, const UInt _bpt, const UInt _chi, const double _angle)
{
	if (_chainIndex >= 0 && _chainIndex < itsChains.size())
	{
		itsChains[_chainIndex]->setChi(_resIndex, _bpt, _chi, _angle);
	}
	else
	{
		cout << "ERROR in protein::setChi(...)\n\tchain index is out of bounds ..." << endl;
		return;
	}
	    // make same change to symmetry linked chains
    UInt independentChainIndex = 0;
    for (UInt i = 0; i < itsIndependentChainsMap.size(); i++)
    {
        if (itsIndependentChainsMap[i] == _chainIndex) independentChainIndex = i;
    }
    UInt numSymmetryLinkedChains = itsChainLinkageMap[independentChainIndex].size();
    if (itsChainLinkageMap[independentChainIndex][0] != -1)
    {
        for (UInt i = 0; i < numSymmetryLinkedChains; i ++)
        {
            itsChains[itsChainLinkageMap[independentChainIndex][i]]->setChi(_resIndex, _bpt, _chi, _angle);
        }
    }

    return;
}

void protein::rotateChain(UInt _chain, const axis _axis, const double _theta)
{
	if (_chain >= 0 && _chain < itsChains.size())
	{
		itsChains[_chain]->rotate(_axis, _theta);
	}
	else
	{
		cout << "ERROR in protein::rotateChain(...)\n\tchain index is out of bounds ..." << endl;
		return;
	}
	return;
}

void protein::rotateChainRelative(UInt _chain, const axis _axis, const double _theta)
{
	if (_chain >= 0 && _chain < itsChains.size())
	{
		itsChains[_chain]->rotateRelative(_axis, _theta);
	}
	else
	{
		cout << "ERROR in protein::rotateChainRelative(...)\n\tchain index is out of bounds ..." << endl;
		return;
	}
	return;
}

void protein::translateChain(UInt _chain, const double _x, const double _y, const double _z)
{
	if (_chain >= 0 && _chain < itsChains.size())
	{
		itsChains[_chain]->translate( _x, _y, _z);
	}
	else
	{
		cout << "ERROR in protein::translateChain(...)\n\tchain index is out of bounds ..." << endl;
		return;
	}
	return;
}

void protein::translateChainR(UInt _chain, const double _x, const double _y, const double _z)
{
	if (_chain >= 0 && _chain < itsChains.size())
	{
		itsChains[_chain]->translateR( _x, _y, _z);
	}
	else
	{
		cout << "ERROR in protein::translateChain(...)\n\tchain index is out of bounds ..." << endl;
		return;
	}
	return;
}

void protein::setSidechainDihedralAngles(UInt _chainIndex, UInt _indexInChain, vector <vector <double> > Angles)
{
	if (_chainIndex >= 0 && _chainIndex < itsChains.size())
	{
		itsChains[_chainIndex]->setSidechainDihedralAngles(_indexInChain, Angles);
	}
	else
	{
		cout << "ERROR in protein::setSidechainDihedralAngles(...)\n\tchain index is out of bounds ..." << endl;
		return;
	}
	return;
}

void protein::setRelativeChi(const UInt _chainIndex, const UInt _resIndex, const UInt _bpt, const UInt _chi, const double _angle)
{
	if (_chainIndex >= 0 && _chainIndex < itsChains.size())
	{
		itsChains[_chainIndex]->setRelativeChi(_resIndex, _bpt, _chi, _angle);
	}
	else
	{
		cout << "ERROR in protein::setRelativeChi(...)\n\tchain index is out of bounds ..." << endl;
		return;
	}

	// make same change to symmetry linked chains
	UInt independentChainIndex = 0;
	for (UInt i = 0; i < itsIndependentChainsMap.size(); i++)
    {
        if (itsIndependentChainsMap[i] == _chainIndex) independentChainIndex = i;
    }
	UInt numSymmetryLinkedChains = itsChainLinkageMap[independentChainIndex].size();
    if (itsChainLinkageMap[independentChainIndex][0] != -1)
    {
        for (UInt i = 0; i < numSymmetryLinkedChains; i ++)
        {
            itsChains[itsChainLinkageMap[independentChainIndex][i]]->setRelativeChi(_resIndex, _bpt, _chi, _angle);
        }
    }

    return;
}

void protein::setRotamerNotAllowed(const UInt _chainIndex, const UInt _residueIndex, const UInt _resType, const UInt _bpt, const UInt _rotamer )
{
    if (_chainIndex >= 0 && _chainIndex < itsChains.size())
    {
        itsChains[_chainIndex]->setRotamerNotAllowed(_residueIndex, _resType, _bpt, _rotamer);
    }
    else
    {
        cout << "Error from protein::setRotamerNotAllowed" << endl;
        cout << " chain index out of bounds : " << _chainIndex << endl;
		return;
    }

	// symmetry mapping code:
	UInt independentChainIndex = 0;
    for (UInt i = 0; i < itsIndependentChainsMap.size(); i++)
    {
        if (itsIndependentChainsMap[i] == _chainIndex) independentChainIndex = i;
    }

    UInt numSymmetryLinkedChains = itsChainLinkageMap[independentChainIndex].size();
    if (itsChainLinkageMap[independentChainIndex][0] != -1)
    {
        for (UInt i = 0; i < numSymmetryLinkedChains; i ++)
        {
            itsChains[itsChainLinkageMap[independentChainIndex][i]]->setRotamerNotAllowed( _residueIndex, _resType, _bpt, _rotamer );
        }
    }

    return;
}

void protein::setCanonicalHelixRotamersOnly(const UInt _chainIndex)
{
	UIntVec activeResidues = itsChains[_chainIndex]->getActiveResidues();
	if (activeResidues.size() > 0)
	{
		for (UInt i = 0; i < activeResidues.size(); i++)
		{
			setCanonicalHelixRotamersOnly(_chainIndex, activeResidues[i]);
		}
	}
	else
		cout << "No active positions on chain " << _chainIndex << endl;

	return;
}

void protein::setCanonicalHelixRotamersOnly(const UInt _chainIndex, const UInt _resIndex)
{
	UIntVec activeResidues = itsChains[_chainIndex]->getActiveResidues();
	bool isActive = false;
	for (UInt i = 0; i < activeResidues.size(); i++) if (activeResidues[i] == _resIndex) isActive = true;
	if (isActive)
	{
		UIntVec allowedResidues = getResAllowed(_chainIndex, _resIndex);
		for (UInt i = 0; i < allowedResidues.size(); i++)
		{
			setCanonicalHelixRotamersOnly( _chainIndex, _resIndex, allowedResidues[i]);
		}
	}
	return;
}

void protein::setCanonicalHelixRotamersOnly( const UInt _chainIndex, const UInt _resIndex, const UInt _resType )
{
	UIntVec allowedResidues = getResAllowed(_chainIndex, _resIndex);
	bool resAllowed = false;
	for (UInt i = 0; i < allowedResidues.size(); i ++)
	{
			if (_resType == allowedResidues[i]) resAllowed = true;
	}
	if (resAllowed)
	{
		if (messagesActive) cout << "Setting canonical helix rotamers ONLY for chain " << _chainIndex << " pos " << _resIndex << " for " << residue::getDataBaseItem(_resType) << endl;
		for (UInt bpt = 0; bpt < residue::getNumBpt(_resType); bpt++)
		{
			UIntVec allowedRotamers;
			allowedRotamers  = itsChains[_chainIndex]->getAllowedRotamers(_resIndex,  _resType, bpt);
			for (UInt j = 0; j < allowedRotamers.size(); j ++)
			{
				if (isValidHelixRotamer(_resType, bpt, allowedRotamers[j]) )
				{
					if (messagesActive) cout << " keeping rotamer - " << allowedRotamers[j] << " ";
				}
				else
				{
					setRotamerNotAllowed ( _chainIndex, _resIndex, _resType, bpt, allowedRotamers[j]);
				}
			}
			if (messagesActive) cout << endl;
		}
	}
	else
	{
		cout << "ERROR in protein::setCanonicalHelixRotamersOnly." << endl;
		cout << " " << residue::getDataBaseItem(_resType) << " not allowed at position " << _chainIndex << " " << _resIndex << endl;
		cout << " rotamer modification aborted for this residue type." << endl;
	}
	return;
}

bool protein::isValidHelixRotamer(UInt _resType, UInt _bpt, UInt _rotamer)
{
	if (_resType == 0) return true;
	if (_resType == 1) { if (_rotamer == 37 || _rotamer == 40 || _rotamer == 43) return true;}
	if (_resType == 2) { if (_rotamer == 7 || _rotamer == 8) return true;}
	if (_resType == 3) { if (_rotamer == 7 || _rotamer == 8) return true;}
	if (_resType == 4) { if (_rotamer == 2) return true;}
	if (_resType == 5) { if (_rotamer == 9 || _rotamer == 21 || _rotamer == 22 || _rotamer == 23 || _rotamer == 28) return true;}
	if (_resType == 6) { if (_rotamer == 13 || _rotamer == 22) return true;}
	if (_resType == 7) return true;
	if (_resType == 8) { if (_rotamer == 3 || _rotamer == 6) return true;}
	if (_resType == 9) { if (_rotamer == 7 || _rotamer == 8) return true;}
	if (_resType == 10){ if (_rotamer == 3 || _rotamer == 7) return true;}
	if (_resType == 11){ if (_rotamer == 40 || _rotamer == 67) return true;}
	if (_resType == 12){ if (_rotamer == 21 || _rotamer == 23) return true;}
	if (_resType == 13){ if (_rotamer == 3 || _rotamer == 6) return true;}
	if (_resType == 14){ if (_rotamer == 1) return true;}
	if (_resType == 15){ if (_rotamer == 0 || _rotamer == 2) return true;}
	if (_resType == 16){ if (_rotamer == 0 || _rotamer == 2) return true;}
	if (_resType == 17){ if (_rotamer == 3 || _rotamer == 5 || _rotamer == 8) return true;}
	if (_resType == 18){ if (_rotamer == 3 || _rotamer == 6) return true;}
	if (_resType == 19){ if (_rotamer == 1 ) return true;}

	return false;
}

void protein::setRotamerWBC(const UInt _chainIndex, const UInt _resIndex, const UInt _bpt, const UInt _rotamer)
{
    //cout << "Setting rotamer res:"<<_resIndex<<" bpt:"<<_bpt<< " rotamer:"<<_rotamer<< endl;
    if(_chainIndex < itsChains.size())
    {   itsChains[_chainIndex]->setRotamerWithoutBuffering(_resIndex,_bpt,_rotamer);
        // find chain in independent chain list
        UInt indChainIndex = 0;
        for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
        {   if (itsIndependentChainsMap[i] == _chainIndex)
            {   indChainIndex = i;
            }
        }

        UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
        if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
        {       for (UInt i=0; i<numSymLinkedChains;i++)
            {
                itsChains[itsChainLinkageMap[indChainIndex][i]]->setRotamerWithoutBuffering(_resIndex,_bpt,_rotamer);
            }
        }
    }
}

void protein::setRotamer(const UInt _chainIndex, const UInt _resIndex, const UInt _bpt, const UInt _rotamer)
{
    itsChains[_chainIndex]->setRotamerWithoutBuffering(_resIndex,_bpt,_rotamer);
    // find chain in independent chain list
    UInt indChainIndex = 0;
    for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
    {   if (itsIndependentChainsMap[i] == _chainIndex)
        {   indChainIndex = i;
        }
    }

    UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
    if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
    {       for (UInt i=0; i<numSymLinkedChains;i++)
        {
            itsChains[itsChainLinkageMap[indChainIndex][i]]->setRotamerWithoutBuffering(_resIndex,_bpt,_rotamer);
        }
    }
}

void protein::setPolarHRotamer(const UInt _chainIndex, const UInt _resIndex, const UInt _rotamer)
{
    itsChains[_chainIndex]->setPolarHRotamerWithoutBuffering(_resIndex,_rotamer);
    // find chain in independent chain list
    UInt indChainIndex = 0;
    for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
    {   if (itsIndependentChainsMap[i] == _chainIndex)
        {   indChainIndex = i;
        }
    }

    UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
    if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
    {       for (UInt i=0; i<numSymLinkedChains;i++)
        {
            itsChains[itsChainLinkageMap[indChainIndex][i]]->setPolarHRotamerWithoutBuffering(_resIndex,_rotamer);
        }
    }
}

bool protein::performRandomRotamerChange(ran& _ran)
{   int chainToModify = chooseTargetChain(_ran);
    cout << "CHAIN " << chainToModify << " ";
    cout << "ROTC ";
    if (chainToModify >= 0)
    {
        vector<chainModBuffer> theBuffers;
        theBuffers = itsChains[chainToModify]->performRandomRotamerChange(_ran);
        if ( theBuffers[0].containsData() )
        {
            itsLastModifiedChain = chainToModify;
            itsLastModificationMethod = 1;
            // find chain in independent chain list
            UInt indChainIndex = 0;
            for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
            {   if (int(itsIndependentChainsMap[i]) == chainToModify)
                {   indChainIndex = i;
                }
            }

            UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
            if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
            {       for (UInt i=0; i<numSymLinkedChains;i++)
                {
                    itsChains[itsChainLinkageMap[indChainIndex][i]]->repeatModification(theBuffers[1]);
                }
            }
            return true;
        }
        else
        {
            //cout << "Protein level has detected abort..." << endl;
        }
    }
    else
    {   cout << "Error from protein::performRandomRotamerChange" << endl;
        cout << "Protein contains no chains" << endl;
    }
    resetAllBuffers();
    return false;
}

bool protein::performRandomRotamerChange(ran& _ran, vector <int> _position)
{   int chainToModify = _position[1];
    cout << "CHAIN " << chainToModify << " ";
    cout << "ROTC ";
    if (chainToModify >= 0)
    {
        vector<chainModBuffer> theBuffers;
        theBuffers = itsChains[chainToModify]->performRandomRotamerChange(_ran, _position);
        if ( theBuffers[0].containsData() )
        {
            itsLastModifiedChain = chainToModify;
            itsLastModificationMethod = 1;
            // find chain in independent chain list
            UInt indChainIndex = 0;
            for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
            {   if (int(itsIndependentChainsMap[i]) == chainToModify)
                {   indChainIndex = i;
                }
            }

            UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
            if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
            {       for (UInt i=0; i<numSymLinkedChains;i++)
                {
                    itsChains[itsChainLinkageMap[indChainIndex][i]]->repeatModification(theBuffers[1]);
                }
            }
            return true;
        }
        else
        {
            cout << "Protein level has detected abort..." << endl;
        }
    }
    else
    {   cout << "Error from protein::performRandomRotamerChange" << endl;
        cout << "Protein contains no chains" << endl;
    }
    resetAllBuffers();
    return false;
}

bool protein::performRandomRotamerRotation(ran& _ran)
{   int chainToModify = chooseTargetChain(_ran);
    cout << "CHAIN " << chainToModify << " ";
    cout << "RR ";
    if (chainToModify >= 0)
    {
        vector<chainModBuffer> theBuffers;
        theBuffers = itsChains[chainToModify]->performRandomRotamerRotation(_ran);
        if ( theBuffers[0].containsData() )
        {   itsLastModifiedChain = chainToModify;
            itsLastModificationMethod = 2;
            // find chain in independent chain list
            UInt indChainIndex = 0;
            for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
            {   if (int(itsIndependentChainsMap[i]) == chainToModify)
                {   indChainIndex = i;
                }
            }

            UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
            if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
            {       for (UInt i=0; i<numSymLinkedChains;i++)
                {
                    itsChains[itsChainLinkageMap[indChainIndex][i]]->repeatModification(theBuffers[1]);
                }
            }
            return true;
        }
        else
        {
            //cout << "Protein level has detected abort..." << endl;
        }
    }
    else
    {   cout << "Error from protein::performRandomRotamerRotation" << endl;
        cout << "Protein contains no chains" << endl;
    }
    resetAllBuffers();
    return false;
}

bool protein::performRandomRotamerRotation(ran& _ran, vector <int> _position)
{   int chainToModify = _position[1];
    cout << "CHAIN " << chainToModify << " ";
    cout << "RR ";
    if (chainToModify >= 0)
    {
        vector<chainModBuffer> theBuffers;
        theBuffers = itsChains[chainToModify]->performRandomRotamerRotation(_ran, _position);
        if ( theBuffers[0].containsData() )
        {   itsLastModifiedChain = chainToModify;
            itsLastModificationMethod = 2;
            // find chain in independent chain list
            UInt indChainIndex = 0;
            for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
            {   if (int(itsIndependentChainsMap[i]) == chainToModify)
                {   indChainIndex = i;
                }
            }

            UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
            if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
            {       for (UInt i=0; i<numSymLinkedChains;i++)
                {
                    itsChains[itsChainLinkageMap[indChainIndex][i]]->repeatModification(theBuffers[1]);
                }
            }
            return true;
        }
        else
        {
            //cout << "Protein level has detected abort..." << endl;
        }
    }
    else
    {   cout << "Error from protein::performRandomRotamerRotation" << endl;
        cout << "Protein contains no chains" << endl;
    }
    resetAllBuffers();
    return false;
}

void protein::commitLastRotamerChange()
{   if(itsLastModifiedChain >= 0 && itsLastModificationMethod == 1)
    {   itsChains[itsLastModifiedChain]->commitLastRotamerChange();
        // find chain in independent chain list
        UInt indChainIndex = 0;
        for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
        {   if (int(itsIndependentChainsMap[i]) == itsLastModifiedChain)
            {   indChainIndex = i;
            }
        }

        UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
        if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
        {       for (UInt i=0; i<numSymLinkedChains;i++)
            {
                itsChains[itsChainLinkageMap[indChainIndex][i]]->commitLastRotamerChange();
            }
        }
        resetAllBuffers();
    }
    else
    {   cout << "Error reported by protein::commitLastRotamerChange()";
        cout << endl << "No last modified chain" << endl;
    }
    cout << "ACCEPTED " << endl;
}

void protein::saveCurrentState()
{
	for (UInt i = 0; i < itsChains.size(); i ++)
	{
		itsChains[i]->saveCurrentState();
	}
	return;
}

void protein::undoState()
{
	for (UInt i = 0; i < itsChains.size(); i ++)
	{
		itsChains[i]->undoState();
	}
	saveCurrentState();
	return;
}

void protein::commitState()
{
	for (UInt i = 0; i < itsChains.size(); i ++)
	{
		itsChains[i]->commitState();
	}
	return;
}

void protein::undoLastRotamerChange()
{   if (itsLastModifiedChain >=0 && itsLastModificationMethod == 1)
    {   itsChains[itsLastModifiedChain]->undoLastRotamerChange();
        // find chain in independent chain list
        UInt indChainIndex = 0;
        for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
        {   if (int(itsIndependentChainsMap[i]) == itsLastModifiedChain)
            {   indChainIndex = i;
            }
        }

        UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
        if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
        {       for (UInt i=0; i<numSymLinkedChains;i++)
            {
                itsChains[itsChainLinkageMap[indChainIndex][i]]->undoLastRotamerChange();
            }
        }
        resetAllBuffers();
    }
    else
    {   cout << "Error reported by protein::undoLastRotamerChange()";
        cout << endl << "No last modified chain" << endl;
    }
    cout << "REJECTED " << endl;
}

void protein::commitLastRotamerRotation()
{   if(itsLastModifiedChain >= 0 && itsLastModificationMethod == 2)
    {   itsChains[itsLastModifiedChain]->commitLastRotamerRotation();
        // find chain in independent chain list
        UInt indChainIndex = 0;
        for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
        {   if (int(itsIndependentChainsMap[i]) == itsLastModifiedChain)
            {   indChainIndex = i;
            }
        }

        UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
        if (/* numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
        {       for (UInt i=0; i<numSymLinkedChains;i++)
            {
                itsChains[itsChainLinkageMap[indChainIndex][i]]->commitLastRotamerRotation();
            }
        }
        resetAllBuffers();
    }
    else
    {   cout << "Error reported by protein::commitLastRotamerRotation()";
        cout << endl << "No last modified chain" << endl;
    }
    cout << "ACCEPTED " << endl;
}

void protein::undoLastRotamerRotation()
{   if (itsLastModifiedChain >=0 && itsLastModificationMethod == 2)
    {   itsChains[itsLastModifiedChain]->undoLastRotamerRotation();
        // find chain in independent chain list
        UInt indChainIndex = 0;
        for (UInt i=0; i<itsIndependentChainsMap.size(); i++)
        {   if (int(itsIndependentChainsMap[i]) == itsLastModifiedChain)
            {   indChainIndex = i;
            }
        }

        UInt numSymLinkedChains = itsChainLinkageMap[indChainIndex].size();
        if ( /*numSymLinkedChains != 1 &&*/ itsChainLinkageMap[indChainIndex][0] != -1)
        {       for (UInt i=0; i<numSymLinkedChains;i++)
            {
                itsChains[itsChainLinkageMap[indChainIndex][i]]->undoLastRotamerRotation();
            }
        }
        resetAllBuffers();
    }
    else
    {   cout << "Error reported by protein::undoLastRotamerRotation()";
        cout << endl << "No last modified chain" << endl;
    }
    cout << "REJECTED " << endl;
}

void protein::listAllowedRotamers(UInt _chain, UInt _resIndex)
{   if (_chain < itsChains.size())
    {   itsChains[_chain]->listAllowedRotamers(_resIndex);
    }
}

double protein::getRMSD(protein* _other)
{
    vector<dblVec> coord1;
    vector<dblVec> coord2;
    atomIterator theIter1(this);
    atomIterator theIter2(_other);
    atom* pAtom;
    bool first = true;
    
    // Load backbone atoms into vector for fit and alignment
    for (;!(theIter1.last());theIter1++)
    {
       pAtom = theIter1.getAtomPointer(); 
       if(pAtom->getName() == "N" || pAtom->getName() == "CA" || pAtom->getName() == "C" || pAtom->getName() == "O"){
            coord1.push_back(pAtom->getCoords());
       }
    }
    for (;!(theIter2.last());theIter2++)
    {
       pAtom = theIter2.getAtomPointer(); 
       if(pAtom->getName() == "N" || pAtom->getName() == "CA" || pAtom->getName() == "C" || pAtom->getName() == "O"){
            coord2.push_back(pAtom->getCoords());
       }
    }
    int diff = 0;
    if(coord1.size() != coord2.size()){
		if (coord2.size() < coord1.size()){ diff = coord1.size()-coord2.size(); first = true;}
		else{diff = coord2.size()-coord1.size(); first = false;}
    }
    int maxsize;
    if (first){maxsize = coord2.size();}else{maxsize = coord1.size();}
	double rotmat[9]; double centroid1[3]; double centroid2[3]; double rmsd = 0; int ierr = 0;
	int list1[maxsize]; int list2[maxsize]; int trials = 1; double bestRMSD= 1E10;
	double newCoord1[maxsize*3]; double newCoord2[maxsize*3]; double newCoord3[maxsize*3]; double rmsdat[maxsize];
	
	if (diff != 0){trials = diff;}
	for (int h = 0; h < trials; h++)
	{
		for (int i=0; i<maxsize; i++)
		{	
			for (int j=0; j<3; j++)
			{
				if(first){
					newCoord1[ (i*3) + j] = coord1[i+h][j];
					newCoord2[ (i*3) + j] = coord2[i+h][j];
				}
				else{
					newCoord1[ (i*3) + j] = coord2[i+h][j];
					newCoord2[ (i*3) + j] = coord1[i+h][j];
				}
			}
			list1[i] = i+1;
			list2[i] = i+1;
		}
		
		// Calculate best fit of backbone atoms, rotation matrix and rmsd using fortran algorithm based on Machlachlan
		bestfit_(newCoord1, &maxsize, newCoord2, &maxsize, &maxsize, newCoord3, list1, list2, &rmsd, &ierr, rotmat, centroid1, centroid2, rmsdat);
		if (rmsd < bestRMSD){
			bestRMSD = rmsd;
		}
	}
	return bestRMSD;
}

void protein::alignToAxis(const axis _axis)
{
    vector<dblVec> coord1;
    vector<dblVec> coord2;
    dblVec coords(3);
    atomIterator theIter(this);
    atom* pAtom;
    
    // Load backbone atoms into vector for alignment to axis
    for (;!(theIter.last());theIter++)
    {
       pAtom = theIter.getAtomPointer(); 
       if (pAtom->getName() == "N" || pAtom->getName() == "CA" || pAtom->getName() == "C" || pAtom->getName() == "O"){
            coord2.push_back(pAtom->getCoords());
       }
    }
    
    // Find longest axis of protein
    double maxdist = -1E10, dist;
	for (UInt i = 0; i < coord2.size(); i++)
	{
		for (UInt j = i+1; j < coord2.size(); j++)
		{
			dist = CMath::distance(coord2[i], coord2[j]);
			if (dist > maxdist){maxdist = dist;}
		}
	}
	
	// Create set of points in space equally spaced of the size of the backbone on the axis
	double increment = maxdist/coord2.size();
	double coord = (maxdist/2)*-1;
	for (UInt i = 0; i < coord2.size(); i++)
	{
		if (_axis == X_axis){
			coords[0] = coord; coords[1] = 0.0; coords[2] = 0.0;
			coord1.push_back(coords);
		}
		if (_axis == Y_axis){
			coords[0] = 0.0; coords[1] = coord; coords[2] = 0.0;
			coord1.push_back(coords);
		}
		if (_axis == Z_axis){
			coords[0] = 0.0; coords[1] = 0.0; coords[2] = coord;
			coord1.push_back(coords);
		}
		coord += increment;
	}
	
    int maxsize = coord2.size();
	double rotmat[9]; double centroid1[3]; double centroid2[3]; double rmsd = 0; int ierr = 0;
	int list1[maxsize]; int list2[maxsize];
	double newCoord1[maxsize*3]; double newCoord2[maxsize*3]; double newCoord3[maxsize*3]; double rmsdat[maxsize];
	for (int i=0; i<maxsize; i++)
	{	
		for (int j=0; j<3; j++)
		{
			newCoord1[ (i*3) + j] = coord1[i][j];
			newCoord2[ (i*3) + j] = coord2[i][j];
		}
		list1[i] = i+1;
		list2[i] = i+1;
	}
	bestfit_(newCoord1, &maxsize, newCoord2, &maxsize, &maxsize, newCoord3, list1, list2, &rmsd, &ierr, rotmat, centroid1, centroid2, rmsdat);

	// Load rotation vector into rotation matrix
	dblMat rotMat(3,3,3);
    for (UInt i=0; i<3; i++)
    {	for (UInt j=0; j<3; j++)
		{
			rotMat[i][j] = rotmat[(j*3) + i];
		}
    }
    
    // Translate protein centroid to zero, rotate and translate centroid to final axis
	for (UInt i = 0; i < getNumChains(); i++)
	{
		translateChain(i, -centroid2[0], -centroid2[1], -centroid2[2]);
		transform(i,rotMat);
		translateChain(i,centroid1[0], centroid1[1], centroid1[2]);
	}
}

void protein::translate(const UInt _index, const dblVec& _dblVec)
{
	if(_index < itsChains.size())
    {
		itsChains[_index]->translate(_dblVec);
    }
}

void protein::translate(const dblVec& _dblVec)
{
	// translate every chain
	for (UInt i = 0; i < itsChains.size(); i ++)
	{
		itsChains[i]->translate(_dblVec);
	}
}

void protein::eulerRotate(UInt _chain, const double _phi, const double _theta, const double _psi)
{
	// calculate rotation matrix
	double a11 = cos(_psi)*cos(_phi) - cos(_theta)*sin(_phi)*sin(_psi);
	double a12 = cos(_psi)*sin(_phi) + cos(_theta)*cos(_phi)*sin(_psi);
	double a13 = sin(_psi)*sin(_theta);
	double a21 = -1*sin(_psi)*cos(_phi) - cos(_theta)*sin(_phi)*cos(_psi);
	double a22 = -1*sin(_psi)*sin(_phi) + cos(_theta)*cos(_phi)*cos(_psi);
	double a23 = cos(_psi)*sin(_theta);
	double a31 = sin(_theta)*sin(_phi);
	double a32 = -1*sin(_theta)*cos(_phi);
	double a33 = cos(_theta);

	dblMat rotMatrix(3,3,0.0);
	rotMatrix[0][0] = a11;
	rotMatrix[1][0] = a12;
	rotMatrix[2][0] = a13;
	rotMatrix[0][1] = a21;
	rotMatrix[1][1] = a22;
	rotMatrix[2][1] = a23;
	rotMatrix[0][2] = a31;
	rotMatrix[1][2] = a32;
	rotMatrix[2][2] = a33;

	transform(_chain,rotMatrix);
	return;
}

void protein::eulerRotate(const double _phi, const double _theta, const double _psi)
{
	// calculate rotation matrix
	double a11 = cos(_psi)*cos(_phi) - cos(_theta)*sin(_phi)*sin(_psi);
	double a12 = cos(_psi)*sin(_phi) + cos(_theta)*cos(_phi)*sin(_psi);
	double a13 = sin(_psi)*sin(_theta);
	double a21 = -1*sin(_psi)*cos(_phi) - cos(_theta)*sin(_phi)*cos(_psi);
	double a22 = -1*sin(_psi)*sin(_phi) + cos(_theta)*cos(_phi)*cos(_psi);
	double a23 = cos(_psi)*sin(_theta);
	double a31 = sin(_theta)*sin(_phi);
	double a32 = -1*sin(_theta)*cos(_phi);
	double a33 = cos(_theta);

	dblMat rotMatrix(3,3,0.0);
	rotMatrix[0][0] = a11;
	rotMatrix[1][0] = a12;
	rotMatrix[2][0] = a13;
	rotMatrix[0][1] = a21;
	rotMatrix[1][1] = a22;
	rotMatrix[2][1] = a23;
	rotMatrix[0][2] = a31;
	rotMatrix[1][2] = a32;
	rotMatrix[2][2] = a33;

	for (UInt i = 0; i < itsChains.size(); i ++)
	{
		transform(i,rotMatrix);
	}
	return;
}

void protein::undoEulerRotate(UInt _chain, const double _phi, const double _theta, const double _psi)
{
	// calculate inverse rotation matrix (transpose of rotation matrix)
	double a11 = cos(_psi)*cos(_phi) - cos(_theta)*sin(_phi)*sin(_psi);
	double a12 = -1*sin(_psi)*cos(_phi) - cos(_theta)*sin(_phi)*cos(_psi);
	double a13 = sin(_theta)*sin(_phi);
	double a21 = cos(_psi)*sin(_phi) + cos(_theta)*cos(_phi)*sin(_psi);
	double a22 = -1*sin(_psi)*sin(_phi) + cos(_theta)*cos(_phi)*cos(_psi);
	double a23 = -1*sin(_theta)*cos(_phi);
	double a31 = sin(_theta)*sin(_psi);
	double a32 = sin(_theta)*cos(_psi);
	double a33 = cos(_theta);

	dblMat rotMatrix(3,3,0.0);
	rotMatrix[0][0] = a11;
	rotMatrix[1][0] = a12;
	rotMatrix[2][0] = a13;
	rotMatrix[0][1] = a21;
	rotMatrix[1][1] = a22;
	rotMatrix[2][1] = a23;
	rotMatrix[0][2] = a31;
	rotMatrix[1][2] = a32;
	rotMatrix[2][2] = a33;

	transform(_chain,rotMatrix);
	return;
}

void protein::undoEulerRotate(const double _phi, const double _theta, const double _psi)
{
	// calculate inverse rotation matrix (transpose of rotation matrix)
	double a11 = cos(_psi)*cos(_phi) - cos(_theta)*sin(_phi)*sin(_psi);
	double a12 = -1*sin(_psi)*cos(_phi) - cos(_theta)*sin(_phi)*cos(_psi);
	double a13 = sin(_theta)*sin(_phi);
	double a21 = cos(_psi)*sin(_phi) + cos(_theta)*cos(_phi)*sin(_psi);
	double a22 = -1*sin(_psi)*sin(_phi) + cos(_theta)*cos(_phi)*cos(_psi);
	double a23 = -1*sin(_theta)*cos(_phi);
	double a31 = sin(_theta)*sin(_psi);
	double a32 = sin(_theta)*cos(_psi);
	double a33 = cos(_theta);

	dblMat rotMatrix(3,3,0.0);
	rotMatrix[0][0] = a11;
	rotMatrix[1][0] = a12;
	rotMatrix[2][0] = a13;
	rotMatrix[0][1] = a21;
	rotMatrix[1][1] = a22;
	rotMatrix[2][1] = a23;
	rotMatrix[0][2] = a31;
	rotMatrix[1][2] = a32;
	rotMatrix[2][2] = a33;

	for (UInt i = 0; i < itsChains.size(); i ++)
	{
		transform(i,rotMatrix);
	}
	return;
}

void protein::translate(const UInt _index, const double _x,const double _y,const double _z)
{	dblVec _vec;
	_vec.newsize(3);
	_vec[0] = _x;
	_vec[1] = _y;
	_vec[2] = _z;
	translate(_index, _vec);
}

void protein::translate(const double _x, const double _y, const double _z)
{
	dblVec _vec;
	_vec.newsize(3);
	_vec[0] = _x;
	_vec[1] = _y;
	_vec[2] = _z;
	for (UInt i=0; i< itsChains.size(); i++)
	{	translate(i,_vec);
	}
}

void protein::transform(const UInt _index, const dblMat& _dblMat)
{   if(_index < itsChains.size())
    {   itsChains[_index]->transform(_dblMat);
    }
}

void protein::rotate(const UInt _index, const axis _axis, const double _theta)
{	point origin;
	// The default is to set this point to the origin
	//cout << "ROTATING CHAIN " << _index << endl;
	origin.setCoords(0.0,0.0,0.0);
	dblVec vec = dblVec(3);
	for (int i = 0; i<vec.dim(); i++)
	{   vec[i] = 0.0;
	}
	if (_axis == X_axis)
	{   vec[0] = 1.0;
	}
	else if (_axis == Y_axis)
	{   vec[1] = 1.0;
	}
	else if (_axis == Z_axis)
	{   vec[2]  = 1.0;
	}
	rotate(_index, origin , vec, _theta);
}

void protein::rotate(const axis _axis, const double _theta)
{	point origin;
	// The default is to set this point to the origin
	origin.setCoords(0.0,0.0,0.0);
	dblVec vec = dblVec(3);
	for (int i = 0; i<vec.dim(); i++)
	{   vec[i] = 0.0;
	}
	if (_axis == X_axis)
	{   vec[0] = 1.0;
	}
	else if (_axis == Y_axis)
	{   vec[1] = 1.0;
	}
	else if (_axis == Z_axis)
	{   vec[2]  = 1.0;
	}
	for (UInt i=0; i<itsChains.size(); i++)
	{
		rotate(i, origin , vec, _theta);
	}
}

void protein::rotate(const UInt _index, const point& _point,const dblVec& _R_axis, const double _theta)
{   if( _index < itsChains.size())
    {   itsChains[_index]->rotate(_point, _R_axis, _theta);
    }
}


void protein::coilcoil(const double _pitch)
{
    if (_pitch == 0.0)
    {
        cout << "ERROR in protein::coilcoil(...)  pitch cannot be zero!!" << endl;
        cout << "\t protein unchanged." << endl;
        return;
    }
    for (UInt i = 0; i < itsChains.size(); i ++)
    {
        itsChains[i]->coilcoil(_pitch);
    }
    return;
}

void protein::coilcoil(UInt _chain, double _pitch)
{
	if (_pitch == 0.0)
	{
		cout << "ERROR in protein::coilcoil(...)  pitch cannot be zero!!" << endl;
		cout << "\t protein unchanged." << endl;
		return;
	}
	if (_chain >=0 && _chain < itsChains.size())
	{
		itsChains[_chain]->coilcoil(_pitch);
	}
	return;
}

// surface area and solvation energy functions.
void protein::initializeSpherePoints()
{
	for (UInt i = 0; i < itsChains.size(); i ++)
	{
		itsChains[i]->initializeSpherePoints();
	}
	return;
}

void protein::initializeSpherePoints(UInt _chain)
{
	if (_chain >=0 && _chain < itsChains.size() )
	{
		itsChains[_chain]->initializeSpherePoints();
	}
	else
	{
		cout << "ERROR in initializeSpherePoints ... chain index out of range" << endl;
	}
	return;
}

void protein::initializeSpherePoints(UInt _chain, UInt _residue)
{
	if (_chain >= 0 && _chain < itsChains.size())
	{
		itsChains[_chain]->initializeSpherePoints(_residue);
	}
	else
	{
		cout << "ERROR in initializeSpherePoints ... chain index out of range" << endl;
	}
	return;
}

double protein::tabulateSurfaceArea()
{
	double surfaceArea = 0.0;
	for (UInt i = 0; i < itsChains.size(); i++)
	{
		surfaceArea += itsChains[i]->tabulateSurfaceArea();
	}

	return surfaceArea;
}

double protein::tabulateSurfaceArea(UInt _chain)
{
	double surfaceArea = 0.0;
	if (_chain >= 0 && _chain < itsChains.size() )
	{
		surfaceArea = itsChains[_chain]->tabulateSurfaceArea();
	}
	else
	{
		cout << "ERROR in tabulateSurfaceArea ... chain index out of range." << endl;
	}

	return surfaceArea;
}

double protein::tabulateSurfaceArea(UInt _chain, UInt _residue)
{
	double surfaceArea = 0.0;
	if (_chain >=0 && _chain < itsChains.size() )
	{
		surfaceArea = itsChains[_chain]->tabulateSurfaceArea(_residue);
	}
	else
	{
		cout << "ERROR in tabulateSurfaceArea ... chain index out of range." << endl;
	}

	return surfaceArea;
}

double protein::tabulateSurfaceArea(UInt _chainIndex, UInt _residueIndex, UInt _atomIndex)
{
	double surfaceArea = 0.0;
	if (_chainIndex >=0 && _chainIndex < itsChains.size() )
	{
		surfaceArea = itsChains[_chainIndex]->tabulateSurfaceArea(_residueIndex, _atomIndex);
	}
	else
	{
		cout << "ERROR in tabulateSurfaceArea ... chain index out of range." << endl;
	}

	return surfaceArea;
}

double protein::getItsSolvationParam()
{
	return itsSolvationParam;
}

void protein::setItsSolvationParam(UInt _param)
{
	itsSolvationParam = _param;
}

void protein::removeSpherePoints()
{
	for (UInt i = 0; i < itsChains.size(); i++ )
	{
		itsChains[i]->removeIntraChainSpherePoints();
		for (UInt j = 0; j < itsChains.size(); j ++)
		{
			if ( i != j ) itsChains[i]->removeInterChainSpherePoints(itsChains[j]);
		}
	}
	return;
}

void protein::removeSpherePoints(UInt _chain)
{
	if  (_chain >= 0 && _chain < itsChains.size() )
	{
		itsChains[_chain]->removeIntraChainSpherePoints();
		for (UInt j = 0; j < itsChains.size(); j ++)
		{
			if (_chain != j) itsChains[_chain]->removeInterChainSpherePoints(itsChains[j]);
		}
	}
	else
	{
		cout << "ERROR in removeSpherePoints ... chain index out of range." << endl;
	}
	return;
}

void protein::removeSpherePoints(UInt _chain, UInt _residue)
{
	if (_chain >=0 && _chain < itsChains.size())
	{
		itsChains[_chain]->removeIntraChainSpherePoints(_residue);
		for (UInt j =0; j < itsChains.size(); j ++)
		{
			if (_chain != j) itsChains[_chain]->removeInterChainSpherePoints(_residue, itsChains[j]);
		}
	}
	else
	{
		cout << "ERROR in removeSpherePoints ... chain index out of range." << endl;
	}
	return;
}

residue* protein::superimposeGLY(const UInt _chain, const UInt _residue)
{
	// get coordinates of a glycine complete with hydrogens superimposed onto the mainchain of this residue
	if (_chain >=0 && _chain < itsChains.size() )
	{
		return itsChains[_chain]->superimposeGLY(_residue);
	}
	else
	{
		cout << "ERROR in protein::superimposeGLY ... chain index out of range." << endl;
		exit(1);
	}
}

dblVec protein::getBackBoneCentroid()
{
	dblVec centroid(3);
	centroid[0] = 0.0; centroid[1] = 0.0; centroid[2] = 0.0;
	for (UInt i = 0; i < itsChains.size(); i ++)
	{
		centroid = centroid + itsChains[i]->getBackBoneCentroid();
	}

	centroid = centroid / (double)itsChains.size();
	return centroid;
}

typedef UIntVec::iterator iterUIntVec;

vector <dblVec> protein::saveCoords( UInt chainIndex, UInt resIndex)
{
	UInt nAtoms = getNumAtoms(chainIndex, resIndex);
	vector <dblVec> allCoords;
	for (UInt i=0; i<nAtoms; i++)
	{
		dblVec coords = getCoords(chainIndex, resIndex, i);
		allCoords.push_back(coords);
	}
	return allCoords;
}

void protein::setAllCoords( UInt chainIndex, UInt resIndex, vector<dblVec> allCoords)
{
	UInt nAtoms = getNumAtoms(chainIndex, resIndex);
	for (UInt i=0; i<nAtoms; i++)
	{
		setCoords(chainIndex, resIndex, i, allCoords[i]);
	}
}

void protein::cofactorRelax(UInt _plateau)
{  
	UInt pastCofactorClashes = getNumHardClashes(),count = 0;
	if (pastCofactorClashes > 0)
	{	
		//--Initialize variables for loop, calculate starting energy and build energy vectors---------------
		double rotX,rotY,rotZ;
		UInt randchain, randres, resnum, chainNum = getNumChains(), cofactorClashes, nobetter = 0;
		srand (time(NULL));

		//--Run optimizaiton loop to relative minima, determined by _plateau----------------------------
		do
		{   //--choose random cofactor
			do{
				randchain = rand() % chainNum;
				resnum = getNumResidues(randchain);
				randres = rand() % resnum;
			}while (!isCofactor(randchain, randres));
			nobetter++;
			count++;
	
			//--cofactor rotation-----------------------------------------------------------------------
			rotX = rand() % 31, rotY = rand() % 31, rotZ = rand() % 31;
			rotateChainRelative(randchain,X_axis,rotX), rotateChainRelative(randchain,Y_axis,rotY), rotateChainRelative(randchain,Z_axis,rotZ);
			cofactorClashes = getNumHardClashes();
			if (cofactorClashes <= pastCofactorClashes){
				nobetter = 0, pastCofactorClashes = cofactorClashes;
			}
			else{
				rotateChainRelative(randchain,Z_axis,rotZ*-1), rotateChainRelative(randchain,Y_axis,rotY*-1), rotateChainRelative(randchain,X_axis,rotX*-1);
			}

		} while (nobetter < _plateau && count < _plateau*10);
	}
	return;
}

void protein::protSampling(UInt iterations)
{
	// Sidechain and backslide sampling with a local dielectric scaling of electrostatics and corresponding Born/Gill implicit solvation energy
	//--Initialize variables for loop, calculate starting energy and build energy vectors----
	UInt randchain, randres, resnum, backboneOrSidechain = 1, changes = 0;
	UInt clashes, clashesStart, bbClashes, bbClashesStart, chainNum = getNumChains();
	double Energy, pastEnergy = protEnergy(), deltaEnergy, sPhi, sPsi, KT = KB*Temperature(), heat = 0.0;
	vector < DouVec > currentSidechainConf, newSidechainConf; vector <double> backboneAngles(2);
	bool sidechainTest, backboneTest, revert, energyTest;
	vector <dblVec> currentCoords;
	//--Run optimizaiton loop to local minima defined by an RT plateau------------------------
	do{
		//--choose random residue and set variables
		randchain = rand() % chainNum, resnum = getNumResidues(randchain), randres = rand() % resnum;
		clashesStart = getNumHardClashes();
		backboneTest = false, sidechainTest = false, energyTest = false, revert = true;
	
		//--Backbone conformation trial--------------------------------------------------------
		backboneOrSidechain = rand() % 2;
		if (randres > 0 && randres < resnum-2 && backboneOrSidechain == 0){
			backboneTest = true; bbClashesStart = getNumHardBackboneClashes();
			sPhi = getPhi(randchain,randres), sPsi = getPsi(randchain,randres);
			backboneAngles = getRandConformationFromBackboneType(sPhi, sPsi);
			setDihedral(randchain,randres,backboneAngles[0],0,0); setDihedral(randchain,randres,backboneAngles[1],1,0);
			bbClashes = getNumHardBackboneClashes();
			if (bbClashes <= bbClashesStart+2){
				energyTest = true; revert = false;
			}
		}
		//--Sidechain conformation trial--------------------------------------------------------
		else{
			sidechainTest = true;
			currentSidechainConf = getSidechainDihedrals(randchain, randres);
			newSidechainConf = randContinuousSidechainConformation(randchain, randres);
			setSidechainDihedralAngles(randchain, randres, newSidechainConf);
			clashes = getNumHardClashes();
			if (clashes <= clashesStart+2){
				energyTest = true; revert = false;
			}
		}

		//--Energy-Test-------------------------------------------------------------------------
		if (energyTest){
			changes++;
			Energy = protEnergy();
			deltaEnergy = Energy - pastEnergy;
			if (deltaEnergy <= KT+heat){
				pastEnergy = Energy; heat = 0;
				cout << Energy << endl;
			}
			else{revert = true; heat++;}
		}
		//--Revert conformation-----------------------------------------------------------------
		if (revert){
			if(backboneTest){
				setDihedral(randchain,randres,sPhi,0,0);
				setDihedral(randchain,randres,sPsi,1,0);
			}
			if(sidechainTest){
				setSidechainDihedralAngles(randchain, randres, currentSidechainConf);
			}
		}
	} while (changes < iterations);
	return;
}

vector <double> protein::getRandConformationFromBackboneType(double _phi, double _psi)
{
	vector <double> angles(2);
	double randtheta = (rand() % 31)-15; // maximum 30 degree window of dihedral sampling (+ or - 15)
	angles[0] = _phi+randtheta; angles[1] = _psi-randtheta; // approximation of RPT constraints via inverse proportionality between phi and psi
	return angles;
}

void protein::saveState(string _fileName)
{
	if (messagesActive) cout << " writing file " << _fileName << endl;
	pdbWriter(this, _fileName);
	return;
}

void protein::saveState(string& _fileName)
{
	if (messagesActive) cout << " writing file " << _fileName << endl;
	pdbWriter(this, _fileName);
	return;
}

//this function will calculate the hammingdistance between two sequences obtained from two pdb files
double protein::getHammingDistance(vector<string>seq1,vector<string>seq2) 
{ 
	double count = 0.0;double percent=0.0;double countsim=0.0;
	for (UInt i=0;i<seq1.size();i++){
        	if(seq1[i]==seq2[i]){
                	if (seq1[i]!="-" || seq2[i]!="X"){
                        	 count++;
                }
         }
        }
	countsim=seq1.size()-count;
	percent=((countsim/double(seq1.size()))*100);
   	return percent;
}  



// Take a copy of every atom position.  The minimiser accepts uphill moves, so
// the conformation it ends on is not in general the best one it found; without
// a snapshot a pass can and does finish worse than it started.  Coordinates are
// the right thing to keep rather than dihedrals: getChi and the energy both read
// them, they capture backbone and cofactor moves as well as sidechain ones, and
// they do not depend on the mutation buffers undoState works through.
void protein::snapshotConformationCU(vector<double>& _x, vector<double>& _y,
                                     vector<double>& _z)
{
	updateDeviceCoords();
	_x = itsCoordX; _y = itsCoordY; _z = itsCoordZ;
}

// Put a snapshot back, in the object graph and on the device.  The bonding tree
// is untouched by this -- only positions move -- so every dihedral accessor
// keeps working and reports the angles the snapshot was taken at.
void protein::restoreConformationCU(const vector<double>& _x, const vector<double>& _y,
                                    const vector<double>& _z)
{
	buildAtomIndex();
	const int n = (int)itsAtomPtrs.size();
	if ((int)_x.size() < n) {return;}
	for (int i = 0; i < n; i++)
	{	itsAtomPtrs[i]->setCoords(_x[i], _y[i], _z[i]); }
	updateDeviceCoords();
	if (itsEnergyContext)
	{	energySetCoords(itsEnergyContext, &itsCoordX[0], &itsCoordY[0], &itsCoordZ[0]);
		// The torsion baseline describes the conformation we just walked away
		// from, and a coordinate upload does not clear it.
		energyInvalidateTorsionBaseline(itsEnergyContext); }
}
