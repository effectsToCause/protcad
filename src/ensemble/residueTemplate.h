#include "pcAssert.h"
#include "atom.h"
#include "amberVDW.h"
#include "amberElec.h"
#include "aaBaseline.h"
#include "typedef.h"
#include "helixPropensity.h"

#ifndef RESIDUETEMPLATE_H
#define RESIDUETEMPLATE_H

#ifndef RESIDUE_H
class residue;
#endif

#ifndef ROTAMERLIB_H
#include "rotamerLib.h"
#endif

class residueTemplate
{
public:
	residueTemplate();
	residueTemplate(const residueTemplate& _rhs);
	~residueTemplate();
	void initialize();
	void reset();
	int getAtomIndexOf(const string& _name) const;
	string getAtomNameOf(UInt index) const;

private:
	void initializeChiDefinitions();
public:
	void addChiDefinitions(const StrVec& _strVect);
	void addAtomTypeDefinitions(const StrVec& _strVect);
	bool chiDefinitionsNonempty() const;
	bool isValidChi(const UInt _bpt, const UInt _index) const;
	UIntVec getAtomsOfChi(const UInt _bpt, const UInt _index) const;
        UIntVec getAtomsOfPolarHChi() const;
	bool getHasPolarHRotamers() const;
	inline int getNumberOfChis(const UInt _bpt) const
	{	if (chiDefinitions.size() == 0)
			return 0;
		UInt chiDefined = chiDefinitions[_bpt].size();
		if( chiDefined >= 4 )
		{	return chiDefined-3;
		}
		else
		{	return -1;
		}
	}
	vector< vector< UInt > > getBondingPattern() {return itsBondingPattern;}
	vector< UInt > getBondingPattern(UInt _atomIndex)
		{ return itsBondingPattern[_atomIndex]; }

	static UInt getHowMany() {return howManyTemplates;}
	//void assignAtomEnergyTypes();

	static double getAmberElecEnergySQ(const int _resType1, const int _atomType1, const int _resType2, const int _atomType2, const double _distanceSquared);
	static double getVDWEnergySQ(const int _type1, const int _type2, const double _distanceSquared);
	static double getVDWRadius(const int _type1);
	static string getVDWTypeName(const int _type1);
	static double getVDWEpsilon(const int _type1);
	static double getPolarizability(const int _type1);
	static double getPolarizabilityFlag(const int _type1);
	static double getVolume(const int _type1);
	void printAtomEnergyTypeDefinitions() const;
	static double getSolvationEnergy(double _surfaceArea, UInt _atomType, UInt _paramSet);
	static bool isClash(const int _type1, const int _type2, const double _distance);

	// variables declaration
	string typeString;
	string getName() {return typeString;}
	UInt typeIndex;
	UInt getType() {return typeIndex;}
	vector<string> atomNameList;
	UIntVec connectivity;
	UIntVec mainChain;
	UIntVec branchPoints;
	vector<bool> isMainChain;
	vector<atom> atomList;
	static UInt howManyTemplates;
        bool hasPolarHRotamers;

	vector< vector< UInt > > itsBondingPattern;

	// chi definitions and rotamer library
	vector<UIntVec> chiDefinitions;
	bool chiDefinitionsInitialized;
	vector <rotamerLib*> itsRotamerLibs;

	// atom type definitions for energy calculations
	vector< vector < string > > itsAtomEnergyTypeDefinitionStrings;
	vector< vector < int > >    itsAtomEnergyTypeDefinitions;
	void convertAtomTypeStringsToIndices(const StrVec& _strVect);
		
	// Energy modeling
	UIntVec itsAtomEnergyTypeIndex;
    static int itsCurrentEnergyType;
	static amberElec itsAmberElec;
	static amberVDW itsAmberVDW;
	static aaBaseline itsAABaseline;
	static bool atomEnergyTypeDefinitonsBuilt;
	static helixPropensity itsHelixPropensity;
};

#endif
