#include "amberVDW.h"
#include "amberParams.h"
#include <cstdlib>

double amberVDW::itsScaleFactor = 1.0;
double amberVDW::itsRadiusScaleFactor = 1.0;
bool amberVDW::linearRepulsionDampening = false;
double amberVDW::itsAttractionScaleFactor = 1.0;
double amberVDW::itsRepulsionScaleFactor = 1.0;

amberVDW::amberVDW()
{	// default constructor
	// read the file until the end.....
#ifdef AMBERVDW_DEBUG
	cout << "amberVDW::amberVDW() called" << endl;
#endif
	itsFileName = "amberVDW.frc";
	R_ref.resize(0);
	EPS.resize(0);
    Pol_ref.resize(0);
	Pol_flag.resize(0);
    Vol_ref.resize(0);
	buildDataBase();
	//cout << " amberVDW database is built " << endl;
#ifdef AMBERVDW_DEBUG
	for (UInt i=0; i< R_ref.size(); i++)
	{	cout << amberAtomTypeNames[i] << "   " << R_ref[i];
		cout << "    " << EPS[i] << endl;
	}
#endif
}

amberVDW::amberVDW(int _Dummy)
{	
#ifdef AMBERVDW_DEBUG
	cout << "amberVDW::amberVDW(int) called" << endl;
#endif
	// another constructor
	// read the file until the end.....
	itsFileName = "amberVDW.frc";
	R_ref.resize(0);
	EPS.resize(0);
    Pol_ref.resize(0);
	Pol_flag.resize(0);
    Vol_ref.resize(0);
	buildDataBase();
	//cout << " amberVDW database is built " << endl;
#ifdef AMBERVDW_DEBUG
	cout << "AmberVDWAtomTypeNames: " << endl;
	for (UInt i=0; i< R_ref.size(); i++)
	{	cout << i << "  "<< amberAtomTypeNames[i] << "   " << R_ref[i];
		cout << "    " << EPS[i] << endl;
	}
#endif
}

//deep copy constructor
amberVDW::amberVDW(const amberVDW& _otherAmberVDW)
{
#ifdef AMBERVDW_DEBUG
	cout << "amberVDW deep copy constructor called " << endl;
#endif
	itsFileName = _otherAmberVDW.itsFileName;
	R_ref = _otherAmberVDW.R_ref;
	EPS = _otherAmberVDW.EPS;
    Pol_ref = _otherAmberVDW.Pol_ref;
	Pol_flag = _otherAmberVDW.Pol_flag;
    Vol_ref = _otherAmberVDW.Vol_ref;
	amberAtomTypeNames = _otherAmberVDW.amberAtomTypeNames;
}

amberVDW::~amberVDW()
{
}

bool amberVDW::isClash(const UInt _type1, const UInt _type2, const double _distance)
{
	double R_ref_pair = itsRadiusScaleFactor * (R_ref[_type1] + R_ref[_type2]);
	if (_distance < R_ref_pair/2) return true;
	return false;
}

double amberVDW::getRadius(const UInt _type1)
{
    double radius  = (R_ref[_type1]) * itsRadiusScaleFactor;
    return radius;
}
double amberVDW::getEpsilon(const UInt _type1)
{
    double eps  = EPS[_type1];
    return eps;
}

double amberVDW::getPolarizability(const UInt _type1)
{
    double polarizability  = Pol_ref[_type1];
    return polarizability;
}

double amberVDW::getPolarizabilityFlag(const UInt _type1)
{
    double polflag  = Pol_flag[_type1];
    return polflag;
}

double amberVDW::getVolume(const UInt _type1)
{
    double volume  = Vol_ref[_type1];
    return volume;
}

double amberVDW::getEnergySQ(const UInt _type1, const UInt _type2, const double _distanceSquared) const //optimized to avoid distance calculation if possible
{
    double energy = 0.0;
    double R_ref_pair = 0.0;
    double EPS_pair = 0.0;
    if (_type1 < R_ref.size())
    {
        if (_type2 < R_ref.size())
        {       
			//cout << EPS[_type1] << " " << EPS[_type2] << " " << R_ref[_type1] << " " << R_ref[_type2] << endl;
			R_ref_pair  = itsRadiusScaleFactor * (R_ref[_type1] + R_ref[_type2]);
            if (EPS[_type1] == EPS[_type2])
				EPS_pair = EPS[_type1]; // save a sqrt operation
			else
				EPS_pair = sqrt( EPS[_type1] * EPS[_type2]);
            if (linearRepulsionDampening) // see Kuhlman & Baker PNAS v97 p10383  (2000) - online supplementary materials
			{
				double distance = sqrt(_distanceSquared);
                if ( (pow(R_ref_pair,2)/_distanceSquared) < 1.12)
                    energy = EPS_pair * (( itsRepulsionScaleFactor * pow(R_ref_pair,12)/pow(_distanceSquared,6)) - ( 2 * itsAttractionScaleFactor * pow(R_ref_pair,6)/pow(_distanceSquared,3)) );
                else
                    energy = 10 - 11.2 * (distance / R_ref_pair );
			}
            else 
			{
				//cout << "R_ref_pair used in getEnergySQ is " << R_ref_pair << endl;
				energy = EPS_pair * (( itsRepulsionScaleFactor * pow(R_ref_pair,12)/pow(_distanceSquared, 6)) - (2 * itsAttractionScaleFactor * pow(R_ref_pair,6)/pow(_distanceSquared,3)));
				//cout << "distancesquared: " << _distanceSquared << " " << energy << endl;
			}
        }
    }
    energy *= itsScaleFactor;
    return energy;
}   


double amberVDW::getEnergy(const UInt _type1, const UInt _type2, const double _distance) const
{
	double energy = 0.0;
	double R_ref_pair = 0.0;
	double EPS_pair = 0.0;
	if (_type1 < R_ref.size())
	{
		if (_type2 < R_ref.size())
		{		
			R_ref_pair  = itsRadiusScaleFactor * (R_ref[_type1] + R_ref[_type2]);
			if (EPS[_type1] == EPS[_type2])
				EPS_pair = EPS[_type1];
			else
				EPS_pair = sqrt( EPS[_type1] * EPS[_type2]);
			if (linearRepulsionDampening) // see Kuhlman & Baker PNAS v97 p10383  (2000) - online supplementary materials
				if ( (R_ref_pair/_distance) < 1.12)
					energy = EPS_pair * ( itsRepulsionScaleFactor * pow( (R_ref_pair/_distance),12) - (2 * itsAttractionScaleFactor * pow((R_ref_pair/_distance),6)));
				else
					energy = 10 - 11.2 * (_distance / R_ref_pair );
			else energy = EPS_pair * ( itsRepulsionScaleFactor * pow( (R_ref_pair/_distance),12) - (2 * itsAttractionScaleFactor * pow((R_ref_pair/_distance),6)));
			//cout << "R_ref_pair used in getEnergy is " << R_ref_pair << endl;
		}
	}
	energy *= itsScaleFactor;
	return energy;
}

double amberVDW::getWaterEnergy(const UInt _type1) const
{
    double energy = 0.0;
    double EPS_pair = 0.0;
    UInt waterType = 54;
    if (_type1 < EPS.size())
    {
        if (EPS[_type1] == EPS[waterType])
            EPS_pair = EPS[_type1];
        else
            EPS_pair = sqrt( EPS[_type1] * EPS[waterType]);
        energy = EPS_pair * -1;
    }
    return energy;
}

void amberVDW::buildDataBase()
{	
	string evname = "PROTCADDIR";
	string path = getEnvironmentVariable(evname);

	path += "/data/";
	string iFile = path + itsFileName;
	ifstream inFile;
	string currentLine;
	StrVec parsedStrings;
	parsedStrings.resize(0);
	
	// read the amber.frc data file
	inFile.open(iFile.c_str());
	if (!inFile)
	{	cout << "Error: unable to open input file: ";
		cout << iFile << endl;
		exit (1);
	}

	while (getline(inFile, currentLine, '\n'))
	{	// ignore the comment line
		// comment line should start with #
		if(currentLine[0] != '#' && currentLine[0] != '@'
			&& currentLine[0] != '!' && currentLine[0] != '>')
		{	
			parsedStrings=Parse::parse(currentLine);
            if (parsedStrings.size() == 7) convertToDataElements(parsedStrings);
			parsedStrings.resize(0);
		}
	}

	inFile.close();
	inFile.clear();

	applyForceFieldParams();
}

// amberVDW.frc is a hand transcription of Amber's nonbonded parameters, and it
// got three of the polar hydrogens wrong: H and HS are too large, and HO was
// given a radius and a well depth at all when ff14SB gives it neither.  A
// hydroxyl hydrogen with R* = 1.1 has a repulsive wall sitting exactly where a
// hydrogen bond has to close, which is worth several kcal/mol per bond and is
// systematically anti-secondary-structure.
//
// Rather than patch the three rows, take the physical parameters from the
// distributed ff14SB files for every type that appears in them.  Types that do
// not (water, ions, nucleic acid, anything GAFF) keep whatever amberVDW.frc
// says, since ff14SB has no opinion about them.
//
// Two things are deliberately preserved:
//   - the row order of amberVDW.frc, because indices from
//     getIndexFromNameString are cached all over the ensemble code and
//     getWaterEnergy indexes water by a hardcoded literal;
//   - the polarizability flag column, which is protcad's own and has no Amber
//     counterpart.
//
// Shell volume is recomputed as 4.18*R*^3, which is the relation the file's own
// volume column already satisfies for every row, so this is a no-op except
// where the radius itself changed.  It matters for HO: the solvation model was
// giving a zero-radius hydrogen 5.6 A^3 of displaced solvent.
//
// Set PROTCAD_VDW_LEGACY=1 to keep the old table, so the effect of the fix can
// be measured rather than silently absorbed.
void amberVDW::applyForceFieldParams()
{
	const char* legacy = getenv("PROTCAD_VDW_LEGACY");
	if (legacy && legacy[0] == '1')
	{
		cout << "amberVDW: PROTCAD_VDW_LEGACY=1, using data/amberVDW.frc as written" << endl;
		return;
	}

	amberParams ff;
	ff.loadFF14SB();

	for (UInt i = 0; i < amberAtomTypeNames.size(); i++)
	{
		if (!ff.hasVdw(amberAtomTypeNames[i])) continue;
		const amberVdwParam& p = ff.vdw(amberAtomTypeNames[i]);
		R_ref[i] = p.radius;
		EPS[i] = p.epsilon;
		if (p.polarizability > 0.0) Pol_ref[i] = p.polarizability;
		Vol_ref[i] = 4.18 * p.radius * p.radius * p.radius;
	}
}

// where specific information about parsed data is intepreted
void amberVDW::convertToDataElements(const StrVec& _parsedStrings)
{
	double tmpDouble;
	sscanf(_parsedStrings[1].c_str(), "%lf", &tmpDouble);
    Pol_flag.push_back(tmpDouble);
	amberAtomTypeNames.push_back(_parsedStrings[2]);
	sscanf(_parsedStrings[3].c_str(), "%lf", &tmpDouble);
	R_ref.push_back(tmpDouble);
	tmpDouble = 0.0;
	sscanf(_parsedStrings[4].c_str(), "%lf", &tmpDouble);
	EPS.push_back(tmpDouble);
    tmpDouble = 0.0;
    sscanf(_parsedStrings[5].c_str(), "%lf", &tmpDouble);
    Pol_ref.push_back(tmpDouble);
    tmpDouble = 0.0;
    sscanf(_parsedStrings[6].c_str(), "%lf", &tmpDouble);
    Vol_ref.push_back(tmpDouble);
}

int amberVDW::getIndexFromNameString(string _name)
{
	for (UInt i=0; i<amberAtomTypeNames.size(); i++)
	{	if (amberAtomTypeNames[i] == _name)
		{
#ifdef AMBERVDW_DEBUG
			cout <<  "In amberVDW::getIndexFromNameString(string _name)" << endl;
			cout <<  "   found atom type " << _name << " at position " << i << endl;
#endif
			return i;
		}
	}
#ifdef AMBERVDW_DEBUG
	cout << "In amberVDW::getIndexFromNameString(string _name)" << endl;
	cout << "    couldnt find atom type: " << _name << endl;
#endif
	// If we're not able to find the atom name here, warn
	// the user, and return a -1, which is out of range normally.
	return -1;
}
