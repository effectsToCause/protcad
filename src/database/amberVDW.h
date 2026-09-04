// filename: amberVDW.h

#include "pcAssert.h"
#include <string.h>
#include <fstream>
#include <vector>
#include "typedef.h"
#include <stdio.h>
#include <math.h>
#include "generalio.h"
#include "parse.h"

#ifndef _AMBER_VDW_H
#define _AMBER_VDW_H

class amberVDW
{
public:
	amberVDW();
	amberVDW(int _dumy);
	amberVDW(const amberVDW& _otherAmberVDW);
	~amberVDW();
	
	double getEnergy(const UInt _type1,const UInt _type2,const double _distance) const;
    double getWaterEnergy(const UInt _type1) const;
	double getEnergySQ(const UInt _type1, const UInt _type2, const double _distanceSquared) const;
	int getIndexFromNameString(string _name);
	string getNameFromIndex(const UInt _index) const
		{ return _index < amberAtomTypeNames.size() ? amberAtomTypeNames[_index] : string(); }
	bool isClash(const UInt _type1, const UInt _type2, const double _distance);
    double getRadius(const UInt _type1);
	double getEpsilon(const UInt _type1);
    double getPolarizability(const UInt _type1);
	double getPolarizabilityFlag(const UInt _type1);
    double getVolume(const UInt _type1);

	static double itsScaleFactor;
	static void setScaleFactor(const double _scale)
		{ itsScaleFactor = _scale;
                  // cout << " vdw scale factor set to: " << _scale << endl;
                 }
	static double getScaleFactor()
		{ return itsScaleFactor; }

	static double itsRadiusScaleFactor;

	static bool linearRepulsionDampening;	// Kuhlman & Baker PNAS v97 p10383

	static double itsAttractionScaleFactor;
	
	static double itsRepulsionScaleFactor;
	
private:
	void buildDataBase();
	void applyForceFieldParams();
	// StrVec is intended non const reference
	void convertToDataElements(const StrVec& _parsedStrings);

private:
	vector< double > R_ref;
	vector< double > EPS;
    vector< double > Pol_ref;
	vector< double > Pol_flag;
    vector< double > Vol_ref;
	vector< string > amberAtomTypeNames;
	string itsFileName;
};

#endif
