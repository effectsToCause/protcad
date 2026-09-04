#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
#include "vec3.h"

#ifndef TYPEDEF_H
#define TYPEDEF_H

// Check for CUDA lib.  The CMake build already defines __CUDA__ on the command
// line for every translation unit; the guard keeps this fallback working for a
// direct compile without redefining the macro (which warned on every TU).
#if __has_include("cuda_runtime.h")
#ifndef __CUDA__
#define __CUDA__
#endif
#endif

//physical constants
#define PI 3.1415926535 //Pi (Ratio of a circle's circumference to its diameter)
#define KB 0.0019872041 //Boltzmann constant (kcal/mol K)
#define EU 2.7182818284 //Eulers number (base of natural log)
#define KC 332.0636     //Coulombs constant (kcal/mol)

//amino acid types
#define Daa 27
#define Nterm 53
#define Cterm 106

//data types
typedef unsigned int UInt;
typedef vector<double> DouVec;
typedef vector<double> DblVec;
typedef vector<UInt> UIntVec;
typedef vector<string> StrVec;
typedef Vec3 dblVec;
typedef Mat3 dblMat;

#ifndef CMATH_H
#include "CMath.h"
#endif

#endif
