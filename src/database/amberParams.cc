// filename: amberParams.cc

#include "amberParams.h"
#include "generalio.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>

namespace
{
    const double kPi = 3.14159265358979323846;

    bool isBlank(const string& s)
    {
        for (size_t i = 0; i < s.size(); ++i)
            if (!isspace((unsigned char)s[i])) return false;
        return true;
    }

    string rstrip(const string& s)
    {
        size_t e = s.find_last_not_of(" \t\r\n");
        return (e == string::npos) ? string() : s.substr(0, e + 1);
    }

    // Pull the atom type occupying a fixed two-character field.  The type
    // fields in these tables are positional, not whitespace delimited: a line
    // reads "CT-C -N -H" and splitting on whitespace merges "C -N" into one
    // token.  Types can also legitimately contain characters that look like
    // delimiters, so the columns are the only reliable guide.
    string field2(const string& line, size_t at)
    {
        if (at >= line.size()) return string();
        string s = line.substr(at, 2);
        size_t b = s.find_first_not_of(" \t");
        if (b == string::npos) return string();
        size_t e = s.find_last_not_of(" \t");
        return s.substr(b, e - b + 1);
    }

    // Numeric fields following the type block.
    vector<double> numbersAfter(const string& line, size_t start)
    {
        vector<double> out;
        if (start >= line.size()) return out;
        istringstream in(line.substr(start));
        string tok;
        while (in >> tok)
        {
            // Comments trail every table in these files and are not delimited.
            // Stop at the first token that is not a number.
            char* end = 0;
            double v = strtod(tok.c_str(), &end);
            if (end == tok.c_str() || (*end != '\0' && *end != '.')) break;
            out.push_back(v);
        }
        return out;
    }
}

amberParams::amberParams() : itsPendingImproper(false) {}
amberParams::~amberParams() {}


string amberParams::key2(const string& a, const string& b)
{ return a + "-" + b; }
string amberParams::key3(const string& a, const string& b, const string& c)
{ return a + "-" + b + "-" + c; }
string amberParams::key4(const string& a, const string& b, const string& c,
                         const string& d)
{ return a + "-" + b + "-" + c + "-" + d; }

string amberParams::dataPath(const string& _fileName) const
{
    string evname = "PROTCADDIR";
    string path = getEnvironmentVariable(evname);
    path += "/data/amber/";
    return path + _fileName;
}

void amberParams::loadFF14SB()
{
    // parm10 carries the shared tables; frcmod.ff14SB replaces the protein
    // backbone and sidechain torsions and adds the 2C/3C/C8/CO types.  Order
    // matters -- the frcmod is a patch and must be applied second.
    loadParm("parm10.dat");
    loadFrcmod("frcmod.ff14SB");
    applyEquivalences();
}

void amberParams::loadParm(const string& _fileName)
{
    parseFile(dataPath(_fileName), false);
}

void amberParams::loadFrcmod(const string& _fileName)
{
    parseFile(dataPath(_fileName), true);
    applyEquivalences();
}

void amberParams::readMass(const string& line)
{
    string t = field2(line, 0);
    if (t.empty()) return;
    vector<double> v = numbersAfter(line, 2);
    if (v.size() < 1) return;
    amberVdwParam& p = itsVdw[t];
    // Polarizability is optional in this section; mass itself is unused here
    // because the sampler never integrates equations of motion.
    if (v.size() >= 2) p.polarizability = v[1];
}

void amberParams::readBond(const string& line)
{
    string a = field2(line, 0), b = field2(line, 3);
    if (a.empty() || b.empty()) return;
    vector<double> v = numbersAfter(line, 5);
    if (v.size() < 2) return;
    amberBondParam p; p.k = v[0]; p.r0 = v[1];
    itsBonds[key2(a, b)] = p;
    itsBonds[key2(b, a)] = p;
}

void amberParams::readAngle(const string& line)
{
    string a = field2(line, 0), b = field2(line, 3), c = field2(line, 6);
    if (a.empty() || b.empty() || c.empty()) return;
    vector<double> v = numbersAfter(line, 8);
    if (v.size() < 2) return;
    amberAngleParam p; p.k = v[0]; p.theta0 = v[1] * kPi / 180.0;
    itsAngles[key3(a, b, c)] = p;
    itsAngles[key3(c, b, a)] = p;
}

void amberParams::readTorsion(const string& line, bool improper)
{
    string a = field2(line, 0), b = field2(line, 3);
    string c = field2(line, 6), d = field2(line, 9);
    if (b.empty() || c.empty()) return;
    if (a.empty()) a = "X";
    if (d.empty()) d = "X";

    vector<double> v = numbersAfter(line, 11);
    amberTorsionTerm term;
    if (improper)
    {
        // No IDIVF field in the improper table.
        if (v.size() < 3) return;
        term.divisor     = 1.0;
        term.barrier     = v[0];
        term.phase       = v[1] * kPi / 180.0;
        term.periodicity = v[2];
    }
    else
    {
        if (v.size() < 4) return;
        term.divisor     = (v[0] != 0.0) ? v[0] : 1.0;
        term.barrier     = v[1];
        term.phase       = v[2] * kPi / 180.0;
        term.periodicity = v[3];
    }

    const bool continues = (term.periodicity < 0.0);
    term.periodicity = fabs(term.periodicity);

    const string k = key4(a, b, c, d);
    if (!itsPendingTerms.empty() && (k != itsPendingKey ||
                                     itsPendingImproper != improper))
    {
        // A new quartet began while a series was still marked open.  That means
        // the file omitted the terminating record; keep what was collected
        // rather than discarding it, and say so.
        itsWarnings.push_back("unterminated torsion series for " + itsPendingKey);
        map<string, vector<amberTorsionTerm> >& tgt =
            itsPendingImproper ? itsImpropers : itsTorsions;
        tgt[itsPendingKey] = itsPendingTerms;
        itsPendingTerms.clear();
    }

    itsPendingKey = k;
    itsPendingImproper = improper;
    itsPendingTerms.push_back(term);

    if (!continues)
    {
        map<string, vector<amberTorsionTerm> >& tgt =
            improper ? itsImpropers : itsTorsions;
        // A later file replaces an earlier definition outright rather than
        // appending to it; that is what makes a frcmod a patch.
        tgt[k] = itsPendingTerms;
        itsPendingTerms.clear();
        itsPendingKey.clear();
    }
}

void amberParams::readNonbon(const string& line)
{
    istringstream in(line);
    string t; double r, e;
    if (!(in >> t >> r >> e)) return;
    amberVdwParam& p = itsVdw[t];
    p.radius = r;
    p.epsilon = e;
    p.defined = true;
}

void amberParams::readEquivalence(const string& line)
{
    istringstream in(line);
    vector<string> toks; string t;
    while (in >> t) toks.push_back(t);
    if (toks.size() < 2) return;
    // The first symbol owns the parameters; the rest borrow them.
    for (size_t i = 1; i < toks.size(); ++i)
        itsEquivalences.push_back(make_pair(toks[i], toks[0]));
}

void amberParams::applyEquivalences()
{
    for (size_t i = 0; i < itsEquivalences.size(); ++i)
    {
        const string& alias = itsEquivalences[i].first;
        const string& src   = itsEquivalences[i].second;
        map<string, amberVdwParam>::const_iterator s = itsVdw.find(src);
        if (s == itsVdw.end() || !s->second.defined) continue;
        amberVdwParam& a = itsVdw[alias];
        // An explicit MOD4 record always wins over an equivalence.
        if (a.defined) continue;
        a.radius  = s->second.radius;
        a.epsilon = s->second.epsilon;
        a.defined = true;
    }
    itsEquivalences.clear();
}

void amberParams::parseFile(const string& _path, bool _isFrcmod)
{
    ifstream in(_path.c_str());
    if (!in)
    {
        cout << "amberParams: unable to open " << _path << endl;
        exit(1);
    }

    // Sections in a parm file are positional: the title, then MASS, a line of
    // hydrophilic types, BOND, ANGLE, DIHE, IMPROPER, the 10-12 table, the
    // equivalence block and finally MOD4, each terminated by a blank line.  A
    // frcmod names its sections instead, so both are driven through one enum
    // with a different way of advancing.
    enum Section { SEC_NONE, SEC_MASS, SEC_HYDROPHILIC, SEC_BOND, SEC_ANGLE,
                   SEC_DIHE, SEC_IMPROPER, SEC_HBOND, SEC_EQUIV, SEC_NONBON,
                   SEC_SKIP, SEC_DONE };
    Section sec = _isFrcmod ? SEC_NONE : SEC_MASS;

    string line;
    bool firstLine = true;
    while (getline(in, line))
    {
        line = rstrip(line);
        if (firstLine) { firstLine = false; continue; }   // title
        if (sec == SEC_DONE) break;

        if (_isFrcmod)
        {
            if (isBlank(line)) { sec = SEC_NONE; continue; }
            if (sec == SEC_NONE)
            {
                string h = line.substr(0, 8);
                if      (h.compare(0, 4, "MASS") == 0) sec = SEC_MASS;
                else if (h.compare(0, 4, "BOND") == 0) sec = SEC_BOND;
                else if (h.compare(0, 4, "ANGL") == 0) sec = SEC_ANGLE;
                else if (h.compare(0, 4, "DIHE") == 0) sec = SEC_DIHE;
                else if (h.compare(0, 4, "IMPR") == 0) sec = SEC_IMPROPER;
                else if (h.compare(0, 4, "NONB") == 0) sec = SEC_NONBON;
                else sec = SEC_SKIP;   // LJEDIT, CMAP and anything else
                continue;
            }
        }
        else
        {
            if (line.compare(0, 3, "END") == 0) { sec = SEC_DONE; continue; }
            if (isBlank(line))
            {
                switch (sec)
                {
                    case SEC_MASS:        sec = SEC_HYDROPHILIC; break;
                    case SEC_BOND:        sec = SEC_ANGLE;    break;
                    case SEC_ANGLE:       sec = SEC_DIHE;     break;
                    case SEC_DIHE:        sec = SEC_IMPROPER; break;
                    case SEC_IMPROPER:    sec = SEC_HBOND;    break;
                    case SEC_HBOND:       sec = SEC_EQUIV;    break;
                    case SEC_EQUIV:       sec = SEC_NONBON;   break;
                    default: break;
                }
                continue;
            }
            if (sec == SEC_HYDROPHILIC) { sec = SEC_BOND; continue; }
            // The vdW block opens with "MOD4 RE"; the kind of nonbonded table
            // is named there and only RE is supported.
            if (sec == SEC_NONBON && line.compare(0, 4, "MOD4") == 0)
            {
                if (line.find("RE") == string::npos)
                    itsWarnings.push_back("nonbonded table is not in RE form: "
                                          + line);
                continue;
            }
        }

        switch (sec)
        {
            case SEC_MASS:     readMass(line); break;
            case SEC_BOND:     readBond(line); break;
            case SEC_ANGLE:    readAngle(line); break;
            case SEC_DIHE:     readTorsion(line, false); break;
            case SEC_IMPROPER: readTorsion(line, true); break;
            case SEC_EQUIV:    readEquivalence(line); break;
            case SEC_NONBON:   readNonbon(line); break;
            default: break;
        }
    }

    if (!itsPendingTerms.empty())
    {
        map<string, vector<amberTorsionTerm> >& tgt =
            itsPendingImproper ? itsImpropers : itsTorsions;
        tgt[itsPendingKey] = itsPendingTerms;
        itsPendingTerms.clear();
        itsPendingKey.clear();
    }
}

bool amberParams::hasVdw(const string& _type) const
{
    map<string, amberVdwParam>::const_iterator i = itsVdw.find(_type);
    return i != itsVdw.end() && i->second.defined;
}

const amberVdwParam& amberParams::vdw(const string& _type) const
{
    static amberVdwParam none;
    map<string, amberVdwParam>::const_iterator i = itsVdw.find(_type);
    return (i == itsVdw.end()) ? none : i->second;
}

vector<string> amberParams::vdwTypes() const
{
    vector<string> out;
    for (map<string, amberVdwParam>::const_iterator i = itsVdw.begin();
         i != itsVdw.end(); ++i)
        if (i->second.defined) out.push_back(i->first);
    return out;
}

const vector<amberTorsionTerm>& amberParams::torsion(
    const string& a, const string& b, const string& c, const string& d) const
{
    // tleap resolves a torsion by looking for the specific quartet first and
    // falling back to the wildcard form, accepting either direction because a
    // dihedral and its reverse are the same angle.
    map<string, vector<amberTorsionTerm> >::const_iterator i;
    string keys[4] = { key4(a, b, c, d), key4(d, c, b, a),
                       key4("X", b, c, "X"), key4("X", c, b, "X") };
    for (int k = 0; k < 4; ++k)
    {
        i = itsTorsions.find(keys[k]);
        if (i != itsTorsions.end()) return i->second;
    }
    return itsEmpty;
}

const vector<amberTorsionTerm>& amberParams::improper(
    const string& a, const string& b, const string& c, const string& d) const
{
    // The third position is the central atom of an Amber improper.  The other
    // three are interchangeable, so every arrangement of them is tried, most
    // specific first.
    const string p[3] = { a, b, d };
    map<string, vector<amberTorsionTerm> >::const_iterator i;
    static const int perm[6][3] = {{0,1,2},{0,2,1},{1,0,2},
                                   {1,2,0},{2,0,1},{2,1,0}};
    for (int wild = 0; wild < 3; ++wild)
    {
        for (int q = 0; q < 6; ++q)
        {
            string x = p[perm[q][0]], y = p[perm[q][1]], z = p[perm[q][2]];
            if (wild >= 1) x = "X";
            if (wild >= 2) y = "X";
            i = itsImpropers.find(key4(x, y, c, z));
            if (i != itsImpropers.end()) return i->second;
        }
    }
    return itsEmpty;
}

bool amberParams::bond(const string& a, const string& b,
                       amberBondParam& out) const
{
    map<string, amberBondParam>::const_iterator i = itsBonds.find(key2(a, b));
    if (i == itsBonds.end()) return false;
    out = i->second;
    return true;
}

bool amberParams::angle(const string& a, const string& b, const string& c,
                        amberAngleParam& out) const
{
    map<string, amberAngleParam>::const_iterator i = itsAngles.find(key3(a, b, c));
    if (i == itsAngles.end()) return false;
    out = i->second;
    return true;
}
