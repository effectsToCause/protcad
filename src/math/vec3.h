// filename: vec3.h
// contents: fixed-size 3-vector and 3x3 matrix replacing TNT::Vector<double>
//           and TNT::Matrix<double>.
//
// Every use of TNT in protcad is at dimension 3: all 14 non-TNT newsize call
// sites are newsize(3) or newsize(3,3), every construction is dblVec(3) or
// dblMat(3,3,...), determinant() is hardcoded 3x3 and rotationMatrix()
// returns R(3,3).  Nothing needs variable-dimension linear algebra, so the
// heap allocation TNT performs per vector -- one new double[3] plus a 1-based
// offset pointer for EVERY atom coordinate, since point::itsCoords is a
// dblVec and atom derives from point -- buys nothing and costs locality.
//
// The API below mirrors TNT's deliberately, including the elementwise
// operator* (TNT's vector operator* is NOT a dot product) and the stream
// formats, so call sites are unchanged and the swap is verifiable by
// byte-identical output rather than by inspection.
//
// One intentional semantic difference: TNT's newsize() allocates without
// initialising, so `dblVec v; v.newsize(3);` left garbage.  These types zero
// initialise.  That is strictly safer, and the byte-identical check across
// the swap is what establishes no caller depended on the garbage.

#ifndef VEC3_H
#define VEC3_H

#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// protcad ships its own assert.h (src/ensemble/assert.h) which shadows
// <cassert> on the include path, so use a local macro.  Off by default:
// operator[] is on the per-atom-coordinate hot path and must stay branchless.
#ifdef PROTCAD_VEC3_CHECKED
#define VEC3_ASSERT(c) do { if(!(c)) { \
    std::fprintf(stderr,"VEC3_ASSERT %s at %s:%d\n",#c,__FILE__,__LINE__); \
    std::abort(); } } while(0)
#else
#define VEC3_ASSERT(c) ((void)0)
#endif

class Vec3
{
  public:
    typedef double value_type;
    typedef double* iterator;
    typedef const double* const_iterator;

    Vec3() { v_[0] = v_[1] = v_[2] = 0.0; }

    Vec3(int N, const double& value = 0.0)
    {
        VEC3_ASSERT(N == 3); (void)N;
        v_[0] = v_[1] = v_[2] = value;
    }

    Vec3(int N, const double* v)
    {
        VEC3_ASSERT(N == 3); (void)N;
        v_[0] = v[0]; v_[1] = v[1]; v_[2] = v[2];
    }

    iterator begin() { return v_; }
    iterator end()   { return v_ + 3; }
    const_iterator begin() const { return v_; }
    const_iterator end()   const { return v_ + 3; }

    Vec3& newsize(int N) { VEC3_ASSERT(N == 3); (void)N; return *this; }

    Vec3& operator=(const double& scalar)
    {
        v_[0] = v_[1] = v_[2] = scalar;
        return *this;
    }

    int dim()  const { return 3; }
    int size() const { return 3; }

    // TNT exposes both a 1-based operator() and a 0-based operator[].
    double&       operator()(int i)       { VEC3_ASSERT(i >= 1 && i <= 3); return v_[i - 1]; }
    const double& operator()(int i) const { VEC3_ASSERT(i >= 1 && i <= 3); return v_[i - 1]; }
    double&       operator[](int i)       { VEC3_ASSERT(i >= 0 && i <  3); return v_[i]; }
    const double& operator[](int i) const { VEC3_ASSERT(i >= 0 && i <  3); return v_[i]; }

  private:
    double v_[3];
};

inline bool operator==(const Vec3& A, const Vec3& B)
{
    return A[0] == B[0] && A[1] == B[1] && A[2] == B[2];
}
inline bool operator!=(const Vec3& A, const Vec3& B) { return !(A == B); }

inline std::ostream& operator<<(std::ostream& s, const Vec3& A)
{
    s << 3 << std::endl;
    for (int i = 0; i < 3; i++) s << A[i] << " " << std::endl;
    s << std::endl;
    return s;
}

inline std::istream& operator>>(std::istream& s, Vec3& A)
{
    int N;
    s >> N;
    for (int i = 0; i < 3; i++) s >> A[i];
    return s;
}

inline Vec3 operator+(const Vec3& A, const Vec3& B)
{
    Vec3 t; for (int i = 0; i < 3; i++) t[i] = A[i] + B[i]; return t;
}
inline Vec3 operator-(const Vec3& A, const Vec3& B)
{
    Vec3 t; for (int i = 0; i < 3; i++) t[i] = A[i] - B[i]; return t;
}
// Elementwise, matching TNT.  This is not a dot product.
inline Vec3 operator*(const Vec3& A, const Vec3& B)
{
    Vec3 t; for (int i = 0; i < 3; i++) t[i] = A[i] * B[i]; return t;
}
inline Vec3 operator*(const Vec3& A, const double& B)
{
    Vec3 t; for (int i = 0; i < 3; i++) t[i] = A[i] * B; return t;
}
inline Vec3 operator*(const double& B, const Vec3& A) { return A * B; }
inline Vec3 operator/(const Vec3& A, const double& B)
{
    Vec3 t; for (int i = 0; i < 3; i++) t[i] = A[i] / B; return t;
}
inline double dot_prod(const Vec3& A, const Vec3& B)
{
    double s = 0.0; for (int i = 0; i < 3; i++) s += A[i] * B[i]; return s;
}

class Mat3
{
  public:
    typedef double value_type;

    Mat3() { set(0.0); }

    Mat3(int M, int N, const double& value = 0.0)
    {
        VEC3_ASSERT(M == 3 && N == 3); (void)M; (void)N;
        set(value);
    }

    Mat3& newsize(int M, int N)
    {
        VEC3_ASSERT(M == 3 && N == 3); (void)M; (void)N;
        return *this;
    }

    Mat3& operator=(const double& scalar) { set(scalar); return *this; }

    int dim(int d) const { VEC3_ASSERT(d == 1 || d == 2); (void)d; return 3; }
    int size()     const { return 9; }

    double*       operator[](int i)       { VEC3_ASSERT(i >= 0 && i < 3); return m_[i]; }
    const double* operator[](int i) const { VEC3_ASSERT(i >= 0 && i < 3); return m_[i]; }

    double&       operator()(int i, int j)
    { VEC3_ASSERT(i >= 1 && i <= 3 && j >= 1 && j <= 3); return m_[i - 1][j - 1]; }
    const double& operator()(int i, int j) const
    { VEC3_ASSERT(i >= 1 && i <= 3 && j >= 1 && j <= 3); return m_[i - 1][j - 1]; }

  private:
    void set(const double& v)
    {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) m_[i][j] = v;
    }
    double m_[3][3];
};

inline std::ostream& operator<<(std::ostream& s, const Mat3& A)
{
    s << 3 << " " << 3 << std::endl;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) s << A[i][j] << " ";
        s << std::endl;
    }
    return s;
}

inline Mat3 operator+(const Mat3& A, const Mat3& B)
{
    Mat3 t;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) t[i][j] = A[i][j] + B[i][j];
    return t;
}
inline Mat3 operator-(const Mat3& A, const Mat3& B)
{
    Mat3 t;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) t[i][j] = A[i][j] - B[i][j];
    return t;
}
// Matrix multiply, matching TNT's matmult ordering.
inline Mat3 operator*(const Mat3& A, const Mat3& B)
{
    Mat3 t;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < 3; k++) sum += A[i][k] * B[k][j];
            t[i][j] = sum;
        }
    return t;
}
inline Vec3 operator*(const Mat3& A, const Vec3& x)
{
    Vec3 t;
    for (int i = 0; i < 3; i++)
    {
        double sum = 0.0;
        for (int k = 0; k < 3; k++) sum += A[i][k] * x[k];
        t[i] = sum;
    }
    return t;
}

#endif
