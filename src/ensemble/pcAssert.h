// filename: pcAssert.h
//
// Was src/ensemble/assert.h, which shadowed the standard <assert.h>/<cassert>
// for all 24 headers that included it -- any translation unit asking for the
// C library assert got this instead.  Renamed so the standard header is
// reachable again.
//
// The original read:
//
//     #define GLOBAL_DEBUG
//     #ifndef GLOBAL_DEBUG
//         #define ASSERT(x)
//     #else
//         ... print to cout, then block on `cin >> buffer` ...
//     #endif
//
// The file defined the very symbol it then tested, so the no-op branch was
// unreachable and every ASSERT was live in optimised builds -- including the
// ones on the chi/phi/psi hot path in residue.cc.  A failure printed to cout
// (corrupting the stdout that the harnesses parse) and then blocked on stdin.
// Under nohup, with stdin closed, cin fails silently and execution continues
// anyway, so the check never actually stopped anything; it only had the power
// to corrupt output.
//
// Default is now the no-op the author's #ifndef plainly intended.  Define
// PROTCAD_DEBUG_ASSERT to get a real check, which aborts rather than
// prompting: a batch run has nobody to answer the prompt.

#ifndef PCASSERT_H
#define PCASSERT_H

#ifndef PROTCAD_DEBUG_ASSERT
	#define ASSERT(x) ((void)0)
#else
	#include <cstdio>
	#include <cstdlib>
	#define ASSERT(x) \
	do { \
		if (!(x)) { \
			fprintf(stderr, "ASSERT failed: %s\n  at %s:%d\n", \
			        #x, __FILE__, __LINE__); \
			abort(); \
		} \
	} while (0)
#endif

#endif
