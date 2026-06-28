// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package mldsa

// Decomplected benchmark suite for the ML-DSA ring primitives. Each
// Benchmark isolates EXACTLY ONE ring operation; no benchmark measures
// two braided things.
//
// The parameter axis of a benchmark is the one that actually changes the
// operation's cost — and only that one:
//
//   - NTT, INTT, MulNTT, InfNorm, Power2Round are parameter-INVARIANT at
//     the ring level (fixed N=256, q=8 380 417, drop length D=13). They
//     are benched once, on the production ring (ML-DSA-65). Adding a
//     P44/P65/P87 sub-axis would measure the identical machine code three
//     times — a false coupling, not a cost curve, so it is omitted.
//   - Decompose and UseHint are gated on Gamma2. ML-DSA-65 and -87 share
//     one Gamma2 ((q-1)/32); ML-DSA-44 uses the other ((q-1)/88). They
//     are benched over the two DISTINCT Gamma2 values (P65, P44).
//   - MatVec (the A*x Module-LWE kernel) scales with the module shape
//     (K,L), so it is benched over all three profiles whose dims differ:
//     4x4, 6x5, 8x7.
//
// NTT and INTT transform in place. Both are constant-time (fixed loop
// bounds, branchless Montgomery reduction), so repeated in-place
// application measures the exact per-call cost: the coefficient values
// drift, but the timing does not depend on them. INTT additionally
// re-normalizes its operand into [0, q) every call, so its operand never
// leaves the documented input range.

import (
	"math/rand"
	"testing"

	"github.com/luxfi/mlwe"
)

// sinkU64 absorbs scalar results so the compiler cannot elide the
// benchmarked call as dead code.
var sinkU64 uint64

// BenchmarkNTT measures one forward NTT (coefficient -> evaluation
// domain). Parameter-invariant; benched on ML-DSA-65.
func BenchmarkNTT(b *testing.B) {
	r := MustNew(Profile65)
	p := randPoly(r, rand.New(rand.NewSource(1)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.NTT(p)
	}
}

// BenchmarkINTT measures one inverse NTT (evaluation -> coefficient
// domain, normalized). Parameter-invariant; benched on ML-DSA-65.
func BenchmarkINTT(b *testing.B) {
	r := MustNew(Profile65)
	p := randPoly(r, rand.New(rand.NewSource(2)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.INTT(p)
	}
}

// BenchmarkMulNTT measures one pointwise NTT-domain product (dst = a o b).
// Operands a and b are not mutated, so every iteration sees identical,
// in-range inputs. Parameter-invariant; benched on ML-DSA-65.
func BenchmarkMulNTT(b *testing.B) {
	r := MustNew(Profile65)
	rng := rand.New(rand.NewSource(3))
	a := randPoly(r, rng)
	c := randPoly(r, rng)
	dst := r.NewPoly()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.MulNTT(dst, a, c)
	}
}

// BenchmarkInfNorm measures one centered infinity-norm reduction.
// Branchless; parameter-invariant; benched on ML-DSA-65.
func BenchmarkInfNorm(b *testing.B) {
	r := MustNew(Profile65)
	p := randPoly(r, rand.New(rand.NewSource(4)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkU64 = r.InfNorm(p)
	}
}

// BenchmarkPower2Round measures one Power2Round split. Gated only on the
// fixed drop length D=13, so parameter-invariant; benched on ML-DSA-65.
// Each call allocates its two output polynomials (see the braiding note
// in the suite report).
func BenchmarkPower2Round(b *testing.B) {
	r := MustNew(Profile65)
	p := randPoly(r, rand.New(rand.NewSource(5)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Power2Round(p)
	}
}

// BenchmarkDecompose measures one Decompose split, over the two distinct
// Gamma2 values ((q-1)/32 for ML-DSA-65, (q-1)/88 for ML-DSA-44).
func BenchmarkDecompose(b *testing.B) {
	rng := rand.New(rand.NewSource(6))
	for _, prof := range []mlwe.Profile{Profile65, Profile44} {
		r := MustNew(prof)
		p := randPoly(r, rng)
		b.Run(prof.Name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				r.Decompose(p)
			}
		})
	}
}

// BenchmarkUseHint measures one UseHint correction, over the two distinct
// Gamma2 values. The hint poly carries random 0/1 coefficients.
func BenchmarkUseHint(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	for _, prof := range []mlwe.Profile{Profile65, Profile44} {
		r := MustNew(prof)
		rp := randPoly(r, rng)
		hint := r.NewPoly()
		for i := range hint.Coeffs {
			hint.Coeffs[i] = uint64(rng.Intn(2))
		}
		b.Run(prof.Name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				r.UseHint(hint, rp)
			}
		})
	}
}

// BenchmarkMatVec measures the A*x Module-LWE kernel over the three
// module shapes whose dimensions differ: ML-DSA-44 (4x4), -65 (6x5),
// -87 (8x7). A and x are coefficient-domain; MatVec performs all forward
// and inverse transforms internally.
func BenchmarkMatVec(b *testing.B) {
	rng := rand.New(rand.NewSource(8))
	for _, prof := range []mlwe.Profile{Profile44, Profile65, Profile87} {
		r := MustNew(prof)
		a := make(mlwe.PolyMat, prof.K)
		for i := range a {
			a[i] = make(mlwe.PolyVec, prof.L)
			for j := range a[i] {
				a[i][j] = randPoly(r, rng)
			}
		}
		x := make(mlwe.PolyVec, prof.L)
		for j := range x {
			x[j] = randPoly(r, rng)
		}
		b.Run(prof.Name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				r.MatVec(a, x)
			}
		})
	}
}
