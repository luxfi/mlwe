// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package modulesis is the ring-generic module-level toolkit shared by
// every Lux Module-LWE instance. It composes the per-polynomial
// primitives of an mlwe.Ring (and the rounding primitives of an
// mlwe.RoundingRing) into the operations that act on module vectors and
// matrices: the matrix-vector product A*x, the vector norms used for
// Module-SIS bound checks, and the vector liftings of Power2Round,
// Decompose, HighBits and UseHint.
//
// Everything here is written against the mlwe interfaces, so it works
// unchanged for ML-DSA's 23-bit ring and Corona's 48-bit ring. There is
// exactly one implementation of each operation, reached either directly
// or through a ring's interface method (e.g. ring/mldsa.Ring.MatVec
// delegates to MatVec here).
package modulesis

import (
	"math/big"

	"github.com/luxfi/mlwe"
)

// MatVec returns A*x over R_q for a coefficient-domain matrix A (K rows
// by L columns) and a coefficient-domain vector x (length L), with the
// result (length K) in the coefficient domain. Forward and inverse NTTs
// are performed internally; inputs are not mutated.
//
// It panics if x's length does not match A's column count, since a
// dimension mismatch is a programmer error, not attacker input.
func MatVec(r mlwe.Ring, a mlwe.PolyMat, x mlwe.PolyVec) mlwe.PolyVec {
	k, l := a.Rows(), a.Cols()
	if len(x) != l {
		panic("mlwe/modulesis: MatVec dimension mismatch")
	}
	// Pre-transform the vector once.
	xhat := make(mlwe.PolyVec, l)
	for j := 0; j < l; j++ {
		xhat[j] = x[j].Clone()
		r.NTT(xhat[j])
	}
	out := make(mlwe.PolyVec, k)
	tmp := r.NewPoly()
	ahat := r.NewPoly()
	for i := 0; i < k; i++ {
		acc := r.NewPoly() // zero
		for j := 0; j < l; j++ {
			copy(ahat.Coeffs, a[i][j].Coeffs)
			r.NTT(ahat)
			r.MulNTT(tmp, ahat, xhat[j])
			r.Add(acc, acc, tmp)
		}
		r.INTT(acc)
		out[i] = acc
	}
	return out
}

// InfNorm returns the centered infinity norm of a module vector: the
// maximum centered coefficient magnitude across all of v's polynomials.
func InfNorm(r mlwe.Ring, v mlwe.PolyVec) uint64 {
	var best uint64
	for i := range v {
		if n := r.InfNorm(v[i]); n > best {
			best = n
		}
	}
	return best
}

// L2Sq returns the exact squared Euclidean norm of a module vector: the
// sum over every coefficient of its centered magnitude squared. The
// result is a big.Int so it is exact for any ring width and module
// dimension (no overflow, no rounding) — the form Module-SIS bound
// checks need, since ||v||_2 <= B iff L2Sq(v) <= B*B.
func L2Sq(r mlwe.Ring, v mlwe.PolyVec) *big.Int {
	q := r.Q()
	half := q / 2
	sum := new(big.Int)
	term := new(big.Int)
	for i := range v {
		for _, c64 := range v[i].Coeffs {
			c := c64 % q
			mag := c
			if c > half {
				mag = q - c
			}
			term.SetUint64(mag)
			term.Mul(term, term)
			sum.Add(sum, term)
		}
	}
	return sum
}

// Power2Round applies the ring's Power2Round coefficient-wise across a
// module vector, returning (v0, v1).
func Power2Round(r mlwe.RoundingRing, v mlwe.PolyVec) (v0, v1 mlwe.PolyVec) {
	v0 = make(mlwe.PolyVec, len(v))
	v1 = make(mlwe.PolyVec, len(v))
	for i := range v {
		v0[i], v1[i] = r.Power2Round(v[i])
	}
	return v0, v1
}

// Decompose applies the ring's Decompose coefficient-wise across a
// module vector, returning (low, high).
func Decompose(r mlwe.RoundingRing, v mlwe.PolyVec) (low, high mlwe.PolyVec) {
	low = make(mlwe.PolyVec, len(v))
	high = make(mlwe.PolyVec, len(v))
	for i := range v {
		low[i], high[i] = r.Decompose(v[i])
	}
	return low, high
}

// HighBits returns the high part of Decompose across a module vector.
func HighBits(r mlwe.RoundingRing, v mlwe.PolyVec) mlwe.PolyVec {
	out := make(mlwe.PolyVec, len(v))
	for i := range v {
		out[i] = r.HighBits(v[i])
	}
	return out
}

// UseHint applies the ring's UseHint coefficient-wise across a module
// vector. hint and v must have equal length.
func UseHint(r mlwe.RoundingRing, hint, v mlwe.PolyVec) mlwe.PolyVec {
	if len(hint) != len(v) {
		panic("mlwe/modulesis: UseHint dimension mismatch")
	}
	out := make(mlwe.PolyVec, len(v))
	for i := range v {
		out[i] = r.UseHint(hint[i], v[i])
	}
	return out
}

// HintWeight returns the total Hamming weight of a hint vector (the
// number of 1 coefficients), used to enforce the FIPS 204 Omega bound.
func HintWeight(hint mlwe.PolyVec) int {
	var w int
	for i := range hint {
		for _, c := range hint[i].Coeffs {
			if c&1 == 1 {
				w++
			}
		}
	}
	return w
}
