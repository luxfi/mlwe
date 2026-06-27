// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package mlwe is the shared Module-LWE base for the Lux post-quantum
// signature schemes.
//
// Both Pulsar (FIPS 204 / ML-DSA, q = 8 380 417, a 23-bit ring) and
// Corona (Ringtail/Raccoon, q = 0x1000000004A01, a 48-bit ring) are
// Module-LWE constructions over R_q = Z_q[X]/(X^256 + 1). They share
// the same module arithmetic, the same SHAKE-derived samplers, the
// same Shamir/Lagrange secret-sharing, the same SP 800-185 transcript
// hashing, and the same bit-packing codecs. This module factors that
// common surface out exactly once.
//
// # Layering
//
// This root package defines only VALUE TYPES (Poly, PolyVec, PolyMat,
// Profile) and small INTERFACES (Ring, RoundingRing, Codec). It has no
// internal dependencies. Concrete rings live under ring/; samplers
// under sample/; the module-level (Module-SIS) norm and rounding
// toolkit under modulesis/; secret sharing under share/; the hash
// transcript under transcript/; bit packing under codec/.
//
// # Coefficient width
//
// A Poly stores coefficients as uint64. This is the single
// representation that holds BOTH the 23-bit ML-DSA modulus and the
// 48-bit Corona modulus without truncation. A concrete Ring is free to
// narrow internally (ring/mldsa keeps a [256]uint32 fast core and
// converts at the interface boundary) so that byte-for-byte FIPS 204
// output and constant-time Montgomery arithmetic are preserved.
//
// # Domain of a Poly
//
// The Poly type does not tag whether its coefficients are in the
// coefficient domain or the NTT (evaluation) domain. The domain is a
// property of the value at a given point in a computation, tracked by
// the caller, exactly as in the reference FIPS 204 code. NTT moves a
// value coefficient->evaluation; INTT moves it back. MulNTT is only
// meaningful on evaluation-domain operands.
package mlwe

// Poly is an element of R_q = Z_q[X]/(X^N + 1) with N coefficients.
// Coeffs[i] is the coefficient of X^i, reduced into [0, q) in the
// coefficient domain (NTT-domain intermediates may be larger but stay
// well below 2^64 for every supported ring).
type Poly struct {
	Coeffs []uint64
}

// Clone returns a deep copy of p.
func (p Poly) Clone() Poly {
	c := make([]uint64, len(p.Coeffs))
	copy(c, p.Coeffs)
	return Poly{Coeffs: c}
}

// PolyVec is a module vector: an ordered list of ring elements. In
// Module-LWE a secret/error vector and a commitment are PolyVecs.
type PolyVec []Poly

// Clone returns a deep copy of v.
func (v PolyVec) Clone() PolyVec {
	out := make(PolyVec, len(v))
	for i := range v {
		out[i] = v[i].Clone()
	}
	return out
}

// PolyMat is a module matrix in row-major order: Mat[i] is row i, a
// PolyVec of length L (the column count). The public matrix A of a
// Module-LWE instance is a PolyMat.
type PolyMat []PolyVec

// Rows reports the number of rows (the module dimension K).
func (m PolyMat) Rows() int { return len(m) }

// Cols reports the number of columns (the secret dimension L), or 0 for
// an empty matrix.
func (m PolyMat) Cols() int {
	if len(m) == 0 {
		return 0
	}
	return len(m[0])
}

// Profile is the parameter profile of a concrete Module-LWE instance.
// It names the ring shape and the rounding parameters that the FIPS 204
// (and Ringtail) decomposition primitives are gated on. A Profile is a
// pure value: equal Profiles describe identical instances.
type Profile struct {
	// Name is the canonical human-readable identifier, e.g.
	// "ML-DSA-65".
	Name string

	// N is the ring degree (256 for every Lux Module-LWE instance).
	N int

	// Q is the prime modulus.
	Q uint64

	// K is the module dimension (row count of A).
	K int

	// L is the secret dimension (column count of A).
	L int

	// Eta is the secret/error coefficient bound for the centered
	// binomial sampler.
	Eta int

	// Tau is the challenge Hamming weight for SampleInBall.
	Tau int

	// Gamma1 is the mask coefficient bound (the ExpandMask range is
	// the open interval (-Gamma1, Gamma1]).
	Gamma1 uint32

	// Gamma2 is the low-order rounding range (alpha = 2*Gamma2). It
	// gates Decompose, HighBits and UseHint.
	Gamma2 uint32

	// Omega is the maximum hint Hamming weight.
	Omega int
}

// Ring is the irreducible per-instance polynomial arithmetic of a
// Module-LWE ring. It is deliberately small: every higher-order
// operation (matrix-vector products, vector norms, vector rounding)
// composes these primitives in the modulesis package and therefore
// works uniformly across rings.
//
// NTT and INTT use the implementation's native (Montgomery) convention.
// The contract a caller may rely on is the MULTIPLICATION round-trip:
//
//	INTT(MulNTT(NTT(a), NTT(b))) == a * b   in R_q
//
// The bare round-trip INTT(NTT(p)) is NOT promised to equal p: a
// constant-time Montgomery ring (such as ring/mldsa) leaves a fixed
// Montgomery factor on the bare round-trip that the multiplication path
// cancels. See ring/mldsa for the exact convention.
type Ring interface {
	// N returns the ring degree.
	N() int

	// Q returns the prime modulus.
	Q() uint64

	// NewPoly returns a freshly allocated zero polynomial with N
	// coefficients.
	NewPoly() Poly

	// NTT transforms p in place from the coefficient domain to the
	// NTT (evaluation) domain.
	NTT(p Poly)

	// INTT transforms p in place from the NTT domain back to the
	// coefficient domain.
	INTT(p Poly)

	// Add sets dst = a + b (coefficient-wise, reduced into [0, q)).
	// dst may alias a or b.
	Add(dst, a, b Poly)

	// Sub sets dst = a - b (coefficient-wise, reduced into [0, q)).
	// dst may alias a or b.
	Sub(dst, a, b Poly)

	// MulNTT sets dst = a o b, the coefficient-wise product of two
	// NTT-domain operands. dst may alias a or b.
	MulNTT(dst, a, b Poly)

	// MatVec returns A*x over R_q for a coefficient-domain matrix A
	// and coefficient-domain vector x, with the result in the
	// coefficient domain. Forward and inverse transforms are handled
	// internally.
	MatVec(a PolyMat, x PolyVec) PolyVec

	// InfNorm returns the centered infinity norm of p: the maximum
	// over coefficients of min(c, q-c). p must be in [0, q).
	InfNorm(p Poly) uint64

	// Codec returns the byte codec bound to this ring's degree.
	Codec() Codec
}

// RoundingRing is a Ring that also supports the FIPS 204 high/low-bit
// decomposition primitives. These are gated on the instance's Gamma2
// (and the Power2Round drop length D), so they live on the concrete
// ring rather than as free functions.
//
// Power2Round, Decompose, HighBits and UseHint all operate
// coefficient-wise and require their inputs reduced into [0, q). Their
// PolyVec liftings are in the modulesis package.
type RoundingRing interface {
	Ring

	// Power2Round splits each coefficient r = r1*2^D + r0 with r0 in
	// (-2^(D-1), 2^(D-1)], returning (r0, r1) with r0 represented as
	// r0 mod q in [0, q).
	Power2Round(p Poly) (r0, r1 Poly)

	// Decompose splits each coefficient r = r1*(2*Gamma2) + r0 with r0
	// centered in (-Gamma2, Gamma2], returning (low, high) with low =
	// r0 mod q in [0, q) and high = r1.
	Decompose(p Poly) (low, high Poly)

	// HighBits returns the high part r1 of Decompose.
	HighBits(p Poly) Poly

	// UseHint applies one FIPS 204 (Algorithm 40) hint bit per
	// coefficient to r, returning the corrected high part. hint
	// coefficients must be 0 or 1.
	UseHint(hint, r Poly) Poly
}

// Codec serializes ring elements to and from bytes using little-endian
// (LSB-first) bit packing, the FIPS 204 wire convention. A Codec is
// bound to a fixed ring degree N.
type Codec interface {
	// N returns the ring degree this codec packs.
	N() int

	// Pack packs the low `bits` bits of each of p's N coefficients
	// into ceil(N*bits/8) bytes, LSB-first.
	Pack(p Poly, bits int) []byte

	// Unpack is the inverse of Pack: it reads N coefficients of `bits`
	// bits each from data.
	Unpack(data []byte, bits int) Poly
}
