// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package mldsa

import (
	"errors"

	"github.com/luxfi/mlwe"
	"github.com/luxfi/mlwe/codec"
	"github.com/luxfi/mlwe/modulesis"
)

// FIPS 204 parameter profiles. N and Q are fixed; the modes differ in
// the module shape (K, L), the secret bound Eta, the challenge weight
// Tau, the mask range Gamma1, the rounding range Gamma2, and the hint
// bound Omega.
var (
	// Profile44 is ML-DSA-44 (NIST PQ Category 2).
	Profile44 = mlwe.Profile{
		Name: "ML-DSA-44", N: 256, Q: mldsaQ,
		K: 4, L: 4, Eta: 2, Tau: 39,
		Gamma1: 1 << 17, Gamma2: mldsaGamma2P44, Omega: 80,
	}
	// Profile65 is ML-DSA-65 (NIST PQ Category 3) — the Lux
	// consensus production target.
	Profile65 = mlwe.Profile{
		Name: "ML-DSA-65", N: 256, Q: mldsaQ,
		K: 6, L: 5, Eta: 4, Tau: 49,
		Gamma1: 1 << 19, Gamma2: mldsaGamma2P65, Omega: 55,
	}
	// Profile87 is ML-DSA-87 (NIST PQ Category 5).
	Profile87 = mlwe.Profile{
		Name: "ML-DSA-87", N: 256, Q: mldsaQ,
		K: 8, L: 7, Eta: 2, Tau: 60,
		Gamma1: 1 << 19, Gamma2: mldsaGamma2P65, Omega: 75,
	}
)

// ErrProfile is returned by New for a profile that is not a FIPS 204
// ML-DSA ring (wrong degree or modulus, or an unsupported Gamma2).
var ErrProfile = errors.New("mldsa: profile is not a FIPS 204 ML-DSA ring")

// Ring is the FIPS 204 ML-DSA Module-LWE ring bound to one parameter
// profile. It implements mlwe.Ring and mlwe.RoundingRing. A Ring is a
// value with no mutable state; it is safe for concurrent use.
type Ring struct {
	profile mlwe.Profile
	gamma2  uint32
	cdc     ringCodec
}

// New returns the ML-DSA ring for the given profile. The profile must
// have N == 256, Q == 8 380 417, and a supported Gamma2 (the FIPS 204
// (q-1)/32 or (q-1)/88).
func New(p mlwe.Profile) (*Ring, error) {
	if p.N != mldsaN || p.Q != mldsaQ {
		return nil, ErrProfile
	}
	if p.Gamma2 != mldsaGamma2P65 && p.Gamma2 != mldsaGamma2P44 {
		return nil, ErrProfile
	}
	return &Ring{profile: p, gamma2: p.Gamma2, cdc: ringCodec{}}, nil
}

// MustNew is the panic-on-error New, for package-level ring values.
func MustNew(p mlwe.Profile) *Ring {
	r, err := New(p)
	if err != nil {
		panic(err)
	}
	return r
}

// Profile returns the ring's parameter profile.
func (r *Ring) Profile() mlwe.Profile { return r.profile }

// N returns the ring degree (256).
func (r *Ring) N() int { return mldsaN }

// Q returns the modulus (8 380 417).
func (r *Ring) Q() uint64 { return mldsaQ }

// NewPoly returns a freshly allocated zero polynomial.
func (r *Ring) NewPoly() mlwe.Poly { return mlwe.Poly{Coeffs: make([]uint64, mldsaN)} }

// Codec returns the LSB-first bit codec for this ring's degree.
func (r *Ring) Codec() mlwe.Codec { return r.cdc }

// toInternal narrows an mlwe.Poly into the fast [256]uint32 core. Values
// are < 18q < 2^28 in every domain we handle, so the cast is lossless.
func toInternal(p mlwe.Poly) poly {
	var out poly
	for i := 0; i < mldsaN && i < len(p.Coeffs); i++ {
		out[i] = uint32(p.Coeffs[i])
	}
	return out
}

// writeBack copies a core poly into an mlwe.Poly's coefficient slice.
// dst must have N coefficients (every dst here is either a caller poly
// from NewPoly or one allocated by NewPoly inside this package).
func writeBack(src *poly, dst mlwe.Poly) {
	for i := 0; i < mldsaN; i++ {
		dst.Coeffs[i] = uint64(src[i])
	}
}

// NTT transforms p in place: coefficient domain -> NTT domain.
func (r *Ring) NTT(p mlwe.Poly) {
	c := toInternal(p)
	c.ntt()
	writeBack(&c, p)
}

// INTT transforms p in place: NTT domain -> coefficient domain,
// normalized into [0, q).
//
// Convention: with the verbatim constant-time Montgomery core, the bare
// round-trip INTT(NTT(p)) equals R*p mod q, NOT p. The contract callers
// rely on is the multiplication round-trip
// INTT(MulNTT(NTT(a), NTT(b))) == a*b, where the inverse-NTT's R factor
// cancels the Montgomery product's R^-1 factor. See ring_test.go.
func (r *Ring) INTT(p mlwe.Poly) {
	c := toInternal(p)
	// The forward NTT leaves coefficients unreduced (< 18q); the inverse
	// NTT's additive branch would overflow uint32 on inputs that large,
	// so reduce to < 2q first (the range the verbatim invNTT expects).
	c.reduceLe2Q()
	c.invNTT()
	c.normalize()
	writeBack(&c, p)
}

// Add sets dst = a + b, reduced into [0, q).
func (r *Ring) Add(dst, a, b mlwe.Poly) {
	ca, cb := toInternal(a), toInternal(b)
	var out poly
	out.add(&ca, &cb)
	out.normalize()
	writeBack(&out, dst)
}

// Sub sets dst = a - b, reduced into [0, q). Inputs are reduced to < 2q
// first so the verbatim sub (which adds 2q-b) stays in range.
func (r *Ring) Sub(dst, a, b mlwe.Poly) {
	ca, cb := toInternal(a), toInternal(b)
	ca.reduceLe2Q()
	cb.reduceLe2Q()
	var out poly
	out.sub(&ca, &cb)
	out.normalize()
	writeBack(&out, dst)
}

// MulNTT sets dst = a o b, the coefficient-wise Montgomery product of
// two NTT-domain operands. Inputs are reduced to < 2q first so the
// Montgomery reduction precondition (a[i]*b[i] < q*2^32) holds for any
// NTT-domain input.
func (r *Ring) MulNTT(dst, a, b mlwe.Poly) {
	ca, cb := toInternal(a), toInternal(b)
	ca.reduceLe2Q()
	cb.reduceLe2Q()
	var out poly
	out.mulHat(&ca, &cb)
	writeBack(&out, dst)
}

// MatVec returns A*x over R_q for coefficient-domain A and x, result in
// the coefficient domain. Delegates to the ring-generic implementation
// in modulesis so there is one matrix-vector product across all rings.
func (r *Ring) MatVec(a mlwe.PolyMat, x mlwe.PolyVec) mlwe.PolyVec {
	return modulesis.MatVec(r, a, x)
}

// InfNorm returns the centered infinity norm of p. Each coefficient is
// reduced into [0, q) (constant-time) and its centered magnitude
// min(c, q-c) is taken; the maximum is returned. Branchless throughout.
func (r *Ring) InfNorm(p mlwe.Poly) uint64 {
	const half = (mldsaQ - 1) / 2
	var best uint32
	for _, c64 := range p.Coeffs {
		c := modQ(uint32(c64))
		// Centered magnitude min(c, q-c) via the FIPS exceeds trick.
		x := int32(half) - int32(c)
		x ^= x >> 31
		mag := uint32(half) - uint32(x)
		// Branchless best = max(best, mag); both < 2^23.
		diff := int32(mag) - int32(best)
		sel := uint32(diff >> 31) // 0 if mag>=best else 0xFFFFFFFF
		best = (mag &^ sel) | (best & sel)
	}
	return uint64(best)
}

// Power2Round splits each coefficient r = r1*2^D + r0 with r0 centered;
// it returns (r0 mod q in [0,q), r1).
func (r *Ring) Power2Round(p mlwe.Poly) (r0, r1 mlwe.Poly) {
	c := toInternal(p)
	var p0, p1 poly
	for i := 0; i < mldsaN; i++ {
		a0plusQ, a1 := power2round(c[i])
		// a0plusQ = q + a0 in (q-2^12, q+2^12]; reduce to [0,q).
		v := a0plusQ
		v -= mldsaQ
		v += uint32(int32(v)>>31) & mldsaQ
		p0[i] = v
		p1[i] = a1
	}
	r0, r1 = r.NewPoly(), r.NewPoly()
	writeBack(&p0, r0)
	writeBack(&p1, r1)
	return r0, r1
}

// Decompose splits each coefficient r = r1*(2*Gamma2) + r0 with r0
// centered; it returns (low = r0 mod q in [0,q), high = r1).
func (r *Ring) Decompose(p mlwe.Poly) (low, high mlwe.Poly) {
	c := toInternal(p)
	var lo, hi poly
	for i := 0; i < mldsaN; i++ {
		a0plusQ, a1 := decompose(modQ(c[i]), r.gamma2)
		v := a0plusQ
		v -= mldsaQ
		v += uint32(int32(v)>>31) & mldsaQ
		lo[i] = v
		hi[i] = a1
	}
	low, high = r.NewPoly(), r.NewPoly()
	writeBack(&lo, low)
	writeBack(&hi, high)
	return low, high
}

// HighBits returns the high part r1 of Decompose.
func (r *Ring) HighBits(p mlwe.Poly) mlwe.Poly {
	c := toInternal(p)
	var hi poly
	for i := 0; i < mldsaN; i++ {
		hi[i] = highBitsCoeff(modQ(c[i]), r.gamma2)
	}
	out := r.NewPoly()
	writeBack(&hi, out)
	return out
}

// UseHint applies one hint bit per coefficient to r, returning the
// corrected high part. hint coefficients must be 0 or 1.
func (r *Ring) UseHint(hint, rp mlwe.Poly) mlwe.Poly {
	ch := toInternal(hint)
	cr := toInternal(rp)
	var out poly
	for i := 0; i < mldsaN; i++ {
		out[i] = useHint(ch[i]&1, modQ(cr[i]), r.gamma2)
	}
	res := r.NewPoly()
	writeBack(&out, res)
	return res
}

// ringCodec implements mlwe.Codec for the ML-DSA ring degree, delegating
// to the generic LSB-first packer in package codec.
type ringCodec struct{}

func (ringCodec) N() int { return mldsaN }

func (ringCodec) Pack(p mlwe.Poly, bits int) []byte {
	return codec.PackBits(p.Coeffs, bits)
}

func (ringCodec) Unpack(data []byte, bits int) mlwe.Poly {
	return mlwe.Poly{Coeffs: codec.UnpackBits(data, mldsaN, bits)}
}

// Static interface checks.
var (
	_ mlwe.Ring         = (*Ring)(nil)
	_ mlwe.RoundingRing = (*Ring)(nil)
	_ mlwe.Codec        = ringCodec{}
)
