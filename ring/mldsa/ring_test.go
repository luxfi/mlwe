// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package mldsa

import (
	"errors"
	"math/rand"
	"testing"

	"github.com/luxfi/mlwe"
)

const q = mldsaQ

func randPoly(r *Ring, rng *rand.Rand) mlwe.Poly {
	p := r.NewPoly()
	for i := range p.Coeffs {
		p.Coeffs[i] = uint64(rng.Int63n(q))
	}
	return p
}

// schoolbookMul computes a*b in R_q = Z_q[X]/(X^256+1) by the negacyclic
// definition (X^256 = -1), a reference independent of the NTT.
func schoolbookMul(a, b []uint64) []uint64 {
	n := len(a)
	acc := make([]int64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			prod := int64(a[i]) * int64(b[j]) % int64(q)
			if k := i + j; k < n {
				acc[k] = (acc[k] + prod) % int64(q)
			} else {
				acc[k-n] = (acc[k-n] - prod) % int64(q)
			}
		}
	}
	out := make([]uint64, n)
	for k, v := range acc {
		v %= int64(q)
		if v < 0 {
			v += int64(q)
		}
		out[k] = uint64(v)
	}
	return out
}

func TestNew_RejectsBadProfile(t *testing.T) {
	bad := mlwe.Profile{Name: "x", N: 512, Q: q, Gamma2: mldsaGamma2P65}
	if _, err := New(bad); !errors.Is(err, ErrProfile) {
		t.Fatalf("wrong N: want ErrProfile, got %v", err)
	}
	bad = mlwe.Profile{Name: "x", N: 256, Q: 12289, Gamma2: mldsaGamma2P65}
	if _, err := New(bad); !errors.Is(err, ErrProfile) {
		t.Fatalf("wrong Q: want ErrProfile, got %v", err)
	}
	bad = mlwe.Profile{Name: "x", N: 256, Q: q, Gamma2: 7}
	if _, err := New(bad); !errors.Is(err, ErrProfile) {
		t.Fatalf("wrong Gamma2: want ErrProfile, got %v", err)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatalf("MustNew did not panic on bad profile")
			}
		}()
		MustNew(bad)
	}()
}

// TestNTT_MultiplicationRoundTrip is the canonical correctness invariant
// of the ring: INTT(MulNTT(NTT(a), NTT(b))) == a*b, checked against the
// independent negacyclic schoolbook product.
func TestNTT_MultiplicationRoundTrip(t *testing.T) {
	r := MustNew(Profile65)
	rng := rand.New(rand.NewSource(7))
	for iter := 0; iter < 64; iter++ {
		a := randPoly(r, rng)
		b := randPoly(r, rng)
		ah, bh := a.Clone(), b.Clone()
		r.NTT(ah)
		r.NTT(bh)
		prod := r.NewPoly()
		r.MulNTT(prod, ah, bh)
		r.INTT(prod)
		want := schoolbookMul(a.Coeffs, b.Coeffs)
		for i := range want {
			if prod.Coeffs[i] != want[i] {
				t.Fatalf("iter %d coeff %d: NTT mul %d != schoolbook %d", iter, i, prod.Coeffs[i], want[i])
			}
		}
	}
}

// TestNTT_InverseConvention documents and pins the Montgomery
// convention: with the verbatim constant-time core, the BARE round-trip
// INTT(NTT(p)) is NOT p but a FIXED scalar multiple c*p mod q. The test
// derives c from the data and asserts it is the same constant for every
// coefficient and across independent polynomials, then pins it. This is
// the factor that the multiplication round-trip cancels.
func TestNTT_InverseConvention(t *testing.T) {
	r := MustNew(Profile65)
	rng := rand.New(rand.NewSource(11))

	derive := func(seed int64) uint64 {
		p := randPoly(r, rng)
		// Ensure coeff 0 is invertible (non-zero) to derive c.
		p.Coeffs[0] = 12345
		work := p.Clone()
		r.NTT(work)
		r.INTT(work)
		c := (work.Coeffs[0] * modInvU64(p.Coeffs[0], q)) % q
		for i := range p.Coeffs {
			if work.Coeffs[i] != (p.Coeffs[i]*c)%q {
				t.Fatalf("INTT(NTT(p)) is not a uniform scalar multiple at coeff %d", i)
			}
		}
		return c
	}

	c := derive(1)
	// Stable across independent polynomials.
	if c2 := derive(2); c2 != c {
		t.Fatalf("scalar not stable: %d vs %d", c, c2)
	}
	// Pin the exact constant: c = R = 2^32 mod q.
	if c != uint64(1<<32)%q {
		t.Fatalf("INTT(NTT(p)) scalar = %d, expected R = %d", c, uint64(1<<32)%q)
	}
	t.Logf("INTT(NTT(p)) == %d * p (mod q); R = 2^32 mod q = %d", c, uint64(1<<32)%q)
}

// modInvU64 is a^-1 mod m by Fermat (m prime), for test derivation only.
func modInvU64(a, m uint64) uint64 {
	res, base, exp := uint64(1), a%m, m-2
	for exp > 0 {
		if exp&1 == 1 {
			res = (res * base) % m
		}
		base = (base * base) % m
		exp >>= 1
	}
	return res
}

func TestAddSub(t *testing.T) {
	r := MustNew(Profile65)
	rng := rand.New(rand.NewSource(3))
	a, b := randPoly(r, rng), randPoly(r, rng)
	sum := r.NewPoly()
	r.Add(sum, a, b)
	diff := r.NewPoly()
	r.Sub(diff, a, b)
	for i := range a.Coeffs {
		if sum.Coeffs[i] != (a.Coeffs[i]+b.Coeffs[i])%q {
			t.Fatalf("Add coeff %d", i)
		}
		want := (a.Coeffs[i] + q - b.Coeffs[i]) % q
		if diff.Coeffs[i] != want {
			t.Fatalf("Sub coeff %d: %d != %d", i, diff.Coeffs[i], want)
		}
	}
	// Add then Sub recovers a.
	back := r.NewPoly()
	r.Sub(back, sum, b)
	for i := range a.Coeffs {
		if back.Coeffs[i] != a.Coeffs[i] {
			t.Fatalf("(a+b)-b coeff %d: %d != %d", i, back.Coeffs[i], a.Coeffs[i])
		}
	}
}

// TestMatVec_VsSchoolbook checks the ring-generic MatVec (reached
// through Ring.MatVec) against a schoolbook K-by-L matrix-vector product.
func TestMatVec_VsSchoolbook(t *testing.T) {
	r := MustNew(Profile65)
	rng := rand.New(rand.NewSource(99))
	const K, L = 6, 5
	a := make(mlwe.PolyMat, K)
	for i := 0; i < K; i++ {
		a[i] = make(mlwe.PolyVec, L)
		for j := 0; j < L; j++ {
			a[i][j] = randPoly(r, rng)
		}
	}
	x := make(mlwe.PolyVec, L)
	for j := 0; j < L; j++ {
		x[j] = randPoly(r, rng)
	}
	got := r.MatVec(a, x)
	if len(got) != K {
		t.Fatalf("MatVec rows %d != %d", len(got), K)
	}
	for i := 0; i < K; i++ {
		want := make([]uint64, mldsaN)
		for j := 0; j < L; j++ {
			term := schoolbookMul(a[i][j].Coeffs, x[j].Coeffs)
			for c := range want {
				want[c] = (want[c] + term[c]) % q
			}
		}
		for c := range want {
			if got[i].Coeffs[c] != want[c] {
				t.Fatalf("row %d coeff %d: %d != %d", i, c, got[i].Coeffs[c], want[c])
			}
		}
	}
}

// TestDecompose_Recompose checks the FIPS 204 Decompose identity for all
// three rounding parameter sets: r == high*(2*gamma2) + centered(low).
func TestDecompose_Recompose(t *testing.T) {
	for _, prof := range []mlwe.Profile{Profile44, Profile65, Profile87} {
		r := MustNew(prof)
		g2 := uint64(prof.Gamma2)
		rng := rand.New(rand.NewSource(int64(prof.Gamma2)))
		p := randPoly(r, rng)
		low, high := r.Decompose(p)
		hb := r.HighBits(p)
		for i := range p.Coeffs {
			if hb.Coeffs[i] != high.Coeffs[i] {
				t.Fatalf("%s coeff %d: HighBits %d != Decompose high %d", prof.Name, i, hb.Coeffs[i], high.Coeffs[i])
			}
			centeredLow := int64(low.Coeffs[i])
			if low.Coeffs[i] > (q-1)/2 {
				centeredLow -= q
			}
			rec := (int64(high.Coeffs[i])*int64(2*g2) + centeredLow) % int64(q)
			if rec < 0 {
				rec += int64(q)
			}
			if uint64(rec) != p.Coeffs[i] {
				t.Fatalf("%s coeff %d: recompose %d != %d", prof.Name, i, rec, p.Coeffs[i])
			}
		}
	}
}

// TestUseHint_RoundTrip checks the FIPS 204 hint correctness property:
// for a true commitment w and a small perturbation e, a one-bit hint
// recovers HighBits(w) from the perturbed value. UseHint(0, w) is also
// HighBits(w).
func TestUseHint_RoundTrip(t *testing.T) {
	r := MustNew(Profile65)
	rng := rand.New(rand.NewSource(5))
	w := randPoly(r, rng)
	w1 := r.HighBits(w)

	// UseHint(0, w) == HighBits(w).
	zeroHint := r.NewPoly()
	if got := r.UseHint(zeroHint, w); !eqPoly(got, w1) {
		t.Fatalf("UseHint(0, w) != HighBits(w)")
	}

	// Perturb by a small e in [-8, 8]; build the hint from the high-bit
	// change and confirm UseHint recovers w1.
	wp := r.NewPoly()
	hint := r.NewPoly()
	for i := range w.Coeffs {
		e := int64(rng.Intn(17) - 8)
		v := (int64(w.Coeffs[i]) + e) % int64(q)
		if v < 0 {
			v += int64(q)
		}
		wp.Coeffs[i] = uint64(v)
	}
	wphb := r.HighBits(wp)
	for i := range hint.Coeffs {
		if wphb.Coeffs[i] != w1.Coeffs[i] {
			hint.Coeffs[i] = 1
		}
	}
	got := r.UseHint(hint, wp)
	if !eqPoly(got, w1) {
		for i := range got.Coeffs {
			if got.Coeffs[i] != w1.Coeffs[i] {
				t.Fatalf("UseHint coeff %d: got %d want %d (wp=%d w=%d)", i, got.Coeffs[i], w1.Coeffs[i], wp.Coeffs[i], w.Coeffs[i])
			}
		}
	}
}

func TestPower2Round_Recompose(t *testing.T) {
	r := MustNew(Profile65)
	rng := rand.New(rand.NewSource(13))
	p := randPoly(r, rng)
	r0, r1 := r.Power2Round(p)
	for i := range p.Coeffs {
		centered := int64(r0.Coeffs[i])
		if r0.Coeffs[i] > (q-1)/2 {
			centered -= q
		}
		rec := (int64(r1.Coeffs[i])<<mldsaD + centered) % int64(q)
		if rec < 0 {
			rec += int64(q)
		}
		if uint64(rec) != p.Coeffs[i] {
			t.Fatalf("coeff %d: recompose %d != %d", i, rec, p.Coeffs[i])
		}
	}
}

func TestInfNorm(t *testing.T) {
	r := MustNew(Profile65)
	p := r.NewPoly()
	// Centered magnitudes: 0->0, 1->1, q-1->1, (q-1)/2 -> (q-1)/2,
	// (q+1)/2 -> (q-1)/2.
	p.Coeffs[0] = 0
	p.Coeffs[1] = 5
	p.Coeffs[2] = q - 3 // centered -3 -> magnitude 3
	p.Coeffs[3] = (q - 1) / 2
	if got := r.InfNorm(p); got != (q-1)/2 {
		t.Fatalf("InfNorm = %d, want %d", got, (q-1)/2)
	}
	// All-zero poly -> 0.
	if got := r.InfNorm(r.NewPoly()); got != 0 {
		t.Fatalf("InfNorm(0) = %d", got)
	}
}

func TestCodec_RoundTrip(t *testing.T) {
	r := MustNew(Profile65)
	cdc := r.Codec()
	if cdc.N() != mldsaN {
		t.Fatalf("Codec.N = %d", cdc.N())
	}
	rng := rand.New(rand.NewSource(21))
	// 10-bit packing (the t1 width).
	p := r.NewPoly()
	for i := range p.Coeffs {
		p.Coeffs[i] = uint64(rng.Intn(1 << 10))
	}
	packed := cdc.Pack(p, 10)
	if len(packed) != mldsaN*10/8 {
		t.Fatalf("packed len %d", len(packed))
	}
	back := cdc.Unpack(packed, 10)
	if !eqPoly(back, p) {
		t.Fatalf("codec round-trip mismatch")
	}
}

func TestRingAccessors(t *testing.T) {
	r := MustNew(Profile65)
	if r.N() != 256 || r.Q() != q {
		t.Fatalf("N/Q wrong")
	}
	if r.Profile().Name != "ML-DSA-65" {
		t.Fatalf("profile name %q", r.Profile().Name)
	}
}

// TestDecompose_UnsupportedGamma2 covers the defensive guard in the
// verbatim decompose: an out-of-scope gamma2 returns (0, 0).
func TestDecompose_UnsupportedGamma2(t *testing.T) {
	if a0, a1 := decompose(100, 999); a0 != 0 || a1 != 0 {
		t.Fatalf("decompose(_, badGamma2) = (%d,%d), want (0,0)", a0, a1)
	}
}

// TestUseHint_BothDirections drives the per-coefficient useHint hint==1
// branches: r0 > 0 (bucket +1) and r0 <= 0 (bucket -1), plus hint == 0.
func TestUseHint_BothDirections(t *testing.T) {
	const g2 = mldsaGamma2P65
	const m = (q - 1) / (2 * g2) // 16 buckets
	// hint==0 returns the plain high bits.
	if got := useHint(0, 12345, g2); got != highBitsCoeff(12345, g2) {
		t.Fatalf("useHint(0,r) != HighBits(r)")
	}
	// r = 1: centered low r0 = 1 > 0  -> (r1+1) mod m.
	if got, want := useHint(1, 1, g2), (highBitsCoeff(1, g2)+1)%m; got != want {
		t.Fatalf("useHint(1, 1) = %d, want %d", got, want)
	}
	// r = 2*g2 - 1: centered low r0 = -1 <= 0 -> (r1+m-1) mod m.
	r := uint32(2*g2 - 1)
	if got, want := useHint(1, r, g2), (highBitsCoeff(r, g2)+m-1)%m; got != want {
		t.Fatalf("useHint(1, 2g2-1) = %d, want %d", got, want)
	}
}

func eqPoly(a, b mlwe.Poly) bool {
	if len(a.Coeffs) != len(b.Coeffs) {
		return false
	}
	for i := range a.Coeffs {
		if a.Coeffs[i] != b.Coeffs[i] {
			return false
		}
	}
	return true
}
