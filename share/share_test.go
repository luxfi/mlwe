// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package share

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"math/big"
	mrand "math/rand"
	"testing"
)

func gf257(t *testing.T) Field {
	f, err := NewPrimeField(257)
	if err != nil {
		t.Fatalf("NewPrimeField(257): %v", err)
	}
	return f
}

func TestNewPrimeField_Validation(t *testing.T) {
	if _, err := NewPrimeField(2); !errors.Is(err, ErrModulusTooSmall) {
		t.Fatalf("p=2 want ErrModulusTooSmall, got %v", err)
	}
	if _, err := NewPrimeField(256); !errors.Is(err, ErrModulusNotPrime) {
		t.Fatalf("p=256 want ErrModulusNotPrime, got %v", err)
	}
	if _, err := NewPrimeField(9); !errors.Is(err, ErrModulusNotPrime) {
		t.Fatalf("p=9 want ErrModulusNotPrime, got %v", err)
	}
	// 2^63 and above is out of range: Add/Sub would overflow uint64.
	if _, err := NewPrimeField(1 << 63); !errors.Is(err, ErrModulusTooLarge) {
		t.Fatalf("p=2^63 want ErrModulusTooLarge, got %v", err)
	}
	// 2^32+1 is now in range (it was rejected as too large at v0.1.0) but
	// is composite (Fermat F5 = 641 * 6700417).
	if _, err := NewPrimeField((1 << 32) + 1); !errors.Is(err, ErrModulusNotPrime) {
		t.Fatalf("p=2^32+1 want ErrModulusNotPrime, got %v", err)
	}
	// FIPS-204 prime.
	if _, err := NewPrimeField(8380417); err != nil {
		t.Fatalf("p=8380417 want ok, got %v", err)
	}
	// Corona's 48-bit NTT prime — the field this change unblocks.
	if _, err := NewPrimeField(0x1000000004A01); err != nil {
		t.Fatalf("p=coronaQ want ok, got %v", err)
	}
	// Largest prime below the 2^63 ceiling: exercises isPrime + mulmod at
	// the very top of the supported range.
	if _, err := NewPrimeField((1 << 63) - 25); err != nil {
		t.Fatalf("p=2^63-25 want ok, got %v", err)
	}
}

// TestFieldArithmetic checks the field axioms used by sharing.
func TestFieldArithmetic(t *testing.T) {
	for _, f := range []Field{gf257(t), MLDSAField, CoronaField} {
		p := f.Modulus()
		// Inverse: a * a^-1 == 1 for a range of a.
		for _, a := range []uint64{1, 2, 3, p - 1, p / 2, 12345 % p} {
			if a == 0 {
				continue
			}
			if got := f.Mul(a, f.Inv(a)); got != 1 {
				t.Fatalf("p=%d: %d * inv = %d, want 1", p, a, got)
			}
		}
		// Add/Sub inverse relationship.
		if f.Sub(f.Add(10, 20), 20) != 10 {
			t.Fatalf("p=%d: (10+20)-20 != 10", p)
		}
		// Reduce wraps.
		if f.Reduce(p) != 0 || f.Reduce(p+5) != 5 {
			t.Fatalf("p=%d: Reduce wrong", p)
		}
		// Sub underflow path.
		if f.Sub(1, 2) != p-1 {
			t.Fatalf("p=%d: 1-2 != p-1", p)
		}
		// Inv(0) defined as 0.
		if f.Inv(0) != 0 {
			t.Fatalf("p=%d: Inv(0) != 0", p)
		}
	}
}

// TestSplitReconstruct_Identity proves Reconstruct(Split(s)) == s for a
// spread of (t, n) and both fields, recovering from a non-trivial
// quorum.
func TestSplitReconstruct_Identity(t *testing.T) {
	for _, f := range []Field{gf257(t), MLDSAField, CoronaField} {
		for _, tn := range [][2]int{{1, 1}, {2, 3}, {3, 5}, {5, 9}} {
			th, n := tn[0], tn[1]
			secret := make([]uint64, 32)
			for i := range secret {
				secret[i] = f.Reduce(uint64(i*7 + 11))
			}
			pts := make([]uint64, n)
			for i := range pts {
				pts[i] = uint64(i + 1)
			}
			shares, err := Split(secret, th, pts, f, rand.Reader)
			if err != nil {
				t.Fatalf("Split: %v", err)
			}
			// Reconstruct from the last t shares (an arbitrary quorum).
			got, err := Reconstruct(shares[n-th:], f)
			if err != nil {
				t.Fatalf("Reconstruct: %v", err)
			}
			if !equalU64(got, secret) {
				t.Fatalf("p=%d t=%d n=%d: secret mismatch", f.Modulus(), th, n)
			}
		}
	}
}

// TestReconstruct_AnyQuorum confirms every t-subset reconstructs to the
// same secret.
func TestReconstruct_AnyQuorum(t *testing.T) {
	for _, f := range []Field{MLDSAField, CoronaField} {
		const th, n = 3, 5
		// Include a near-modulus slot so Corona exercises the wide multiply.
		secret := []uint64{111, 222, f.Modulus() - 1}
		pts := []uint64{1, 2, 3, 4, 5}
		shares, err := Split(secret, th, pts, f, rand.Reader)
		if err != nil {
			t.Fatalf("p=%d Split: %v", f.Modulus(), err)
		}
		// All C(5,3) = 10 quorums.
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				for k := j + 1; k < n; k++ {
					q := []Share{shares[i], shares[j], shares[k]}
					got, err := Reconstruct(q, f)
					if err != nil {
						t.Fatalf("p=%d Reconstruct {%d,%d,%d}: %v", f.Modulus(), i, j, k, err)
					}
					if !equalU64(got, secret) {
						t.Fatalf("p=%d quorum {%d,%d,%d} = %v != %v", f.Modulus(), i, j, k, got, secret)
					}
				}
			}
		}
	}
}

// TestPrivacy_TMinus1IndependentOfSecret is the security property: any
// t-1 shares are consistent with EVERY secret. Concretely, fix t-1
// shares H; completing them with two different values at a fresh t-th
// point reconstructs to two different secrets. Hence H alone fixes
// nothing about the secret.
func TestPrivacy_TMinus1IndependentOfSecret(t *testing.T) {
	for _, f := range []Field{MLDSAField, CoronaField} {
		const th = 4
		// Build t-1 = 3 fixed shares at points 1,2,3 (single-slot secret).
		H := []Share{
			{X: 1, Y: []uint64{1000}},
			{X: 2, Y: []uint64{2000}},
			{X: 3, Y: []uint64{3000}},
		}
		xt := uint64(4)
		q0 := append(append([]Share{}, H...), Share{X: xt, Y: []uint64{5000}})
		q1 := append(append([]Share{}, H...), Share{X: xt, Y: []uint64{9999}})
		if len(q0) != th || len(q1) != th {
			t.Fatalf("quorum size wrong")
		}
		s0, err := Reconstruct(q0, f)
		if err != nil {
			t.Fatalf("p=%d Reconstruct q0: %v", f.Modulus(), err)
		}
		s1, err := Reconstruct(q1, f)
		if err != nil {
			t.Fatalf("p=%d Reconstruct q1: %v", f.Modulus(), err)
		}
		// Same H, different t-th share -> different secret. So H carries no
		// information about the secret.
		if s0[0] == s1[0] {
			t.Fatalf("p=%d: t-1 shares determined the secret: both quorums gave %d", f.Modulus(), s0[0])
		}
	}
}

// TestLagrange_EvalsPolynomial confirms the basis interpolates a known
// polynomial both at 0 and at arbitrary points.
func TestLagrange_EvalsPolynomial(t *testing.T) {
	f := gf257(t)
	// P(x) = 3 + 5x + 2x^2 over GF(257).
	poly := func(x uint64) uint64 {
		return f.Add(f.Add(3, f.Mul(5, x)), f.Mul(2, f.Mul(x, x)))
	}
	xs := []uint64{1, 2, 3}
	ys := make([]uint64, len(xs))
	for i, x := range xs {
		ys[i] = poly(x)
	}
	for _, atX := range []uint64{0, 4, 7, 100} {
		lam, err := Lagrange(xs, atX, f)
		if err != nil {
			t.Fatalf("Lagrange: %v", err)
		}
		var got uint64
		for i := range xs {
			got = f.Add(got, f.Mul(lam[i], ys[i]))
		}
		if got != poly(atX) {
			t.Fatalf("interp at %d = %d, want %d", atX, got, poly(atX))
		}
	}
}

func TestSplit_Errors(t *testing.T) {
	f := gf257(t)
	secret := []uint64{1, 2}
	if _, err := Split(nil, 2, []uint64{1, 2}, f, rand.Reader); !errors.Is(err, ErrEmptySecret) {
		t.Fatalf("empty secret: %v", err)
	}
	if _, err := Split(secret, 3, []uint64{1, 2}, f, rand.Reader); !errors.Is(err, ErrThreshold) {
		t.Fatalf("t>n: %v", err)
	}
	if _, err := Split(secret, 0, []uint64{1, 2}, f, rand.Reader); !errors.Is(err, ErrThreshold) {
		t.Fatalf("t<1: %v", err)
	}
	if _, err := Split(secret, 2, []uint64{0, 1}, f, rand.Reader); !errors.Is(err, ErrZeroPoint) {
		t.Fatalf("zero point: %v", err)
	}
	if _, err := Split(secret, 2, []uint64{1, 1}, f, rand.Reader); !errors.Is(err, ErrDuplicatePoint) {
		t.Fatalf("dup point: %v", err)
	}
	// Short randomness source -> error from readFieldElement.
	if _, err := Split(secret, 2, []uint64{1, 2}, f, bytes.NewReader([]byte{0x01})); err == nil {
		t.Fatalf("short rand: expected error")
	}
}

func TestReconstruct_Errors(t *testing.T) {
	f := gf257(t)
	if _, err := Reconstruct(nil, f); !errors.Is(err, ErrNotEnoughShares) {
		t.Fatalf("empty: %v", err)
	}
	// Inconsistent slot counts.
	bad := []Share{{X: 1, Y: []uint64{1, 2}}, {X: 2, Y: []uint64{1}}}
	if _, err := Reconstruct(bad, f); err == nil {
		t.Fatalf("inconsistent slots: expected error")
	}
	// Duplicate eval point reaches Lagrange's checkPoints.
	dup := []Share{{X: 1, Y: []uint64{1}}, {X: 1, Y: []uint64{2}}}
	if _, err := Reconstruct(dup, f); !errors.Is(err, ErrDuplicatePoint) {
		t.Fatalf("dup point: %v", err)
	}
}

func TestLagrange_Errors(t *testing.T) {
	f := gf257(t)
	if _, err := Lagrange([]uint64{0, 1}, 5, f); !errors.Is(err, ErrZeroPoint) {
		t.Fatalf("zero point: %v", err)
	}
	if _, err := Lagrange([]uint64{2, 2}, 5, f); !errors.Is(err, ErrDuplicatePoint) {
		t.Fatalf("dup point: %v", err)
	}
}

// TestIsPrime exercises the Miller-Rabin branches directly. The witness
// set is {2,3,5,7,11,13,17,19,23,29,31,37}, deterministic well past 2^63.
func TestIsPrime(t *testing.T) {
	// 97, 193 have n-1 divisible by a high power of 2, so a witness reaches
	// n-1 only after one or more squarings — exercising the inner
	// Miller-Rabin loop's success break. coronaQ, 2^61-1 (M61) and 2^63-25
	// drive a real large prime through the wide-mulmod squaring path.
	primes := []uint64{3, 5, 7, 11, 13, 37, 61, 97, 193, 257, 7919, 8380417,
		2147483647, 4294967291, coronaQ, (1 << 61) - 1, (1 << 63) - 25}
	for _, p := range primes {
		if !isPrime(p) {
			t.Fatalf("isPrime(%d) = false, want true", p)
		}
	}
	// 1763 = 41*43 and 1000003*1000033 survive trial division by the
	// witness set (all factors > 37) and must be rejected by a Miller-Rabin
	// round, exercising the composite-return path over both small and wide
	// moduli.
	composites := []uint64{0, 1, 4, 15, 25, 256, 561, 1105, 1763,
		8380418, 2147483645, 1000003 * 1000033}
	for _, c := range composites {
		if isPrime(c) {
			t.Fatalf("isPrime(%d) = true, want false", c)
		}
	}
}

// TestCorona_ModulusGroundTruth pins CoronaField to the value used by the
// Corona signer (sign/config.go: Q = 0x1000000004A01) and verifies the
// field invariants the share layer relies on. It also guards against the
// 2^48+1 transcription error (that value is composite: 65537 * 4294901761).
func TestCorona_ModulusGroundTruth(t *testing.T) {
	const want = 0x1000000004A01
	if CoronaField.Modulus() != want {
		t.Fatalf("CoronaField modulus = %#x, want %#x", CoronaField.Modulus(), uint64(want))
	}
	if want != (uint64(1)<<48)+0x4A01 {
		t.Fatalf("Corona modulus != 2^48 + 0x4A01")
	}
	if want == (uint64(1)<<48)+1 {
		t.Fatalf("Corona modulus must not be 2^48+1 (composite)")
	}
	q := new(big.Int).SetUint64(want)
	if !q.ProbablyPrime(64) {
		t.Fatalf("Corona modulus %#x is not prime", uint64(want))
	}
	if !isPrime(want) {
		t.Fatalf("package isPrime rejected the Corona prime %#x", uint64(want))
	}
	// q ≡ 1 (mod 2N), N = 256: required for the negacyclic NTT.
	if want%512 != 1 {
		t.Fatalf("Corona modulus not ≡ 1 (mod 512): %d", want%512)
	}
}

// TestMulmod_BigIntCrossCheck validates the 128-bit mulmod against a
// math/big reference over edge operands (clustered at 0/1/2 and just below
// p) and many randomized pairs, for moduli spanning the supported range:
// GF(257), the FIPS-204 prime, Corona's 48-bit prime, and the largest
// prime below the 2^63 ceiling.
func TestMulmod_BigIntCrossCheck(t *testing.T) {
	for _, p := range []uint64{257, 8380417, coronaQ, (1 << 63) - 25} {
		bp := new(big.Int).SetUint64(p)
		ref := func(a, b uint64) uint64 {
			prod := new(big.Int).Mul(new(big.Int).SetUint64(a), new(big.Int).SetUint64(b))
			return prod.Mod(prod, bp).Uint64()
		}
		edges := []uint64{0, 1, 2, 3, p / 2, p - 3, p - 2, p - 1}
		for _, a := range edges {
			for _, b := range edges {
				if got := mulmod(a, b, p); got != ref(a, b) {
					t.Fatalf("p=%d mulmod(%d,%d)=%d want %d", p, a, b, got, ref(a, b))
				}
			}
		}
		rng := mrand.New(mrand.NewSource(int64(p) ^ 0x5DEECE66D))
		for i := 0; i < 20000; i++ {
			a, b := rng.Uint64()%p, rng.Uint64()%p
			got, exp := mulmod(a, b, p), ref(a, b)
			if got != exp {
				t.Fatalf("p=%d mulmod(%d,%d)=%d want %d", p, a, b, got, exp)
			}
			// The public Field.Mul must agree where the field is constructible.
			if p != (1<<63)-25 {
				var f Field
				switch p {
				case 257:
					f = mustField(t, 257)
				case 8380417:
					f = MLDSAField
				case coronaQ:
					f = CoronaField
				}
				if fm := f.Mul(a, b); fm != exp {
					t.Fatalf("p=%d Field.Mul(%d,%d)=%d want %d", p, a, b, fm, exp)
				}
			}
		}
	}
}

// TestLagrange_Corona exercises Lagrange interpolation over the 48-bit
// field with near-modulus coefficients, at X=0 (the reconstruction case)
// and at non-zero X, so num/den/Inv/Mul all run through the wide path.
func TestLagrange_Corona(t *testing.T) {
	f := CoronaField
	p := f.Modulus()
	c0, c1, c2 := p-1, p-12345, p/2 // P(x) = c0 + c1 x + c2 x^2
	poly := func(x uint64) uint64 {
		return f.Add(f.Add(c0, f.Mul(c1, x)), f.Mul(c2, f.Mul(x, x)))
	}
	xs := []uint64{7, 99999, p - 1}
	ys := make([]uint64, len(xs))
	for i, x := range xs {
		ys[i] = poly(x)
	}
	for _, atX := range []uint64{0, 1, 424242, p - 2} {
		lam, err := Lagrange(xs, atX, f)
		if err != nil {
			t.Fatalf("Lagrange: %v", err)
		}
		var got uint64
		for i := range xs {
			got = f.Add(got, f.Mul(lam[i], ys[i]))
		}
		if got != poly(atX) {
			t.Fatalf("interp at %d = %d, want %d", atX, got, poly(atX))
		}
	}
}

// TestReadFieldElement_Rejection covers the unbiased sampler's rejection
// branch: an all-0xFF block decodes to MaxUint64, which is at or above the
// rejection limit for any odd prime, so it is discarded; the next in-range
// block is accepted. It also confirms exactly two blocks were consumed.
func TestReadFieldElement_Rejection(t *testing.T) {
	f := CoronaField
	p := f.Modulus()
	var small [8]byte
	binary.BigEndian.PutUint64(small[:], 12345)
	r := bytes.NewReader(append(bytes.Repeat([]byte{0xFF}, 8), small[:]...))
	got, err := readFieldElement(r, f)
	if err != nil {
		t.Fatalf("readFieldElement: %v", err)
	}
	if got != 12345%p {
		t.Fatalf("got %d, want %d", got, 12345%p)
	}
	if r.Len() != 0 {
		t.Fatalf("expected both 8-byte blocks consumed, %d bytes left", r.Len())
	}
}

func mustField(t *testing.T, p uint64) Field {
	t.Helper()
	f, err := NewPrimeField(p)
	if err != nil {
		t.Fatalf("NewPrimeField(%d): %v", p, err)
	}
	return f
}

func equalU64(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
