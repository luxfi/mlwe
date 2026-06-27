// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package share

import (
	"bytes"
	"crypto/rand"
	"errors"
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
	if _, err := NewPrimeField((1 << 32) + 1); !errors.Is(err, ErrModulusTooLarge) {
		t.Fatalf("p>2^32 want ErrModulusTooLarge, got %v", err)
	}
	if _, err := NewPrimeField(8380417); err != nil {
		t.Fatalf("p=q want ok, got %v", err)
	}
}

// TestFieldArithmetic checks the field axioms used by sharing.
func TestFieldArithmetic(t *testing.T) {
	for _, f := range []Field{gf257(t), MLDSAField} {
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
	for _, f := range []Field{gf257(t), MLDSAField} {
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
	f := MLDSAField
	const th, n = 3, 5
	secret := []uint64{111, 222, 333}
	pts := []uint64{1, 2, 3, 4, 5}
	shares, err := Split(secret, th, pts, f, rand.Reader)
	if err != nil {
		t.Fatalf("Split: %v", err)
	}
	// All C(5,3) = 10 quorums.
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			for k := j + 1; k < n; k++ {
				q := []Share{shares[i], shares[j], shares[k]}
				got, err := Reconstruct(q, f)
				if err != nil {
					t.Fatalf("Reconstruct {%d,%d,%d}: %v", i, j, k, err)
				}
				if !equalU64(got, secret) {
					t.Fatalf("quorum {%d,%d,%d} reconstructed %v != %v", i, j, k, got, secret)
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
	f := MLDSAField
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
		t.Fatalf("Reconstruct q0: %v", err)
	}
	s1, err := Reconstruct(q1, f)
	if err != nil {
		t.Fatalf("Reconstruct q1: %v", err)
	}
	// Same H, different t-th share -> different secret. So H carries no
	// information about the secret.
	if s0[0] == s1[0] {
		t.Fatalf("t-1 shares determined the secret: both quorums gave %d", s0[0])
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

// TestIsPrime exercises the Miller-Rabin branches directly.
func TestIsPrime(t *testing.T) {
	// 17, 97, 193 have n-1 divisible by a high power of 2, so a witness
	// reaches n-1 only after one or more squarings — exercising the inner
	// Miller-Rabin loop's success break.
	primes := []uint64{3, 5, 7, 11, 13, 17, 61, 97, 193, 257, 7919, 8380417, 2147483647, 4294967291}
	for _, p := range primes {
		if !isPrime(p) {
			t.Fatalf("isPrime(%d) = false, want true", p)
		}
	}
	// 323=17*19, 437=19*23, 1763=41*43 survive trial division by the
	// small-prime set and must be rejected by a Miller-Rabin witness.
	composites := []uint64{0, 1, 4, 15, 25, 256, 323, 437, 561, 1105, 1763, 8380418, 2147483645}
	for _, c := range composites {
		if isPrime(c) {
			t.Fatalf("isPrime(%d) = true, want false", c)
		}
	}
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
