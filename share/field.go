// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package share is the single Shamir secret-sharing and Lagrange
// interpolation surface for the Lux Module-LWE stack. It replaces the
// three field-specialized copies the Pulsar reference carried (GF(257)
// byte sharing, GF(q) wide sharing, and the large-committee wrapper)
// with one generic implementation over a Field interface.
//
// A secret is a vector of field elements ([]uint64); a Share is that
// vector evaluated at a non-zero point. Sharing a 32-byte seed is the
// special case secret = the 32 bytes lifted into the field.
package share

import (
	"errors"
	"math/bits"
)

// Field is a prime field GF(p). The implementation in this package
// supports any prime p in (2, 2^63): every product a*b is formed as a
// full 128-bit value and then reduced (see mulmod), so a single code
// path covers the whole supported modulus range — both the FIPS-204
// prime GF(8380417) and Corona's 48-bit NTT-friendly prime
// GF(0x1000000004A01).
type Field interface {
	// Modulus returns the prime p.
	Modulus() uint64
	// Reduce returns x mod p for any x.
	Reduce(x uint64) uint64
	// Add returns (a + b) mod p for a, b in [0, p).
	Add(a, b uint64) uint64
	// Sub returns (a - b) mod p for a, b in [0, p).
	Sub(a, b uint64) uint64
	// Mul returns (a * b) mod p for a, b in [0, p).
	Mul(a, b uint64) uint64
	// Inv returns a^-1 mod p for a in [1, p). Inv(0) is undefined and
	// returns 0.
	Inv(a uint64) uint64
}

// Field errors.
var (
	ErrModulusTooLarge = errors.New("mlwe/share: modulus must be < 2^63")
	ErrModulusNotPrime = errors.New("mlwe/share: modulus is not prime")
	ErrModulusTooSmall = errors.New("mlwe/share: modulus must be > 2")
)

// primeField is a GF(p) with p an odd prime in (2, 2^63). Every modular
// product goes through mulmod's 128-bit path, so the full supported range
// (FIPS-204 and Corona moduli included) is served by one path with no
// width-dependent branch.
type primeField struct{ p uint64 }

// NewPrimeField returns GF(p). p must be an odd prime in (2, 2^63). The
// upper bound keeps Add/Sub within uint64 (a + p < 2^64) and keeps the
// 128-bit product's high word below p, so mulmod's reduction never
// overflows.
func NewPrimeField(p uint64) (Field, error) {
	if p <= 2 {
		return nil, ErrModulusTooSmall
	}
	if p >= 1<<63 {
		return nil, ErrModulusTooLarge
	}
	if !isPrime(p) {
		return nil, ErrModulusNotPrime
	}
	return primeField{p: p}, nil
}

func (f primeField) Modulus() uint64        { return f.p }
func (f primeField) Reduce(x uint64) uint64 { return x % f.p }

func (f primeField) Add(a, b uint64) uint64 {
	s := a + b // a, b < p < 2^63, so a + b < 2^64: no overflow.
	if s >= f.p {
		s -= f.p
	}
	return s
}

func (f primeField) Sub(a, b uint64) uint64 {
	return (a + f.p - b) % f.p // a + p < 2^64 for p < 2^63: no overflow.
}

func (f primeField) Mul(a, b uint64) uint64 {
	return mulmod(a, b, f.p)
}

func (f primeField) Inv(a uint64) uint64 {
	if a == 0 {
		return 0
	}
	return modPow(a%f.p, f.p-2, f.p) // Fermat: a^(p-2) = a^-1 (p prime)
}

// mulmod returns a*b mod p for a, b in [0, p) and p in (2, 2^63).
//
// It forms the exact 128-bit product with bits.Mul64 (constant-time) and
// reduces it with one bits.Div64. The Div64 no-overflow precondition
// (hi < p) holds for every such p: with a, b < p the product is < p^2,
// so hi = floor(a*b / 2^64) <= floor((p-1)^2 / 2^64) < p, because
// (p-1)^2 < p*2^64 for all p <= 2^64.
//
// Timing: bits.Div64 lowers to a hardware divide whose latency depends on
// its operands, so mulmod is NOT constant-time. That is acceptable for
// this package: its multiplications run over public party identifiers
// (the Lagrange basis) and a one-shot share recombination, not a
// secret-dependent per-message hot path. A constant-time field would
// reduce with Montgomery or Barrett arithmetic instead.
func mulmod(a, b, p uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	_, rem := bits.Div64(hi, lo, p)
	return rem
}

// modPow computes base^exp mod m by square-and-multiply, reducing every
// product through mulmod so it is correct across the full (2, 2^63) range.
func modPow(base, exp, m uint64) uint64 {
	result := uint64(1)
	b := base % m
	for exp > 0 {
		if exp&1 == 1 {
			result = mulmod(result, b, m)
		}
		b = mulmod(b, b, m)
		exp >>= 1
	}
	return result
}

// MLDSAField is GF(q) for the FIPS 204 prime q = 8 380 417. This is the
// field over which the Pulsar threshold layer Shamir-shares a seed and
// over which Lagrange coefficients align with R_q arithmetic.
var MLDSAField Field = primeField{p: 8380417}

// coronaQ is Corona's canonical 48-bit NTT-friendly prime modulus,
// q = 0x1000000004A01 (= 2^48 + 0x4A01), matching the constant Q in the
// Corona signer (sign/config.go). q ≡ 1 (mod 512), so it carries the
// negacyclic NTT over Z_q[X]/(X^256 + 1).
const coronaQ = 0x1000000004A01

// CoronaField is GF(q) for Corona's modulus q = 0x1000000004A01. Corona
// (the Lux Ringtail/Raccoon-line Module-LWE threshold signer) Shamir-
// shares over this field; its ~2^48 modulus is why this package reduces
// products through the 128-bit mulmod path.
var CoronaField Field = primeField{p: coronaQ}

// millerRabinWitnesses is used both for trial division and as the
// Miller-Rabin base set. Sharing one set is what guarantees every base
// a < n by the time the rounds run: a value n that survives trial
// division is > 37 and coprime to each base. This set is a deterministic
// witness for all n < 3.3e24, covering the entire (2, 2^63) range.
var millerRabinWitnesses = []uint64{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

// isPrime is a deterministic Miller-Rabin test. With millerRabinWitnesses
// it is exact for every n far past the 2^63 modulus ceiling. All modular
// squarings go through mulmod, so it is correct across the supported range.
func isPrime(n uint64) bool {
	if n < 2 {
		return false
	}
	for _, p := range millerRabinWitnesses {
		if n == p {
			return true
		}
		if n%p == 0 {
			return false
		}
	}
	// n is now odd, > 37, and coprime to every witness, so each base a
	// below satisfies a < n.
	d := n - 1
	r := 0
	for d&1 == 0 {
		d >>= 1
		r++
	}
	for _, a := range millerRabinWitnesses {
		x := modPow(a, d, n)
		if x == 1 || x == n-1 {
			continue
		}
		composite := true
		for i := 0; i < r-1; i++ {
			x = mulmod(x, x, n)
			if x == n-1 {
				composite = false
				break
			}
		}
		if composite {
			return false
		}
	}
	return true
}
