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

import "errors"

// Field is a prime field GF(p). The implementations in this package
// require p prime and p <= 2^32 so that a single 64-bit multiply holds
// any product a*b without overflow. (Wider fields — e.g. Corona's
// 48-bit modulus — need a 128-bit multiply path and are a later
// addition; the interface is stable across that change.)
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
	ErrModulusTooLarge = errors.New("mlwe/share: modulus must be <= 2^32")
	ErrModulusNotPrime = errors.New("mlwe/share: modulus is not prime")
	ErrModulusTooSmall = errors.New("mlwe/share: modulus must be > 2")
)

// primeField is a GF(p) with p prime and p <= 2^32.
type primeField struct{ p uint64 }

// NewPrimeField returns GF(p). p must be an odd prime in (2, 2^32].
func NewPrimeField(p uint64) (Field, error) {
	if p <= 2 {
		return nil, ErrModulusTooSmall
	}
	if p > 1<<32 {
		return nil, ErrModulusTooLarge
	}
	if !isPrime(p) {
		return nil, ErrModulusNotPrime
	}
	return primeField{p: p}, nil
}

func (f primeField) Modulus() uint64      { return f.p }
func (f primeField) Reduce(x uint64) uint64 { return x % f.p }

func (f primeField) Add(a, b uint64) uint64 {
	s := a + b
	if s >= f.p {
		s -= f.p
	}
	return s
}

func (f primeField) Sub(a, b uint64) uint64 {
	return (a + f.p - b) % f.p
}

func (f primeField) Mul(a, b uint64) uint64 {
	return (a * b) % f.p
}

func (f primeField) Inv(a uint64) uint64 {
	if a == 0 {
		return 0
	}
	return modPow(a%f.p, f.p-2, f.p) // Fermat: a^(p-2) = a^-1 (p prime)
}

// modPow computes base^exp mod m by square-and-multiply.
func modPow(base, exp, m uint64) uint64 {
	result := uint64(1)
	b := base % m
	for exp > 0 {
		if exp&1 == 1 {
			result = (result * b) % m
		}
		b = (b * b) % m
		exp >>= 1
	}
	return result
}

// MLDSAField is GF(q) for the FIPS 204 prime q = 8 380 417. This is the
// field over which the Pulsar threshold layer Shamir-shares a seed and
// over which Lagrange coefficients align with R_q arithmetic.
var MLDSAField Field = primeField{p: 8380417}

// isPrime is a deterministic Miller-Rabin test, exact for all n < 2^32
// with the witness set {2, 7, 61}.
func isPrime(n uint64) bool {
	if n < 2 {
		return false
	}
	for _, p := range []uint64{2, 3, 5, 7, 11, 13, 61} {
		if n == p {
			return true
		}
		if n%p == 0 {
			return false
		}
	}
	d := n - 1
	r := 0
	for d&1 == 0 {
		d >>= 1
		r++
	}
	for _, a := range []uint64{2, 7, 61} {
		x := modPow(a, d, n)
		if x == 1 || x == n-1 {
			continue
		}
		composite := true
		for i := 0; i < r-1; i++ {
			x = (x * x) % n
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
