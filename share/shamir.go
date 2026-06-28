// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package share

import (
	"encoding/binary"
	"errors"
	"io"
)

// Share is one party's Shamir share of a vector secret: the secret
// polynomial(s) evaluated at the non-zero point X. Y[k] is the share of
// secret slot k.
type Share struct {
	X uint64
	Y []uint64
}

// Sharing errors.
var (
	ErrThreshold       = errors.New("mlwe/share: threshold must satisfy 1 <= t <= n")
	ErrZeroPoint       = errors.New("mlwe/share: evaluation point 0 is reserved for the secret")
	ErrDuplicatePoint  = errors.New("mlwe/share: duplicate evaluation point")
	ErrNotEnoughShares = errors.New("mlwe/share: fewer shares than the threshold")
	ErrEmptySecret     = errors.New("mlwe/share: empty secret")
)

// Split shares secret (a vector of field elements) across the parties at
// evalPoints with reconstruction threshold t, over field f. Each party
// p gets Share{X: evalPoints[p], Y: secret evaluated at that point}. The
// degree-(t-1) polynomial coefficients above the constant term are drawn
// from rand (8 bytes per coefficient, reduced into the field), so the
// distribution of any t-1 shares is independent of the secret.
//
// evalPoints must be non-zero and distinct, and there must be at least t
// of them. For cryptographic use pass crypto/rand as rand; for
// deterministic sharing pass a fixed XOF stream.
func Split(secret []uint64, t int, evalPoints []uint64, f Field, rand io.Reader) ([]Share, error) {
	n := len(evalPoints)
	if len(secret) == 0 {
		return nil, ErrEmptySecret
	}
	if t < 1 || n < t {
		return nil, ErrThreshold
	}
	if err := checkPoints(evalPoints); err != nil {
		return nil, err
	}

	slots := len(secret)
	// coeffs[d][k]: degree-d coefficient for slot k. Degree 0 is the
	// secret itself.
	coeffs := make([][]uint64, t)
	coeffs[0] = make([]uint64, slots)
	for k := range secret {
		coeffs[0][k] = f.Reduce(secret[k])
	}
	for d := 1; d < t; d++ {
		coeffs[d] = make([]uint64, slots)
		for k := 0; k < slots; k++ {
			c, err := readFieldElement(rand, f)
			if err != nil {
				return nil, err
			}
			coeffs[d][k] = c
		}
	}

	shares := make([]Share, n)
	for i, x := range evalPoints {
		y := make([]uint64, slots)
		for k := 0; k < slots; k++ {
			// Horner: ((c_{t-1} * x + c_{t-2}) * x + ...) + c_0.
			acc := coeffs[t-1][k]
			for d := t - 2; d >= 0; d-- {
				acc = f.Add(f.Mul(acc, x), coeffs[d][k])
			}
			y[k] = acc
		}
		shares[i] = Share{X: x, Y: y}
	}
	return shares, nil
}

// Reconstruct recovers the vector secret from a quorum of shares by
// Lagrange interpolation to X = 0. It needs at least as many shares as
// the (implicit) threshold; supplying more is fine and supplying fewer
// yields a different (wrong) value, so callers must pass exactly a valid
// quorum. All shares must share the same slot count.
func Reconstruct(shares []Share, f Field) ([]uint64, error) {
	if len(shares) == 0 {
		return nil, ErrNotEnoughShares
	}
	xs := make([]uint64, len(shares))
	slots := len(shares[0].Y)
	for i, s := range shares {
		xs[i] = s.X
		if len(s.Y) != slots {
			return nil, errors.New("mlwe/share: inconsistent share slot counts")
		}
	}
	lambda, err := Lagrange(xs, 0, f)
	if err != nil {
		return nil, err
	}
	out := make([]uint64, slots)
	for k := 0; k < slots; k++ {
		var acc uint64
		for i := range shares {
			acc = f.Add(acc, f.Mul(lambda[i], shares[i].Y[k]))
		}
		out[k] = acc
	}
	return out, nil
}

// Lagrange returns the Lagrange basis coefficients lambda_i for
// interpolating a polynomial, given on the points xs, to the value atX:
//
//	P(atX) = sum_i lambda_i * P(xs[i]),  lambda_i = prod_{j!=i} (atX - xs[j]) / (xs[i] - xs[j])
//
// With atX = 0 these are the reconstruction coefficients used by
// Reconstruct; with a non-zero atX they evaluate the interpolated
// polynomial at an arbitrary point (used by reshare and by tests).
// xs must be distinct.
func Lagrange(xs []uint64, atX uint64, f Field) ([]uint64, error) {
	if err := checkPoints(xs); err != nil {
		return nil, err
	}
	t := len(xs)
	lambda := make([]uint64, t)
	for i := 0; i < t; i++ {
		num := uint64(1)
		den := uint64(1)
		for j := 0; j < t; j++ {
			if i == j {
				continue
			}
			num = f.Mul(num, f.Sub(atX, xs[j]))
			den = f.Mul(den, f.Sub(xs[i], xs[j]))
		}
		lambda[i] = f.Mul(num, f.Inv(den))
	}
	return lambda, nil
}

// checkPoints verifies points are non-zero and distinct.
func checkPoints(xs []uint64) error {
	seen := make(map[uint64]struct{}, len(xs))
	for _, x := range xs {
		if x == 0 {
			return ErrZeroPoint
		}
		if _, dup := seen[x]; dup {
			return ErrDuplicatePoint
		}
		seen[x] = struct{}{}
	}
	return nil
}

// readFieldElement returns one field element drawn uniformly from [0, p)
// by rejection sampling over 8-byte big-endian blocks. A block is kept
// only if it is below the largest multiple of p that fits in 64 bits;
// blocks at or above that bound are discarded, so the result carries no
// modulo bias for any p < 2^63 (and the masking coefficients in Split are
// therefore exactly uniform, giving t-1 shares perfect — information-
// theoretic — independence from the secret). Because p < 2^63 the
// rejection probability is below 1/2, so this terminates after fewer than
// two blocks in expectation on any well-behaved reader. For cryptographic
// sharing pass crypto/rand; for deterministic sharing pass a fixed XOF
// stream.
func readFieldElement(r io.Reader, f Field) (uint64, error) {
	p := f.Modulus()
	limit := (^uint64(0) / p) * p // largest multiple of p that is < 2^64
	var buf [8]byte
	for {
		if _, err := io.ReadFull(r, buf[:]); err != nil {
			return 0, err
		}
		if u := binary.BigEndian.Uint64(buf[:]); u < limit {
			return u % p, nil
		}
	}
}
