// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package share

// Decomplected benchmark suite for Shamir sharing and the field
// multiply. Each Benchmark isolates EXACTLY ONE operation.
//
// Reconstruct is composed of Lagrange (the O(t^2) basis) and a dot
// product (the O(t*slots) combine). Benching Split, Reconstruct and
// Lagrange SEPARATELY is what makes that split visible: Lagrange alone
// shows how much of Reconstruct is the basis.
//
// Two axes vary the cost and both are carried:
//   - field: GF(257) (byte sharing), GF(8 380 417) (ML-DSA), and the
//     48-bit Corona field. The field changes the operand width fed to
//     mulmod's bits.Div64, hence its latency.
//   - (t, n): the threshold/party shape. Split is O(n*t), Lagrange and
//     Reconstruct are O(t^2), so a small spread makes the curve visible.
//
// BenchmarkMulmod is the leaf: ONE field multiply across three operand
// magnitudes. mulmod has a single 128-bit code path (bits.Mul64 +
// bits.Div64) regardless of width, so this is not a "small vs wide path"
// dispatch — it isolates the DATA-DEPENDENT hardware-divide latency that
// the source comment flags as non-constant-time.

import (
	"fmt"
	"math/rand"
	"testing"
)

type benchField struct {
	name string
	f    Field
}

// benchFields returns the three production fields. GF(257) is built here
// because it is not a package var (CoronaField and MLDSAField are).
func benchFields(b *testing.B) []benchField {
	gf257, err := NewPrimeField(257)
	if err != nil {
		b.Fatalf("NewPrimeField(257): %v", err)
	}
	return []benchField{
		{"GF257", gf257},
		{"MLDSAField", MLDSAField},
		{"CoronaField", CoronaField},
	}
}

type tn struct{ t, n int }

// benchTN is a representative spread of real threshold committees: the
// 2-of-3 bridge quorum, a small 3-of-5, and a mid 5-of-9.
var benchTN = []tn{{2, 3}, {3, 5}, {5, 9}}

const benchSlots = 32 // a 32-byte seed lifted to 32 field slots

func benchSecret(rng *rand.Rand) []uint64 {
	s := make([]uint64, benchSlots)
	for i := range s {
		s[i] = rng.Uint64()
	}
	return s
}

// evalPoints returns n distinct non-zero points (1..n), valid in every
// field here since n is far below 257.
func evalPoints(n int) []uint64 {
	xs := make([]uint64, n)
	for i := range xs {
		xs[i] = uint64(i + 1)
	}
	return xs
}

// BenchmarkSplit measures one Split (degree-(t-1) polynomial evaluation
// at n points, including the masking-coefficient draws), per field and
// (t, n).
func BenchmarkSplit(b *testing.B) {
	secret := benchSecret(rand.New(rand.NewSource(1)))
	for _, bf := range benchFields(b) {
		for _, c := range benchTN {
			pts := evalPoints(c.n)
			b.Run(fmt.Sprintf("%s/t%d_n%d", bf.name, c.t, c.n), func(b *testing.B) {
				randr := rand.New(rand.NewSource(42))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := Split(secret, c.t, pts, bf.f, randr); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkReconstruct measures one Reconstruct (Lagrange-to-0 plus the
// share combine) from a minimal t-share quorum, per field and (t, n).
// The Split that produces the shares is setup, outside the timer.
func BenchmarkReconstruct(b *testing.B) {
	secret := benchSecret(rand.New(rand.NewSource(2)))
	for _, bf := range benchFields(b) {
		for _, c := range benchTN {
			pts := evalPoints(c.n)
			shares, err := Split(secret, c.t, pts, bf.f, rand.New(rand.NewSource(7)))
			if err != nil {
				b.Fatal(err)
			}
			quorum := shares[:c.t]
			b.Run(fmt.Sprintf("%s/t%d_n%d", bf.name, c.t, c.n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := Reconstruct(quorum, bf.f); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkLagrange measures one Lagrange basis computation to X=0 over t
// points (O(t^2) field ops), per field and t. This isolates the basis
// from Reconstruct's share-combine step.
func BenchmarkLagrange(b *testing.B) {
	for _, bf := range benchFields(b) {
		for _, c := range benchTN {
			xs := evalPoints(c.t)
			b.Run(fmt.Sprintf("%s/t%d", bf.name, c.t), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := Lagrange(xs, 0, bf.f); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkMulmod measures ONE field multiply across three operand
// magnitudes (8-bit, 23-bit, 48-bit). Same code path each time; the
// benchmark surfaces bits.Div64's data-dependent latency.
func BenchmarkMulmod(b *testing.B) {
	cases := []struct {
		name    string
		x, y, p uint64
	}{
		{"GF257", 200, 199, 257},
		{"MLDSAField", 8380416, 8380415, 8380417},
		{"CoronaField", coronaQ - 1, coronaQ - 2, coronaQ},
	}
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkU64 = mulmod(c.x, c.y, c.p)
			}
		})
	}
}

// sinkU64 absorbs the field-multiply result so it cannot be elided.
var sinkU64 uint64
