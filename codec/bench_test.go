// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package codec

// Decomplected benchmark suite for the bit codec and bounded framing.
// Each Benchmark isolates EXACTLY ONE operation.
//
//   - PackBits / UnpackBits are benched over the real FIPS 204 widths
//     (4-bit w1, 10-bit t1, 13-bit t0, 20-bit gamma1) on a full N=256
//     polynomial, since their cost is N*bits.
//   - ReadVector is benched once at a representative element count; it
//     reuses the package test's getU64/WriteVector framing helpers so the
//     decoder under test is the exact one the round-trip test exercises.

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

var benchWidths = []int{4, 10, 13, 20}

// benchCoeffs returns 256 coefficients masked to the given bit width.
func benchCoeffs(bits int) []uint64 {
	rng := rand.New(rand.NewSource(int64(bits)))
	mask := uint64(1)<<uint(bits) - 1
	c := make([]uint64, 256)
	for i := range c {
		c[i] = rng.Uint64() & mask
	}
	return c
}

// BenchmarkPackBits measures one Pack of a 256-coefficient polynomial,
// per FIPS 204 bit width.
func BenchmarkPackBits(b *testing.B) {
	for _, bits := range benchWidths {
		c := benchCoeffs(bits)
		b.Run(fmt.Sprintf("%dbit", bits), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				PackBits(c, bits)
			}
		})
	}
}

// BenchmarkUnpackBits measures one Unpack back to 256 coefficients, per
// FIPS 204 bit width.
func BenchmarkUnpackBits(b *testing.B) {
	for _, bits := range benchWidths {
		data := PackBits(benchCoeffs(bits), bits)
		b.Run(fmt.Sprintf("%dbit", bits), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				UnpackBits(data, 256, bits)
			}
		})
	}
}

// BenchmarkReadVector measures one bounded-frame read of 256 u64
// elements. The frame is built once (setup) with WriteVector/putU64; the
// timed loop wraps a fresh reader over it and reads via getU64.
func BenchmarkReadVector(b *testing.B) {
	const n = 256
	var buf bytes.Buffer
	elems := make([]uint64, n)
	for i := range elems {
		elems[i] = uint64(i)
	}
	if err := WriteVector(&buf, elems, putU64); err != nil {
		b.Fatal(err)
	}
	data := buf.Bytes()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ReadVector(bytes.NewReader(data), n, getU64); err != nil {
			b.Fatal(err)
		}
	}
}
