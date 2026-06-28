// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package transcript

// Decomplected benchmark suite for the SP 800-185 constructions. Each
// Benchmark isolates EXACTLY ONE construction; the SP 800-185 string
// encoders (LeftEncode/RightEncode/EncodeString/BytePad) are measured
// only as part of the construction that uses them, since on their own
// they are a handful of bytes and never a hot path in isolation.
//
// The axis is the input size, since for a sponge the cost is the number
// of absorbed blocks. CShake256 and KMAC256 are benched over a small,
// large and bulk message; TupleHash256 is benched over two
// representative tuple SHAPES: a many-small-fields Fiat-Shamir transcript
// and a few-large-fields bulk bind, which is where its per-part
// EncodeString framing differs from a flat hash.

import (
	"fmt"
	"testing"
)

var benchSizes = []int{64, 1024, 8192}

func sizeName(n int) string {
	if n >= 1024 {
		return fmt.Sprintf("%dKiB", n/1024)
	}
	return fmt.Sprintf("%dB", n)
}

func filled(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i)
	}
	return b
}

// makeParts builds count byte slices of size each, for the multi-part
// TupleHash bind.
func makeParts(count, size int) [][]byte {
	parts := make([][]byte, count)
	for i := range parts {
		parts[i] = filled(size)
	}
	return parts
}

// BenchmarkCShake256 measures one cSHAKE256 with a non-empty
// customization (the real cSHAKE framing, not the SHAKE reduction),
// 32-byte output, over input sizes.
func BenchmarkCShake256(b *testing.B) {
	for _, n := range benchSizes {
		in := filled(n)
		b.Run(sizeName(n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				CShake256("", "mlwe-bench", in, 32)
			}
		})
	}
}

// BenchmarkKMAC256 measures one KMAC256 (32-byte key, 32-byte output)
// over message sizes.
func BenchmarkKMAC256(b *testing.B) {
	key := filled(32)
	for _, n := range benchSizes {
		msg := filled(n)
		b.Run(sizeName(n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				KMAC256(key, msg, 32, "")
			}
		})
	}
}

// BenchmarkTupleHash256 measures one TupleHash256 (32-byte output) over
// two representative tuple shapes.
func BenchmarkTupleHash256(b *testing.B) {
	cases := []struct {
		name  string
		parts [][]byte
	}{
		{"4x32B", makeParts(4, 32)},    // Fiat-Shamir transcript: many small fields
		{"3x4KiB", makeParts(3, 4096)}, // bulk bind: few large fields
	}
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				TupleHash256(c.parts, 32, "")
			}
		})
	}
}
