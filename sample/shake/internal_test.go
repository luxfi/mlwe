// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package shake

import "testing"

// TestSampleInBall_BufferRefill is a white-box test of the sampler's
// SHAKE buffer-refill path. The FIPS 204 weights (tau <= 60) consume far
// fewer than one 136-byte block of rejection bytes, so refill never
// fires at production parameters; a large tau forces hundreds of draws,
// exercising the refill branch directly.
func TestSampleInBall_BufferRefill(t *testing.T) {
	const n = 256
	const tau = 220 // ~hundreds of rejection draws, well past one block
	out := make([]uint64, n)
	sampleInBall(out, []byte("refill"), tau, 8380417, n)
	weight := 0
	for _, v := range out {
		switch v {
		case 0:
		case 1, 8380417 - 1:
			weight++
		default:
			t.Fatalf("coeff %d not in {0,+/-1}", v)
		}
	}
	if weight != tau {
		t.Fatalf("weight %d, want %d", weight, tau)
	}
}
