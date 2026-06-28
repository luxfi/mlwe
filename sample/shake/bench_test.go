// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package shake_test

// Decomplected benchmark suite for the FIPS 204 SHAKE samplers. Each
// Benchmark isolates EXACTLY ONE sampler.
//
// Every sampler's cost genuinely varies across the parameter profile, so
// each is benched over all three:
//
//   - ExpandA      scales with the matrix shape K*L (rejection-sampled).
//   - ExpandS      scales with K+L and the secret bound Eta (CBD).
//   - ExpandMask   scales with L and the mask width Gamma1.
//   - SampleInBall scales with the challenge weight Tau.
//
// Seeds are fixed, so each iteration recomputes the identical (seed,
// nonce) expansion: the rejection loops run a deterministic number of
// SHAKE squeezes, giving reproducible timing.

import (
	"testing"

	"github.com/luxfi/mlwe"
	"github.com/luxfi/mlwe/ring/mldsa"
	"github.com/luxfi/mlwe/sample/shake"
)

var benchProfiles = []mlwe.Profile{mldsa.Profile44, mldsa.Profile65, mldsa.Profile87}

// BenchmarkExpandA measures one ExpandA (public matrix A from rho, NTT
// domain), per profile (K*L grows 16 -> 30 -> 56).
func BenchmarkExpandA(b *testing.B) {
	var rho [32]byte
	for i := range rho {
		rho[i] = byte(i)
	}
	for _, p := range benchProfiles {
		b.Run(p.Name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				shake.ExpandA(p, rho)
			}
		})
	}
}

// BenchmarkExpandS measures one ExpandS (secret vectors s1, s2 by
// centered binomial), per profile.
func BenchmarkExpandS(b *testing.B) {
	var rhoPrime [64]byte
	for i := range rhoPrime {
		rhoPrime[i] = byte(i)
	}
	for _, p := range benchProfiles {
		b.Run(p.Name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				shake.ExpandS(p, rhoPrime)
			}
		})
	}
}

// BenchmarkExpandMask measures one ExpandMask (signing mask y in
// (-gamma1, gamma1]), per profile.
func BenchmarkExpandMask(b *testing.B) {
	var rho [64]byte
	for i := range rho {
		rho[i] = byte(i)
	}
	for _, p := range benchProfiles {
		b.Run(p.Name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				shake.ExpandMask(p, rho, 0)
			}
		})
	}
}

// BenchmarkSampleInBall measures one SampleInBall (tau nonzero +/-1
// challenge), per profile (Tau grows 39 -> 49 -> 60).
func BenchmarkSampleInBall(b *testing.B) {
	seed := []byte("mlwe-bench-sample-in-ball-seed-0")
	for _, p := range benchProfiles {
		b.Run(p.Name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				shake.SampleInBall(p, seed)
			}
		})
	}
}
