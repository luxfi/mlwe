// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package zeroize provides best-effort secret-buffer wiping.
//
// Threat model: a process holding reconstructed secret material (a
// Shamir-recombined seed, an expanded secret vector, a nonce) risks
// exfiltration via coredump, /proc/self/mem, or swap if the secret is
// left live after use. Go has no native memzero and the GC may copy
// buffers, so this is defense in depth, not a guarantee.
//
// One package, one concern: every secret-bearing buffer in the Lux
// Module-LWE stack is wiped through exactly these helpers, so the audit
// footprint of "where do we clear secrets" is a single file. Call the
// matching helper on a secret buffer at the return site of the function
// that produced it. Explicit (not deferred) calls keep the secret's
// lifetime locally legible.
package zeroize

// Bytes overwrites every byte of b with zero.
//
// The loop is written over the backing array of a reference type so a
// future compiler is less able to elide it as a dead store. If Go ever
// exposes a guaranteed memclr for secrets, route this through it.
func Bytes(b []byte) {
	for i := range b {
		b[i] = 0
	}
}

// U32 overwrites every element of s with zero. Used for GF(257) /
// 23-bit lane buffers (per-byte Shamir lanes, NTT-domain coefficients).
func U32(s []uint32) {
	for i := range s {
		s[i] = 0
	}
}

// U64 overwrites every element of s with zero. Used for wide-lane
// secret material (48-bit Corona coefficients, mlwe.Poly coefficients).
func U64(s []uint64) {
	for i := range s {
		s[i] = 0
	}
}
