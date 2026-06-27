// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package transcript is the single SP 800-185 (cSHAKE / KMAC /
// TupleHash) surface for the Lux Module-LWE stack.
//
// Every protocol hash in Pulsar and Corona routes through this package.
// Keeping it in one place means the audit footprint of "which hash, in
// which mode, with which domain tag" is one file, and that rotating a
// customization tag is a single, deliberate edit.
//
// The vended primitives are exactly the SP 800-185 constructions:
//
//   - CShake256  (FIPS 202 §6.3 + SP 800-185 §3.3)
//   - KMAC256    (SP 800-185 §4)
//   - TupleHash256 (SP 800-185 §5) — the injective multi-part bind
//
// plus the SP 800-185 §2.3 string encoders (LeftEncode, RightEncode,
// EncodeString, BytePad) that the constructions are built from. The
// encoders are exported because they are independently useful and
// independently testable against the spec's worked examples.
//
// Domain separation is the caller's responsibility and is expressed
// through the function-name N and customization S parameters. Distinct
// (N, S) pairs yield independent random oracles; this package never
// hardcodes a tag.
package transcript

import (
	"encoding/binary"

	"golang.org/x/crypto/sha3"
)

// CShake256 returns the first outLen bytes of cSHAKE256(input, N,
// customization) per SP 800-185 §3.3.
//
// When functionName and customization are both empty, cSHAKE256 is
// defined to equal SHAKE256 (SP 800-185 §3.3); this package preserves
// that, which is the anchoring KAT against the FIPS 202 reference.
func CShake256(functionName, customization string, input []byte, outLen int) []byte {
	h := sha3.NewCShake256([]byte(functionName), []byte(customization))
	_, _ = h.Write(input)
	out := make([]byte, outLen)
	_, _ = h.Read(out)
	return out
}

// KMAC256 returns KMAC256(key, msg, outLen, customization) per
// SP 800-185 §4:
//
//	KMAC256(K, X, L, S) = cSHAKE256(
//	    bytepad(encode_string(K), 136) || X || right_encode(L),
//	    L, "KMAC", S)
//
// where L is the output length in BITS and 136 is the SHA3-256 rate in
// bytes ((1600 - 2*256)/8). The function-name N is fixed to "KMAC".
func KMAC256(key, msg []byte, outLen int, customization string) []byte {
	body := BytePad(EncodeString(key), 136)
	body = append(body, msg...)
	body = append(body, RightEncode(uint64(outLen)*8)...)
	return CShake256("KMAC", customization, body, outLen)
}

// TupleHash256 returns TupleHash256(parts, outLen, customization) per
// SP 800-185 §5:
//
//	TupleHash256(X, L, S) = cSHAKE256(
//	    encode_string(X[0]) || ... || encode_string(X[n-1])
//	    || right_encode(L),
//	    L, "TupleHash", S)
//
// where L is the output length in BITS. The encode_string wrapping of
// each part makes the encoding injective: no concatenation of one tuple
// can collide with the encoding of a different tuple, so TupleHash is
// the correct primitive for binding an ordered set of fields into a
// transcript digest. The function-name N is fixed to "TupleHash".
func TupleHash256(parts [][]byte, outLen int, customization string) []byte {
	var body []byte
	for _, p := range parts {
		body = append(body, EncodeString(p)...)
	}
	body = append(body, RightEncode(uint64(outLen)*8)...)
	return CShake256("TupleHash", customization, body, outLen)
}

// LeftEncode returns left_encode(x) per SP 800-185 §2.3.1. x is the
// value to encode; callers encoding a byte length pre-multiply by 8.
func LeftEncode(x uint64) []byte {
	if x == 0 {
		return []byte{0x01, 0x00}
	}
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], x)
	i := 0
	for i < 7 && buf[i] == 0 {
		i++
	}
	out := make([]byte, 0, 9-i)
	out = append(out, byte(8-i))
	out = append(out, buf[i:]...)
	return out
}

// RightEncode returns right_encode(x) per SP 800-185 §2.3.1.
func RightEncode(x uint64) []byte {
	if x == 0 {
		return []byte{0x00, 0x01}
	}
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], x)
	i := 0
	for i < 7 && buf[i] == 0 {
		i++
	}
	out := make([]byte, 0, 9-i)
	out = append(out, buf[i:]...)
	out = append(out, byte(8-i))
	return out
}

// EncodeString returns encode_string(s) = left_encode(bit_len(s)) || s
// per SP 800-185 §2.3.2.
func EncodeString(s []byte) []byte {
	out := LeftEncode(uint64(len(s)) * 8)
	return append(out, s...)
}

// BytePad returns bytepad(x, w) = left_encode(w) || x || 0-padding to a
// multiple of w bytes, per SP 800-185 §2.3.3.
func BytePad(x []byte, w int) []byte {
	prefix := LeftEncode(uint64(w))
	out := make([]byte, 0, len(prefix)+len(x)+w)
	out = append(out, prefix...)
	out = append(out, x...)
	for len(out)%w != 0 {
		out = append(out, 0x00)
	}
	return out
}
