// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package transcript

import (
	"bytes"
	"encoding/hex"
	"testing"

	"golang.org/x/crypto/sha3"
)

// TestCShake256_ReducesToShake is the anchoring KAT: SP 800-185 §3.3
// defines cSHAKE256(X, L, "", "") to equal SHAKE256(X, L). SHAKE256 is
// the NIST FIPS 202 function (the stdlib implementation is
// NIST-validated), so matching it pins our cSHAKE to the reference.
func TestCShake256_ReducesToShake(t *testing.T) {
	for _, in := range [][]byte{nil, []byte("a"), bytes.Repeat([]byte{0xAB}, 200)} {
		for _, l := range []int{16, 32, 64, 137} {
			got := CShake256("", "", in, l)
			want := make([]byte, l)
			sha3.ShakeSum256(want, in)
			if !bytes.Equal(got, want) {
				t.Fatalf("cSHAKE256(X=%x,L=%d) != SHAKE256: %x vs %x", in, l, got, want)
			}
		}
	}
}

// TestCShake256_NISTSample replays the published NIST SP 800-185
// cSHAKE256 Sample #3: N = "", S = "Email Signature", X = 00010203,
// L = 512 bits.
func TestCShake256_NISTSample(t *testing.T) {
	in := []byte{0x00, 0x01, 0x02, 0x03}
	wantHex := "" +
		"d008828e2b80ac9d2218ffee1d070c48" +
		"b8e4c87bff32c9699d5b6896eee0edd1" +
		"64020e2be0560858d9c00c037e34a969" +
		"37c561a74c412bb4c746469527281c8c"
	want, _ := hex.DecodeString(wantHex)
	got := CShake256("", "Email Signature", in, 64)
	if !bytes.Equal(got, want) {
		t.Fatalf("cSHAKE256 NIST sample #3 mismatch:\n got  %x\n want %x", got, want)
	}
}

// TestEncoders pins the SP 800-185 §2.3 string encoders to hand-derived
// values from the spec definitions.
func TestEncoders(t *testing.T) {
	cases := []struct {
		got  []byte
		want []byte
		name string
	}{
		{LeftEncode(0), []byte{0x01, 0x00}, "left_encode(0)"},
		{LeftEncode(1), []byte{0x01, 0x01}, "left_encode(1)"},
		{LeftEncode(256), []byte{0x02, 0x01, 0x00}, "left_encode(256)"},
		{LeftEncode(65536), []byte{0x03, 0x01, 0x00, 0x00}, "left_encode(65536)"},
		{RightEncode(0), []byte{0x00, 0x01}, "right_encode(0)"},
		{RightEncode(1), []byte{0x01, 0x01}, "right_encode(1)"},
		{RightEncode(256), []byte{0x01, 0x00, 0x02}, "right_encode(256)"},
		{EncodeString(nil), []byte{0x01, 0x00}, "encode_string(empty)"},
		{EncodeString([]byte("X")), []byte{0x01, 0x08, 'X'}, "encode_string(X)"},
		{BytePad([]byte{0x41}, 4), []byte{0x01, 0x04, 0x41, 0x00}, "bytepad(0x41,4)"},
	}
	for _, c := range cases {
		if !bytes.Equal(c.got, c.want) {
			t.Fatalf("%s = % x, want % x", c.name, c.got, c.want)
		}
	}
}

// TestBytePad_AlignsToWidth confirms bytepad always produces a multiple
// of w bytes.
func TestBytePad_AlignsToWidth(t *testing.T) {
	for _, w := range []int{8, 136, 168} {
		for inLen := 0; inLen < 300; inLen += 37 {
			out := BytePad(bytes.Repeat([]byte{0xCC}, inLen), w)
			if len(out)%w != 0 {
				t.Fatalf("bytepad(len=%d,w=%d) len %d not a multiple of w", inLen, w, len(out))
			}
		}
	}
}

// TestKMAC256_MatchesSpecConstruction validates that KMAC256 applies the
// exact SP 800-185 §4 formula. Combined with the cSHAKE KAT and the
// encoder KATs, this pins KMAC256 to the standard without depending on a
// transcribed external digest.
func TestKMAC256_MatchesSpecConstruction(t *testing.T) {
	key := []byte("a 32-byte key for KMAC testing!!")
	msg := []byte{0x00, 0x01, 0x02, 0x03}
	const outLen = 64
	got := KMAC256(key, msg, outLen, "My Tagged Application")

	body := BytePad(EncodeString(key), 136)
	body = append(body, msg...)
	body = append(body, RightEncode(uint64(outLen)*8)...)
	want := CShake256("KMAC", "My Tagged Application", body, outLen)

	if !bytes.Equal(got, want) {
		t.Fatalf("KMAC256 does not match its SP 800-185 construction")
	}
}

// TestTupleHash256_MatchesSpecConstruction validates the §5 formula.
func TestTupleHash256_MatchesSpecConstruction(t *testing.T) {
	parts := [][]byte{[]byte("alpha"), {0x00, 0x01}, []byte("gamma")}
	const outLen = 48
	got := TupleHash256(parts, outLen, "ctx")

	var body []byte
	for _, p := range parts {
		body = append(body, EncodeString(p)...)
	}
	body = append(body, RightEncode(uint64(outLen)*8)...)
	want := CShake256("TupleHash", "ctx", body, outLen)

	if !bytes.Equal(got, want) {
		t.Fatalf("TupleHash256 does not match its SP 800-185 construction")
	}
}

// TestTupleHash256_Injective is the core domain/boundary property: the
// encode_string wrapping makes the tuple encoding injective, so no
// regrouping of the same concatenated bytes can collide. ("a","bc") and
// ("ab","c") share the concatenation "abc" but must hash differently.
func TestTupleHash256_Injective(t *testing.T) {
	h1 := TupleHash256([][]byte{[]byte("a"), []byte("bc")}, 32, "")
	h2 := TupleHash256([][]byte{[]byte("ab"), []byte("c")}, 32, "")
	h3 := TupleHash256([][]byte{[]byte("abc")}, 32, "")
	if bytes.Equal(h1, h2) || bytes.Equal(h1, h3) || bytes.Equal(h2, h3) {
		t.Fatalf("TupleHash256 collided across regroupings: not injective")
	}
}

// TestDomainTagInjectivity confirms distinct customization tags yield
// independent oracles for the same input, across all three primitives.
func TestDomainTagInjectivity(t *testing.T) {
	x := []byte("same input")
	if bytes.Equal(CShake256("F", "TAG-A", x, 32), CShake256("F", "TAG-B", x, 32)) {
		t.Fatalf("cSHAKE256 ignored the customization tag")
	}
	if bytes.Equal(CShake256("N-A", "S", x, 32), CShake256("N-B", "S", x, 32)) {
		t.Fatalf("cSHAKE256 ignored the function name")
	}
	if bytes.Equal(KMAC256(x, x, 32, "TAG-A"), KMAC256(x, x, 32, "TAG-B")) {
		t.Fatalf("KMAC256 ignored the customization tag")
	}
	if bytes.Equal(TupleHash256([][]byte{x}, 32, "TAG-A"), TupleHash256([][]byte{x}, 32, "TAG-B")) {
		t.Fatalf("TupleHash256 ignored the customization tag")
	}
}
