// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package shake_test

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/luxfi/mlwe"
	"github.com/luxfi/mlwe/ring/mldsa"
	"github.com/luxfi/mlwe/sample/shake"
	"golang.org/x/crypto/sha3"
)

const q = 8380417

func profileFor(mode string) (mlwe.Profile, bool) {
	switch mode {
	case "Pulsar-44":
		return mldsa.Profile44, true
	case "Pulsar-65":
		return mldsa.Profile65, true
	case "Pulsar-87":
		return mldsa.Profile87, true
	}
	return mlwe.Profile{}, false
}

// expandedPublicKey reproduces the FIPS 204 ML-DSA public key (rho ||
// PackT1(t1)) from a 32-byte seed using ONLY the mlwe Phase 0
// primitives: the SHAKE samplers (ExpandA, ExpandS), the ring arithmetic
// (NTT, MulNTT, INTT, Add, Power2Round), and the bit codec (10-bit pack).
// Matching the committed vectors therefore byte-validates all three
// against the FIPS 204 reference.
func expandedPublicKey(t *testing.T, prof mlwe.Profile, seed [32]byte) []byte {
	r := mldsa.MustNew(prof)

	// FIPS 204 KeyGen seed expansion: SHAKE256(seed || K || L) -> 128 B.
	var eSeed [128]byte
	h := sha3.NewShake256()
	_, _ = h.Write(seed[:])
	_, _ = h.Write([]byte{byte(prof.K), byte(prof.L)})
	_, _ = h.Read(eSeed[:])
	var rho [32]byte
	var rhoPrime [64]byte
	copy(rho[:], eSeed[:32])
	copy(rhoPrime[:], eSeed[32:96])

	a := shake.ExpandA(prof, rho) // K x L, NTT domain
	s1, s2 := shake.ExpandS(prof, rhoPrime)

	// s1Hat = NTT(s1).
	s1Hat := s1.Clone()
	for j := range s1Hat {
		r.NTT(s1Hat[j])
	}
	// t = A . s1 + s2 (dot product over NTT-domain A and s1Hat).
	tvec := make(mlwe.PolyVec, prof.K)
	tmp := r.NewPoly()
	for i := 0; i < prof.K; i++ {
		acc := r.NewPoly()
		for j := 0; j < prof.L; j++ {
			r.MulNTT(tmp, a[i][j], s1Hat[j])
			r.Add(acc, acc, tmp)
		}
		r.INTT(acc)
		r.Add(acc, acc, s2[i])
		tvec[i] = acc
	}

	// Power2Round and pack t1 at 10 bits.
	cdc := r.Codec()
	pub := make([]byte, 0, 32+320*prof.K)
	pub = append(pub, rho[:]...)
	for i := 0; i < prof.K; i++ {
		_, t1 := r.Power2Round(tvec[i])
		pub = append(pub, cdc.Pack(t1, 10)...)
	}
	return pub
}

// TestKeygenPublicKey_KAT is the headline known-answer test: reproduce
// the FIPS 204 public key from seed for every committed vector and
// require a byte-for-byte match. These vectors were generated to equal
// cloudflare/circl's ML-DSA output, so a match proves the lifted ring,
// the SHAKE samplers, and the codec are FIPS 204 exact.
func TestKeygenPublicKey_KAT(t *testing.T) {
	type entry struct {
		Mode      string `json:"mode"`
		Seed      string `json:"seed"`
		PublicKey string `json:"public_key"`
	}
	data, err := os.ReadFile(filepath.Join("testdata", "keygen.json"))
	if err != nil {
		t.Fatalf("read vectors: %v", err)
	}
	var entries []entry
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatalf("parse vectors: %v", err)
	}
	if len(entries) == 0 {
		t.Fatalf("no vectors")
	}
	var perMode = map[string]int{}
	for i, e := range entries {
		prof, ok := profileFor(e.Mode)
		if !ok {
			t.Fatalf("entry %d: unknown mode %q", i, e.Mode)
		}
		seedBytes, _ := hex.DecodeString(e.Seed)
		var seed [32]byte
		copy(seed[:], seedBytes)
		got := expandedPublicKey(t, prof, seed)
		if hex.EncodeToString(got) != e.PublicKey {
			t.Fatalf("entry %d (%s): public key mismatch\n got  %s...\n want %s...",
				i, e.Mode, hex.EncodeToString(got)[:64], e.PublicKey[:64])
		}
		perMode[e.Mode]++
	}
	t.Logf("validated FIPS 204 public-key KAT: %v", perMode)
}

// TestExpandA_Properties checks ExpandA is deterministic, seed-sensitive,
// and produces in-range coefficients of the right shape.
func TestExpandA_Properties(t *testing.T) {
	prof := mldsa.Profile65
	var rho1, rho2 [32]byte
	rho2[0] = 1
	a := shake.ExpandA(prof, rho1)
	if a.Rows() != prof.K || a.Cols() != prof.L {
		t.Fatalf("ExpandA shape %dx%d, want %dx%d", a.Rows(), a.Cols(), prof.K, prof.L)
	}
	for i := range a {
		for j := range a[i] {
			if len(a[i][j].Coeffs) != prof.N {
				t.Fatalf("poly[%d][%d] len %d", i, j, len(a[i][j].Coeffs))
			}
			for _, c := range a[i][j].Coeffs {
				if c >= q {
					t.Fatalf("coeff %d out of [0,q)", c)
				}
			}
		}
	}
	// Determinism and seed-sensitivity.
	b := shake.ExpandA(prof, rho1)
	if !sameMat(a, b) {
		t.Fatalf("ExpandA not deterministic")
	}
	if sameMat(a, shake.ExpandA(prof, rho2)) {
		t.Fatalf("ExpandA ignored the seed")
	}
}

// TestExpandS_Range checks the centered binomial output lies in
// [q-eta, q+eta] (i.e. centered in [-eta, eta]) for eta in {2, 4}.
func TestExpandS_Range(t *testing.T) {
	for _, prof := range []mlwe.Profile{mldsa.Profile65 /*eta 4*/, mldsa.Profile87 /*eta 2*/} {
		var rp [64]byte
		rp[0] = 7
		s1, s2 := shake.ExpandS(prof, rp)
		eta := uint64(prof.Eta)
		check := func(v mlwe.PolyVec, name string) {
			for k := range v {
				for _, c := range v[k].Coeffs {
					// centered value in [-eta, eta] => c in {0..eta} U {q-eta..q-1}.
					ok := c <= eta || c >= q-eta
					if !ok {
						t.Fatalf("%s %s[%d] coeff %d outside +/-%d", prof.Name, name, k, c, eta)
					}
				}
			}
		}
		check(s1, "s1")
		check(s2, "s2")
		if len(s1) != prof.L || len(s2) != prof.K {
			t.Fatalf("%s ExpandS lengths %d,%d", prof.Name, len(s1), len(s2))
		}
		// Determinism.
		s1b, _ := shake.ExpandS(prof, rp)
		for k := range s1 {
			for i := range s1[k].Coeffs {
				if s1[k].Coeffs[i] != s1b[k].Coeffs[i] {
					t.Fatalf("%s ExpandS not deterministic", prof.Name)
				}
			}
		}
	}
}

// TestExpandMask_Range checks the mask coefficients fall in the FIPS 204
// range (-gamma1, gamma1], represented mod q, for both gamma1 widths.
func TestExpandMask_Range(t *testing.T) {
	for _, prof := range []mlwe.Profile{mldsa.Profile44 /*gamma1 2^17*/, mldsa.Profile65 /*2^19*/} {
		var rho [64]byte
		rho[0] = 3
		y := shake.ExpandMask(prof, rho, 0)
		if len(y) != prof.L {
			t.Fatalf("%s ExpandMask len %d", prof.Name, len(y))
		}
		g1 := uint64(prof.Gamma1)
		for k := range y {
			for _, c := range y[k].Coeffs {
				// centered in (-g1, g1] => c in [0, g1] U [q-g1+1, q-1].
				centered := int64(c)
				if c > q/2 {
					centered -= q
				}
				if centered <= -int64(g1) || centered > int64(g1) {
					t.Fatalf("%s mask coeff centered %d outside (-%d, %d]", prof.Name, centered, g1, g1)
				}
			}
		}
		// kappa changes the output.
		y2 := shake.ExpandMask(prof, rho, 1)
		if y[0].Coeffs[0] == y2[0].Coeffs[0] && y[0].Coeffs[1] == y2[0].Coeffs[1] {
			t.Fatalf("%s ExpandMask ignored kappa", prof.Name)
		}
	}
}

// TestSampleInBall_Structure checks the challenge has exactly tau
// nonzero coefficients, all +/-1, and is deterministic / seed-sensitive.
func TestSampleInBall_Structure(t *testing.T) {
	prof := mldsa.Profile65
	seed := []byte("challenge-seed-0")
	c := shake.SampleInBall(prof, seed)
	nonzero := 0
	for _, v := range c.Coeffs {
		switch v {
		case 0:
		case 1, q - 1:
			nonzero++
		default:
			t.Fatalf("SampleInBall coeff %d not in {0,1,-1}", v)
		}
	}
	if nonzero != prof.Tau {
		t.Fatalf("SampleInBall weight %d, want tau=%d", nonzero, prof.Tau)
	}
	// Determinism.
	c2 := shake.SampleInBall(prof, seed)
	for i := range c.Coeffs {
		if c.Coeffs[i] != c2.Coeffs[i] {
			t.Fatalf("SampleInBall not deterministic")
		}
	}
	// Seed-sensitivity.
	c3 := shake.SampleInBall(prof, []byte("challenge-seed-1"))
	same := true
	for i := range c.Coeffs {
		if c.Coeffs[i] != c3.Coeffs[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatalf("SampleInBall ignored the seed")
	}
}

// TestSampleInBall_Stress runs SampleInBall over many seeds, asserting
// the exact tau weight and {0,+1,-1} alphabet each time. Beyond being a
// property test, the volume of draws exercises the SHAKE buffer-refill
// path inside the sampler that a single seed rarely reaches.
func TestSampleInBall_Stress(t *testing.T) {
	prof := mldsa.Profile87 // largest tau = 60
	for s := 0; s < 4096; s++ {
		seed := []byte{byte(s), byte(s >> 8), 0xA5}
		c := shake.SampleInBall(prof, seed)
		w := 0
		for _, v := range c.Coeffs {
			switch v {
			case 0:
			case 1, q - 1:
				w++
			default:
				t.Fatalf("seed %d: coeff %d not in {0,+/-1}", s, v)
			}
		}
		if w != prof.Tau {
			t.Fatalf("seed %d: weight %d != tau %d", s, w, prof.Tau)
		}
	}
}

func sameMat(a, b mlwe.PolyMat) bool {
	if a.Rows() != b.Rows() || a.Cols() != b.Cols() {
		return false
	}
	for i := range a {
		for j := range a[i] {
			for k := range a[i][j].Coeffs {
				if a[i][j].Coeffs[k] != b[i][j].Coeffs[k] {
					return false
				}
			}
		}
	}
	return true
}
