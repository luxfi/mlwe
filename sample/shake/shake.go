// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package shake is the FIPS 204 (ML-DSA) family of SHAKE-derived
// samplers, lifted at the value level from the audited Pulsar
// reference. Every sampler is byte-for-byte the FIPS 204 / circl
// distribution over the same (seed, nonce), so the public keys and
// signatures derived through them match the reference library exactly.
//
// Four samplers, four FIPS 204 algorithms:
//
//   - ExpandA      (Alg 32) — public matrix A from rho, NTT domain
//   - ExpandS      (Alg 33) — secret vectors s1,s2 by centered binomial
//   - ExpandMask   (Alg 34) — signing mask y in (-gamma1, gamma1]
//   - SampleInBall (Alg 29) — challenge c with tau nonzero +/-1 coeffs
//
// These are the SHAKE samplers of the ML-DSA 23-bit ring; the Corona
// (Ringtail) Gaussian sampler is a separate concern and lives under
// sample/gaussian. Every sampler is parameterized by an mlwe.Profile so
// it serves ML-DSA-44/65/87 uniformly, but the rejection bounds and the
// centered-binomial arithmetic are FIPS 204 specific by construction.
package shake

import (
	"encoding/binary"
	"math/bits"

	"github.com/luxfi/mlwe"
	"golang.org/x/crypto/sha3"
)

// ExpandA derives the K-by-L public matrix A from the 32-byte seed rho.
// Each entry is sampled uniformly in [0, q) directly in the NTT
// (evaluation) domain, exactly as FIPS 204 ExpandA: no forward NTT is
// applied to the result.
func ExpandA(p mlwe.Profile, rho [32]byte) mlwe.PolyMat {
	a := make(mlwe.PolyMat, p.K)
	for i := 0; i < p.K; i++ {
		a[i] = make(mlwe.PolyVec, p.L)
		for j := 0; j < p.L; j++ {
			a[i][j] = mlwe.Poly{Coeffs: make([]uint64, p.N)}
			rejNTTPoly(a[i][j].Coeffs, &rho, uint16(i)<<8|uint16(j), p.Q, p.N)
		}
	}
	return a
}

// ExpandS derives the secret vectors s1 (length L) and s2 (length K)
// from the 64-byte seed rhoPrime via the centered binomial sampler with
// bound Eta. Coefficients are stored un-normalized in [q-Eta, q+Eta]
// (the FIPS 204 representation), which is < 2q and so a valid NTT input.
func ExpandS(p mlwe.Profile, rhoPrime [64]byte) (s1, s2 mlwe.PolyVec) {
	eta := uint64(p.Eta)
	s1 = make(mlwe.PolyVec, p.L)
	for i := 0; i < p.L; i++ {
		s1[i] = mlwe.Poly{Coeffs: make([]uint64, p.N)}
		cbdEta(s1[i].Coeffs, &rhoPrime, uint16(i), eta, p.Q, p.N)
	}
	s2 = make(mlwe.PolyVec, p.K)
	for i := 0; i < p.K; i++ {
		s2[i] = mlwe.Poly{Coeffs: make([]uint64, p.N)}
		cbdEta(s2[i].Coeffs, &rhoPrime, uint16(i+p.L), eta, p.Q, p.N)
	}
	return s1, s2
}

// ExpandMask derives the signing mask vector y (length L) from the
// 64-byte seed rho and the round counter kappa. Coefficient i of poly l
// uses nonce kappa+l. Coefficients are uniform in (-gamma1, gamma1],
// stored normalized in [0, q).
func ExpandMask(p mlwe.Profile, rho [64]byte, kappa uint16) mlwe.PolyVec {
	gamma1Bits := uint32(bits.TrailingZeros32(p.Gamma1))
	y := make(mlwe.PolyVec, p.L)
	for i := 0; i < p.L; i++ {
		y[i] = mlwe.Poly{Coeffs: make([]uint64, p.N)}
		expandMaskPoly(y[i].Coeffs, &rho, kappa+uint16(i), gamma1Bits, p.Q, p.N)
	}
	return y
}

// SampleInBall derives the challenge polynomial c from seed (the FIPS
// 204 commitment hash c-tilde). c has exactly Tau coefficients in
// {-1, +1} and the rest zero, with -1 stored as q-1.
func SampleInBall(p mlwe.Profile, seed []byte) mlwe.Poly {
	c := mlwe.Poly{Coeffs: make([]uint64, p.N)}
	sampleInBall(c.Coeffs, seed, p.Tau, p.Q, p.N)
	return c
}

// rejNTTPoly samples coefficients uniformly from SHAKE-128(seed||nonce)
// by rejection, in [0, q). FIPS 204 RejNTTPoly / circl polyDeriveUniform.
func rejNTTPoly(out []uint64, seed *[32]byte, nonce uint16, q uint64, n int) {
	var iv [34]byte
	copy(iv[:32], seed[:])
	iv[32] = byte(nonce)
	iv[33] = byte(nonce >> 8)
	h := sha3.NewShake128()
	_, _ = h.Write(iv[:])
	var buf [168]byte
	i := 0
	for i < n {
		_, _ = h.Read(buf[:])
		for j := 0; j+3 <= 168 && i < n; j += 3 {
			t := (uint32(buf[j]) | (uint32(buf[j+1]) << 8) | (uint32(buf[j+2]) << 16)) & 0x7fffff
			if uint64(t) < q {
				out[i] = uint64(t)
				i++
			}
		}
	}
}

// cbdEta samples coefficients in [-eta, eta] (centered binomial) from
// SHAKE-256(seed||nonce), storing them in [q-eta, q+eta]. FIPS 204
// RejBoundedPoly / circl polyDeriveUniformLeqEta. Supports eta in {2,4}.
func cbdEta(out []uint64, seed *[64]byte, nonce uint16, eta, q uint64, n int) {
	var iv [66]byte
	copy(iv[:64], seed[:])
	iv[64] = byte(nonce)
	iv[65] = byte(nonce >> 8)
	h := sha3.NewShake256()
	_, _ = h.Write(iv[:])
	var buf [136]byte
	i := 0
	for i < n {
		_, _ = h.Read(buf[:])
		for j := 0; j < 136 && i < n; j++ {
			t1 := uint32(buf[j]) & 15
			t2 := uint32(buf[j]) >> 4
			if eta == 2 {
				if t1 <= 14 {
					t1 -= ((205 * t1) >> 10) * 5
					out[i] = q + eta - uint64(t1)
					i++
				}
				if t2 <= 14 && i < n {
					t2 -= ((205 * t2) >> 10) * 5
					out[i] = q + eta - uint64(t2)
					i++
				}
			} else if eta == 4 {
				if uint64(t1) <= 2*eta {
					out[i] = q + eta - uint64(t1)
					i++
				}
				if uint64(t2) <= 2*eta && i < n {
					out[i] = q + eta - uint64(t2)
					i++
				}
			}
		}
	}
}

// expandMaskPoly samples one mask polynomial from SHAKE-256(seed||nonce)
// by unpacking a (gamma1Bits+1)-bit-packed stream into coefficients
// uniform in (-gamma1, gamma1], normalized to [0, q). FIPS 204
// ExpandMask per-poly / circl polyDeriveUniformLeGamma1.
func expandMaskPoly(out []uint64, seed *[64]byte, nonce uint16, gamma1Bits uint32, q uint64, n int) {
	var iv [66]byte
	copy(iv[:64], seed[:])
	iv[64] = byte(nonce)
	iv[65] = byte(nonce >> 8)
	h := sha3.NewShake256()
	_, _ = h.Write(iv[:])
	size := int((gamma1Bits + 1) * uint32(n) / 8)
	buf := make([]byte, size)
	_, _ = h.Read(buf)
	unpackGamma1(out, buf, gamma1Bits, q, n)
}

// unpackGamma1 unpacks a gamma1-bit-packed polynomial (centered uniform
// in (-gamma1, gamma1]) into out, normalized to [0, q). gamma1Bits is
// 17 or 19. Lifted from circl polyUnpackLeGamma1.
func unpackGamma1(out []uint64, buf []byte, gamma1Bits uint32, q uint64, n int) {
	gamma1 := uint32(1) << gamma1Bits
	qq := uint32(q)
	if gamma1Bits == 17 {
		j := 0
		size := (17 + 1) * n / 8
		for i := 0; i+9 <= size; i += 9 {
			p0 := uint32(buf[i]) | (uint32(buf[i+1]) << 8) | (uint32(buf[i+2]&0x3) << 16)
			p1 := uint32(buf[i+2]>>2) | (uint32(buf[i+3]) << 6) | (uint32(buf[i+4]&0xf) << 14)
			p2 := uint32(buf[i+4]>>4) | (uint32(buf[i+5]) << 4) | (uint32(buf[i+6]&0x3f) << 12)
			p3 := uint32(buf[i+6]>>6) | (uint32(buf[i+7]) << 2) | (uint32(buf[i+8]) << 10)
			p0 = gamma1 - p0
			p1 = gamma1 - p1
			p2 = gamma1 - p2
			p3 = gamma1 - p3
			p0 += uint32(int32(p0)>>31) & qq
			p1 += uint32(int32(p1)>>31) & qq
			p2 += uint32(int32(p2)>>31) & qq
			p3 += uint32(int32(p3)>>31) & qq
			out[j] = uint64(p0)
			out[j+1] = uint64(p1)
			out[j+2] = uint64(p2)
			out[j+3] = uint64(p3)
			j += 4
		}
	} else if gamma1Bits == 19 {
		j := 0
		size := (19 + 1) * n / 8
		for i := 0; i+5 <= size; i += 5 {
			p0 := uint32(buf[i]) | (uint32(buf[i+1]) << 8) | (uint32(buf[i+2]&0xf) << 16)
			p1 := uint32(buf[i+2]>>4) | (uint32(buf[i+3]) << 4) | (uint32(buf[i+4]) << 12)
			p0 = gamma1 - p0
			p1 = gamma1 - p1
			p0 += uint32(int32(p0)>>31) & qq
			p1 += uint32(int32(p1)>>31) & qq
			out[j] = uint64(p0)
			out[j+1] = uint64(p1)
			j += 2
		}
	}
}

// sampleInBall fills out with a challenge polynomial: tau coefficients
// in {-1,+1} (the -1 stored as q-1), the rest zero. FIPS 204
// SampleInBall / circl polyDeriveUniformBall.
func sampleInBall(out []uint64, seed []byte, tau int, q uint64, n int) {
	var buf [136]byte
	h := sha3.NewShake256()
	_, _ = h.Write(seed)
	_, _ = h.Read(buf[:])

	signs := binary.LittleEndian.Uint64(buf[:8])
	bufOff := 8

	for i := uint16(uint16(n) - uint16(tau)); i < uint16(n); i++ {
		var b uint16
		for {
			if bufOff >= 136 {
				_, _ = h.Read(buf[:])
				bufOff = 0
			}
			b = uint16(buf[bufOff])
			bufOff++
			if b <= i {
				break
			}
		}
		out[i] = out[b]
		out[b] = 1
		// XOR-trick: 1 ^ (1 | (q-1)) = q-1, selecting -1 when sign set.
		out[b] ^= (-(signs & 1)) & (1 | (q - 1))
		signs >>= 1
	}
}
