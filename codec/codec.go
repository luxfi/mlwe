// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package codec provides the bit-packing and bounded framing primitives
// shared by every Lux Module-LWE wire format.
//
// Two concerns live here, kept separate:
//
//   - PackBits / UnpackBits: little-endian (LSB-first) fixed-width
//     coefficient packing. This is the FIPS 204 packing convention
//     (PackW1, PackT1, etc. are all LSB-first b-bit packings) and the
//     Ringtail packing convention. One implementation, every width.
//
//   - ReadVector: a length-prefixed frame reader that HARD-CAPS the
//     declared element count before allocating. Every length prefix in
//     a Lux wire format must be read through this so that a hostile
//     frame claiming a 4-billion-element vector cannot drive an
//     unbounded allocation or unbounded recursion.
package codec

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

// Framing errors.
var (
	// ErrFrameTooLarge is returned by ReadVector when a frame's
	// declared element count exceeds the caller's hard cap. This is the
	// recursion-bomb / allocation-bomb guard.
	ErrFrameTooLarge = errors.New("mlwe/codec: declared vector length exceeds maximum")

	// ErrBitWidth is returned when a bit width is outside (0, 64].
	ErrBitWidth = errors.New("mlwe/codec: bit width must be in (0, 64]")
)

// PackBits packs the low `bits` bits of each coefficient into a byte
// slice, LSB-first within each coefficient and LSB-first within each
// byte. The output length is ceil(len(coeffs)*bits/8).
//
// This reproduces the FIPS 204 b-bit packers byte-for-byte: e.g.
// PackBits(t1, 10) equals FIPS 204 PolyPackT1, and PackBits(w1, 4)
// equals PolyPackW1 for ML-DSA-65/87.
//
// bits must be in (0, 64]; PackBits panics otherwise, since a wrong
// width is a programmer error, not attacker-controlled input.
func PackBits(coeffs []uint64, bits int) []byte {
	if bits <= 0 || bits > 64 {
		panic(ErrBitWidth)
	}
	out := make([]byte, (len(coeffs)*bits+7)/8)
	bitpos := 0
	for _, c := range coeffs {
		for b := 0; b < bits; b++ {
			if (c>>uint(b))&1 == 1 {
				out[bitpos>>3] |= 1 << uint(bitpos&7)
			}
			bitpos++
		}
	}
	return out
}

// UnpackBits is the inverse of PackBits: it reads n coefficients of
// `bits` bits each, LSB-first, from data.
//
// UnpackBits is total and panic-free on its data argument: if data is
// shorter than ceil(n*bits/8), the missing high bits read as zero. This
// keeps the function safe on hostile input; callers that need to reject
// a short frame must validate the frame length first (e.g. via a
// length read through ReadVector). bits must be in (0, 64].
func UnpackBits(data []byte, n, bits int) []uint64 {
	if bits <= 0 || bits > 64 {
		panic(ErrBitWidth)
	}
	out := make([]uint64, n)
	bitpos := 0
	for i := 0; i < n; i++ {
		var v uint64
		for b := 0; b < bits; b++ {
			byteIdx := bitpos >> 3
			if byteIdx < len(data) && (data[byteIdx]>>uint(bitpos&7))&1 == 1 {
				v |= 1 << uint(b)
			}
			bitpos++
		}
		out[i] = v
	}
	return out
}

// ReadVector reads a 4-byte big-endian length prefix from r, rejects it
// if it exceeds max, and then decodes exactly that many elements by
// calling dec once per element.
//
// max is the caller's trusted upper bound on the element count (e.g.
// the parameter set's module dimension, or a validator-set size). The
// length is checked BEFORE any per-element allocation, so a frame
// claiming an enormous count fails fast with ErrFrameTooLarge instead
// of attempting a huge make or unbounded recursion.
//
// dec is responsible for reading one element's bytes from r and must
// return a non-nil error on a short read; ReadVector propagates it.
func ReadVector[T any](r io.Reader, max uint32, dec func(io.Reader) (T, error)) ([]T, error) {
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return nil, fmt.Errorf("mlwe/codec: read length prefix: %w", err)
	}
	n := binary.BigEndian.Uint32(lenBuf[:])
	if n > max {
		return nil, fmt.Errorf("%w: declared %d, max %d", ErrFrameTooLarge, n, max)
	}
	out := make([]T, 0, n)
	for i := uint32(0); i < n; i++ {
		v, err := dec(r)
		if err != nil {
			return nil, fmt.Errorf("mlwe/codec: decode element %d/%d: %w", i, n, err)
		}
		out = append(out, v)
	}
	return out, nil
}

// WriteVector writes a 4-byte big-endian length prefix followed by each
// element via enc. It is the inverse of ReadVector and exists so that a
// producer and a bounded consumer share one framing. enc must write
// exactly the bytes that dec will read.
func WriteVector[T any](w io.Writer, elems []T, enc func(io.Writer, T) error) error {
	var lenBuf [4]byte
	binary.BigEndian.PutUint32(lenBuf[:], uint32(len(elems)))
	if _, err := w.Write(lenBuf[:]); err != nil {
		return fmt.Errorf("mlwe/codec: write length prefix: %w", err)
	}
	for i, e := range elems {
		if err := enc(w, e); err != nil {
			return fmt.Errorf("mlwe/codec: encode element %d/%d: %w", i, len(elems), err)
		}
	}
	return nil
}
