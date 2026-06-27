// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package codec

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math/rand"
	"testing"
)

// TestPackUnpack_RoundTripEveryWidth proves Unpack(Pack(x)) == x for
// every bit width 1..32 over random coefficients masked to the width.
func TestPackUnpack_RoundTripEveryWidth(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const n = 256
	for bits := 1; bits <= 32; bits++ {
		mask := uint64(1)<<uint(bits) - 1
		in := make([]uint64, n)
		for i := range in {
			in[i] = rng.Uint64() & mask
		}
		packed := PackBits(in, bits)
		if want := (n*bits + 7) / 8; len(packed) != want {
			t.Fatalf("bits=%d: packed len %d, want %d", bits, len(packed), want)
		}
		out := UnpackBits(packed, n, bits)
		for i := range in {
			if out[i] != in[i] {
				t.Fatalf("bits=%d: coeff %d round-trip %d != %d", bits, i, out[i], in[i])
			}
		}
	}
}

// TestPackBits_KnownVector pins the LSB-first packing to a hand-computed
// example so a future refactor can't silently change the wire order.
func TestPackBits_KnownVector(t *testing.T) {
	// [1,2,3] at 4 bits, LSB-first: 0x21, 0x03.
	got := PackBits([]uint64{1, 2, 3}, 4)
	want := []byte{0x21, 0x03}
	if !bytes.Equal(got, want) {
		t.Fatalf("PackBits([1,2,3],4) = % x, want % x", got, want)
	}
	back := UnpackBits(want, 3, 4)
	if back[0] != 1 || back[1] != 2 || back[2] != 3 {
		t.Fatalf("UnpackBits round-trip = %v", back)
	}
}

// TestUnpackBits_ShortDataZeroPads confirms the unpacker is total: a
// short buffer yields zeros for the missing high bits rather than
// reading out of bounds or panicking.
func TestUnpackBits_ShortDataZeroPads(t *testing.T) {
	out := UnpackBits([]byte{0xFF}, 4, 4) // 1 byte = 2 nibbles of data
	want := []uint64{0xF, 0xF, 0, 0}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("coeff %d = %d, want %d", i, out[i], want[i])
		}
	}
}

func TestPackBits_RejectsBadWidth(t *testing.T) {
	for _, bits := range []int{0, -1, 65} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("PackBits(bits=%d) did not panic", bits)
				}
			}()
			PackBits([]uint64{1}, bits)
		}()
	}
}

func TestUnpackBits_RejectsBadWidth(t *testing.T) {
	for _, bits := range []int{0, -1, 65} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("UnpackBits(bits=%d) did not panic", bits)
				}
			}()
			UnpackBits([]byte{0}, 1, bits)
		}()
	}
}

// failWriter fails on the Nth write (1-indexed), to exercise WriteVector
// error paths.
type failWriter struct{ failOn, n int }

func (f *failWriter) Write(p []byte) (int, error) {
	f.n++
	if f.n >= f.failOn {
		return 0, errors.New("forced write failure")
	}
	return len(p), nil
}

func TestWriteVector_LengthPrefixError(t *testing.T) {
	if err := WriteVector(&failWriter{failOn: 1}, []uint64{1, 2}, putU64); err == nil {
		t.Fatalf("expected length-prefix write error")
	}
}

func TestWriteVector_ElementError(t *testing.T) {
	// First write (length prefix) succeeds, the element encode fails.
	if err := WriteVector(&failWriter{failOn: 2}, []uint64{1, 2}, putU64); err == nil {
		t.Fatalf("expected element write error")
	}
}

// putU64 / readU64 are a trivial element codec for ReadVector tests.
func putU64(w io.Writer, v uint64) error {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], v)
	_, err := w.Write(b[:])
	return err
}

func getU64(r io.Reader) (uint64, error) {
	var b [8]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint64(b[:]), nil
}

// TestReadVector_RoundTrip confirms WriteVector/ReadVector are inverse.
func TestReadVector_RoundTrip(t *testing.T) {
	elems := []uint64{1, 2, 3, 1 << 40, 0}
	var buf bytes.Buffer
	if err := WriteVector(&buf, elems, putU64); err != nil {
		t.Fatalf("WriteVector: %v", err)
	}
	got, err := ReadVector(&buf, 16, getU64)
	if err != nil {
		t.Fatalf("ReadVector: %v", err)
	}
	if len(got) != len(elems) {
		t.Fatalf("len %d != %d", len(got), len(elems))
	}
	for i := range elems {
		if got[i] != elems[i] {
			t.Fatalf("elem %d: %d != %d", i, got[i], elems[i])
		}
	}
}

// TestReadVector_RejectsOversizedFrame is the explicit DoS regression: a
// frame that declares a 4-billion-element count must be rejected at the
// length check, BEFORE any per-element allocation, instead of attempting
// a huge make or unbounded read.
func TestReadVector_RejectsOversizedFrame(t *testing.T) {
	var frame [4]byte
	binary.BigEndian.PutUint32(frame[:], 0xFFFFFFFF) // ~4.29e9 elements
	// No body bytes at all: a non-bounded reader would try to allocate
	// for 4.29e9 elements first; ours must reject on the length alone.
	got, err := ReadVector(bytes.NewReader(frame[:]), 1024, getU64)
	if !errors.Is(err, ErrFrameTooLarge) {
		t.Fatalf("expected ErrFrameTooLarge, got err=%v got=%v", err, got)
	}
}

// TestReadVector_AtMaxBoundary confirms a count equal to max is accepted
// and a count of max+1 is rejected.
func TestReadVector_AtMaxBoundary(t *testing.T) {
	const max = 4
	elems := []uint64{10, 20, 30, 40}
	var ok bytes.Buffer
	_ = WriteVector(&ok, elems, putU64)
	if _, err := ReadVector(&ok, max, getU64); err != nil {
		t.Fatalf("count==max should be accepted: %v", err)
	}

	var over bytes.Buffer
	_ = WriteVector(&over, append(elems, 50), putU64) // 5 > max
	if _, err := ReadVector(&over, max, getU64); !errors.Is(err, ErrFrameTooLarge) {
		t.Fatalf("count==max+1 should be rejected, got %v", err)
	}
}

// TestReadVector_TruncatedBody surfaces a short read inside an element.
func TestReadVector_TruncatedBody(t *testing.T) {
	var frame []byte
	var lp [4]byte
	binary.BigEndian.PutUint32(lp[:], 3) // declares 3 elements
	frame = append(frame, lp[:]...)
	frame = append(frame, 0x00, 0x01) // only 2 bytes of body (need 24)
	if _, err := ReadVector(bytes.NewReader(frame), 8, getU64); err == nil {
		t.Fatalf("expected error on truncated body")
	}
}

// TestReadVector_TruncatedLengthPrefix surfaces a short length read.
func TestReadVector_TruncatedLengthPrefix(t *testing.T) {
	if _, err := ReadVector(bytes.NewReader([]byte{0x00, 0x01}), 8, getU64); err == nil {
		t.Fatalf("expected error on truncated length prefix")
	}
}
