// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package zeroize

import "testing"

func TestBytes(t *testing.T) {
	b := []byte{1, 2, 3, 4, 5}
	Bytes(b)
	for i, v := range b {
		if v != 0 {
			t.Fatalf("byte %d not zeroed: %d", i, v)
		}
	}
	Bytes(nil) // must not panic
}

func TestU32(t *testing.T) {
	s := []uint32{9, 8, 7}
	U32(s)
	for i, v := range s {
		if v != 0 {
			t.Fatalf("u32 %d not zeroed: %d", i, v)
		}
	}
	U32(nil)
}

func TestU64(t *testing.T) {
	s := []uint64{1 << 40, 1 << 50}
	U64(s)
	for i, v := range s {
		if v != 0 {
			t.Fatalf("u64 %d not zeroed: %d", i, v)
		}
	}
	U64(nil)
}
