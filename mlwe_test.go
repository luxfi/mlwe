// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package mlwe

import "testing"

func TestPolyClone(t *testing.T) {
	p := Poly{Coeffs: []uint64{1, 2, 3}}
	c := p.Clone()
	c.Coeffs[0] = 99
	if p.Coeffs[0] != 1 {
		t.Fatalf("Clone aliased the backing array")
	}
	if len(c.Coeffs) != 3 || c.Coeffs[1] != 2 || c.Coeffs[2] != 3 {
		t.Fatalf("Clone copied wrong values: %v", c.Coeffs)
	}
}

func TestPolyVecClone(t *testing.T) {
	v := PolyVec{{Coeffs: []uint64{1}}, {Coeffs: []uint64{2}}}
	c := v.Clone()
	c[0].Coeffs[0] = 7
	if v[0].Coeffs[0] != 1 {
		t.Fatalf("PolyVec.Clone aliased element backing array")
	}
	if len(c) != 2 || c[1].Coeffs[0] != 2 {
		t.Fatalf("PolyVec.Clone wrong: %v", c)
	}
}

func TestPolyMatDims(t *testing.T) {
	var empty PolyMat
	if empty.Rows() != 0 || empty.Cols() != 0 {
		t.Fatalf("empty mat dims %d x %d", empty.Rows(), empty.Cols())
	}
	m := PolyMat{
		PolyVec{{Coeffs: []uint64{0}}, {Coeffs: []uint64{0}}, {Coeffs: []uint64{0}}},
		PolyVec{{Coeffs: []uint64{0}}, {Coeffs: []uint64{0}}, {Coeffs: []uint64{0}}},
	}
	if m.Rows() != 2 || m.Cols() != 3 {
		t.Fatalf("mat dims %d x %d, want 2 x 3", m.Rows(), m.Cols())
	}
}
