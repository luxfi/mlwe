// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package modulesis_test

import (
	"math/big"
	"math/rand"
	"testing"

	"github.com/luxfi/mlwe"
	"github.com/luxfi/mlwe/modulesis"
	"github.com/luxfi/mlwe/ring/mldsa"
)

const q = 8380417

func randPoly(r mlwe.Ring, rng *rand.Rand) mlwe.Poly {
	p := r.NewPoly()
	for i := range p.Coeffs {
		p.Coeffs[i] = uint64(rng.Int63n(q))
	}
	return p
}

// schoolbookMul: independent negacyclic reference (X^256 = -1).
func schoolbookMul(a, b []uint64) []uint64 {
	n := len(a)
	acc := make([]int64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			prod := int64(a[i]) * int64(b[j]) % int64(q)
			if k := i + j; k < n {
				acc[k] = (acc[k] + prod) % int64(q)
			} else {
				acc[k-n] = (acc[k-n] - prod) % int64(q)
			}
		}
	}
	out := make([]uint64, n)
	for k, v := range acc {
		v %= int64(q)
		if v < 0 {
			v += int64(q)
		}
		out[k] = uint64(v)
	}
	return out
}

func TestMatVec_VsSchoolbook(t *testing.T) {
	r := mldsa.MustNew(mldsa.Profile87) // K=8, L=7
	rng := rand.New(rand.NewSource(1))
	const K, L = 8, 7
	a := make(mlwe.PolyMat, K)
	for i := 0; i < K; i++ {
		a[i] = make(mlwe.PolyVec, L)
		for j := 0; j < L; j++ {
			a[i][j] = randPoly(r, rng)
		}
	}
	x := make(mlwe.PolyVec, L)
	for j := 0; j < L; j++ {
		x[j] = randPoly(r, rng)
	}
	got := modulesis.MatVec(r, a, x)
	for i := 0; i < K; i++ {
		want := make([]uint64, r.N())
		for j := 0; j < L; j++ {
			term := schoolbookMul(a[i][j].Coeffs, x[j].Coeffs)
			for c := range want {
				want[c] = (want[c] + term[c]) % q
			}
		}
		for c := range want {
			if got[i].Coeffs[c] != want[c] {
				t.Fatalf("row %d coeff %d: %d != %d", i, c, got[i].Coeffs[c], want[c])
			}
		}
	}
}

func TestMatVec_DimensionMismatchPanics(t *testing.T) {
	r := mldsa.MustNew(mldsa.Profile65)
	a := mlwe.PolyMat{mlwe.PolyVec{r.NewPoly(), r.NewPoly()}} // 1x2
	x := mlwe.PolyVec{r.NewPoly()}                            // length 1 != 2
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on dimension mismatch")
		}
	}()
	modulesis.MatVec(r, a, x)
}

func TestInfNorm_Vector(t *testing.T) {
	r := mldsa.MustNew(mldsa.Profile65)
	p0 := r.NewPoly()
	p0.Coeffs[0] = 7 // mag 7
	p1 := r.NewPoly()
	p1.Coeffs[0] = q - 100 // centered -100 -> mag 100
	v := mlwe.PolyVec{p0, p1}
	if got := modulesis.InfNorm(r, v); got != 100 {
		t.Fatalf("InfNorm vec = %d, want 100", got)
	}
}

func TestL2Sq(t *testing.T) {
	r := mldsa.MustNew(mldsa.Profile65)
	p := r.NewPoly()
	p.Coeffs[0] = 3     // mag 3
	p.Coeffs[1] = q - 4 // centered -4 -> mag 4
	p.Coeffs[2] = 0     // 0
	v := mlwe.PolyVec{p}
	got := modulesis.L2Sq(r, v)
	if got.Cmp(big.NewInt(9+16)) != 0 {
		t.Fatalf("L2Sq = %s, want 25", got.String())
	}
}

func TestVectorRounding_Decompose(t *testing.T) {
	r := mldsa.MustNew(mldsa.Profile65)
	rng := rand.New(rand.NewSource(2))
	v := mlwe.PolyVec{randPoly(r, rng), randPoly(r, rng), randPoly(r, rng)}
	low, high := modulesis.Decompose(r, v)
	hb := modulesis.HighBits(r, v)
	g2 := int64(2 * mldsa.Profile65.Gamma2)
	for k := range v {
		for i := range v[k].Coeffs {
			if hb[k].Coeffs[i] != high[k].Coeffs[i] {
				t.Fatalf("vec %d coeff %d: HighBits != Decompose high", k, i)
			}
			centered := int64(low[k].Coeffs[i])
			if low[k].Coeffs[i] > (q-1)/2 {
				centered -= q
			}
			rec := (int64(high[k].Coeffs[i])*g2 + centered) % q
			if rec < 0 {
				rec += q
			}
			if uint64(rec) != v[k].Coeffs[i] {
				t.Fatalf("vec %d coeff %d recompose %d != %d", k, i, rec, v[k].Coeffs[i])
			}
		}
	}
}

func TestVectorRounding_Power2Round(t *testing.T) {
	r := mldsa.MustNew(mldsa.Profile65)
	rng := rand.New(rand.NewSource(3))
	v := mlwe.PolyVec{randPoly(r, rng), randPoly(r, rng)}
	v0, v1 := modulesis.Power2Round(r, v)
	for k := range v {
		for i := range v[k].Coeffs {
			centered := int64(v0[k].Coeffs[i])
			if v0[k].Coeffs[i] > (q-1)/2 {
				centered -= q
			}
			rec := (int64(v1[k].Coeffs[i])<<13 + centered) % q
			if rec < 0 {
				rec += q
			}
			if uint64(rec) != v[k].Coeffs[i] {
				t.Fatalf("vec %d coeff %d recompose %d != %d", k, i, rec, v[k].Coeffs[i])
			}
		}
	}
}

func TestVectorUseHint_AndWeight(t *testing.T) {
	r := mldsa.MustNew(mldsa.Profile65)
	rng := rand.New(rand.NewSource(4))
	w := mlwe.PolyVec{randPoly(r, rng), randPoly(r, rng)}
	w1 := modulesis.HighBits(r, w)
	zeroHint := mlwe.PolyVec{r.NewPoly(), r.NewPoly()}
	got := modulesis.UseHint(r, zeroHint, w)
	for k := range w {
		for i := range w[k].Coeffs {
			if got[k].Coeffs[i] != w1[k].Coeffs[i] {
				t.Fatalf("UseHint(0,w) != HighBits(w) at vec %d coeff %d", k, i)
			}
		}
	}
	// HintWeight counts set bits.
	hint := mlwe.PolyVec{r.NewPoly(), r.NewPoly()}
	hint[0].Coeffs[1] = 1
	hint[0].Coeffs[5] = 1
	hint[1].Coeffs[9] = 1
	if w := modulesis.HintWeight(hint); w != 3 {
		t.Fatalf("HintWeight = %d, want 3", w)
	}
}

func TestUseHint_DimensionMismatchPanics(t *testing.T) {
	r := mldsa.MustNew(mldsa.Profile65)
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on UseHint dimension mismatch")
		}
	}()
	modulesis.UseHint(r, mlwe.PolyVec{r.NewPoly()}, mlwe.PolyVec{r.NewPoly(), r.NewPoly()})
}
