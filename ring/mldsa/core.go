// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package mldsa is the concrete FIPS 204 (ML-DSA) Module-LWE ring:
// R_q = Z_q[X]/(X^256 + 1) with q = 8 380 417 = 2^23 - 2^13 + 1.
//
// The arithmetic core in this file is lifted VERBATIM, at the value
// level, from the audited Pulsar reference (pulsar/mldsa_lattice.go and
// pulsar/boundary.go). Verbatim means identical arithmetic, which means
// identical FIPS 204 bytes: a public key or signature produced through
// these primitives is byte-for-byte what cloudflare/circl's
// mldsa{44,65,87} produces, and verifies under the unmodified library.
//
// The constant-time Montgomery convention (R = 2^32) is preserved
// exactly, so no secret-dependent branch, index, or modulo is
// introduced. The exported mlwe.Ring / mlwe.RoundingRing wrapper lives
// in ring.go; this file is the package-private fast core operating on a
// fixed [256]uint32 polynomial.
package mldsa

// Ring parameters per FIPS 204 §4, fixed across all three parameter
// sets (ML-DSA-44/65/87).
const (
	mldsaN        = 256        // ring degree
	mldsaQ        = 8380417    // 2^23 - 2^13 + 1
	mldsaQinv     = 4236238847 // -(q^-1) mod 2^32
	mldsaROver256 = 41978      // (256)^-1 * R^2 mod q, R = 2^32
	mldsaD        = 13         // dropped low bits (Power2Round)

	mldsaGamma2P65 = 261888 // (q-1)/32 for ML-DSA-65/87
	mldsaGamma2P44 = 95232  // (q-1)/88 for ML-DSA-44
)

// poly is the ML-DSA polynomial in R_q. Coefficients are stored mod q
// (or mod 2q in transient states); arithmetic is performed in either
// Montgomery form (NTT-domain work) or standard form.
type poly [mldsaN]uint32

// reduceLe2Q computes y < 2q with y == x (mod q). Constant-time.
func reduceLe2Q(x uint32) uint32 {
	x1 := x >> 23
	x2 := x & 0x7FFFFF
	return x2 + (x1 << 13) - x1
}

// le2qModQ computes x mod q for 0 <= x < 2q.
func le2qModQ(x uint32) uint32 {
	x -= mldsaQ
	mask := uint32(int32(x) >> 31)
	return x + (mask & mldsaQ)
}

// modQ computes x mod q via two-step reduction.
func modQ(x uint32) uint32 { return le2qModQ(reduceLe2Q(x)) }

// montReduceLe2Q reduces x * R^-1 mod q assuming x < q * 2^32. Result
// in [0, 2q).
func montReduceLe2Q(x uint64) uint32 {
	m := (x * mldsaQinv) & 0xffffffff
	return uint32((x + m*uint64(mldsaQ)) >> 32)
}

// power2round returns a0+q and a1 with a = a1*2^D + a0 and
// -2^(D-1) < a0 <= 2^(D-1). FIPS 204 Power2Round.
func power2round(a uint32) (a0plusQ, a1 uint32) {
	a0 := a & ((1 << mldsaD) - 1)
	a0 -= (1 << (mldsaD - 1)) + 1
	a0 += uint32(int32(a0)>>31) & (1 << mldsaD)
	a0 -= (1 << (mldsaD - 1)) - 1
	a0plusQ = mldsaQ + a0
	a1 = (a - a0) >> mldsaD
	return
}

// decompose splits 0 <= a < q into a0, a1 with a = a1*alpha + a0 for
// alpha = 2*gamma2. Returns a0+q (in [1, q) for normalized input) and
// a1. FIPS 204 Decompose.
func decompose(a uint32, gamma2 uint32) (a0plusQ, a1 uint32) {
	a1 = (a + 127) >> 7
	if gamma2 == mldsaGamma2P65 {
		a1 = (a1*1025 + (1 << 21)) >> 22
		a1 &= 15
	} else if gamma2 == mldsaGamma2P44 {
		a1 = (a1*11275 + (1 << 23)) >> 24
		a1 ^= uint32(int32(43-a1)>>31) & a1
	} else {
		return 0, 0
	}
	alpha := 2 * gamma2
	a0plusQ = a - a1*alpha
	a0plusQ += uint32(int32(a0plusQ-(mldsaQ-1)/2)>>31) & mldsaQ
	return
}

// reduceLe2Q normalizes all coefficients of p to < 2q in place.
func (p *poly) reduceLe2Q() {
	for i := 0; i < mldsaN; i++ {
		p[i] = reduceLe2Q(p[i])
	}
}

// normalize reduces each coefficient to [0, q).
func (p *poly) normalize() {
	for i := 0; i < mldsaN; i++ {
		p[i] = modQ(p[i])
	}
}

// add sets p = a + b (per-coefficient, no reduction).
func (p *poly) add(a, b *poly) {
	for i := 0; i < mldsaN; i++ {
		p[i] = a[i] + b[i]
	}
}

// sub sets p = a - b mod 2q assuming coefficients of b are < 2q.
func (p *poly) sub(a, b *poly) {
	for i := 0; i < mldsaN; i++ {
		p[i] = a[i] + (2*mldsaQ - b[i])
	}
}

// mulHat sets p = a * b pointwise (NTT-domain Montgomery mul). Assumes
// a[i]*b[i] < 2^32 * q.
func (p *poly) mulHat(a, b *poly) {
	for i := 0; i < mldsaN; i++ {
		p[i] = montReduceLe2Q(uint64(a[i]) * uint64(b[i]))
	}
}

// exceeds reports whether the centered-rep max-abs coefficient of p is
// >= bound. Assumes p normalized. Leaks position-of-break (allowed
// under FIPS 204 rejection sampling) but not the value.
func (p *poly) exceeds(bound uint32) bool {
	for i := 0; i < mldsaN; i++ {
		x := int32((mldsaQ-1)/2) - int32(p[i])
		x ^= x >> 31
		x = int32((mldsaQ-1)/2) - x
		if uint32(x) >= bound {
			return true
		}
	}
	return false
}

// power2Round splits p into p0PlusQ and p1 per FIPS 204 Power2Round.
// Requires p normalized.
func (p *poly) power2Round(p0PlusQ, p1 *poly) {
	for i := 0; i < mldsaN; i++ {
		p0PlusQ[i], p1[i] = power2round(p[i])
	}
}

// polyDotHat sets p = sum_i a[i] * b[i] pointwise in Montgomery form.
func polyDotHat(p *poly, a, b []poly) {
	if len(a) != len(b) {
		panic("mldsa: polyDotHat length mismatch")
	}
	var t poly
	*p = poly{}
	for i := range a {
		t.mulHat(&a[i], &b[i])
		p.add(&t, p)
	}
}

// nttZetas: powers of the 512th root of unity zeta=1753 in Montgomery
// form per FIPS 204 §3.6. Byte-identical to circl's Zetas.
var nttZetas = [mldsaN]uint32{
	4193792, 25847, 5771523, 7861508, 237124, 7602457, 7504169,
	466468, 1826347, 2353451, 8021166, 6288512, 3119733, 5495562,
	3111497, 2680103, 2725464, 1024112, 7300517, 3585928, 7830929,
	7260833, 2619752, 6271868, 6262231, 4520680, 6980856, 5102745,
	1757237, 8360995, 4010497, 280005, 2706023, 95776, 3077325,
	3530437, 6718724, 4788269, 5842901, 3915439, 4519302, 5336701,
	3574422, 5512770, 3539968, 8079950, 2348700, 7841118, 6681150,
	6736599, 3505694, 4558682, 3507263, 6239768, 6779997, 3699596,
	811944, 531354, 954230, 3881043, 3900724, 5823537, 2071892,
	5582638, 4450022, 6851714, 4702672, 5339162, 6927966, 3475950,
	2176455, 6795196, 7122806, 1939314, 4296819, 7380215, 5190273,
	5223087, 4747489, 126922, 3412210, 7396998, 2147896, 2715295,
	5412772, 4686924, 7969390, 5903370, 7709315, 7151892, 8357436,
	7072248, 7998430, 1349076, 1852771, 6949987, 5037034, 264944,
	508951, 3097992, 44288, 7280319, 904516, 3958618, 4656075,
	8371839, 1653064, 5130689, 2389356, 8169440, 759969, 7063561,
	189548, 4827145, 3159746, 6529015, 5971092, 8202977, 1315589,
	1341330, 1285669, 6795489, 7567685, 6940675, 5361315, 4499357,
	4751448, 3839961, 2091667, 3407706, 2316500, 3817976, 5037939,
	2244091, 5933984, 4817955, 266997, 2434439, 7144689, 3513181,
	4860065, 4621053, 7183191, 5187039, 900702, 1859098, 909542,
	819034, 495491, 6767243, 8337157, 7857917, 7725090, 5257975,
	2031748, 3207046, 4823422, 7855319, 7611795, 4784579, 342297,
	286988, 5942594, 4108315, 3437287, 5038140, 1735879, 203044,
	2842341, 2691481, 5790267, 1265009, 4055324, 1247620, 2486353,
	1595974, 4613401, 1250494, 2635921, 4832145, 5386378, 1869119,
	1903435, 7329447, 7047359, 1237275, 5062207, 6950192, 7929317,
	1312455, 3306115, 6417775, 7100756, 1917081, 5834105, 7005614,
	1500165, 777191, 2235880, 3406031, 7838005, 5548557, 6709241,
	6533464, 5796124, 4656147, 594136, 4603424, 6366809, 2432395,
	2454455, 8215696, 1957272, 3369112, 185531, 7173032, 5196991,
	162844, 1616392, 3014001, 810149, 1652634, 4686184, 6581310,
	5341501, 3523897, 3866901, 269760, 2213111, 7404533, 1717735,
	472078, 7953734, 1723600, 6577327, 1910376, 6712985, 7276084,
	8119771, 4546524, 5441381, 6144432, 7959518, 6094090, 183443,
	7403526, 1612842, 4834730, 7826001, 3919660, 8332111, 7018208,
	3937738, 1400424, 7534263, 1976782,
}

// nttInvZetas: inverse twiddle factors per FIPS 204 §3.6. Byte-
// identical to circl's InvZetas.
var nttInvZetas = [mldsaN]uint32{
	6403635, 846154, 6979993, 4442679, 1362209, 48306, 4460757,
	554416, 3545687, 6767575, 976891, 8196974, 2286327, 420899,
	2235985, 2939036, 3833893, 260646, 1104333, 1667432, 6470041,
	1803090, 6656817, 426683, 7908339, 6662682, 975884, 6167306,
	8110657, 4513516, 4856520, 3038916, 1799107, 3694233, 6727783,
	7570268, 5366416, 6764025, 8217573, 3183426, 1207385, 8194886,
	5011305, 6423145, 164721, 5925962, 5948022, 2013608, 3776993,
	7786281, 3724270, 2584293, 1846953, 1671176, 2831860, 542412,
	4974386, 6144537, 7603226, 6880252, 1374803, 2546312, 6463336,
	1279661, 1962642, 5074302, 7067962, 451100, 1430225, 3318210,
	7143142, 1333058, 1050970, 6476982, 6511298, 2994039, 3548272,
	5744496, 7129923, 3767016, 6784443, 5894064, 7132797, 4325093,
	7115408, 2590150, 5688936, 5538076, 8177373, 6644538, 3342277,
	4943130, 4272102, 2437823, 8093429, 8038120, 3595838, 768622,
	525098, 3556995, 5173371, 6348669, 3122442, 655327, 522500,
	43260, 1613174, 7884926, 7561383, 7470875, 6521319, 7479715,
	3193378, 1197226, 3759364, 3520352, 4867236, 1235728, 5945978,
	8113420, 3562462, 2446433, 6136326, 3342478, 4562441, 6063917,
	4972711, 6288750, 4540456, 3628969, 3881060, 3019102, 1439742,
	812732, 1584928, 7094748, 7039087, 7064828, 177440, 2409325,
	1851402, 5220671, 3553272, 8190869, 1316856, 7620448, 210977,
	5991061, 3249728, 6727353, 8578, 3724342, 4421799, 7475901,
	1100098, 8336129, 5282425, 7871466, 8115473, 3343383, 1430430,
	6527646, 7031341, 381987, 1308169, 22981, 1228525, 671102,
	2477047, 411027, 3693493, 2967645, 5665122, 6232521, 983419,
	4968207, 8253495, 3632928, 3157330, 3190144, 1000202, 4083598,
	6441103, 1257611, 1585221, 6203962, 4904467, 1452451, 3041255,
	3677745, 1528703, 3930395, 2797779, 6308525, 2556880, 4479693,
	4499374, 7426187, 7849063, 7568473, 4680821, 1600420, 2140649,
	4873154, 3821735, 4874723, 1643818, 1699267, 539299, 6031717,
	300467, 4840449, 2867647, 4805995, 3043716, 3861115, 4464978,
	2537516, 3592148, 1661693, 4849980, 5303092, 8284641, 5674394,
	8100412, 4369920, 19422, 6623180, 3277672, 1399561, 3859737,
	2118186, 2108549, 5760665, 1119584, 549488, 4794489, 1079900,
	7356305, 5654953, 5700314, 5268920, 2884855, 5260684, 2091905,
	359251, 6026966, 6554070, 7913949, 876248, 777960, 8143293,
	518909, 2608894, 8354570, 4186625,
}

// ntt executes the in-place forward NTT on p. Assumes coefficients < 2q
// (standard form); output coefficients bounded by 18q.
func (p *poly) ntt() {
	k := 0
	for l := uint(mldsaN / 2); l > 0; l >>= 1 {
		for offset := uint(0); offset < mldsaN-l; offset += 2 * l {
			k++
			zeta := uint64(nttZetas[k])
			for j := offset; j < offset+l; j++ {
				t := montReduceLe2Q(zeta * uint64(p[j+l]))
				p[j+l] = p[j] + (2*mldsaQ - t)
				p[j] += t
			}
		}
	}
}

// invNTT executes the in-place inverse NTT and multiplies by the
// Montgomery factor R. Coefficients bounded by 2q on output.
func (p *poly) invNTT() {
	k := 0
	for l := uint(1); l < mldsaN; l <<= 1 {
		for offset := uint(0); offset < mldsaN-l; offset += 2 * l {
			zeta := uint64(nttInvZetas[k])
			k++
			for j := offset; j < offset+l; j++ {
				t := p[j]
				p[j] = t + p[j+l]
				t += 256*mldsaQ - p[j+l]
				p[j+l] = montReduceLe2Q(zeta * uint64(t))
			}
		}
	}
	for j := 0; j < mldsaN; j++ {
		p[j] = montReduceLe2Q(mldsaROver256 * uint64(p[j]))
	}
}

// centeredLowBits returns the FIPS Decompose low part of a, centered
// into (-gamma2, gamma2]. a must be normalized to [0, q). Lifted from
// pulsar/boundary.go.
func centeredLowBits(a uint32, gamma2 uint32) int32 {
	a0plusQ, _ := decompose(a, gamma2)
	a0 := int32(a0plusQ)
	if a0plusQ > (mldsaQ-1)/2 {
		a0 -= mldsaQ
	}
	return a0
}

// highBitsCoeff returns the FIPS Decompose high part a1 of a.
func highBitsCoeff(a uint32, gamma2 uint32) uint32 {
	_, a1 := decompose(a, gamma2)
	return a1
}

// useHint applies one FIPS 204 (Algorithm 40) hint bit to coefficient
// r, returning the corrected high part r1. m = (q-1)/(2*gamma2)
// high-bit buckets. r must be normalized to [0, q). Lifted from
// pulsar/boundary.go.
func useHint(hbit, r, gamma2 uint32) uint32 {
	m := uint32((mldsaQ - 1) / (2 * gamma2))
	r0plusQ, r1 := decompose(r, gamma2)
	if hbit == 0 {
		return r1
	}
	r0 := int32(r0plusQ)
	if r0plusQ > (mldsaQ-1)/2 {
		r0 -= mldsaQ
	}
	if r0 > 0 {
		return (r1 + 1) % m
	}
	return (r1 + m - 1) % m
}
