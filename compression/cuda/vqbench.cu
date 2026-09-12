// Microbenchmark of the expert weight pipeline for three storage formats, with the read, decode and
// tensor-core stages separable.  Mirrors the access structure of dsv41/cuda/fp4_tc.cu: a warp owns 8
// weight rows, lane (g = lane/4, t = lane%4) owns row n0+g and a 64-k slice per step, so a warp walks
// 256 k per iteration.  Per 64 weights a lane moves
//
//   FORMAT 0  FP4      32 B   (the stored format: 4 bit/weight, bit-placement decode)
//   FORMAT 1  VQ12     24 B   (3.0 bit/weight: 16 groups of 4 weights, 12-bit index, two planes)
//   FORMAT 2  VQ14     28 B   (3.5 bit/weight: 14-bit index, two planes -- 8 bit + 6 bit)
//
// plus 2 E8M0 scale bytes in every case.  The two-plane layout keeps every load 4/8/16-byte aligned and
// replaces bit-field extraction with a byte read (and, for VQ14, one 6-bit extract).  A codebook entry
// is 4 E2M1 codes, so after the LUT the existing e2m1x8_to_bf16 bit placement is reused unchanged.
//
// DECODE=0 stops after the loads (pure read bandwidth), DOMMA=0 stops after the decode.
#include <cuda_bf16.h>
#include <stdint.h>

#define WARPS 4

__device__ __forceinline__ uint32_t bf16x2_fma0(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0u));
    return r;
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 8 nibbles (bytes 0..3 = k 0..7, low nibble = even k) -> 4 bf16x2 words scaled by f2
__device__ __forceinline__ void e2m1x8_to_bf16(uint32_t w, uint32_t f2, uint32_t* o) {
    uint32_t a = ((w & 0x00070007u) << 6) | ((w & 0x00080008u) << 12);
    uint32_t b = ((w & 0x07000700u) >> 2) | ((w & 0x08000800u) << 4);
    uint32_t c = ((w & 0x00700070u) << 2) | ((w & 0x00800080u) << 8);
    uint32_t d = ((w & 0x70007000u) >> 6) | (w & 0x80008000u);
    o[0] = bf16x2_fma0(a, f2);
    o[1] = bf16x2_fma0(b, f2);
    o[2] = bf16x2_fma0(c, f2);
    o[3] = bf16x2_fma0(d, f2);
}

// LUT_BITS: 12 -> 4096 entries (16 kB as uint32), 14 -> 16384 entries (32 kB as uint16)
template <int FORMAT, int DECODE, int DOMMA, int LUT_BITS, int KPL>
__device__ __forceinline__ void vqbench_body(
        const uint8_t* __restrict__ Wlo, const uint8_t* __restrict__ Whi, const uint8_t* __restrict__ S,
        const uint16_t* __restrict__ LUT, float* __restrict__ out, int N, int K, int nexp,
        long long stride_lo, long long stride_hi, long long stride_se)
{
    extern __shared__ uint32_t smem[];
    if (DECODE && FORMAT > 0) {
        const int n_ent = 1 << LUT_BITS;
        if (LUT_BITS == 12) {
            for (int i = threadIdx.x; i < n_ent; i += WARPS * 32) smem[i] = LUT[i];
        } else {   // 16384 entries kept as uint16: 32 kB instead of 64 kB
            uint16_t* s16 = reinterpret_cast<uint16_t*>(smem);
            for (int i = threadIdx.x; i < n_ent; i += WARPS * 32) s16[i] = LUT[i];
        }
        __syncthreads();
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * WARPS + warp) * 8;
    if (n0 >= N) return;
    const int n = n0 + g;
    const int e = blockIdx.y % nexp;

    // Tiled layout (the one dsv41 already uses for FP4): the 8 rows a warp owns are stored
    // contiguously per 256-k step, so a warp's read per step is one contiguous 8 * BPS byte range
    // instead of 8 scattered pieces.  Applied to every format so the comparison is fair.
    const int KPS = 4 * KPL;                         // k per warp per step
    const int BPS_LO = (FORMAT == 0) ? KPS / 2 : KPS / 4;   // bytes per row per step, low plane
    const int BPS_HI = (FORMAT == 1) ? KPS / 8 : 3 * KPS / 16;
    const int nt = n0 >> 3;
    const long long tile_lo = (long long)nt * (K / KPS) * 8 * BPS_LO;
    const uint8_t* plo = Wlo + (long long)e * stride_lo + tile_lo + g * BPS_LO + t * (BPS_LO / 4);
    const uint8_t* phi = (FORMAT == 0) ? nullptr
        : Whi + (long long)e * stride_hi + (long long)nt * (K / KPS) * 8 * BPS_HI + g * BPS_HI + t * (BPS_HI / 4);
    const int step_lo = 8 * BPS_LO, step_hi = 8 * BPS_HI;
    const int SPS = KPS / 32;                        // scale bytes per row per step
    const uint8_t* srow = S + (long long)e * stride_se + (long long)nt * (K / KPS) * 8 * SPS + g * SPS + (KPL / 32) * t;
    const int sstep = 8 * SPS;

    float c[4] = {0.f, 0.f, 0.f, 0.f};
    uint32_t xa[4] = {0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u};
    uint32_t acc = 0;
    const int steps = K / KPS;
    const int NW = KPL / 8, NG = KPL / 4;
    for (int it = 0; it < steps; ++it) {
        uint32_t words[16];                  // KPL weights = KPL/8 words of 8 nibbles
        if (FORMAT == 0) {
#pragma unroll
            for (int v = 0; v < NW / 4; ++v) {
                const uint4 a = __ldg(reinterpret_cast<const uint4*>(plo + it * step_lo + 16 * v));
                words[4 * v] = a.x; words[4 * v + 1] = a.y; words[4 * v + 2] = a.z; words[4 * v + 3] = a.w;
            }
        } else {
            uint4 lo4[2];
#pragma unroll
            for (int v = 0; v < NG / 16; ++v) lo4[v] = __ldg(reinterpret_cast<const uint4*>(plo + it * step_lo + 16 * v));
            uint32_t idx[32];
            const uint8_t* lob = reinterpret_cast<const uint8_t*>(lo4);
            if (FORMAT == 1) {
                uint32_t hw[4];
#pragma unroll
                for (int v = 0; v < NG / 8; ++v) hw[v] = __ldg(reinterpret_cast<const uint32_t*>(phi + it * step_hi) + v);
#pragma unroll
                for (int i = 0; i < NG; ++i)
                    idx[i] = lob[i] | (((hw[i >> 3] >> (4 * (i & 7))) & 0xF) << 8);
            } else {
                const uint32_t* hp = reinterpret_cast<const uint32_t*>(phi + it * step_hi);
                uint32_t hh[6];
#pragma unroll
                for (int v = 0; v < 3 * NG / 16; ++v) hh[v] = __ldg(hp + v);
#pragma unroll
                for (int i = 0; i < NG; ++i) {
                    const int bit = 6 * i, wi = bit >> 5, sh = bit & 31;
                    uint32_t v = hh[wi] >> sh;
                    if (sh > 26) v |= hh[wi + 1] << (32 - sh);
                    idx[i] = lob[i] | ((v & 0x3F) << 8);
                }
            }
            if (DECODE) {
#pragma unroll
                for (int i = 0; i < NW; ++i) {
                    const uint32_t a = (LUT_BITS == 12) ? smem[idx[2 * i]]
                                                        : (uint32_t)reinterpret_cast<const uint16_t*>(smem)[idx[2 * i]];
                    const uint32_t b = (LUT_BITS == 12) ? smem[idx[2 * i + 1]]
                                                        : (uint32_t)reinterpret_cast<const uint16_t*>(smem)[idx[2 * i + 1]];
                    words[i] = (a & 0xFFFFu) | (b << 16);
                }
            } else {
#pragma unroll
                for (int i = 0; i < NW; ++i) words[i] = idx[2 * i] ^ (idx[2 * i + 1] << 16);
            }
        }
        uint32_t sb[4];
#pragma unroll
        for (int v = 0; v < KPL / 32; ++v) sb[v] = __ldg(srow + it * sstep + v);
        if (!DECODE) {
#pragma unroll
            for (int i = 0; i < NW; ++i) acc ^= words[i];
#pragma unroll
            for (int v = 0; v < KPL / 32; ++v) acc ^= sb[v] << v;
            continue;
        }
        // E8M0 scale folded into the decode multiply, as in fp4_tc.cu (2^(s-127) x 2^126)
        uint32_t fs[4];
#pragma unroll
        for (int v = 0; v < KPL / 32; ++v) fs[v] = ((sb[v] - 1) << 7) | ((sb[v] - 1) << 23);
        uint32_t bf[4];
#pragma unroll
        for (int i = 0; i < NW; ++i) {
            e2m1x8_to_bf16(words[i], fs[i >> 2], bf);
            if (DOMMA) {
#pragma unroll
                for (int j = 0; j < 2; ++j) mma16816(c, xa, bf + 2 * j);
            } else {
                acc ^= bf[0] ^ bf[1] ^ bf[2] ^ bf[3];
            }
        }
    }
    if (acc == 0xdeadbeefu || c[0] == 1e30f)     // never true: keeps the work alive
        out[n0 + lane] = c[0] + c[1] + c[2] + c[3] + (float)acc;
}

#define ENTRY(NAME, F, D, M, L, KPL_) \
extern "C" __global__ void __launch_bounds__(WARPS * 32) NAME( \
        const uint8_t* a, const uint8_t* b, const uint8_t* s, const uint16_t* l, float* o, \
        int N, int K, int E, long long sa, long long sb, long long ss) { \
    vqbench_body<F, D, M, L, KPL_>(a, b, s, l, o, N, K, E, sa, sb, ss); }


ENTRY(fp4_read, 0, 0, 0, 12, 64)
ENTRY(fp4_read_k128, 0, 0, 0, 12, 128)
ENTRY(fp4_dec, 0, 1, 0, 12, 64)
ENTRY(fp4_dec_k128, 0, 1, 0, 12, 128)
ENTRY(fp4_full, 0, 1, 1, 12, 64)
ENTRY(fp4_full_k128, 0, 1, 1, 12, 128)
ENTRY(vq12_read, 1, 0, 0, 12, 64)
ENTRY(vq12_read_k128, 1, 0, 0, 12, 128)
ENTRY(vq12_dec, 1, 1, 0, 12, 64)
ENTRY(vq12_dec_k128, 1, 1, 0, 12, 128)
ENTRY(vq12_full, 1, 1, 1, 12, 64)
ENTRY(vq12_full_k128, 1, 1, 1, 12, 128)
ENTRY(vq14_read, 2, 0, 0, 14, 64)
ENTRY(vq14_read_k128, 2, 0, 0, 14, 128)
ENTRY(vq14_dec, 2, 1, 0, 14, 64)
ENTRY(vq14_dec_k128, 2, 1, 0, 14, 128)
ENTRY(vq14_full, 2, 1, 1, 14, 64)
ENTRY(vq14_full_k128, 2, 1, 1, 14, 128)
