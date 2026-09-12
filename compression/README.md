# DeepSeek-V4.1-Flash FP4 expert weights — 圧縮余地の情報理論的調査（Phase 1）

§0-§11 は**可逆**圧縮、§12 は**非可逆**を許した場合。

結論を先に: **payload（FP4 nibble）側に実用的な可逆圧縮の余地は無い。探索は打ち切りを推奨する。**
残っているのは「圧縮」ではなく**形式の無駄**が 2 箇所だけ:
1. **block scale（E8M0）が 8 bit で 11 値しか運んでいない** → 4 bit 化で expert が 2.94% 小さくなる（decode はほぼゼロコスト、VRAM と HBM 帯域にも効く）。
2. **FP4 の code 0 と 8（±0）が同じ値** → zero-magnitude weight の符号 bit 11.55% が未使用。実カーネルで
   bit-identical を確認済み（§11）。統合それ自体は 0 byte だが、空いた bit に scale を埋め込めば 5.9% になる。

| 問い | 答え |
|---|---|
| 1. 4 bit/weight → 3.5 bit 以下に完全可逆で落とせるか | **不可**。測定した全条件付きモデルの下限は 3.75 bit/weight（held-out）。 |
| 2. 3.2 bit 以下（20%圧縮）は狙えるか | **不可**。理論的にも届かない（下記「なぜ無理か」）。 |
| 3. 展開込みで NVMe streaming を速くできるか | **実質不可**。最良方式 0.890x で decode 予算は 1.24 ms/expert = 15.1 GB/s の decoder が必要（直列時）。zstd/lz4 は実測で 1.4〜2.6 倍**遅くなる**。 |
| 4. 無理なら定量的に証明 | 下記 §9。独立性 control と dictionary scaling law で定量化した。 |
| 5. 非可逆を許すなら？（追加質問） | **こちらは桁違いに有望**。VQ（4 weight / group）+ routing 混合精度で **平均 3.5 bit = 0.853x、wikitext-2 PPL 3.0765 → 3.1002（+0.77%）を実測**。VRAM と HBM 帯域が −14.7%（期待 +7% スループット）。→ §12 |

expert 1 個 = 18.80 MB（payload 17.69 MB + scale 1.11 MB）、weight 35.39 M 個 = 4.25 bit/weight。
以下 ratio はすべて「この 4.25 bit/weight（= 18.80 MB）に対する比」か「payload 4 bit に対する比」を明記する。

---

## 0. 方法と再現

```
../.venv-lc/bin/python sample.py            # checkpoint から block sample を作る (data/sample.npz)
../.venv-lc/bin/python phase1.py            # A baseline / B 条件付き entropy / B2 richer / C bit-plane / D PRNG / E dictionary
../.venv-lc/bin/python phase2_fg.py         # F expert 間構造 / G scale
../.venv-lc/bin/python phase4_extra.py      # layer 別分布, scale alphabet, ±0 alias
../.venv-lc/bin/python scan_scales.py       # 全 scale tensor の全数走査 (17 GB read)
../.venv-lc/bin/python phase3_io.py         # ストレージ帯域の実測と decode 予算
../.venv-lc/bin/python phase5_summary.py    # 最終表 (results/summary.*)
```

- checkpoint は read-only の mmap（`st.py`）でしか触らない。**書き換えは一切していない**。既存推論コードにも手を入れていない。
- sample: layer `0,5,11,17,23,29,35,39` × 16 expert × {w1,w2,w3} × 20 行 = **1,003,520 block = 32.1 M weight**。
- 抽出の正しさの検証: 既知値 H(W)=3.888 に対し **3.8931**、zstd payload 既知 0.973 に対し **0.9747** を再現。
  さらに「32-weight block 内の最大 magnitude code は必ず 6 か 7」という E2M1 × amax 正規化の構造的性質も再現できた
  （1M block で code 5 以下が最大値になった block はゼロ）。→ nibble/scale の切り出しは正しい。
- **理論値と実圧縮率は必ず分けて報告する**。条件付き entropy は plug-in（楽観値）と held-out cross entropy
  （半分で表を作り、残り半分で評価 = 実際に表を配った符号器が払う bit 数）の両方を出し、表の格納コストも加算した。

---

## 1. H(W | scale_bin, position) — 最重要項目

| context | plug-in H | **held-out H** | bit/weight | payload 比 |
|---|---|---|---|---|
| なし H(W) | 3.8931 | 3.8933 | 3.8933 | 0.9733 |
| scale（E8M0 実値。sample では 6 値、全モデルで 11 値） | 3.7704 | 3.7709 | 3.7709 | **0.9427** |
| position_in_block (0..31) | 3.8931 | 3.8933 | 3.8933 | 0.9733 |
| **scale × position (192 ctx)** | 3.7704 | **3.7710** | 3.7710 | **0.9427** |
| layer | 3.8927 | 3.8929 | — | 0.9732 |
| layer × matrix | 3.8926 | 3.8928 | — | 0.9732 |
| expert × matrix（表は expert ごとに保存） | 3.8921 | 3.8928 | +0.0001 表 | 0.9732 |
| layer × matrix × scale × position (3136 ctx) | 3.7680 | 3.7707 | 3.7707 | 0.9427 |

- **position は完全に無情報**（32 位置すべて同一分布）。scale だけが効き、その効果は 0.122 bit。
  scale を 16/32/64 bin に分ける実験は無意味（そもそも全モデルで 11 値しか存在しない）。
- layer / matrix / expert はすべて**ゼロ**。expert 固有テーブルは価値なし。
- 表のコストは全モデル 543.7 G weight に償却すると無視できる（3136 ctx でも 1e-7 bit/weight 未満）。

## 2. Markov-1 / Markov-2

| context | plug-in | held-out | payload 比 |
|---|---|---|---|
| W(i-1) | 3.8673 | 3.8675 | 0.9669 |
| W(i-1), W(i-2) | 3.8447 | 3.8454 | 0.9613 |
| W(i-1..3) | 3.8243 | 3.8289 | 0.9572 |
| position × block 内 running max | 3.8204 | 3.8211 | 0.9553 |
| **scale × position × runmax × W(i-1)（6867 ctx）** | 3.7445 | **3.7497** | **0.9374** ← 全測定中の最良 |
| layer × matrix × scale × pos × runmax × W(i-1)（98405 ctx） | 3.7159 | 3.7874 | 0.9469（**過学習**） |

- 隣接 nibble 相関は本物だがごく小さい（order-1 で 0.026 bit）。order を上げても飽和する。
- 最大 context は plug-in では 3.716 まで下がるが held-out では **悪化**する（3.787）。
  つまり **3.75 bit/weight が本当の飽和点**で、これ以上はデータに存在しない。
- 2-pass の明示的 side info（block ごとに block class を 1〜4 bit 送る）も試したが、
  得られる利得（0.13〜0.15 bit）より side info のコスト（0.031〜0.125 bit/weight）の方が伸びが悪く、最良でも 3.796。

## 3. bit-plane 条件付き entropy

E2M1 = `s ee m`（bit3 = 符号、bit2:1 = 2 bit 指数、bit0 = 1 bit 仮数）。

| plane | P(1) | H | H&#124;pos | H&#124;scale,pos | H&#124;直前 nibble | H&#124;pos,runmax |
|---|---|---|---|---|---|---|
| bit3 sign | 0.4999 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bit2 exp-hi | 0.4306 | 0.9861 | 0.9861 | 0.9153 | 0.9738 | 0.9496 |
| bit1 exp-lo | 0.4440 | 0.9909 | 0.9909 | 0.9874 | 0.9903 | 0.9884 |
| bit0 mantissa | 0.4130 | 0.9781 | 0.9781 | 0.9677 | 0.9734 | 0.9679 |

- 符号 bit は**厳密に 1.0000 bit**。どんな context を与えても 1.0000 から動かない（= weight の符号は完全にランダム）。
  これだけで payload の 25% は原理的に圧縮不可能と確定する。
- 「symbol 全体では一様でも bit-plane は偏っている」という仮説は**否定**された。むしろ逆で、
  plane を独立に符号化すると **3.8708 bit**（0.9677x）となり、同じ context の joint symbol 符号（3.7710）より**悪い**。
- plane を上位から順に条件付けて連鎖させると合計は 3.8933 = H(W) に厳密に一致する（当然だが、実装の健全性チェックになる）。
- **bit-plane 分解に利得は無い。**

## 4. block dictionary（256 / 1K / 4K / 16K / 64K）

32-weight block = 128 bit。1,003,520 block 中 **完全一致は 0 件**（重複率 0.0000%）。
ランダムな 2 block の Hamming 距離は平均 63.01/128。

| D | NN Hamming | residual zero-nibble | H(residual) | index+residual+辞書 | payload 比 |
|---|---|---|---|---|---|
| 256 | 46.01 | 0.173 | 3.7638 | 4.0138 | 1.0035 |
| 1,024 | 43.45 | 0.195 | 3.6917 | 4.0042 | 1.0010 |
| 4,096 | 41.15 | 0.218 | 3.6185 | 3.9935 | 0.9984 |
| 16,384 | 38.69 | 0.247 | 3.5297 | 3.9673 | 0.9918 |
| 65,536 | 35.24 | 0.300 | 3.3862 | 3.8862 | 0.9716 |
| 983,520（sample 全 block を辞書にする上界、辞書容量は**無視**） | 34.47 | 0.290 | 3.3562 | 3.9783 | 0.9949 |
| **control: nibble を block 間でシャッフル（構造ゼロの合成データ）, D=65536** | **36.66** | — | 3.4512 | — | — |

- 決定的なのは control 行。**構造を完全に破壊した合成データでも NN Hamming 36.66 で、実データの 35.24 とほぼ同じ**。
  つまり観測された「近さ」の 99% は 128 bit 空間で 65536 本引いたときの順序統計に過ぎず、block 構造ではない。
- どの D でも index 込みで **4 bit/weight を下回らない**（辞書容量を無視した上界ですら 0.9949）。
- k-medoids 等で辞書を最適化しても、上の「全 block 辞書（= 任意の D=10^6 辞書の上界）」を超えられないので無意味。

## 5. PRNG XOR は本当に無意味か → **無意味であることが証明される**

| 方式 | residual H | zero-nibble 率 | zstd-9 |
|---|---|---|---|
| baseline | 3.8931 | 0.0583 | 0.9747 |
| XOR PRNG（matrix 単位 seed） | **4.0000** | 0.0625 | **1.0000** |
| XOR PRNG（expert 単位 seed） | 4.0000 | 0.0625 | 1.0000 |
| XOR PRNG（block 単位 seed） | 4.0000 | 0.0625 | 1.0000 |

理論: 既知系列との XOR は各位置で**全単射**なので、条件付き entropy H(W|context) は定義上不変。
一方 order-0 分布は 16 値に均され、実測どおり entropy は 3.8931 → 4.0000 に**増える**。
zero-nibble 率も 1/16 = 0.0625 に収束しており、これは「無相関」の教科書的な値。seed 保存コストを議論するまでもない。
**この方向は完全に終了。**

## 6. expert 間構造（§F）と scale（§G）

expert 間（layer 17、48 expert、同一 (row,k) 位置で整列）:

| 指標 | 値 | 偶然値 |
|---|---|---|
| expert ペアの nibble 一致率（最良ペア） | 0.0726 | 0.0709 |
| 隣接 expert (i, i+1) | 0.0709 | 0.0709 |
| 最良一致 expert との XOR residual H | **3.9971** | — |
| 参照 expert 0 との XOR residual H | 3.9974 | — |
| 同一 expert の w1 vs w3 一致率 | 0.0720 | 0.0709 |
| 整列 block の 47 expert 中最近傍 Hamming | 49.59 | ランダムペア 63.04（47 本引いた順序統計としては妥当） |
| その最近傍との block XOR + index | 4.0219 bit/weight | — |

「隣接 expert」だけでなく **「最も近い expert / block」でも改善しない**。excess 一致率は +0.0017 で、XOR 残差の entropy は
むしろ 4.0 に近づく（分布が均されるため）。expert 間の冗長性は存在しない。

scale（E8M0、全モデル 16,986,931,200 byte を全数走査）:

| 方式 | bit/scale | bit/weight | expert 全体（payload は raw のまま） |
|---|---|---|---|
| 現状 raw | 8.0000 | 0.2500 | 4.2500 (1.0000x) |
| **固定 4 bit 化（全モデルで 11 値、LUT 16 entry）** | **4.0** | 0.1250 | 4.1250 (**0.9706x**) |
| order-0 entropy coding（全数走査） | **1.0041** | 0.0314 | 4.0314 (**0.9486x**) |
| zstd-9 | 1.353 | 0.0423 | 4.0423 (0.9511x) |

空間相関の確認（4 layer × 4 expert の部分 sample。この sample 単体の H0 は 1.0086）:
H(s) = 1.0086 → H(s&#124;左隣) = 1.0059 → H(s&#124;左,上) = 1.0050 bit/byte。

- **全モデルの全 scale byte が 117..127 の 11 値しか取らない**（tensor 単位の最大幅は 10）。
  つまり 8 bit のうち実質 3.4 bit しか使っていない。
- ただし scale の空間相関はほぼゼロ（左/上を条件にしても 1.0086 → 1.0050）。エントロピー符号以上の工夫は無駄。
- 「scale だけ 5 倍縮んだ」の正味の効果は **expert 全体で 5.1%**（entropy coding 時）、**2.9%**（4 bit 固定長時）。

---

## 7. 損益分岐（§I、実測）

ストレージ帯域の実測（O_DIRECT、page cache バイパス、checkpoint ファイル上）:

| read size | 帯域 | expert 1 個 (18.8 MB) |
|---|---|---|
| 1 MB | 976 MB/s | 19.25 ms |
| 4 MB | 1423 MB/s | 13.22 ms |
| 19 MB | 1613 MB/s | 11.66 ms |
| 64 MB | **1667 MB/s** | **11.28 ms** |

（`/mnt/ssd` は NVMe ではなく SSD の RAID0。baseline = 18.8 MB / 1667 MB/s = **11.28 ms/expert**）

| ratio | 圧縮後 | I/O 時間 | decode 予算（直列） | 必要 decoder 速度 |
|---|---|---|---|---|
| 0.890（実測最良） | 16.7 MB | 10.04 ms | **1.24 ms** | **15.1 GB/s** |
| 0.90 | 16.9 MB | 10.15 ms | 1.13 ms | 16.7 GB/s |
| 0.80 | 15.0 MB | 9.02 ms | 2.26 ms | 8.3 GB/s |
| 0.70 | 13.2 MB | 7.90 ms | 3.38 ms | 5.6 GB/s |
| 0.60 | 11.3 MB | 6.77 ms | 4.51 ms | 4.2 GB/s |
| 0.50 | 9.4 MB | 5.64 ms | 5.64 ms | 3.3 GB/s |

I/O と decode を完全にオーバーラップできる実装なら要求は緩み、0.890x を保つには **1.87 GB/s** の
decoder があればよい（GPU rANS なら到達可能な水準）。ただし得られるのは **11.0% の時間短縮（1.24 ms/expert）**だけ。
なお実測した汎用 codec は逆効果で、zstd-9 は decode 17.2 ms（1 core, 1144 MB/s）で合計 **27.7 ms**、
lz4 でも 14.9 ms と、**生読みの 11.28 ms より遅い**。

---

## 8. 最終表（§J）

（`results/summary.csv` / `.md` / `.json` に同じものを保存。ratio は expert 全体 18.8 MB 比）

| 方式 | 理論 bit/weight | 実圧縮率(payload) | expert bit/weight | metadata 込み ratio | decode | I/O ms | decode ms | 合計 ms | 採用判断 |
|---|---|---|---|---|---|---|---|---|---|
| raw FP4（現状） | 4.000 | 1.0000 | 4.250 | 1.0000 | — | 11.28 | 0.00 | 11.28 | baseline |
| zstd-3 | — | 0.9762 | 3.952 | 0.9299 | 1052 MB/s | 10.49 | 19.13 | 29.62 | 捨てる |
| zstd-9 | — | 0.9747 | 3.941 | 0.9272 | 1144 MB/s | 10.46 | 17.22 | 27.68 | 捨てる |
| lz4 | — | 1.0001 | 4.122 | 0.9698 | 4331 MB/s | 10.94 | 3.92 | 14.86 | 捨てる |
| order-0 nibble rANS | 3.893 | — | 4.143 | 0.9748 | 未実装 | 11.00 | — | — | 捨てる |
| conditional entropy (scale) | 3.771 | — | 4.021 | 0.9461 | 未実装 | 10.67 | — | — | 捨てる |
| scale + position | 3.771 | — | 4.021 | 0.9461 | 未実装 | 10.67 | — | — | 捨てる |
| Markov-1 | 3.868 | — | 4.117 | 0.9688 | 未実装 | 10.93 | — | — | 捨てる |
| Markov-2 | 3.845 | — | 4.095 | 0.9636 | 未実装 | 10.87 | — | — | 捨てる |
| bit-plane（plane 独立） | 3.871 | — | 4.121 | 0.9696 | 未実装 | 10.94 | — | — | 捨てる |
| PRNG XOR | 4.000 | 1.0000 | 4.250 | 1.0000 | — | 11.28 | — | — | 捨てる（原理的に無効） |
| learned/context predictor（最良、held-out） | 3.750 | — | 4.000 | 0.9411 | 未実装 | 10.62 | — | — | 捨てる |
| block dictionary 64K + XOR residual | 3.886 | — | 4.136 | 0.9732 | 未実装 | 10.98 | — | — | 捨てる |
| nearest-block residual (1M 辞書) | 3.978 | — | 4.228 | 0.9949 | 未実装 | 11.22 | — | — | 捨てる |
| scale のみ 4 bit 化 | — | 1.0000 | 4.125 | **0.9706** | ほぼ 0 | 10.95 | 0.06 | 11.00 | **形式修正として採用可** |
| scale のみ entropy coding | — | 1.0000 | 4.031 | 0.9486 | ~1 GB/s 必要 | 10.70 | 1.11 | 11.81 | 保留 |
| **best combined（context rANS + coded scales）** | **3.781** | — | **3.781** | **0.8897** | 15.1 GB/s（直列）/ 1.9 GB/s（重畳） | 10.04 | — | 10.04 | **見送り** |
| （参考）±0 alias 統合 ※可逆でない | 3.633 | — | 3.664 | 0.8622 | 同上 | 9.73 | — | 9.73 | 別議論 |

採用基準（ratio > 0.90 → 捨てる / 0.80〜0.90 → decode がほぼ無料なら検討）に照らすと、
**単独で 0.90 を切る方式は「best combined」だけ**であり、その decode は context 適応 rANS
（6867 context、直前 nibble と running max に逐次依存）で、「ほぼ無料」からは程遠い。

---

## 9. なぜ無理か（定量的な証明）

1. **符号の 1 bit は完全にランダム**。P(sign=1)=0.4999、どの context でも H=1.0000。payload の 25% は確定的に非圧縮。
2. **量子化器がすでにほぼ最適**。block ごとに amax で正規化し power-of-2 scale を付けた E2M1 は、
   残った分布の entropy が 3.893/4 = 97.3%。一様 4 bit 符号が捨てている冗長性は**最初から 2.7% しかない**。
3. **条件付けで得られるのは追加 0.14 bit だけ**。使える side information は block scale のみで 0.122 bit、
   系列相関が 0.026 bit、running max が 0.05 bit。すべて足し合わせても held-out で 3.75 bit（= 残り 3.75/4 = 93.7%）で飽和する。
   context を増やすと held-out が悪化する（3.716 plug-in → 3.787 test）ことから、**これは推定精度の限界ではなくデータの限界**。
4. **block 構造は存在しない**。独立性を強制した control データが実データと同じ最近傍統計（36.66 vs 35.24 bit）を再現する。
   1M block に完全一致が 1 件も無いことも、128 bit ブロックの実効 entropy が 124 bit 級であることと整合する
   （誕生日境界は 2^62 block）。
5. **辞書の scaling law**。D を 256→65536（256 倍）にして NN Hamming は 46.0→35.2、
   すなわち **倍にするごとに 1.35 bit しか近づかない**。residual を実用域（例えば 128 bit 中 10 bit 違い）に
   するには 2^34 級の辞書が必要で、その辞書本体だけで 2^34 × 16 B = 275 GB ≒ モデル本体より大きくなる。
6. **expert 間・matrix 間の冗長性ゼロ**。最良一致 expert でも一致率は偶然値 +0.0017。
   MoE の expert は互いに独立に学習された別方向であり、XOR 残差は逆に entropy を増やす。

以上より、「FP4 payload は情報理論的にすでに飽和しており、可逆圧縮で取り返せるのは最大 6.3%
（4.000 → 3.750 bit/weight）、しかもそれには逐次 context 適応 rANS が必要」というのが定量的な答えである。

---

## 10. 判断と次の一手

**可逆圧縮探索は打ち切り（payload について）。** 理由は上記 §9、判断根拠は §8 の表。
ユーザ基準の「最良方式でも 0.90 を超えない場合は終了」に照らして、best combined = 0.8897 は境界上だが、
decode が無料でない以上、投資対効果は無い（1 expert あたり 1.24 ms、しかも I/O 経路にしか効かない）。

加えて**重要な構造上の事実**: 現行 runtime は expert を VRAM（EP）または host RAM に常駐させており、
NVMe から stream していない。常駐形式を entropy 符号化すると GEMM が weight を直接読めなくなるため、
token ごとに展開が必要になり、圧縮の利得（最大 11%）を遥かに超える代償を払う。
**可逆圧縮が意味を持つのは I/O 境界だけで、そこですら上限が 11% である。**

一方、**やる価値のある形式修正が 1 件だけある**（これは「圧縮」ではない）:

- **E8M0 scale を 4 bit にする**。全モデル 17.0 G byte の scale が 117..127 の 11 値しか取らないことを全数確認済み。
  16 entry の固定 LUT で完全可逆、decode は shift 1 回で FP4 kernel に融合できる。
  - checkpoint / ディスク: expert 18.80 MB → 18.25 MB（**0.9706x**）
  - VRAM: 常駐 expert が同じ比率で縮む（4 GPU EP、1 GPU あたり約 1000 expert で **約 550 MB** 相当）
  - **HBM 帯域**: expert 段は帯域律速（1 layer 128 行で 1.74 ms / 3.6 ms、実効 ~1 TB/s）なので、
    読む byte が 2.9% 減れば expert 段がほぼそのまま 2.9% 速くなる見込み（end-to-end で 1〜1.5%）。
  - 代償: `cuda/fp4_tc.cu` / `fp4_tcw.cu` の scale 読み出しと `quant.tile_fp4_scales`、loader の変更が必要。
    （本 Phase では**何も変更していない**。やるなら別タスクとして提案する。）

参考（可逆ではない）: FP4 の code 0 と 8 はどちらも 0.0 に dequantize される（±0）。
統合すると payload から正確に P(magnitude=0) = 0.1166 bit/weight が消える
（符号 bit の条件付き entropy が測定上ちょうど 1.0000 なので、この差し引きは近似ではなく厳密）。
最良 context モデル 3.7497 → 3.6331 bit、expert 全体で 0.8622x。
**推論結果は変わらない**（積も和も RN で同一）が checkpoint の byte は変わるため、
「完全可逆」の要件からは外れる。採否は別途判断すること。

---

## 11. ±0（code 0 と 8）の統合について

E2M1 の code 0 = +0.0、code 8 = -0.0 で、**dequantize 後の値は同じ**。全 code 8 を code 0 に書き換えたら何が起きるかを
実カーネルで検証した（`test_zero_merge.py`、layer 17 の expert 0..5 の w1、-0 nibble は 5.77%）。

| 検査 | 結果 |
|---|---|
| dequantize 後の bf16 bit pattern | 4,086,681 個だけ差異、**その全てが zero**（値としては完全一致） |
| fp32 参照 GEMM | **bit-identical** |
| `cukern.fp4_gemm_tc`（tiled、8 token × 6 expert、実カーネル） | **bit-identical**, max&#124;dy&#124; = 0 |
| `cukern.fp4_gemv_pairs` | **bit-identical**, max&#124;dy&#124; = 0 |

理由も厳密: 累算器は +0.0 で始まり、x × (±0.0) = ±0.0。RN では任意の c について c + (±0.0) = c、
かつ +0.0 + (-0.0) = +0.0 なので、**+0.0 で始まった累算器が符号付きゼロで差を生むことは原理的にない**。
`fp4_tc.cu` の decode は sign bit を bf16 の bit15 に置くだけなので -0.0 は正しく -0.0 になり、上の議論がそのまま通る。
CPU 経路（`moe_cpu.cpp`）も nibble → bf16 の vpermw 表引きなので同じ。
**「一緒にして良いか」への答えは Yes（推論出力は 1 bit も変わらない）。**

ただし **統合それ自体は 1 byte も減らさない**。4 bit 固定長 packing のままなら 15 値でも 16 値でも 4 bit だからである。
byte にするには「空いた符号 bit をどう使うか」を決める必要があり、選択肢は以下の 4 つ:

| 方式 | expert サイズ | disk | VRAM | HBM 帯域 | decode コスト | 必要な変更 |
|---|---|---|---|---|---|---|
| ±0 統合のみ（4 bit packing 据え置き） | 18.801 MB | **0%** | 0% | 0% | 無し | 無し |
| 15 値の base-15 packing（13 symbol を 51 bit 等） | 18.460 MB | 1.8% | 1.8% | 1.8% | 除算（乗算+shift）| **カーネルの bit 配置 decode が崩れる** → 非現実的 |
| **scale を 4 bit 化**（±0 とは無関係） | 18.248 MB | **2.9%** | **2.9%** | **2.9%** | shift 1 回 | scale 読み出し経路 |
| **空いた符号 bit に scale を埋め込む** | 17.695 MB | **5.9%** | 0% | 0% | expert ごとに 1 pass + 11 symbol rANS | loader のみ（カーネル変更**不要**） |
| ±0 統合 + payload 全体を context rANS | 16.210 MB | 13.8% | 0% | 0% | 逐次 context rANS | 全面的 |

**符号 bit 側チャネル**が ±0 の知見を一番うまく使う方法である:
- zero-magnitude weight は 11.55%、つまり **1 expert あたり 510.8 kB 分の符号 bit が完全に未使用**。
- entropy 符号化した scale は **138.8 kB**（1.0041 bit × 1,105,920）で、容量に 3.7 倍の余裕がある。
  なお 4 bit 固定長の scale（553.0 kB）は**入らない**ので、埋め込むなら scale 側の entropy 符号化は必須。
- 側チャネルの位置は payload だけから決まる（`nibble & 7 == 0` か否か）ので、scale 無しで復号できる。循環参照は無い。
- さらに良いことに、これは「統合」ですらない: **符号 bit を自由データ領域と宣言するだけ**で、
  復号後も ±0 が混在するが上で証明したとおり出力は変わらない。**FP4 カーネルは一切変更不要**。
- 代償: expert ごとに payload 17.7 MB を 1 pass 走査して zero nibble の符号 bit を集める前処理が要る。
  VRAM 上には結局 scale を展開して置くので **VRAM も HBM 帯域も減らない**。効くのは disk / PCIe / ロード時間だけ。

**推奨**: 近い将来やるなら **scale の 4 bit 化**（2.9%、しかも VRAM と HBM 帯域にも効く、変更が小さい）。
符号 bit チャネルは disk 5.9% と倍だが、帯域には効かず前処理 pass が要るので、
checkpoint 配布サイズやロード時間を削りたくなった時の選択肢として保留するのが妥当。
±0 統合単体（0%）は実施理由が無い。

---

## 12. 非可逆を許した場合（Lossy）

可逆側の上限は 11%（しかも disk 限定）だったが、**非可逆側は桁が違う**。ここは VRAM と HBM 帯域に直接効く。
出発点が既に量子化済み（E2M1 + per-32 E8M0）である点、つまり**ノイズの上乗せになる**ことを前提に測った。
スクリプト: `lossy.py`（L1-L3）、`lossy_alloc.py`（L4）。結果: `results/lossy.{csv,json}`, `results/lossy_rd.json`, `results/lossy_alloc.{csv,json}`。

### 12.1 基準線: FP4 自身のノイズ

Gaussian → per-32 amax → pow2 scale → E2M1 を模擬すると、**checkpoint の FP4 化自体が既に rel-RMSE 11.54%（SQNR 18.8 dB）**
の誤差を持っている。以下の数値はすべて「この上に何 % 積むか」として読む。
（expert 出力レベルでは weight 誤差の約 1.65 倍なので、現 checkpoint は元の bf16 に対して expert 出力で約 19% 相当。）

### 12.2 理論限界 R(D)（Blahut-Arimoto、16 値離散源、二乗誤差）

| rate | 追加 rel-RMSE | SQNR | FP4 ノイズ比（電力） |
|---|---|---|---|
| 4.0 bit（= H 3.893 以上）| 0 | ∞ | 0 |
| 3.5 bit | **4.59%** | 26.8 dB | 0.16x |
| 3.0 bit | **9.46%** | 20.5 dB | 0.67x |
| 2.5 bit | 15.81% | 16.0 dB | 1.88x |
| 2.0 bit | 23.56% | 12.6 dB | 4.17x |

**3.5 bit なら理論的には既存ノイズの 16% しか足さない。** ここが非可逆側にレバレッジがある理由。

### 12.3 実方式（実 expert layer 17 / expert 5 で測定、weight 誤差 / expert 出力誤差）

| 方式 | bit/w | ratio(4.25 比) | ‖ΔW‖/‖W‖ | expert 出力誤差 |
|---|---|---|---|---|
| scalar Lloyd-Max 8 levels | 3.000 | 0.765 | 23.73% | 38.72% |
| E2M0（仮数 bit を捨てる） | 3.000 | 0.765 | 25.94% | 50.25% |
| 2:4 構造化スパース（A100 mma.sp） | 3.000 | 0.765 | 36.86% | 57.76% |
| scale block 32→64 | 4.000 | 1.000 | 45.18% | 131.83% |
| **VQ dim4 / 4096 entries** | 3.000 | 0.765 | **14.90%** | 24.50% |
| VQ dim4 / 4096、E2M1 格子に snap | 3.000 | 0.765 | 16.28% | 26.77% |
| **VQ dim4 / 16384 entries** | 3.500 | 0.882 | **8.06%** | 13.28% |
| VQ dim4 / 16384、E2M1 格子に snap | 3.500 | 0.882 | 9.33% | 15.40% |
| VQ dim4 / 32768、E2M1 格子に snap | 3.750 | 0.941 | 5.11% | 8.40% |

- **スカラー量子化は論外**（3 bit で 23.7%、理論限界 9.46% の 2.5 倍）。分布が偏っているので固定長スカラーは rate を捨てている。
- **ベクトル量子化（4 weight = 1 group）が正解**。3 bit で 14.9%、3.5 bit で 8.1% と理論限界にかなり近い。
- **E2M1 格子に snap した変種**（codebook の各成分を E2M1 の 16 値に丸める）は誤差が 1〜2 割増えるだけで、
  **codebook 1 entry = 4 nibble** になるため LUT を引いた後は**既存の `fp4_tc.cu` の bit 配置 decode がそのまま使える**。
  3 bit なら LUT は 4096 × 16 bit = 8 kB（shared memory に常駐可）、3.5 bit なら 16384 × 16 bit = 32 kB。
- 2:4 スパースと scale block の粗化は**どちらも捨てる**（同じ rate で VQ より遥かに悪い）。

### 12.4 routing を使った混合精度（MoE 固有の切り札）

`results/route_telemetry.pt`（5 タスク × 40 layer × 384 expert）の routing mass は非常に偏っている:

| top-k | 平均 | japanese | coding | math | english | translation |
|---|---|---|---|---|---|---|
| 64 | 71.4% | 79.3% | 61.0% | 68.7% | 70.7% | 77.2% |
| 128 | 88.5% | 90.8% | 83.3% | 87.6% | 89.4% | 91.2% |
| 192 | 95.9% | 96.5% | 94.2% | 95.4% | 96.2% | 97.4% |

そこで `min Σ_e mass_e ε(b_e)²  s.t.  mean b_e ≤ B` を Lagrange sweep で解く（ε は 12.3 の実測曲線）:

**expert ごとの下限を 3.0 bit にした安全側の配分**

| 平均 bit/w | ratio(4bit scale 込) | 一律配分の誤差 | **混合配分** | 最悪タスク | 任意タスク上界 | bit 内訳 |
|---|---|---|---|---|---|---|
| 3.750 | 0.912 | 8.41% | **1.44%** | 1.66% | 12.88% | 4bit:10260 / 3.75:1393 / 3.5:309 / 3.25:218 / 3.0:3180 |
| **3.500** | **0.853** | 15.39% | **5.00%** | 5.93% | 18.20% | 4bit:5241 / 3.75:2627 / 3.5:694 / 3.25:508 / 3.0:6290 |
| 3.250 | 0.794 | 21.29% | 10.80% | 12.77% | 22.61% | 4bit:1553 / 3.75:2296 / 3.5:820 / 3.25:637 / 3.0:10054 |

下限 2.5 bit まで許すとさらに良く、平均 3.5 bit で混合 3.41%（最悪タスク 4.02%）だが、
**任意タスク上界が 21.0%** に上がる。下限 3.0 bit の方が未知タスクに対して頑健。

- 「任意タスク上界」= その配分で expert を一様に引いた場合の誤差。**未知のタスクが冷たい expert を叩いた時の最悪値**で、
  混合配分のリスクはここに集約される。配分は 5 タスクの平均で決めて同じ 5 タスクで評価しているので、
  「最悪タスク」列は in-sample であることに注意。実運用前に telemetry のタスク数とトークン数を増やすべき。

### 12.5 システム上の見返り（ここが本命）

| 構成 | expert 合計 | 4-GPU EP の 1 GPU あたり | HBM 帯域 | 期待 end-to-end |
|---|---|---|---|---|
| 現状 4.25 bit/w | 288.8 GB | 72.2 GB | 基準 | 基準 |
| scale 4 bit のみ（可逆） | 280.4 GB | 70.1 GB | −2.9% | +1〜1.5% |
| 平均 3.75 bit + 4bit scale | 263.4 GB | 65.9 GB | −8.8% | +4% |
| **平均 3.5 bit + 4bit scale** | **246.4 GB** | **61.6 GB** | **−14.7%** | **+7%** |
| 平均 3.0 bit + 4bit scale | 212.3 GB | 53.1 GB | −26.5% | +12% |

expert 段は帯域律速（1 layer 128 行で 1.74 ms / 3.6 ms、実効 ~1 TB/s）なので、読む byte の削減がほぼそのまま効く。
VRAM の余裕は KV cache / batch size に回せる。

### 12.6 品質評価（PPL）— 実測

GPU 4-7 が空いたので実測した。wikitext-2 test、ctx 2048、16 chunk（32,752 token）、全 40 layer の routed expert を置換。
**VQ の codebook entry は 4 個の E2M1 code なので、量子化後も正当な FP4 テンソルであり、カーネルは一切変更せずに評価できる**
（`vq_build.py` が 2 byte → 2 byte の LUT を書き出し、`ppl.py` が load 後の GPU 上のテンソルを書き換える。checkpoint は不変）。

| 構成 | 平均 bit/w | ratio(scale 8bit) | ratio(scale 4bit) | PPL | baseline 比 |
|---|---|---|---|---|---|
| baseline（現 FP4） | 4.000 | 1.000 | 0.971 | **3.0765** | — |
| VQ 3.75 一律 | 3.750 | 0.941 | 0.912 | 3.0941 | +0.57% |
| **routing 混合 平均 3.5** | 3.501 | 0.882 | **0.853** | **3.1002** | **+0.77%** |
| routing 混合 3.5 + calib(shrink 1024) | 3.501 | 0.882 | 0.853 | 3.1076 | +1.01% |
| routing 混合 3.5 + calib | 3.501 | 0.882 | 0.853 | 3.1113 | +1.13% |
| VQ 3.5 一律 + calib | 3.500 | 0.882 | 0.853 | 3.1256 | +1.60% |
| VQ 3.5 一律 | 3.500 | 0.882 | 0.853 | 3.1295 | +1.72% |
| VQ 3.0 を後半 20 layer のみ | 3.500 | 0.882 | 0.853 | 3.1469 | +2.29% |
| routing 混合 平均 3.25 | 3.251 | 0.824 | 0.794 | 3.1472 | +2.30% |
| VQ 3.0 一律 + calib(shrink 1024) | 3.000 | 0.765 | 0.735 | 3.1867 | +3.58% |
| VQ 3.0 一律 + calib | 3.000 | 0.765 | 0.735 | 3.1929 | +3.78% |
| VQ 3.0 一律 | 3.000 | 0.765 | 0.735 | 3.2067 | +4.23% |

ratio 列は 2 つ: scale を現状の 8 bit のままの場合と、§11 の 4 bit 化（可逆）を併用した場合。

- **§12.4 の予測どおり、routing 混合精度が同じ rate で劣化を半分にする**（3.5 bit: 一律 +1.72% → 混合 +0.77%）。
  内訳は 4 bit:6681 / 3.5 bit:2016 / 3.0 bit:6663 expert（40 layer × 384）。
- 一律 3.0 bit（0.735x、VRAM/帯域 −26.5%）でも **+4.2%** に留まる。
- 「後半 20 layer だけ 3.0 bit」は平均 3.5 bit で +2.29% と、混合配分（+0.77%）に大きく劣る。
  **層で切るより expert で切る方が良い**ことが実測で確認できた。

注意: PPL は同一テキストに対する決定論的な値なので構成間の比較は厳密だが、32,752 token という
サンプルサイズと wikitext-2 という 1 ドメインに基づく。採用前には他ドメイン（コード・日本語）と
下流ベンチでの確認が必要。

### 12.7 Calibration-aware 量子化

`calib_collect.py` が MoE forward を**ラップ**して（置き換えではない。元の経路はそのまま走る）
per-expert の活性化二乗和を集める:

- `H13[layer, expert, j] = Σ_t gate_t² · x_tj²`（w1/w3 の入力、5120 次元）
- `H2[layer, expert, j] = Σ_t hq_tj²`（w2 の入力 = SwiGLU 中間、2304 次元。gate は hq に既に入っている）

`vq_encode.py` が `Σ_j h_j (w_j − c_j)²` を最小化する entry を選ぶ（codebook と decoder は不変。
**エンコーダだけが変わる** = GPTQ/AWQ と同じ立て付け）。全 entry 探索は不要で、
65536 個の 4-code tuple それぞれについて非重み付き距離での上位 M 個を事前計算しておけば足りる
（実測: M=16 で最適解との歪み差 +0.002%(w1) / +0.195%(w2)）。この近似で **全モデルのエンコードが 9 分**（M=64 では 2.1 時間）。

重み付けの効果（weighted 歪みの削減）:

| | h の max/median | 非重み付き VQ の重み付き歪み |
|---|---|---|
| w1（入力は層共通の hidden） | 4.7 | 最適比 +10.4% |
| **w2（入力は SwiGLU 中間）** | **34.3** | 最適比 **+59.0%** |

w2 の Hessian が強く偏っているため、calibration の効果は w2 に集中する。

実測 PPL では **一律配分では改善、混合配分では悪化**した:

| 構成 | plain VQ | + calib | + calib & shrink(1024) |
|---|---|---|---|
| 3.0 bit 一律 | 3.2067 | 3.1929 | **3.1867** |
| 3.5 bit 一律 | 3.1295 | 3.1256 | — |
| 混合 平均 3.5 | **3.1002** | 3.1113 | 3.1076 |

原因は特定済み:

- 混合配分で 3.0 bit に落とされる expert は定義上 **routing mass が小さい = calibration token も少ない**。
- 実測: 3.0 bit 群の **95.3% が 2304 token 未満**（w2 の Hessian 対角の次元にすら届かない）、20.4% は 64 token 未満。
- つまり「一番強く量子化する expert の Hessian が一番ノイジー」。per-expert 推定が層平均より悪くなる。

対策は (a) calibration token を増やす（現在 24 window × 2048 = 49k token、expert あたり期待 768 token。
w1 の次元 5120 の 4 倍を狙うなら ~640 window 必要 = 約 1 時間）、
(b) 層平均への縮小推定（James-Stein 型、`ppl.py --calib-shrink`、擬似トークン数で指定）。
(b) は実装・測定済みで、一律配分では効いた（3.0 bit: 3.1929 → 3.1867）が、混合配分では
plain VQ（3.1002）を超えられなかった（3.1076）。**現在の 49k token では、混合配分の cold expert に
calibration を効かせるにはデータが足りない**というのが結論。

なお calibration による差は全て PPL ±0.4% 以内で、**VQ を使うか（scalar 比 −24pp）と
routing 混合配分か（同 rate で −0.95pp）という 2 つの一次要因に比べれば三次的**である。

### 12.8 VQ decode カーネルの構造（設計、実装前）

現 `fp4_tc.cu` / `fp4_tcw.cu` は「16 byte ロード = 32 nibble = 32 weight（= scale 1 個分）」を 1 lane が持つ。
VQ ではここが「12 bit × 8 group」になる。

1. **レート選択は 3.0 bit が圧倒的に実装しやすい**。
   - 3.0 bit: 1 group 12 bit、32 weight = 8 group = **12 byte**。lane が 128 k を担当すれば 48 byte = **16 byte ロード 3 本**（全て整列）。
   - 3.5 bit: 14 bit、32 weight = 14 byte。128 k で 56 byte = 3.5 本と半端。256 k 単位（112 byte = 7 本）にすれば整列するが、
     ステージするレジスタが倍になる。
2. **2 プレーン配置で bit 抽出を消す**。index の下位 8 bit を byte プレーン、上位 4 bit を nibble プレーンに分けて格納すると、
   128 k 分が「32 byte（下位）+ 16 byte（上位）」= 16 byte ロード 3 本になり、funnel shift が不要になる。
3. **LUT は shared memory**。4096 entry × 16 bit（= 4 nibble）。u16 のままだと 2-way bank conflict になるので
   u32 にパディングして **16 kB**。16384 entry（3.5 bit）は 64 kB になり x タイルと同居できない
   → **これも 3.0 bit を選ぶ理由**。
   - ランダム gather なので 32 lane が 32 bank に散ると期待最大バケットは 3〜4（= LDS が 3〜4 発）。
     ただし ncu では現カーネルは **load bound**（long_scoreboard 待ち）なので LDS レイテンシは隠れる見込み。
     隠れない場合は 16 kB の表を 2 部複製（32 kB）して warp 前半・後半で別コピーを引く。
4. **レジスタは減る方向**。LUT の出力 u32 = 4 nibble、2 個で 8 nibble = 既存 `e2m1x8_to_bf16` の入力そのもの
   なので decode 後段は無改造。ステージする圧縮バイトは 25% 減るので、
   register が厳しかった `fp4_tcw.cu` の 2 段プリフェッチはむしろ楽になる。
5. **混合精度**: expert ごとに rate が違うが、grouped GEMM は元々 expert ごとに group を分けているので
   per-expert の rate フラグで 2 種類のカーネル（従来 4 bit / VQ 3 bit）を出し分ければよい。
   production では **rate は 2 種類に限る**（カーネル爆発を避ける）。
6. **最初にやるべき計測**: 本番カーネルを触る前に、「圧縮バイト読み + LUT + 同じ mma ループ」だけの
   マイクロベンチを書き、1/8/32/64 token per expert で **実効 GB/s（元の重みバイト数 / 時間）**を現行と比較する。
   合格ラインは「現行比 0.75 倍の時間」= 25% のバイト削減が LUT に食われないこと。
7. エンコード側はオフラインで、実測 9 分（GPU、全 15,360 expert）。

### 12.8b VQ decode カーネルのマイクロベンチ（実測）

`cuda/vqbench.cu` + `vqbench.py`。`fp4_tc.cu` と同じ warp/lane 構造（warp が 8 行、lane が t で k を 4 分割、
tiled レイアウト）で、**read だけ / +decode / +MMA** を分離して測った。A100 80GB、16 expert 相当（377M weight）。

| format | bit/w | bytes/expert | read | +decode | +MMA | read GB/s | FP4 比 |
|---|---|---|---|---|---|---|---|
| FP4（64k/lane） | 4.0 | 12.53 MB | 0.126 ms | 0.138 | 0.143 | 1592 | 1.000x |
| VQ12（64k/lane） | 3.0 | 9.58 MB | 0.113 ms | 0.134 | 0.156 | 1354 | 1.096x |
| VQ14（64k/lane） | 3.5 | 11.06 MB | 0.164 ms | 0.183 | 0.211 | 1082 | 1.475x |
| FP4（128k/lane） | 4.0 | 12.53 MB | 0.131 ms | 0.144 | 0.139 | 1534 | 0.975x |
| **VQ12（128k/lane）** | 3.0 | 9.58 MB | **0.103 ms** | 0.137 | 0.165 | **1492** | 1.154x |
| VQ14（128k/lane） | 3.5 | 11.06 MB | 0.136 ms | 0.169 | 0.202 | 1302 | 1.415x |

読み方:

1. **バイト削減は read 段では実現する**。lane あたり 128 k にすると VQ12 の read は 0.103 ms（1492 GB/s）で、
   FP4 の 0.131 ms より **21% 速い**（バイトは 24% 少ない = ほぼ丸ごと効いている）。
   64 k/lane だとロード粒度が小さすぎて 1354 GB/s に落ちるので、**粒度は 128 k/lane 必須**。
2. **しかし LUT decode がそれ以上に食う**。VQ12 の decode 段は +0.034 ms（FP4 は +0.013 ms）。
   4 weight ごとに 1 回の shared memory ロードが必要で、インデックスはランダムなので bank conflict が乗る。
   結果として **合計は FP4 比 1.15x 遅い**。帯域律速だったカーネルが issue 律速に変わってしまう。
3. **VQ14（3.5 bit）はさらに悪い**（1.42x）。6 bit フィールドの抽出と、16384 entry を u16 で置くことによる
   2-way bank conflict の両方が効く。**3.5 bit はカーネル的には筋が悪い**。
4. レジスタは 31〜56 で、FP4（40）と大差なし。レジスタ溢れは起きていない。

**結論はハードウェアで割れる**:

| 用途 | ボトルネック | VQ12(3.0bit) の効果 |
|---|---|---|
| A100 VRAM 常駐（現行 EP） | HBM 1.5 TB/s | **負け**（read −21% だが decode で +19% → 合計 1.15x 遅い） |
| NVMe / unified memory streaming（DGX Spark, Mac Studio） | SSD 1.67 GB/s | **圧勝**（転送 −24%、decode は expert あたり 15 us で SSD 8.6 ms の 0.2%） |

expert 1 個（35.4M weight）の decode は実測 **15 us**。NVMe 読み出しが 8.6 ms なので、
streaming 用途では decode コストは事実上ゼロ。**低 bit 化がそのまま tok/s に乗る。**

A100 常駐側で勝つには LUT を消す必要がある。方向は **代数的 codebook**
（QTIP / QuIP# 系の trellis・ハッシュ生成型）で、12 bit インデックスから 4 個の値を
shared memory を触らず ALU だけで作る。これなら decode は FP4 の bit 配置と同程度になり、
read の −21% がそのまま残る。R-D 特性は VQ のまま維持できる。

### 12.9 残る未確認事項

1. **他ドメインの PPL / 下流ベンチ**: 今回は wikitext-2 のみ。コード・日本語・長文での確認が必要。
   特に混合精度は routing に依存するので、**未知タスクでの劣化**が最大のリスク（§12.4 の「任意タスク上界」）。
2. **calibration token の増量**: 現在 49k token。cold expert の Hessian が推定できていない（§12.7）。
   640 window（1.3M token、約 1 時間）まで増やすか、縮小推定を使う。
3. **GPTQ 型の誤差フィードバック**: 今回は Hessian 対角のみ（AWQ 相当）。
   非対角（Cholesky + 逐次量子化）まで入れれば 3.0 bit の +4.2% はさらに下がる余地がある。
4. **カーネル実測**: §12.8 のマイクロベンチ。理論上の帯域削減が実速度になるかは未確認。
5. **telemetry の拡充**: タスク数・トークン数を増やし、混合配分を hold-out タスクで検証する。

### 12.10 lossy 側の判断（PPL 実測後）

- **一律 3 bit のスカラー量子化は却下**（expert 出力誤差 38.7%）。VQ でなければ話にならない。
- **VQ dim4（E2M1 snap）+ routing 混合精度、平均 3.5 bit が最有力**。
  実測 PPL 3.0765 → **3.1002（+0.77%）**、ratio **0.853**（VRAM −14.7%、HBM 帯域 −14.7%、期待 +7% スループット）。
- より攻めるなら一律 3.0 bit（ratio 0.735、VRAM/帯域 −26.5%、期待 +12%）で **+4.2%**。
  混合配分 + calibration の改善余地を考えると、+3% 程度までは詰められる見込み。
- **次の意思決定に必要なのはカーネル実測**（§12.8）。品質側はもう「やれる」と言える水準にある。

---

## 13. 生成物

| ファイル | 内容 |
|---|---|
| `results/phase1.{json,csv,md}` | A/B/B2/C/D/E の全測定値 |
| `results/phase1_fg.{json,csv}` | F（expert 間）/ G（scale） |
| `results/phase1_extra.json` | layer 別 entropy、scale alphabet、±0 alias |
| `results/scale_alphabet_full.json`, `results/scan_scales.log` | 全 scale tensor 走査（16,986,931,200 byte） |
| `results/phase1_io.{json,csv}` | ストレージ帯域と decode 予算 |
| `results/summary.{csv,md,json}` | §8 の最終表 |
| `test_zero_merge.py` → §11 | ±0 統合が実カーネルで bit-identical であることの検証 |
| `lossy.py`, `lossy_alloc.py` → §12 | 非可逆側の rate-distortion、VQ、routing 混合精度 |
| `results/lossy.{csv,json}`, `results/lossy_rd.json`, `results/lossy_alloc.{csv,json}` | §12 の全数値 |
| `vq_build.py` → `results/vq_{3.0,3.5,3.75}.npz` | E2M1-snap VQ の codebook と 2byte→2byte LUT |
| `ppl.py`, `sweep_ppl.sh`, `sweep_calib.sh`, `sweep_mixed.sh`, `sweep_shrink.sh` | PPL 評価（VQ / calibration / routing 混合）|
| `calib_collect.py` → `results/calib_hess.pt` | per-expert 活性化 Hessian 対角（MoE forward をラップして収集）|
| `vq_encode.py` | activation 重み付きエンコーダ（候補上位 M のみ探索）|
| `mk_ppl_summary.py` → `results/ppl_summary.{csv,json}`, `results/ppl.jsonl` | §12.6 の PPL 表 |
| `data/sample.npz` | 1,003,520 block の sample（53 MB、再生成可） |
