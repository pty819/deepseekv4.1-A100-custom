| section | method | bit/weight | ratio vs 4 bit | note |
|---|---|---|---|---|
| A | raw FP4 payload | 4.0000 | 1.0000 | stored form |
| A | order-0 nibble entropy H(W) | 3.8931 | 0.9733 | static Huffman/rANS limit |
| A | order-0 byte entropy | 3.8793 | 0.9698 | pairs of nibbles as one symbol |
| A | order-0 scale entropy | 0.0315 | 0.1262 | 6 distinct E8M0 bytes; bits/weight column is the scale share |
| A | zstd level 1 (payload) | 3.8973 | 0.9743 | real codec on whole expert payload |
| A | zstd level 3 (payload) | 3.9033 | 0.9758 | real codec on whole expert payload |
| A | zstd level 9 (payload) | 3.8973 | 0.9743 | real codec on whole expert payload |
| A | zstd level 19 (payload) | 3.8990 | 0.9747 | real codec on whole expert payload |
| A | lz4 (payload) | 4.0002 | 1.0001 | real codec |
| A | gzip -6 (payload) | 3.9130 | 0.9782 | real codec |
| A | zstd level 9 (scales only) | 0.0423 | 0.1691 | scale bytes are 5.9% of an expert |
| B | H(W) | 3.8933 | 0.9733 | order-0 |
| B | H(W | scale) | 3.7709 | 0.9427 | exact E8M0 byte |
| B | H(W | position_in_block) | 3.8933 | 0.9733 |  |
| B | H(W | scale, position) | 3.7710 | 0.9427 |  |
| B | H(W | layer) | 3.8929 | 0.9732 |  |
| B | H(W | layer, matrix) | 3.8928 | 0.9732 |  |
| B | H(W | expert, matrix) [per-expert tables] | 3.8929 | 0.9732 | table cost charged per expert |
| B | H(W | layer, matrix, scale, position) | 3.7707 | 0.9427 | practical lower-bound candidate |
| B | H(W_i | W_i-1)  Markov-1 | 3.8675 | 0.9669 |  |
| B | H(W_i | W_i-1, W_i-2)  Markov-2 | 3.8454 | 0.9613 |  |
| B | H(W_i | W_i-1..3)  Markov-3 | 3.8289 | 0.9572 |  |
| B | H(W | position, running block max) | 3.8211 | 0.9553 | sequentially decodable |
| B | H(W | pos, runmax, W_i-1) | 3.8095 | 0.9524 |  |
| B | H(W | layer, matrix, scale, pos, runmax) | 3.7600 | 0.9400 | everything decodable, combined |
| B | H(W | pos, running mean magnitude) | 3.8088 | 0.9522 | sequentially decodable |
| B | H(W | scale, pos, runmax, W_i-1) | 3.7497 | 0.9374 |  |
| B | H(W | scale, pos, runmean, W_i-1) | 3.7596 | 0.9399 |  |
| B | H(W | layer, mat, scale, pos, runmax, W_i-1) | 3.7874 | 0.9469 | largest context tried |
| B2 | two-pass block class (2) + scale + pos | 3.7957 | 0.9489 | explicit 1 bit per 32-weight block of side info |
| B2 | two-pass block class (4) + scale + pos | 3.8106 | 0.9527 | explicit 2 bit per 32-weight block of side info |
| B2 | two-pass block class (8) + scale + pos | 3.8358 | 0.9589 | explicit 3 bit per 32-weight block of side info |
| B2 | two-pass block class (16) + scale + pos | 3.8650 | 0.9662 | explicit 4 bit per 32-weight block of side info |
| C | bit3 sign | - | - | P(1)=0.4999 |
| C | bit2 exp-hi | - | - | P(1)=0.4306 |
| C | bit1 exp-lo | - | - | P(1)=0.4440 |
| C | bit0 mantissa | - | - | P(1)=0.4130 |
| C | bit-plane coders summed (each cond. scale,pos) | 3.8708 | 0.9677 | independent planes; >= joint symbol entropy |
| C | bit-plane chain (cond. on higher planes) | 3.8933 | 0.9733 | identical to symbol entropy |
| D | PRNG XOR (per matrix) | 4.0000 | 1.0000 | zero-nibble rate 0.0625, zstd-9 1.0000 |
| D | PRNG XOR (per expert) | 4.0000 | 1.0000 | zero-nibble rate 0.0625, zstd-9 1.0000 |
| D | PRNG XOR (per block) | 4.0000 | 1.0000 | zero-nibble rate 0.0625, zstd-9 1.0000 |
| E | exact duplicate block rate | - | - | 0.0000% of 1,003,520 blocks |
| E | random pair Hamming distance | - | - | mean 63.01/128 bit |
| E | block dictionary D=256 (random templates) | 4.0138 | 1.0035 | NN Hamming 46.01/128, zero-nibble 0.1728, H(res|pos)=3.7630 |
| E | block dictionary D=1024 (random templates) | 4.0042 | 1.0010 | NN Hamming 43.45/128, zero-nibble 0.1953, H(res|pos)=3.6911 |
| E | block dictionary D=4096 (random templates) | 3.9935 | 0.9984 | NN Hamming 41.15/128, zero-nibble 0.2177, H(res|pos)=3.6180 |
| E | block dictionary D=16384 (random templates) | 3.9673 | 0.9918 | NN Hamming 38.69/128, zero-nibble 0.2472, H(res|pos)=3.5292 |
| E | block dictionary D=65536 (random templates) | 3.8862 | 0.9716 | NN Hamming 35.24/128, zero-nibble 0.2995, H(res|pos)=3.3857 |
| E | nearest of all 983,520 sampled blocks | 3.9783 | 0.9946 | NN Hamming 34.47/128; dictionary storage NOT counted (optimistic bound) |
| E | control: independence-shuffled blocks, D=65536 | - | - | NN Hamming 36.66/128, H(res)=3.4512; matching the real data means no block structure |
