| method | theoretical bit/weight | real ratio (payload) | expert bit/weight | ratio incl. metadata | decode MB/s | I/O ms | decode ms | total ms | break-even decode MB/s | overlapped ms | decode MB/s needed to overlap | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw FP4 (stored form) | 4.000 | 1.0000 | 4.250 | 1.0000 | - | 11.28 | 0.00 | 11.28 | nan | 11.28 | 1667 | reject (>0.90) |
| zstd-3 | - | 0.9762 | 3.952 | 0.9299 | 992 | 10.49 | 18.95 | 29.44 | 23763 | 18.95 | 1792 | reject (>0.90) |
| zstd-9 | - | 0.9747 | 3.941 | 0.9272 | 1168 | 10.46 | 16.10 | 26.56 | 22900 | 16.10 | 1798 | reject (>0.90) |
| lz4 | - | 1.0001 | 4.122 | 0.9698 | 4962 | 10.94 | 3.79 | 14.73 | 55136 | 10.94 | 1719 | reject (>0.90) |
| order-0 nibble rANS | 3.893 | - | 4.143 | 0.9748 | - | 11.00 | 0.00 | 11.00 | 66263 | 11.00 | 1710 | reject (>0.90) |
| H(W | scale_bin, position) | 3.771 | - | 4.021 | 0.9461 | - | 10.67 | 0.00 | 10.67 | 30932 | 10.67 | 1762 | reject (>0.90) |
| Markov-1 | 3.868 | - | 4.117 | 0.9688 | - | 10.93 | 0.00 | 10.93 | 53460 | 10.93 | 1720 | reject (>0.90) |
| Markov-2 | 3.845 | - | 4.095 | 0.9636 | - | 10.87 | 0.00 | 10.87 | 45818 | 10.87 | 1730 | reject (>0.90) |
| bit-plane coders (cond. scale,pos) | 3.871 | - | 4.121 | 0.9696 | - | 10.94 | 0.00 | 10.94 | 54826 | 10.94 | 1719 | reject (>0.90) |
| PRNG XOR | 4.000 | 1.0000 | 4.250 | 1.0000 | - | 11.28 | 0.00 | 11.28 | nan | 11.28 | 1667 | reject (>0.90) |
| block dictionary 65536 + XOR residual | 3.886 | - | 4.136 | 0.9732 | - | 10.98 | 0.00 | 10.98 | 62245 | 10.98 | 1713 | reject (>0.90) |
| nearest-block residual (1M dictionary) | 3.978 | - | 4.228 | 0.9949 | - | 11.22 | 0.00 | 11.22 | 326427 | 11.22 | 1675 | reject (>0.90) |
| best context model (payload only) | 3.750 | - | 4.000 | 0.9411 | - | 10.62 | 0.00 | 10.62 | 28300 | 10.62 | 1771 | reject (>0.90) |
| scales: 4-bit packing only | - | 1.0000 | 4.125 | 0.9706 | 20000 | 10.95 | 0.06 | 11.00 | 56668 | 10.95 | 1717 | reject (>0.90) |
| scales: entropy coded only | - | - | 4.031 | 0.9486 | 1000 | 10.70 | 1.11 | 11.81 | 32401 | 10.70 | 1757 | reject (>0.90) |
| best combined (context rANS + coded scales) | 3.781 | - | 3.781 | 0.8897 | - | 10.04 | 0.00 | 10.04 | 15106 | 10.04 | 1873 | only if decode is ~free |
| +-0 alias merged (NOT byte-lossless) | 3.633 | - | 3.664 | 0.8622 | - | 9.73 | 0.00 | 9.73 | 12098 | 9.73 | 1933 | only if decode is ~free |
