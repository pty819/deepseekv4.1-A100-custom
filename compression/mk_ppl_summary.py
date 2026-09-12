"""Consolidate results/ppl.jsonl into the comparison table (16-chunk runs only)."""
import csv, json

rows = []
for line in open("results/ppl.jsonl"):
    d = json.loads(line)
    if d["chunks"] != 16:
        continue
    b = d.get("mean_bits") or (d.get("bits") if d.get("vq", "none") != "none" else 4.0)
    if d["tag"] == "vq3.0-half":
        b = 3.5                      # half the layers at 3.0 bit = the same average rate
    rows.append({"tag": d["tag"], "mean_bits": round(b, 3),
                 "ratio_scales_8bit": round((b + 0.25) / 4.25, 4),
                 "ratio_scales_4bit": round((b + 0.125) / 4.25, 4),
                 "ppl": round(d["ppl"], 4), "delta_pct": None,
                 "calib": d.get("calib", "none"), "shrink": d.get("shrink", ""),
                 "tokens": d["tokens"]})
base = [r for r in rows if r["tag"] == "baseline"][0]["ppl"]
for r in rows:
    r["delta_pct"] = round((r["ppl"] / base - 1) * 100, 2)
rows.sort(key=lambda r: r["ppl"])
with open("results/ppl_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow(r)
json.dump(rows, open("results/ppl_summary.json", "w"), indent=1)
print(f"{'config':<30}{'bit/w':>7}{'ratio8':>8}{'ratio4':>8}{'PPL':>9}{'delta':>8}  calibration")
for r in rows:
    c = "-" if r["calib"] in ("none", "") else ("yes" + (f" shrink{int(r['shrink'])}" if r["shrink"] else ""))
    print(f"{r['tag']:<30}{r['mean_bits']:>7.3f}{r['ratio_scales_8bit']:>8.3f}{r['ratio_scales_4bit']:>8.3f}"
          f"{r['ppl']:>9.4f}{r['delta_pct']:>7.2f}%  {c}")
