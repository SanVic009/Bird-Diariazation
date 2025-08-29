import argparse, pandas as pd, numpy as np

def hysteresis_binary(x, low, high):
    # x: [T] probabilities
    T = len(x)
    y = np.zeros(T, dtype=np.int32)
    active = False
    for t in range(T):
        if not active and x[t] >= high:
            active = True
        elif active and x[t] < low:
            active = False
        y[t] = 1 if active else 0
    return y

def to_segments(times, mask, min_dur=0.6, merge_gap=0.3):
    segs = []
    t0 = None
    for i, m in enumerate(mask):
        t = times[i]
        if m and t0 is None:
            t0 = t
        if (not m or i==len(mask)-1) and t0 is not None:
            t1 = times[i] if not m else times[i]
            if t1 - t0 >= min_dur:
                segs.append([t0, t1])
            t0 = None
    # merge small gaps
    merged = []
    for s in segs:
        if not merged:
            merged.append(s)
        else:
            if s[0] - merged[-1][1] <= merge_gap:
                merged[-1][1] = s[1]
            else:
                merged.append(s)
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="CSV from infer.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--low", type=float, default=0.45)
    ap.add_argument("--high", type=float, default=0.55)
    ap.add_argument("--min_dur", type=float, default=0.6)
    ap.add_argument("--merge_gap", type=float, default=0.3)
    args = ap.parse_args()

    df = pd.read_csv(args.probs)
    times = df["time_s"].values
    labels = [c for c in df.columns if c != "time_s"]

    rows = []
    for lab in labels:
        probs = df[lab].values.astype(float)
        mask = hysteresis_binary(probs, low=args.low, high=args.high)
        segs = to_segments(times, mask, min_dur=args.min_dur, merge_gap=args.merge_gap)
        for s,e in segs:
            rows.append({"label": lab, "start_s": float(s), "end_s": float(e), "duration_s": float(e-s)})
    out_df = pd.DataFrame(rows).sort_values(["start_s","label"])
    out_df.to_csv(args.out, index=False)
    print(f"Wrote timeline to {args.out}")

if __name__ == "__main__":
    main()
