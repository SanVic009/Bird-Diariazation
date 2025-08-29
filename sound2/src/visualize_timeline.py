import argparse, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline_csv", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.timeline_csv)
    labels = sorted(df["label"].unique().tolist())
    label_to_y = {lab:i for i,lab in enumerate(labels)}

    plt.figure()
    for _, row in df.iterrows():
        y = label_to_y[row["label"]]
        plt.plot([row["start_s"], row["end_s"]], [y,y], linewidth=6)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Time (s)")
    plt.title("Bird species timeline")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)

if __name__ == "__main__":
    main()
