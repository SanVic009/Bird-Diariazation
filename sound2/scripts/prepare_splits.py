import argparse, os, json, pandas as pd, numpy as np, sklearn.model_selection as skms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Path to metadata CSV")
    ap.add_argument("--out_dir", required=True, help="Where to write splits and labels")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--min_samples", type=int, default=2, help="Minimum samples per class")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "splits"), exist_ok=True)

    # Read and filter the data
    df = pd.read_csv(args.metadata)
    if "primary_label" not in df.columns or "filename" not in df.columns:
        raise ValueError("Expected columns 'primary_label' and 'filename' in metadata.")

    # Count samples per class and filter
    class_counts = df["primary_label"].value_counts()
    valid_classes = class_counts[class_counts >= args.min_samples].index
    df_filtered = df[df["primary_label"].isin(valid_classes)].reset_index(drop=True)
    
    print(f"Filtered out {len(class_counts) - len(valid_classes)} classes with < {args.min_samples} samples")
    print(f"Remaining classes: {len(valid_classes)}")

    # Build label map from filtered data
    labels = sorted(df_filtered["primary_label"].unique().tolist())
    label_to_idx = {l:i for i,l in enumerate(labels)}
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump({"labels": labels, "label_to_idx": label_to_idx}, f, indent=2)

    # Split the data, trying stratified first, falling back to non-stratified if needed
    try:
        train_df, temp_df = skms.train_test_split(
            df_filtered, 
            test_size=args.val_ratio+args.test_ratio, 
            stratify=df_filtered["primary_label"], 
            random_state=42
        )
        
        rel = args.test_ratio / (args.val_ratio + args.test_ratio) if (args.val_ratio+args.test_ratio)>0 else 0.5
        val_df, test_df = skms.train_test_split(
            temp_df, 
            test_size=rel, 
            stratify=temp_df["primary_label"], 
            random_state=42
        )
    except ValueError as e:
        print("Warning: Could not perform stratified split, falling back to random split")
        train_df, temp_df = skms.train_test_split(
            df_filtered, 
            test_size=args.val_ratio+args.test_ratio, 
            random_state=42
        )
        
        rel = args.test_ratio / (args.val_ratio + args.test_ratio) if (args.val_ratio+args.test_ratio)>0 else 0.5
        val_df, test_df = skms.train_test_split(
            temp_df, 
            test_size=rel, 
            random_state=42
        )

    train_df.to_csv(os.path.join(args.out_dir, "splits/train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "splits/val.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "splits/test.csv"), index=False)

    print(f"Labels: {len(labels)} classes")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
if __name__ == "__main__":
    main()
