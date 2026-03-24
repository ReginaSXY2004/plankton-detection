from pathlib import Path
import pandas as pd

CSV_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\track_results.csv"
OUT_SUMMARY = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\track_summary.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    # 去掉没有 ID 的情况
    df = df[df["track_id"] >= 0].copy()

    if df.empty:
        print("没有可分析的 track_id 数据。")
        return

    summary = (
        df.groupby("track_id")
        .agg(
            first_frame=("frame_idx", "min"),
            last_frame=("frame_idx", "max"),
            num_frames=("frame_idx", "nunique"),
            mean_conf=("conf", "mean"),
            mean_w=("w", "mean"),
            mean_h=("h", "mean"),
        )
        .reset_index()
    )

    summary["lifespan"] = summary["last_frame"] - summary["first_frame"] + 1
    summary["continuity_ratio"] = summary["num_frames"] / summary["lifespan"]

    summary = summary.sort_values(
        by=["num_frames", "continuity_ratio"],
        ascending=[False, False]
    )

    out_path = Path(OUT_SUMMARY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"统计完成: {out_path}")
    print("\n前20个最长轨迹：")
    print(summary.head(20).to_string(index=False))

    short_tracks = (summary["num_frames"] <= 2).sum()
    print(f"\n只出现 1-2 帧的短轨迹数量: {short_tracks}")
    print(f"总轨迹数量: {len(summary)}")


if __name__ == "__main__":
    main()