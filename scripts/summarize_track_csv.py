from pathlib import Path
import pandas as pd

CSV_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\confirmed_microbes.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    if df.empty:
        print("没有确认目标数据。")
        return

    print("总记录数:", len(df))
    print("已计数个体数:", int(df["counted"].sum()))
    print("已保存最佳图个体数:", int(df["saved"].sum()))

    good = df[(df["counted"] == True)]
    if not good.empty:
        print("\n已计数目标的最佳图质量统计：")
        print(good[["best_conf", "best_sharpness", "best_w", "best_h"]].describe())

    print("\n前20条：")
    print(df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()