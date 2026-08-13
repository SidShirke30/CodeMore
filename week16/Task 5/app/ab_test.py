from pathlib import Path
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]

def load_results(path=ROOT / "data" / "ab_results.csv"):
    return pd.read_csv(path)

def summarize(df):
    rows = []
    for model, g in df.groupby("model_version"):
        rows.append({
            "model_version": model,
            "requests": len(g),
            "accuracy": g["correct"].mean(),
            "avg_latency_ms": g["latency_ms"].mean(),
            "error_rate": 1 - g["correct"].mean(),
        })
    return pd.DataFrame(rows)

def accuracy_z_test(df, control="A", treatment="B"):
    a = df[df.model_version == control]["correct"]
    b = df[df.model_version == treatment]["correct"]
    p1, p2 = a.mean(), b.mean()
    n1, n2 = len(a), len(b)
    pooled = (a.sum() + b.sum()) / (n1 + n2)
    se = (pooled * (1 - pooled) * (1/n1 + 1/n2)) ** 0.5
    z = (p2 - p1) / se if se else 0.0
    p_value = 2 * norm.sf(abs(z))
    return {"z_stat": z, "p_value": p_value, "accuracy_difference": p2 - p1}

def choose_model(summary, test, alpha=0.05):
    a = summary.loc[summary.model_version == "A"].iloc[0]
    b = summary.loc[summary.model_version == "B"].iloc[0]
    if test["p_value"] < alpha and b.accuracy > a.accuracy:
        return "B"
    if b.accuracy >= a.accuracy and b.avg_latency_ms < a.avg_latency_ms:
        return "B"
    return "A"

if __name__ == "__main__":
    df = load_results()
    summary = summarize(df)
    test = accuracy_z_test(df)
    winner = choose_model(summary, test)
    print(summary.to_string(index=False))
    print(f"z={test['z_stat']:.3f}, p={test['p_value']:.4f}")
    print(f"Recommended model: {winner}")
