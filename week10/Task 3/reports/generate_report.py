from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from ab_testing import metrics, two_proportion_z_test

m = metrics()
a = m["A"]
b = m["B"]

accuracy_a = "N/A"
accuracy_b = "N/A"

if a["accuracy"] is not None:
    accuracy_a = f"{a['accuracy']:.2%}"
if b["accuracy"] is not None:
    accuracy_b = f"{b['accuracy']:.2%}"

z_score = None
if a["labeled_requests"] and b["labeled_requests"]:
    success_a = round(a["accuracy"] * a["labeled_requests"])
    success_b = round(b["accuracy"] * b["labeled_requests"])
    z_score = two_proportion_z_test(
        success_a, a["labeled_requests"],
        success_b, b["labeled_requests"]
    )

text = "# A/B Test Model Comparison Report\n\n"
text += "## Results\n\n"
text += "| Metric | Model A | Model B |\n|---|---:|---:|\n"
text += f"| Requests | {a['requests']} | {b['requests']} |\n"
text += f"| Labeled requests | {a['labeled_requests']} | {b['labeled_requests']} |\n"
text += f"| Accuracy | {accuracy_a} | {accuracy_b} |\n"
text += f"| Average latency | {a['average_latency_ms'] or 0:.3f} ms | {b['average_latency_ms'] or 0:.3f} ms |\n\n"

text += "## Statistical comparison\n\n"
if z_score is None:
    text += "A z-score could not be calculated because labeled outcomes are not available for both variants.\n"
else:
    text += f"Two-proportion z-test statistic: **{z_score:.4f}**. "
    text += "This is a screening statistic; a production analysis should calculate a two-sided p-value and confidence interval before declaring a winner.\n\n"

text += "## Recommendation\n\n"
text += "Select the model using the pre-registered primary metric, statistical uncertainty, latency, and business impact. Do not choose a winner from a small sample based only on raw accuracy.\n"

report = ROOT / "reports" / "ab_test_report.md"
report.write_text(text, encoding="utf-8")
print(f"Report written to {report}")
