def test_comparison_contains_three_strategies():
    from pathlib import Path
    text = Path("comparison/deployment_comparison.md").read_text(encoding="utf-8")
    for strategy in ["Containerization", "Serverless Functions", "Managed ML Services"]:
        assert strategy in text
