from pathlib import Path
import joblib
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "app" / "model.joblib"
DATA_PATH = ROOT / "data" / "sample_data.csv"
OUTPUT_PATH = ROOT / "explanations" / "prediction_explanation.html"

def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    class_names = bundle["class_names"]

    data = pd.read_csv(DATA_PATH)
    X = data[feature_names].values

    # Explain one concrete instance.
    instance_index = 0
    instance = X[instance_index]

    explainer = LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=len(feature_names),
    )

    prediction = int(model.predict([instance])[0])
    probabilities = model.predict_proba([instance])[0]

    print("Explained instance:", instance)
    print("Predicted class:", class_names[prediction])
    print("Prediction probabilities:")
    for name, probability in zip(class_names, probabilities):
        print(f"  {name}: {probability:.4f}")

    print("\nLocal feature contributions:")
    for feature, weight in explanation.as_list(label=prediction):
        direction = "supports" if weight > 0 else "opposes"
        print(f"  {feature}: {weight:.4f} ({direction} the prediction)")

    explanation.save_to_file(OUTPUT_PATH)
    print(f"Interactive explanation saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
