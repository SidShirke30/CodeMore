from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "service": "model-api"})


@app.get("/")
def index():
    return jsonify({
        "message": "Week 15 AI model deployment pipeline",
        "version": "1.0.0"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
