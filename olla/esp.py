from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/command', methods=['POST'])
def command():
    data = request.json

    print("Received:", data)

    # 🔥 Replace this with AI later
    level = data.get("level")

    if level > 5000:
        return jsonify({"action": "forward"})
    elif level > 3000:
        return jsonify({"action": "left"})
    elif level > 2000:
        return jsonify({"action": "right"})
    else:
        return jsonify({"action": "stop"})

app.run(host='0.0.0.0', port=5000)