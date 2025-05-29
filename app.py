from flask import Flask, request, jsonify, render_template
from token_compare import get_token_counts

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/count', methods=['POST'])
def count_tokens():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': "Missing 'text' field in request body"}), 400

    input_text = data['text']
    counts = get_token_counts(input_text)
    return jsonify(counts)

if __name__ == '__main__':
    app.run(debug=True)