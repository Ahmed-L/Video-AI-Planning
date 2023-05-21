from flask import Flask, jsonify, request, Blueprint

from VideoAIRecognition.inputstream import video

app = Flask(__name__)

app.register_blueprint(video, url_prefix='/api')


@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})


@app.route('/api/post', methods=['POST'])
def handle_post():
    data = request.get_json()
    # Process the data
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run()
