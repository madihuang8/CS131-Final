import threading
from flask import Flask, Response, render_template, request, jsonify
import guitar

app = Flask(__name__)

state = {'chord': 'G'}
state_lock = threading.Lock()

guitar.start(state, state_lock)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(guitar.generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot')
def snapshot():
    frame = guitar.get_snapshot()
    if frame is None:
        return '', 503
    return Response(bytes(frame), mimetype='image/jpeg')


@app.route('/set_chord', methods=['POST'])
def set_chord():
    chord = request.json.get('chord', '')
    with state_lock:
        state['chord'] = chord
    return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='127.0.0.1', port=5000)
