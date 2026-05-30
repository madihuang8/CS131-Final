import threading
from flask import Flask, Response, render_template, request, jsonify
import guitar
import fretboard

app = Flask(__name__)

state = {'chord': 'G'}
state_lock = threading.Lock()


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


@app.route('/lock_fretboard', methods=['POST'])
def lock_fretboard():
    locked = fretboard.lock_current_fretboard()
    return jsonify({'ok': locked, 'locked': locked})


@app.route('/reset_fretboard', methods=['POST'])
def reset_fretboard():
    fretboard.reset_locked_fretboard()
    return jsonify({'ok': True, 'locked': False})


@app.route('/fretboard_status')
def fretboard_status():
    return jsonify({'locked': fretboard.is_fretboard_locked()})


if __name__ == '__main__':
    guitar.start(state, state_lock)
    app.run(debug=False, threaded=True, host='127.0.0.1', port=5000)
