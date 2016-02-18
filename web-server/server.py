# System imports
import sys, subprocess, time
import numpy as np
from os import path
from flask.ext.cors import CORS
from flask import *
from flask.json import jsonify
from werkzeug import secure_filename
from flask_extensions import *

tools_path = os.path.abspath(os.path.join('../tools'))
sys.path.append(tools_path)
tensorflow_path = os.path.abspath(os.path.join('../tensorflow/googly'))
sys.path.append(tensorflow_path)
preprocessing_path = os.path.abspath(os.path.join('../preprocessing'))
sys.path.append(preprocessing_path)

#from predict import predict

import deepaudio as experiment
import image_input
import tensorflow as tf
from preprocessing_commons import wav_to_images_in_memory

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '/home/vegeboy/workspace/uni/iLID-Data/experiment_train_7',
                           """Directory where to read model checkpoints.""")

probabilities_op = None
x = None
sess = None

#lib_path = os.path.abspath(os.path.join('../evaluation'))
#sys.path.append(lib_path)
#from predict import predict
#from convert_to_mono_wav import convert as convert_to_mono_wav


static_assets_path = path.join(path.dirname(__file__), "dist")
app = Flask(__name__, static_folder= static_assets_path)
CORS(app)


# ----- Routes ----------

@app.route("/", defaults={"fall_through": ""})
@app.route("/<path:fall_through>")
def index(fall_through):
    if fall_through:
        return redirect(url_for("index"))
    else:
        return app.send_static_file("index.html")


@app.route("/dist/<path:asset_path>")
def send_static(asset_path):
    return send_from_directory(static_assets_path, asset_path)


@app.route("/audio/<path:audio_path>")
def send_audio(audio_path):
    return send_file_partial(path.join(app.config["UPLOAD_FOLDER"], audio_path))


@app.route("/api/upload", methods=["POST"])
def uploadAudio():

    def is_allowed(filename):
        return len(filter(lambda ext: ext in filename, ["wav", "mp3", "ogg"])) > 0

    file = request.files.getlist("audio")[0]

    if file and is_allowed(file.filename):
        filename = secure_filename(file.filename)
        file_path = path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # convert_to_mono_wav(file_path, True)

        response = jsonify(get_prediction(file_path))
    else:
        response = bad_request("Invalid file")

    return response


@app.route("/api/example/<int:example_id>")
def use_example(example_id):
    if example_id <= 3:
        filename = "audio%s.wav" % example_id
        file_path = path.join(app.config["UPLOAD_FOLDER"], "examples", filename)
        response = jsonify(get_prediction(file_path))
    else:
        response = bad_request("Invalid Example")

    return response


def bad_request(reason):
    response = jsonify({"error" : reason})
    response.status_code = 400
    return response


# -------- Prediction & Features --------
def get_prediction(file_path):

    LABEL_MAP = {
        0 : "English",
        1 : "German",
        2 : "French",
        3 : "Spanish"
    }

    # TODO remove this for production
    # predictions = [[0.3, 0.7]]
    images = wav_to_images_in_memory(file_path)
    with tf.Session() as sess:
        probabilities = sess.run([probabilities_op], feed_dict={x: images})
        predictions = np.mean(predictions, axis=0).tolist()

    print predictions

    pred_with_label = {LABEL_MAP[index] : prob for index, prob in enumerate(predictions)}

    file_path = file_path + "?cachebuster=%s" % time.time()
    result = {
        "audio" : {
            "url" : "%s" % file_path,
        },
        "predictions" : pred_with_label
    }

    return result


def initialize_model():
    global probabilities
    global x
    """Eval for a number of steps."""
    with tf.Graph().as_default():
    # Get images and labels for 10.
        x = tf.placeholder(tf.float32, [None, 39, 600, 1])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = experiment.inference(x)

        # Calculate predictions.
        probabilities = tf.nn.softmax(logits)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            experiment.MOVING_AVERAGE_DECAY)
        variables_to_restore = {}
        for v in tf.all_variables():
          if v in tf.trainable_variables():
            restore_name = variable_averages.average_name(v)
          else:
            restore_name = v.op.name
          variables_to_restore[restore_name] = v

        saver = tf.train.Saver(variables_to_restore)

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        with tf.Session() as sess:
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)

if __name__ == "__main__":
    # Start the server
    app.config.update(
        DEBUG = True,
        SECRET_KEY = "asassdfs",
        CORS_HEADERS = "Content-Type",
        UPLOAD_FOLDER = "audio",
        MODEL = os.path.join("model", "berlin_net_iter_10000.caffemodel"),
        PROTOTXT = os.path.join("model", "net_mel_2lang_bn_deploy.prototxt")
    )
    initialize_model()
    # Make sure all frontend assets are compiled
    # subprocess.Popen("webpack")

    # Start the Flask app
    app.run(port=9009)
