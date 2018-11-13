from flask import Flask,send_file
import flask
from PIL import Image
import io
from completeface import evaluate

from fromsketch import pix2pix
import os

app = Flask(__name__,static_url_path='')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route("/fromsketch",methods=["POST"])
def fromsketch():
    test_path = os.path.join(os.path.join(os.path.dirname(__file__), 'fromsketch'),pix2pix.a.test_dir)
    out_path = os.path.join(os.path.join(os.path.dirname(__file__),'fromsketch'),pix2pix.a.output_dir)
    if flask.request.method == "POST":
        image = flask.request.files["image"].read()
        # image = io.BytesIO(image)
        image = Image.open(io.BytesIO(image))
        image.save(test_path+'/test.jpg')
        pix2pix.main()
        byte_io = io.BytesIO()
        with open(out_path + '/images/test-outputs.png', 'rb')as f:
            byte_image = f.read()
        byte_io.write(byte_image)
        byte_io.seek(0)
    return send_file(byte_io, mimetype="image/jpeg")
@app.route("/completeface", methods=["POST"])
def completeface():
    test_path = os.path.join(os.path.dirname(__file__), evaluate.args.test_dir)
    out_path = os.path.join(os.path.dirname(__file__), evaluate.args.outf)
    test_list_path = os.path.join(os.path.dirname(__file__),'completeface/list_test.txt')
    with open(test_list_path,"wb") as f:
        f.write("test_data/test.jpg")
    if flask.request.method == "POST":
        image = flask.request.files["image"].read()
        # image = io.BytesIO(image)
        image = Image.open(io.BytesIO(image))
        image.save(test_path + '/test.jpg')
        evaluate.evaluate()
        byte_io = io.BytesIO()
        with open(out_path + '/1_x_bar_bar.png', 'rb')as f:
            byte_image = f.read()
        byte_io.write(byte_image)
        byte_io.seek(0)
    return send_file(byte_io, mimetype="image/jpeg")

if __name__ == '__main__':
    app.run()