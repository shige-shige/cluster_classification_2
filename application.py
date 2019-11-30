# -*- coding: utf-8 -*-

# standard module
import os

# 3rd party module
from flask import flash, Flask, redirect, render_template, request, url_for
from flask import send_from_directory
from keras.models import Sequential, load_model
from PIL import Image
from werkzeug.utils import secure_filename
import keras
import numpy as np
import sys


# クラスター毎のディレクトリ
CLASSES = ['MORE', 'Ray', 'VERY']
NUM_CLASSES = len(CLASSES)

# イメージサイズを規定
IMAGE_SIZE_W = 50
IMAGE_SIZE_H = 100

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg', 'PNG', 'JPG', 'JPEG'])

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@application.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print('============>POST開始')
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイル名がありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
            keras.backend.clear_session()
            model = load_model('./app/visual_cnn.h5')
            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((IMAGE_SIZE_W, IMAGE_SIZE_H))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)
            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            os.remove('./uploads/' + filename)

            return render_template('index.html', filename=filename, predicted=CLASSES[predicted], percentage=percentage)
    return render_template('index.html')


# @application.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(application.config['UPLOAD_FOLDER'], filename)


@application.route('/reset')
def reset_site():
    return redirect(url_for('upload_file'))


def main():
    application.debug = True
    # application.run(host='127.0.0.1', port=5000)
    application.run()


if __name__ == '__main__':
    main()
