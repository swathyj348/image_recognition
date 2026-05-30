import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from .infer import predict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'model_full.pth')

ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            try:
                pred = predict(MODEL_PATH, save_path)
                result = pred
            except Exception as e:
                result = f'Error: {e}'
    return render_template('index.html', result=result, filename=filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
