from get_text import extract_text, pdf2image, docx2pdf2image, check_cv, pdf2text, docx2pdf
from flask import Flask, jsonify, request, send_file
import mimetypes
import warnings
import os
import cv2
import io
#from docx2pdf import convert
from PIL import Image
import numpy as np
import json
import uuid


warnings.filterwarnings("ignore")

# Lấy kích thước size định dạng files upload
def get_size(file):
    if file.content_length:
        return file.content_length

    try:
        pos = file.tell()
        file.seek(0, 2)  # seek to end
        size = file.tell()
        file.seek(pos)  # back to original position
        return size
    except (AttributeError, IOError):
        pass

    # in-memory file object that doesn't support seeking or tell
    return 0  # assume small enough

app = Flask(__name__)


@app.route('/extract_cv', methods=['POST', 'GET'])
def extract_cv():
    data_body = dict(request.form)
    error = None
    data = None
    ext_pdf = ['.pdf']
    ext_doc = ['.docx', '.doc']
    ext_img = ['.jpg', '.png']
    if len(dict(request.files)):
        file = request.files['file']
        getsize = get_size(file)
        output_path = './results'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if 0 < getsize < 1000000:
            fname = file.filename
            extension = os.path.splitext(fname)[1]
            if extension in ext_pdf:
                fname = str(uuid.uuid4())+ extension
                file.save(os.path.join('./results', fname))
                file_pdf = './results/'+fname
                image = pdf2image(file_pdf, 150)
            elif extension in ext_doc:
                fname = str(uuid.uuid4())+ extension
                file.save(os.path.join('./results', fname))
                docx2pdf('./results/'+fname, "results")
                file_pdf = './results/'+fname.replace(extension, '.pdf')
                image = docx2pdf2image(file_pdf)
            elif extension in ext_img:
                image_byte = file.read()
                image = Image.open(io.BytesIO(image_byte))
                image = np.array(image)
                if len(image.shape) > 2 and image.shape[2] == 4:
                    # convert the image from RGBA2RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                return jsonify({'status': 0,
                                'error_code': 400,
                                'message': 'Vui lòng nhập định dạng file là .jpg, .png, .pdf, .doc hoặc .docx'})
        else:
            return jsonify({'status': 0,
                            'error_code': 400,
                            'message': 'vui lòng nhập tệp có kích cỡ trong khoảng 0 < getsize < 1000000'})
        label_list = check_cv(image)
        if 'infor' not in label_list:
            result = pdf2text(file_pdf)
            return json.dumps({'status': 1,
                                'error_code': 200,
                                'message': 'trich xuat thanh cong',
                                'information': result})
        result = extract_text(image)
    return json.dumps({'status': 1,
                       'error_code': 200,
                       'message': 'trích xuất thành công thông tin trong cv',
                       'information': result})

@app.route('/<path:image_filename>')
def get_avatar(image_filename):
    image_path = os.path.join('./results', image_filename)
    mimetype, _ = mimetypes.guess_type(image_path)
    return send_file(image_path, mimetype=mimetype, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)


