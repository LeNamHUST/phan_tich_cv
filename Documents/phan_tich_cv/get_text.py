import os
import cv2
from extract_feature import resume_extract, cv_extract
from modules import YOLO_Det
from pdf2image import convert_from_path
import numpy as np
#import easyocr
from paddleocr import PaddleOCR, draw_ocr
import uuid
import fitz
from sua_chinh_ta import terminal_input
import subprocess

def pdf2image(pdf_path, dpi):
    pages = convert_from_path(pdf_path, dpi)
    #pages = convert_from_path(pdf_path)
    image = np.vstack([np.asarray(page) for page in pages])
    return image

def docx2pdf2image(pdf_path):
    pages = convert_from_path(pdf_path)
    image = np.vstack([np.asarray(page) for page in pages])
    return image

def docx2pdf(doc_path, path):

    subprocess.call(['soffice',
                  '--headless',
                 '--convert-to',
                 'pdf',
                 '--outdir',
                 path,
                 doc_path])
    return doc_path


def check_cv(image):
    output_path = './results'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    yolo_det_weight = './weights/best/best.pt'
    det_model = YOLO_Det(weight_path=yolo_det_weight)
    boxes_list, label_list, detect_image, confs = det_model(image, output_path= None, return_result=True)
    cv2.imwrite(os.path.join(output_path, 'detect_image.jpg'), detect_image)
    return label_list

def pdf2text(file_pdf):
    doc = fitz.open(file_pdf)
    text = ""
    for page in doc:
        text += page.get_text()
    result = cv_extract(text)
    return result


def extract_text(image):
    output_path = './results'
    yolo_det_weight = './weights/best/best.pt'
    det_model = YOLO_Det(weight_path=yolo_det_weight)
    boxes_list, label_list, detect_image, confs = det_model(image, output_path= output_path, return_result=True)
    texts, labels, conf = [], [], []
    #reader = easyocr.Reader(['vi'])
    
    ocr = PaddleOCR(use_angle_cls=True,
                lang="en",
                rec_model_dir='/home/ducchinh/AI_365/timviec365_elasticsearch/pdf2text/weights_2',
                rec_image_shape="3, 32, 320",
                rec_char_dict_path='/home/ducchinh/AI_365/timviec365_elasticsearch/pdf2text/util/vi_dict.txt',
                rec_algorithm='CRNN',
                use_space_char=False,
                use_gpu=False,
                ocr_version='PP-OCRv2')
    
    for i, box in enumerate(boxes_list):
        print(box)
        cropped_img = image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
        if label_list[i] == 'avatar' and confs[i] > 0.7:
            avatar_path = os.path.join(output_path, str(uuid.uuid4()) + '.png')
            avatar = cropped_img.copy()
            cv2.imwrite(avatar_path, avatar[:, :, ::-1])
            avatar_path = 'http://43.239.223.137:8000/' + avatar_path.split('/')[-1].replace('results\\', '/')
        else:
            #results=ocr.ocr(img_binary, cls=True)
            #text = '\n'.join([t[1] for t in reader.readtext(cropped_img)])
            text = '\n'.join(terminal_input(result[1][0]) for result in (ocr.ocr(cropped_img, cls=True))[0])
            texts.append(text)
            labels.append(label_list[i])
            conf.append(confs[i])
        if ('avatar' not in label_list) or (label_list[i] == 'avatar' and confs[i] <= 0.7):
            avatar_path = ('khong_tim_thay_avatar')
    result = resume_extract(texts, labels, conf)
    result['Avatar'] = avatar_path
    return result








