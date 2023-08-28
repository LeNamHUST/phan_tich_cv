import re
import unidecode
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import spacy



key_exp = ['kinh nghiệm làm việc',
           'kinh nghiệm',
           'experience'
           ]

key_obj = ['mục tiêu nghề nghiệp',
           'mục tiêu',
           'objective'
           ]

key_edu = ['trình độ học vấn',
           'education'
           ]

key_job = ['thông tin nghề nghiệp']

key_skill = ['các kỹ năng',
             'kỹ năng',
             'skill'
             ]

key_award = ['giải thưởng và thành tích',
             'giải thưởng',
             'awards'
             ]

key_cert = ['chứng chỉ',
            'certifications'
            ]

key_hobby = ['sở thích và tính cách',
             'sở thích',
             'hobbies'
             ]

key_ref = ['người tham chiếu',
           'người liên hệ',
           'references'
           ]

key_act = ['hoạt động',
           'activities'
           ]

key_pro = ['dự án tham gia',
           'dự án',
           'projects'
           ]

features = {'key_exp': key_exp,
            'key_obj': key_obj,
            'key_edu': key_edu,
            'key_job': key_job,
            'key_skill': key_skill,
            'key_award': key_award,
            'key_cert': key_cert,
            'key_hobby': key_hobby,
            'key_ref': key_ref,
            'key_act': key_act,
            'key_pro': key_pro
            }

map_features = {'key_gen': 'Gender',
                'key_exp': 'Experience',
                'key_obj': 'Objective',
                'key_edu': 'Education',
                'key_job': 'JobInformation',
                'key_skill': 'Skill',
                'key_add': 'Address',
                'key_award': 'Awards',
                'key_cert': 'Certifications',
                'key_hobby': 'Hobbies',
                'key_ref': 'References',
                'key_act': 'Activities',
                'key_pro': 'Project'
                }

infos = ['Avatar',
         'FullName',
         'DateOfBirth',
         'Gender',
         'Email',
         'PhoneNumber',
         'Address',
         'Education',
         'Job',
         'Salary',
         'Workplace',
         'Objective',
         'Skill',
         'Awards',
         'Certifications',
         'Hobbies',
         'References',
         'Experience',
         'Activities',
         'Project',
         'AdditionalInformation'
         ]



def check_upper_text(text):
    count = 0
    for i in text:
        if i == i.upper():
            count += 1
    if count > len(text) / 2:
        return True
    return False


def remove_accent(text):
    return unidecode.unidecode(text)

def pre_text(text):
    text = text.replace('�', ' ')
    text = text.replace('\n', ' ')
    return text

def show_text(arr):
    if len(arr) > 0:
        text = arr[0]
        for i in range(1, len(arr)):
            text = text+', '+arr[i]
    else:
        text = ''
    return text

#################################################################
############ lấy thông tin về địa chỉ ###########################
#################################################################

def extract_address(text):
    address = []
    tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base", model_max_length=50)
    model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
    ner_results = nlp(text)
    for ent in ner_results:
        if (ent['entity_group'] == 'LOCATION'):
            address.append(ent['word'])
    return show_text(address)

######################################################
############ lấy thông tin giới tính #################
######################################################


def extract_gender(text):
    text = text.split('\n')
    patterns = ['Nam', 'Nữ', 'Male', 'Female']
    for pattern in patterns:
        if pattern in text:
            return pattern
    else:
        return None


######################################################
############ lấy thông tin email #####################
######################################################


def extraxt_email(text):
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    if (len(email) != 0):
        return email[0]
    else:
        return None


######################################################
############ lấy thông tin số điện thoại #############
######################################################


def extract_phone(text):
    text = text.split('\n')
    patterns = ['[0-9]{10}', '[0-9]{5} [0-9]{5}',
                '[0-9]{4} [0-9]{3} [0-9]{3}',
                '[0-9]{4}-[0-9]{3}-[0-9]{3}',
                '[0-9]{3}-[0-9]{3}-[0-9]{4}']
    for pattern in patterns:
        for t in text:
            if (re.search(pattern, t)):
                return t
    else:
        return None


#####################################
######## lấy ngày tháng năm sinh ####
#####################################



def extract_date_of_birth(text):
    date_of_birth = re.findall(r'\d{2}[/-]\d{2}[/-]\d{4}', text)
    if (len(date_of_birth) != 0):
        return date_of_birth[0]
    return None


######################################################

def extract_feature_from_keys(text):
    text_sp = text.split('\n')
    for key in features.keys():
        for k in features[key]:
            for i in range(len(text_sp)):
                if k in (text_sp[i]).lower():
                    return map_features[key], text.replace(text_sp[i], '')
    return 'AdditionalInformation', text


def resume_extract(texts, labels, conf):
    nlp_ner = spacy.load("model")

    result = {}

    for info in infos:
        result[info] = ''
    result['AdditionalInformation'] = []

    for i, text in enumerate(texts):
        if labels[i] == 'block':
            feature, res_text = extract_feature_from_keys(text)
            if feature == 'JobInformation':
                doc = nlp_ner(res_text)
                for entity in doc.ents:
                    if entity.label_ == 'SALARY':
                        result['Salary'] = entity.text
                result['Workplace'] = extract_address(res_text)
            elif feature != 'AdditionalInformation':
                result[feature] = res_text
            else:
                result['AdditionalInformation'].append(res_text)
        elif labels[i] == 'infor' and conf[i] > 0.7:
            result['Address'] = extract_address(text)
            result['DateOfBirth'] = extract_date_of_birth(text)
            result['PhoneNumber'] = extract_phone(text)
            result['Email'] = extraxt_email(text)
            result['Gender'] = extract_gender(text)
        elif labels[i] == 'name':
            result['FullName'] = text.replace('\n', ' ')
        else:
            result['Job'] = text
    result['AdditionalInformation'] = '\n'.join(result['AdditionalInformation'])
    return result

def cv_extract(text):
    result = {}

    nlp_ner = spacy.load("model")
    doc = nlp_ner(text)
    arr_education = []
    arr_exp = []
    arr_skill = []
    arr_cer = []
    arr_hobbies = []
    arr_award = []
    arr_add = []
    for entity in doc.ents:
        if entity.label_ == 'NAME':
            result['FullName'] = pre_text(entity.text)
        elif entity.label_ == 'DATEOFBIRTH':
            result['DateOfBirth'] = pre_text(entity.text)
        elif entity.label_ == 'GENDER':
            result['Gender'] = pre_text(entity.text)
        elif entity.label_ == 'PHONE':
            result['Phone'] = pre_text(entity.text)
        elif entity.label_ == 'EMAIL':
            result['Email'] = pre_text(entity.text)
        elif entity.label_ == 'ADDRESS':
            result['Address'] = pre_text(entity.text)
        elif entity.label_ == 'JOB':
            result['Job'] = pre_text(entity.text)
        elif entity.label_ == 'EDUCATION':
            arr_education.append(pre_text(entity.text))
            result['Education'] = show_text(arr_education)
        elif entity.label_ == 'EXP':
            arr_exp.append(pre_text(entity.text))
            result['Experience'] = show_text(arr_exp)
        elif entity.label_ == 'SKILL':
            arr_skill.append(pre_text(entity.text))
            result['Skill'] = show_text(arr_skill)
        elif entity.label_ == 'SALARY':
            result['Salary'] = pre_text(entity.text)
        elif entity.label_ == 'WORKSPACE':
            result['Workspace'] = pre_text(entity.text)
        elif entity.label_ == 'HOBBIES':
            arr_hobbies.append(pre_text(entity.text))
            result['Hobbies'] = show_text(arr_hobbies)
        elif entity.label_ == 'AWARDS':
            arr_award.append(pre_text(entity.text))
            result['Awards'] = show_text(arr_award)
        elif entity.label_ == 'CERTIFICATE':
            arr_cer.append(pre_text(entity.text))
            result['Certificate'] = show_text(arr_cer)
        elif entity.label_ == 'ADDITIONALINFORMATION':
            arr_add.append(pre_text(entity.text))
            result['AdditionalInformation'] = show_text(arr_add)
    return result

