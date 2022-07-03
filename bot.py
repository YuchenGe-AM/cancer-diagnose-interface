import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
import numpy as np
import string  # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from trained import *

f = open('skin cancer.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()  # converts to lowercase

sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()


#
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "ai", "next")

GREETING_RESPONSES = ["Hi", "Hey", "*nods*", "Hi there", "Hello", "I am glad! You are talking to me",
                      "You're welcome. Just doing my job",
                      "You'd better talk with the doctor and you need further treatment"]


def greeting(sentence):
    for word in sentence.split():
        for i in range(len(GREETING_INPUTS)):
            if word.lower() == GREETING_INPUTS[i]:
                return GREETING_RESPONSES[i]


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo_response = robo_response #+ "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


# cnn part
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg
from skimage.transform import resize
from PIL import Image, UnidentifiedImageError

# 在这里加模型！加模型！加模型！# 在这里加模型！加模型！加模型！# 在这里加模型！加模型！加模型！# 在这里加模型！加模型！加模型！
models = [vgg16_model, vgg19_model, xception_model, resnet50_model, inception_model, inception_resnet_model]
model = models[5]
loaded_model = model(load_weights=True)
loaded_model.load_weights('inception_resnet_model_weight_1.h5')

def get_class(str):
    '''
    try:
        test_image = Image.open(str)
        Image.open
    except BaseException:
        return 'false'
    else:
    '''
    test_image = mpimg.imread(str)
    image_resize = resize(test_image, (115, 175), mode='constant')
    #tf.expand_dims(image_resize, 0)
    # predict the result
    image_resize = np.array([image_resize.reshape(115, 175, 3)])
    result = loaded_model.predict(image_resize)
    # cancer classes
    # 这里改一改 # 这里改一改 # 这里改一改 # 这里改一改 # 这里改一改 # 这里改一改 # 这里改一改
    classes = {0: ('ba','benign_adenosis'),
               1: ('bf','benign_fibroadenoma'),
               2: ('bpt','benign_phyllodes_tumor'),
               3: ('bta','benign_tubular_adenoma'),
               4: ('mdc','malignant_ductal_carcinoma'),
               5: ('mlc','malignant_lobular_carcinoma'),
               6: ('mmc','malignant_mucinous_carcinoma'),
               7: ('mpc','malignant_papillary_carcinoma')}
    return classes.get(np.argmax(result))[1]


def is_path(str):
    pics = ['bmp', 'png', 'jpg', 'jpeg', 'tiff', 'gif', 'pcx', 'tga', 'exif', 'fpx', 'svg', 'psd', 'cdr', 'pc', 'dxf',
            'ufo', 'eps', 'ai', 'raw']
    if (str.find('.') == -1):
        return -1
    elif (str[str.rfind('.') + 1::] in pics):
        return str
    else:
        return -1
    # /Users/yuxizheng/xizheng/proj_past_7007/Week_5/test_pics_with_label/ISIC_0034299_bcc_1.jpg


def chat(user_response):
    rob_response = "AI Doctor: Hi! I am a chatbot to tell you the diagnosis, please show me your skin picture."
    # check input is path
    path = is_path(user_response)
    if (path == -1):
        user_response = user_response.lower()
    # process user response
    if (user_response != 'bye'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            rob_response = "AI Doctor: You are welcome. Happy to help."
        elif (path != -1):
            r = get_class(path)
            rob_response = "AI Doctor: Please wait few second, your picture is processing."
            if (r == 'false'):
                rob_response = "AI Doctor: Sorry, cannot find the picture through your input path. Please try again."
            else:
                rob_response = "AI Doctor: The diagnosis shows that you are having " + r + '\n' + "AI Doctor: " + response(
                    r)
                sent_tokens.remove(r)
        else:
            if (greeting(user_response) != None):
                rob_response = "AI Doctor: " + greeting(user_response)
            else:
                rob_response = "AI Doctor: " + response(user_response)

                sent_tokens.remove(user_response)
    else:
        rob_response = "AI Doctor: Bye! Take care."
    return rob_response
