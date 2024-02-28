import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

model = tf.keras.models.load_model('Faceid.h5',custom_objects={'L1Dist':L1Dist})
def update(*args):
    ret, frame = capture.read()
    frame=frame[120:120+300,200:200+300,:]
    
    buf = cv2.flip(frame, 0).tostring()
    img_texture = img_texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    img_texture.blit_buffer(buf, colorfmt='bgr' , bufferfmt='ubyte')
    image=img_texture

def preprocess(file_path):
    byte_img=tf.io.read_file(file_path)
    img=tf.io.decode_jpeg(byte_img)
    img=tf.image.resize(img,(105,105))
    img=img/250
    return img

def verify(*args):
    detect_threshold=0.9
    veri_threshold=0.8
    
    SAVE_PATH=os.path.join('varification','input_image','input_image.jpg')
    ret, frame = capture.read()
    frame = frame[120:120+300, 200:200+300 :]
    cv2.imwrite(SAVE_PATH, frame)
    
    results=[]
    for images in os.listdir(os.path.join('verification','ver_img')):

        input_image=preprocess(os.path.join('verification','input_image','inp_img.jpg'))
        valid_image=preprocess(os.path.join('verification','ver_img',images))
        result=model.predict(list(np.expand_dims([input_image,valid_image],axis=1)))
        results.append(result)
    detection=np.sum(np.array(results)>detect_threshold)
    verification=detection/len(os.listdir(os.path.join('verification','ver_img')))
    verified= verification > veri_threshold   
    return results, verified

while(True):
    capture = cv2.VideoCapture(0)
    update()
    if cv2.waitKey(1) & 0xff==27:
        break



