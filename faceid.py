#import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.loader import Logger
import cv2
import tensorflow as tf
import os
import numpy as np

#build app and layout
class CamApp(App):
    def build(self):
        #main layout components
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="verify", on_press=self.verify, size_hint=(1, .1))
        self.vertification_label = Label(text="verification uninitiated", size_hint=(1, .1))

        #add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.vertification_label)

        #load kears model
        self.model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist': L1Dist})

        #setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout
    
    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        #flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    #load image from fike and convert to 100x100px 
    def preprocess(self, file_path):
        #read in image from file path
        byte_img = tf.io.read_file(file_path)
        #load in the image
        img = tf.io.decode_jpeg(byte_img)
        #preprocessing steps - resizing the image
        img = tf.image.resize(img,(100,100))
        #scale image to be between 0 and 1
        img = img / 255.0
        return img
    
    #vertification function to verify person
    def verify(self, *args):
        #specify thresholds
        detection_threshold = 0.99
        verification_threshold = 0.8

        #capture input image from the webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret , frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        #build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'vertificaton_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'vertificaton_images', image))
           
            #make predictions
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        #detection threshold: metric above which a prediction which is considered positive
        detection = np.sum(np.array(results) > detection_threshold)

        #vertification threshold: percentage of positive predictions needed for verification
        verification = detection / len(os.listdir(os.path.join('application_data', 'vertificaton_images')))
        verified = verification > verification_threshold

        #set vertification test
        self.vertification_label.text = 'verified' if verified == True else 'unverified'

        #log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
         

        return results, verified

if __name__ == '__main__':
    CamApp().run()