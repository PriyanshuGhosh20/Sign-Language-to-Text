from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
from keras.models import model_from_json
import operator
import time
import sys
import matplotlib.pyplot as plt
import enchant
from string import ascii_uppercase

class Application:
    def __init__(self):
        self.enchant_dict = enchant.Dict("en_US")
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # Load models dynamically
        model_names = ['bw', 'bw_dru', 'bw_tkdi', 'bw_smn']
        self.loaded_models = {}
        for model_name in model_names:
            json_file_path = f'model/model-{model_name}.json'
            h5_file_path = f'model/model-{model_name}.h5'

            with open(json_file_path, 'r') as json_file:
                model_json = json_file.read()

            loaded_model = model_from_json(model_json)
            loaded_model.load_weights(h5_file_path, by_name=True)

            self.loaded_models[model_name] = loaded_model

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0

        print("Loaded models from disk")

        self.root = tk.Tk()
        self.root.title("Sign language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1920x1080")

        self.panel = tk.Label(self.root)
        self.panel.place(x=135, y=10, width=640, height=640)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=460, y=95, width=310, height=310)

        self.T = tk.Label(self.root)
        self.T.place(x=31, y=17)
        self.T.config(text="Sign Language to Text", font=("courier", 40, "bold"))

        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=640)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=640)
        self.T1.config(text="Character :", font=("Courier", 40, "bold"))

        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=700)

        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=700)
        self.T2.config(text="Word :", font=("Courier", 40, "bold"))

        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=760)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=760)
        self.T3.config(text="Sentence :", font=("Courier", 40, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=820)
        self.T4.config(text="Suggestions", fg="red", font=("Courier", 40, "bold"))

        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=890)

        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=890)

        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=890)

        self.bt4 = tk.Button(self.root, command=self.action4, height=0, width=0)
        self.bt4.place(x=125, y=950)

        self.bt5 = tk.Button(self.root, command=self.action5, height=0, width=0)
        self.bt5.place(x=425, y=950)

        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            self.predict(res)

        self.current_image2 = Image.fromarray(res)
        imgtk = ImageTk.PhotoImage(image=self.current_image2)
        self.panel2.imgtk = imgtk
        self.panel2.config(image=imgtk)
        self.panel3.config(text=self.current_symbol, font=("Courier", 50))
        self.panel4.config(text=self.word, font=("Courier", 40))
        self.panel5.config(text=self.str, font=("Courier", 40))

        # Check if self.word is not empty before getting suggestions
        if self.word:
            predicts = self.enchant_dict.suggest(self.word)
            self.update_suggestions(predicts)

        self.root.after(30, self.video_loop)

    def update_suggestions(self, predicts):
        buttons = [self.bt1, self.bt2, self.bt3, self.bt4, self.bt5]
        for i, button in enumerate(buttons):
            if i < len(predicts):
                button.config(text=predicts[i], font=("Courier", 20))
            else:
                button.config(text="")

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        result = self.loaded_models['bw'].predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_models['bw_dru'].predict(test_image.reshape(1, 128, 128, 1))
        result_tkdi = self.loaded_models['bw_tkdi'].predict(test_image.reshape(1, 128, 128, 1))
        result_smn = self.loaded_models['bw_smn'].predict(test_image.reshape(1, 128, 128, 1))

        prediction = {'blank': result[0][0]}
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        # LAYER 1
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        # LAYER 2
        if self.current_symbol in ['D', 'R', 'U']:
            prediction = {'D': result_dru[0][0], 'R': result_dru[0][1], 'U': result_dru[0][2]}
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['D', 'I', 'K', 'T']:
            prediction = {'D': result_tkdi[0][0], 'I': result_tkdi[0][1], 'K': result_tkdi[0][2], 'T': result_tkdi[0][3]}
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['M', 'N', 'S']:
            prediction1 = {'M': result_smn[0][0], 'N': result_smn[0][1], 'S': result_smn[0][2]}
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            if prediction1[0][0] == 'S':
                self.current_symbol = prediction1[0][0]
            else:
                self.current_symbol = prediction[0][0]

        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 60:
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def action1(self):
        predicts = self.enchant_dict.suggest(self.word)
        if len(predicts) > 0:
            self.word = ""
            self.str += " "
            self.str += predicts[0]
            self.update_suggestions(predicts[1:])

    def action2(self):
        predicts = self.enchant_dict.suggest(self.word)
        if len(predicts) > 1:
            self.word = ""
            self.str += " "
            self.str += predicts[1]
            self.update_suggestions(predicts[2:])

    def action3(self):
        predicts = self.enchant_dict.suggest(self.word)
        if len(predicts) > 2:
            self.word = ""
            self.str += " "
            self.str += predicts[2]
            self.update_suggestions(predicts[3:])

    def action4(self):
        predicts = self.enchant_dict.suggest(self.word)
        if len(predicts) > 3:
            self.word = ""
            self.str += " "
            self.str += predicts[3]
            self.update_suggestions(predicts[4:])

    def action5(self):
        predicts = self.enchant_dict.suggest(self.word)
        if len(predicts) > 4:
            self.word = ""
            self.str += " "
            self.str += predicts[4]
            self.update_suggestions(predicts[5:])

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")
pba = Application()
pba.root.mainloop()
