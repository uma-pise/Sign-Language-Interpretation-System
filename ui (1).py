import operator
import shutil
import threading
from collections import OrderedDict
from queue import Queue
from tkinter import filedialog
import pathlib
from tkinter import *
import os
import cv2
from os import path

import collections
from PIL import ImageTk
import numpy as np

from collections import OrderedDict


from keras.models import Sequential
from keras.models import model_from_json
from sklearn.metrics import classification_report

print(cv2.__version__)

master = Tk()
master.title('Hand Gesture Recognition in Video Frame Sequences')
master.minsize(width=500, height=500)


cnt=0
action=""
def browse_button():
    master.sourceFile = filedialog.askopenfilename(parent=master, initialdir="/", title='Please select a directory')
    print(master.sourceFile)
    global su
    su = master.sourceFile
    e1 = Label(master, text=master.sourceFile)
    e1.grid(row=0, column=1)


def extract_button():
    import cv2
    print(cv2.__version__)

    import os
    arr = os.listdir('D:\\Priyanka\\Btech\\Project\\NewProgram')
    count = 0
    # os.mkdir("ILoveYou")
    for i in range(0, 1):

        vidcap = cv2.VideoCapture(su)
        # success,image = vidcap.read()

        success = True

        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        sec = 0
        frameRate = 0.25  # Change this number to 1 for each 1 second

        success, image = vidcap.read()
        # count = 0
        while success:
            # convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = cv2.resize(image, (350, 650))
            y = 300
            h = 350
            x = 50
            w = 300
            image = image[y:y + h, x:x + w]
            cv2.imwrite("Data/Test/test_pr\\frame%d.jpg" % count, image)  # save frame as JPEG file
            sec = sec + frameRate
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1

    Label(master, text='Number of Frames Extracted : ').grid(row=6)
    Label(master, text=count).grid(row=6, column=1)


def most_frequent(List):
        return max(set(List), key=List.count)

def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

model_file = open('Data/Model/model.json', 'r')
model = model_file.read()
model_file.close()
model = model_from_json(model)
    # Getting weights
model.load_weights("Data/Model/weights.h5")
#Y = predict(model, X)
import numpy as np

labels = ['test_pr']##, 'GoodEve','Hi','HowAreU','ILoveYou','Sorry',]
img_size = 224
import os
import cv2
def predict_button():
    global action
    # os.system('onlypredict.py')

    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    # Y = predict(model, X)
    import numpy as np

    labels = ['test_pr']  ##, 'GoodEve','Hi','HowAreU','ILoveYou','Sorry',]
    img_size = 224
    import os
    import cv2

    val = get_data('Data/Test')
    x_val = []
    y_val = []
    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)

    x_val = np.array(x_val) / 255
    x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)

    predictions = model.predict(x_val)
    # predictions = predictions.reshape(1,-1)[0]
    print(predictions)
#print(classification_report(y_val, predictions, target_names = ['ExcuseMe (Class 0)','Hi (Class 1)','Sorry(Class 2)']))
    Y = np.argmax(predictions, axis=1)


    print(Y)
    V=Y.tolist()
    #List = [2, 1, 2, 2, 1, 3]
    print(most_frequent(V))
    num=most_frequent(V)

    if num== 0:
        action="Excuse Me"
    else:
        if num==1 :
            action="Good Evening"
        else:
            if num==2:
                action="Hi"

            else:
                if num == 3:
                    action = "How are you"

                else:
                     if num == 4:
                        action = "I love you"
                     else:
                         if num==5:
                             action="Sorry"

def videoToSpeech_button():
    # Import the required module for text
    # to speech conversion
    import gtts
    from gtts import gTTS

    # This module is imported so that we can
    # play the converted audio
    import os

    # The text that you want to convert to audio
    mytext = action

    # Language in which you want to convert
    language = 'en'

    # Passing the text and language to the engine,
    # here we have marked slow=False. Which tells
    # the module that the converted audio should
    # have a high speed
    myobj = gTTS(text=mytext, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome
    myobj.save("welcome.mp3")

    # Playing the converted file
    os.system("start welcome.mp3")
    from playsound import playsound

    tts = gtts.gTTS("Hello world")
    from io import BytesIO

    mp3_fp = BytesIO()

    tts = gTTS('hello', lang='en')

    tts.write_to_fp(mp3_fp)

    # playsound("hello.mp3")


def speechToVideo_button():
    #Convert speech to text

    # Python program to translate
    # speech to text and text to speech
    import os
    
    import speech_recognition as sr
    import pyttsx3
    
    # Initialize the recognizer
    r = sr.Recognizer()
    count=0
    words=[]
    # Function to convert text to
    # speech
    def SpeakText(command):
        # Initialize the engine
        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()
    
    
    # Loop infinitely for user to
    # speak
    
    #while (1):
    
        # Exception handling to handle
        # exceptions at the runtime
    try:
    
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                #while(1):
    
                    # wait for a second to let the recognizer
                    # adjust the energy threshold based on
                    # the surrounding noise level
                    r.adjust_for_ambient_noise(source2, duration=0.2)
    
                    # listens for the user's input
                    print("speak")
                    audio2 = r.listen(source2)
    
                    # Using google to recognize audio
                    MyText = r.recognize_google(audio2)
                    MyText = MyText.lower()
                    MyText=MyText.replace(" ", "")
    
                    words.append(MyText)
                    count=count+1
    
                    print("Did you say " + MyText)
    
                    #if MyText =="bye":
                          # break
    
    except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
    
    except sr.UnknownValueError:
            print("unknown error occured")
    
    print(words)
    video=""
    for i in words :
        f=open("Data/indexfile.txt")
        for line in f :
            #print(line)
    
            if line.strip().casefold()==i.casefold() :
                print(line,i)
    
                video=video+","+line.strip()
    
    
    
    arr=video.split(",")
    for j in arr :
        print(j)
        if j=='' :
            continue
        else :
            os.system("Gestures\\"+j+".mp4")

def play_video():
    import os
    os.system(su)

b1 = Button(master, text='Browse', command=browse_button, width=25).grid(row=1, column=1, pady=10)
b2 = Button(master, text='Extract Frames', width=25, command=extract_button).grid(row=4, column=0, padx=10, pady=10)
b3=Button(master, text='Predict', width=25, command=predict_button).grid(row=8, column=0, padx=10, pady=10)
b4 = Button(master, text='Convert to speech', width=25, command=videoToSpeech_button).grid(row=13, column=0,padx=10, pady=10)
b5 = Button(master, text='Convert to video', width=25, command=speechToVideo_button).grid(row=16, column=0, padx=10, pady=10)
b6 = Button(master, text='Play Video', width=25, command=play_video).grid(row=1, column=2, pady=10)
mainloop()
