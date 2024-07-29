from tkinter import *
from tkinter import ttk
from tkinter import filedialog

import Crop_pred as pred
 
 
 
class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Crop Yield Predition")
        self.minsize(640, 400)
        self.configure(background='gray')
       # self.wm_iconbitmap('icon.ico')
 
        self.labelFrame = ttk.LabelFrame(self, text = "Open File")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)
        
        
        
        self.button()
        
        self.predict()
 
 
 
    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse A File",command = self.fileDialog)
        self.button.grid(column = 1, row = 1)
        
        
    def predict(self):
        self.pred = ttk.Button(self.labelFrame, text = "Crop Yield Prediction",command = self.prediction)
        self.pred.grid(column = 1, row = 3)
 
 
    def fileDialog(self):
 
        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("csv files","*.csv"),("all files","*.*")))
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)
        
    def prediction(self):
        print("Crop Yield Prediction")
        pred.crop()
 
 
 
 
 
root = Root()
root.mainloop()