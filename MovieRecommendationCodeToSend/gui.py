# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:59:01 2021

@author: Lohith
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:13:45 2021

@author: Lohith
"""

import tkinter as tk

from tkinter import *

import knnmovierecommend as kmrecommend


    


window = tk.Tk()

window.title('MovieRecommendationAPP')

window.configure(background = 'green')

window.geometry('1250x700')

window.grid_rowconfigure(0, weight = 1)

window.grid_columnconfigure(0, weight = 1)


lb = tk.Label(window, text = ' MOVIE RECOMMENDATION APPLICATION', bg = 'black', fg = 'white', font = ('times', 30, 'italic bold underline'))
lb.place(x = 350, y = 20)


# age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal


mn = tk.Label(window,text = "Enter the movie Name : ", bg = 'blue', fg ='white')

mn.place(x =100, y = 100)




mname = StringVar()
mnEntry = tk.Entry(window, textvariable = mname)
mnEntry.place(x = 300, y = 100)




lbout = tk.Label(window,text = "Recommended Movies are: ", bg = 'blue', fg ='white')

lbout.place(x =100, y = 200)



recmov = StringVar()
outEntry = tk.Entry(window, textvariable = recmov)
outEntry.place(height=100, width=800)
outEntry.place(x = 300, y = 200)



def pred():
    print('called the logic!!!!!!!!!!')  
    x1 = mname.get()
    print(x1)
    recmovie = kmrecommend.get_movie_recommendation(x1)

    print(recmovie)
    recmov.set(recmovie)
    



but = tk.Button(window, text = 'Recommend', command = pred, bg = 'red', fg = 'white', width = 20, height =1)
but.place(x = 500, y = 100)

window.mainloop()