from tkinter import Tk
import tkinter as tk
import random
from tkinter import Text,Entry,Button,END,Scrollbar
from chatbot import chatbot_response

                    
                   


root = Tk() 
root.title("Customer Asistant")


# Create the chatbot's text area
text_area = Text(root, bd=0, bg="#17202A", font="Helvica 14", wrap=tk.WORD,width=50, height=20,fg="#EAECEE")
text_area.pack()

text_area.config(state=tk.DISABLED)

# Create the user's input field
input_field = Entry(root, width=20,bg="blue",fg="#FFF")
input_field.pack()

# Create the send button
send_button = Button(root, text="Send", command=lambda: send_message())
quit_button = Button(root, text="Quit", command=lambda: t_quit())
send_button.pack()
quit_button.pack()


def t_quit():
    global root
    root.quit()
    exit(0)

def send_message():
  # Get the user's input
  user_input = input_field.get()

  # Clear the input field
  input_field.delete(0, END)

  # Generate a response from the chatbot
  _,response = chatbot_response(user_input)
  text_area.config(state=tk.NORMAL)
  # Display the response in the chatbot's text area
  text_area.insert(END, f"User: {user_input}\n")
  text_area.insert(END, f"Chatbot: {response}\n")

  text_area.config(state=tk.DISABLED)

root.mainloop()