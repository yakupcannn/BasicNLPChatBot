import random
import json
import torch
from model import ChatBotNeuralNet
from nltk_utils import bag_of_words,tokenize,stem

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("intents.json","r") as f:
    intents = json.load(f)

FILE ="report.pth"
data = torch.load(FILE)
N_INPUT =data["input_size"]
N_HIDDEN = data["hidden_size"]
N_OUTPUT = data["output_size"]
TOKEN_DICT =data["tokenized_words"]
TAGS = data["tags"]
MODEL_STATE = data["model_state"]
model = ChatBotNeuralNet(N_INPUT,N_HIDDEN,N_OUTPUT).to(DEVICE)
model.load_state_dict(MODEL_STATE)
model.eval()

def chatbot_response(user_input):
    t_sentence = tokenize(user_input)
   #t_sentence = [stem(_w) for _w in t_sentence if _w not in ignore_chars]
    X = bag_of_words(t_sentence,TOKEN_DICT)
    X = X.reshape((1,X.shape[0]))
    X = torch.from_numpy(X)
    output = model(X)
    _,predicted = torch.max(output,dim=-1)
    pred_tag = TAGS[predicted.item()]

    probs = torch.softmax(output,dim=-1)
    th_prob = probs[0][predicted.item()]
    
    if th_prob.item()>0.75:
        for intent in intents["intents"]:
            if pred_tag == intent["tag"]:
                return pred_tag,random.choice(intent['responses'])

    else:
        return pred_tag,"I'm sorry.I don't understand you"
if __name__ =="__main__":                
    ignore_chars = ["?","!","."]
    bot_name = "ChatBot"
    print("I'm ready to talk with you")

    while True:
        sentence = input("You: ")
        print(f"YOU:{sentence}")
        tag,response = chatbot_response(sentence)
        print(f"{bot_name}: {response}")
        if tag =="goodbye":
            break


     
                   
  





