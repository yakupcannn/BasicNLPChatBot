import json
from nltk_utils import tokenize,stem,bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import ChatBotNeuralNet

class ChatBotDataset(Dataset):
  def __init__(self):
    self.X_data = X_train
    self.y_data = y_train
    self.n_samples = len(X_train)

  def __getitem__(self, index):
    return self.X_data[index],self.y_data[index]
  
  def __len__(self):
    return self.n_samples


with open("intents.json","r") as f:
  intents = json.load(f)

tokenized_words = []
tags = []
dataset = []
ignore_chars = ["?","!","."]
for intent in intents["intents"]:
  tag = intent["tag"]
  tags.append(tag)
  for pattern in intent["patterns"]:
    w = tokenize(pattern)
    tokenized_words.extend(w)
    dataset.append((w,tag))

tokenized_words = [stem(_w) for _w in tokenized_words if _w not in ignore_chars]
tokenized_words = sorted(set(tokenized_words))
tags = sorted(set(tags))

print(len(dataset), "patterns")
print(len(tags), "tags:", tags)
print(len(tokenized_words), "unique stemmed words:", tokenized_words)


X_train,y_train = [],[]
for (p_sentence,tag) in dataset:
  bag = bag_of_words(p_sentence,tokenized_words)
  X_train.append(bag)
  y_train.append(tags.index(tag))

X_train =np.array(X_train)
y_train = np.array(y_train)

print(len(X_train))
print(len(y_train))
print(len(X_train[0]))
##HyperParameters
BATCH_SIZE = 8
LEARNING_RATE = 0.001
INPUT_SIZE = len(X_train[0])
HIDDEN_SIZE = 8
OUT_SIZE = len(tags)
N_EPOCHS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(INPUT_SIZE, OUT_SIZE)



chatbot_dataset = ChatBotDataset()
train_loader = DataLoader(dataset=chatbot_dataset,batch_size=BATCH_SIZE,shuffle=True)


model = ChatBotNeuralNet(n_input=INPUT_SIZE,n_hidden=HIDDEN_SIZE,n_classes=OUT_SIZE).to(DEVICE)

##Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lr=LEARNING_RATE,params=model.parameters())

## Model Training
losses = 0 
for epoch in range(N_EPOCHS):
  for (words,labels) in train_loader:
    words = words.to(DEVICE)
    labels = labels.to(DEVICE)

    ##Forward Pass
    pred_outs = model(words)
    loss = criterion(pred_outs,labels)

    ##Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses +=loss.item()

    if (epoch + 1) % 100 == 0:
      print(f"Epoch:{epoch+1}/{N_EPOCHS} | Loss:{loss.item():.4f}")

print(f"Final Loss:{loss.item():.4f}")
print(f"Average Loss:{losses/N_EPOCHS}")

report = {
  "model_state" : model.state_dict(),
  "input_size" : INPUT_SIZE,
  "hidden_size" : HIDDEN_SIZE,
  "output_size" : OUT_SIZE,
  "tokenized_words":tokenized_words,
  "tags":tags
}

FILE = "report.pth"
torch.save(report,FILE)

print(f"TRAINING IS COMPLETED AND MODEL IS SAVED TO {FILE}")


    





  

