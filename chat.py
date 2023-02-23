import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

with open('resources/intents.json', 'r') as f:
    intents = json.load(f)

FILE = "resources/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
# print("Let's chat! type 'quit' to exit")
def get_answer(res):
    sentence = res

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _,predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
                # print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        return "Sorry not able to answer this, you can try asking me something else"
        # print(f"{bot_name}: I am not trained to answer this once i am more capable i will give you the answer....")
