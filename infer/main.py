from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, request, render_template, json, send_from_directory
from PIL import Image
import os

app = Flask(__name__)
app.config["IMAGE"] = "./images/"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def transformer(input):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    return transform(input)

def inference(input):
    model = Net()
    model.load_state_dict(torch.load('../train/mnist_cnn.pt'))
    model.eval()
    output = model(input).squeeze().argmax().item()
    return output

def get_inference(file):
    res = {}
    if not file:
        res['status'] = 'image missing'
        return res
    res['status'] = 'success'
    image = Image.open(file.stream)
    output = inference(process(image))
    res['result'] = output
    return res

def process(input):
    input = transformer(input)
    return input.unsqueeze(0)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == "GET":
        return render_template("upload.html")
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config["IMAGE"], image.filename))
            status = get_inference(image)
            return render_template("upload.html", uploaded_image=image.filename, inf_status=status)

@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    return send_from_directory(app.config["IMAGE"], filename)

app.run(host='0.0.0.0', port=8000, debug=True)