import gdown
from utils import Downloader
from mapping import MLP
from model import *
from generate import generate_beam
subprocess.run(["git", "clone", "https://github.com/openai/CLIP.git"])
subprocess.run(["pip", "install", "CLIP"])

from CLIP.clip import *
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
downloader = Downloader()

device = "cuda" if torch.cuda.is_available() else "cpu"

current_directory = os.getcwd()
save_path = os.path.join(os.path.dirname(current_directory), "Caption-Craft/pretrained_models")
os.makedirs(save_path, exist_ok=True)
model_path = os.path.join(save_path, 'model_weights.pkl')

#downloader.download_file("14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT", model_path)

clip_model, preprocess = load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prefix_length = 10

model = ClipCaptionModel(prefix_length)

#model.load_state_dict(torch.load(model_path, map_location="cpu"),strict=False)

model = model.eval()
model = model.to(device)

app = Flask(__name__)

def model_predict(img, model, path=False):
    if path:
        im = io.imread(img)
        img = PIL.Image.fromarray(im)
    pil_image = img
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    return generated_text_prefix


@app.route('/', methods = ['GET'])
def index(): 
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        f = request.files['file']

        if f.filename == '':
            return "No selected file"

        # Process the file without saving it
        image = PIL.Image.open(f.stream)
        preds = model_predict(image, model, path=False)
        return preds
    return None

if name == 'main':
    app.run(debug=True)
app = Flask(_name_)

def model_predict(img, model, path=False):
    if path:
        im = io.imread(img)
        img = PIL.Image.fromarray(im)
    pil_image = img
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    return generated_text_prefix
