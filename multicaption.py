import subprocess

subprocess.run(["git", "clone", "https://github.com/openai/CLIP.git"])
subprocess.run(["pip", "install", "CLIP"])
subprocess.run(["pip", "install", "ftfy"])
subprocess.run(["pip", "install", "eagleview==1.1"])

import eagleview
from eagleview.figshow import ImageMatrix
import torch.nn as nn
import torch
import numpy as np
import skimage.io as io
import PIL.Image
from CLIP.clip import load
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import trange
import gdown
import ftfy
import os
import csv

class MLP(nn.Module):
    def forward(self, x):
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class ClipCaptionModel(nn.Module):

    def forward(self, tokens, prefix, mask= None, labels= None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length, prefix_size= 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )



def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

class Downloader(object):
    def __init__(self):
        pass
    def download_file(self, file_id, file_dst):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output=file_dst)

downloader = Downloader()
device = "cuda" if torch.cuda.is_available() else "cpu"
current_directory = os.getcwd()
save_path = os.path.join(os.path.dirname(current_directory), "Caption-Craft/pretrained_models")
os.makedirs(save_path, exist_ok=True)
model_path = os.path.join(save_path, 'model_weights.pkl')

if not os.path.exists(model_path):
    downloader.download_file("14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT", model_path)

clip_model, preprocess = load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
prefix_length = 10
model = ClipCaptionModel(prefix_length)

model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
model = model.eval()
model = model.to(device)

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

#code for .csv generation
def process_images_in_folder(folder_path, model):
    output_csv_file = "image_captions.csv"
    with open(output_csv_file, 'w', newline='') as csvfile:
        fieldnames = ['Sl_No', 'Image_name', 'Image_caption']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        sl_no = 1
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                caption = model_predict(image_path, model, path=True)
                filename_without_extension = os.path.splitext(filename)[0]

                writer.writerow({'Sl_No': sl_no, 'Image_name': filename_without_extension, 'Image_caption': caption})
                sl_no += 1

#matrix image generation
folder_path = "/content/Caption"
process_images_in_folder(folder_path, model)


im = ImageMatrix('content/Caption', '/content/image_captions.csv')
im.rand((3, 2)).display_image(
    check_col='Image_name',
    display_label=True,
    display_cols=['Sl_No', 'Image_caption'],
    display_name=True,
    print_all=False,
    x=200,
    y=3900,
    fig_size=(20, 16),
    fontsize=10,
    text_color='white',
    label_background_color='black'
)
