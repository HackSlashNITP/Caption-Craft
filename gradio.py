from PIL import Image
def pre(image_path):
  image = Image.open(image_path)
  return image

def predict(inp):
  image = pre(inp)
  image = preprocess(image).unsqueeze(0).to(device)
  with torch.no_grad():
      prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
      prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
      generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
  return generated_text_prefix

import gradio as gr
iface = gr.Interface(fn=predict, inputs=gr.Image(type="filepath"), outputs=gr.Text())
iface.launch()

