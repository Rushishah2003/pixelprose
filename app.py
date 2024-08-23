from model import processor,model
from PIL import Image
import requests
from flask import Flask,jsonify,render_template,request
app = Flask(__name__)
@app.route('/', methods=['GET'])
def index():
  return render_template('image_caption.html')

@app.route('/submit',methods=['POST'])
def name():
  url=request.form.get('url')
  print(url)
  raw_image= Image.open(requests.get(url, stream=True).raw).convert('RGB')
  text = None
  inputs = processor(raw_image, text, return_tensors="pt")
  out = model.generate(**inputs,max_new_tokens=50)
  caption = processor.decode(out[0], skip_special_tokens=True)
  return render_template('image_caption.html', caption=caption)

if __name__ == '__main__':
  app.run(host='0.0.0.0',port=5000,debug=True)

