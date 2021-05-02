import io
import torch
from torchvision import transforms
from torchvision.transforms.transforms import Resize
import models
from PIL import Image
from flask import Flask, render_template, request

app = Flask (__name__)
 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_mnist', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['img']
        img_bytes = img.read()
        result = mnist_predict(img_bytes)
        return str(result)
    
def mnist_predict(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    input_data = mnist_transform(img)
    input_data = input_data.unsqueeze(0)
    output = LeNet5_MNIST_model(input_data)
    result = int(torch.argmax(output))
    return result

if __name__=='__main__':

    ## set transform
    mnist_transform = transforms.Compose([ transforms.ToTensor(), transforms.Resize((28,28)), transforms.Normalize((0.5,), (0.5,)) ])

    ## load model
    LeNet5_MNIST_model = models.LeNet5()
    LeNet5_MNIST_model.load_state_dict(torch.load('./saved_models/LeNet5.MNIST.pth.tar',map_location=torch.device('cpu' if torch.cuda.is_available else 'cuda'))['model_state_dict'])
    LeNet5_MNIST_model.eval()

    ## run server
    app.run(debug=True)