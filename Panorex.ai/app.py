from flask import Flask, request, render_template, send_file
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import *
from models import *

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Modeli ve diğer gerekli bilgileri yükleme
checkpoint = torch.load('detector2.pth')
img_size = checkpoint['img_size']
out_size = checkpoint['out_size']
out_c = checkpoint['out_c']
n_classes = checkpoint['n_classes']
roi_size = checkpoint['roi_size']
class_names = checkpoint['class_names']

detector = TwoStageDetector(img_size, out_size, out_c, n_classes, roi_size)
detector.load_state_dict(checkpoint['detector_state_dict'])
detector.eval()

# Resim ön işleme fonksiyonu
def preprocess_image(image):
    img_width = 320
    img_height = 240
    preprocess = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    return preprocess(image).unsqueeze(0)

def visualize_detection(image, detections, class_names, width_scale_factor=32, height_scale_factor=30, min_score=0.9, text_color='r', text_bg_color='none', save_path=None):
    # Resmi görsel olarak görüntüleme
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    ax = plt.gca()

    boxes, scores, labels = detections

    # Her bir tespiti çizme
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < min_score:
            continue  # Minimum doğruluk eşiği altındaki tespitleri atla

        x_min, y_min, x_max, y_max = box.tolist()

        # Koordinatları ölçeklendirme ve pozisyonları ayarlama
        x_min_scaled = x_min * width_scale_factor
        y_min_scaled = y_min * height_scale_factor
        x_max_scaled = x_max * width_scale_factor
        y_max_scaled = y_max * height_scale_factor

        width_scaled = x_max_scaled - x_min_scaled
        height_scaled = y_max_scaled - y_min_scaled

        # Kutuyu çizme
        rect = patches.Rectangle((x_min_scaled, y_min_scaled), width_scaled, height_scaled, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Etiketi ekleme
        class_name = class_names[label.item()]
        plt.text(x_min_scaled, y_min_scaled, f'{class_name}: {score:.2f}', color=text_color, backgroundcolor=text_bg_color)

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Eklediğimiz kapatma işlemi

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        image = Image.open(file.stream)
        input_image = preprocess_image(image)

        with torch.no_grad():
            detector.eval()  # Modeli değerlendirme moduna geçirme
            proposals, conf_scores, classes = detector.inference(input_image)
        
        visualize_detection(image,(proposals, conf_scores, classes), class_names, width_scale_factor=290, height_scale_factor=180, min_score=0.96, text_color='black', text_bg_color='none', save_path="static/detected_objects.png")

        return send_file("static/detected_objects.png", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
