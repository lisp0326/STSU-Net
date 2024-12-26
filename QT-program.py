import sys
import torch
import os
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QApplication
from model/STSU.STSU import UNetWithAttention
from Separated import split_image_by_color_main
from Combined import merge_images_by_group
from Marked import process_images_main


def predict_segmentation(model_path, image_path, output_dir, num_classes=3, image_size=(224, 224)):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

   
    model = UNetWithAttention(num_classes=num_classes).to(device)
    #model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    
    image = Image.open(image_path)
    image = image.resize(image_size)
    image_np = np.array(image)

    if len(image_np.shape) == 2:  
        image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:  
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    image_tensor = image_tensor.to(device)

   
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted = predicted.squeeze().cpu().numpy()

   
    predicted_image = Image.fromarray((predicted * 127).astype(np.uint8))  
    predicted_image_path = os.path.join(output_dir, "predicted_output.png")
    predicted_image.save(predicted_image_path)
    print(f"Predicted image saved to {predicted_image_path}")

    return predicted_image_path, []



class ImageSegmentationApp(QtWidgets.QWidget):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image segmentation tools')
        self.setGeometry(100, 100, 1000, 600)

        
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        button_layout = QVBoxLayout()

       
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.original_label)

       
        self.predicted_label = QLabel("Predicted Image")
        self.predicted_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.predicted_label)

       
        self.split_labels = []

        
        self.upload_button = QPushButton('Import images')
        self.upload_button.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_button)

        
        self.predict_button = QPushButton('Predictive segmentation')
        self.predict_button.clicked.connect(self.predict_image)
        self.predict_button.setEnabled(False)
        button_layout.addWidget(self.predict_button)

        
        self.split_button = QPushButton('Calculated length')
        self.split_button.clicked.connect(self.split_image)
        self.split_button.setEnabled(False)
        button_layout.addWidget(self.split_button)

        
        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        self.image_layout = image_layout

    def clear_results(self):
        
        self.original_label.clear()
        self.predicted_label.clear()

        for label in self.split_labels:
            self.image_layout.removeWidget(label)
            label.deleteLater()

        self.split_labels = []

    def upload_image(self):
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Opening an Image File", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            
            self.clear_results()

            self.image_path = file_path
            pixmap = QtGui.QPixmap(self.image_path)
            self.original_label.setPixmap(pixmap.scaled(256, 256, aspectRatioMode=1))
            self.predict_button.setEnabled(True)

    def predict_image(self):
        if hasattr(self, 'image_path'):
            output_dir = "output_parts"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            
            self.predicted_image_path, _ = predict_segmentation(self.model_path, self.image_path, output_dir)

            
            self.split_dir = os.path.join(output_dir, 'split_results')
            if not os.path.exists(self.split_dir):
                os.makedirs(self.split_dir)
            split_image_by_color_main(output_dir, self.split_dir)

            
            pixmap = QtGui.QPixmap(self.predicted_image_path)
            self.predicted_label.setPixmap(pixmap.scaled(256, 256, aspectRatioMode=1))
            self.split_button.setEnabled(True)

    def split_image(self):
        if hasattr(self, 'split_dir'):
            
            processed_dir = os.path.join(self.split_dir, 'processed_results')
            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)
            process_images_main(self.split_dir, processed_dir, resolution_factor=17)

            
            final_dir = os.path.join(processed_dir, 'final_results')
            if not os.path.exists(final_dir):
                os.makedirs(final_dir)
            merge_images_by_group(processed_dir, final_dir, blacken_height=1)

            
            final_images = [f for f in os.listdir(final_dir) if f.endswith('.png')]
            for img_name in final_images:
                img_path = os.path.join(final_dir, img_name)

                
                label = QLabel("Final result")
                label.setAlignment(Qt.AlignCenter)
                pixmap = QtGui.QPixmap(img_path)
                label.setPixmap(pixmap.scaled(256, 256, aspectRatioMode=1))
                self.image_layout.addWidget(label)
                self.split_labels.append(label)



def main():
    app = QApplication(sys.argv)
    model_path = "STSU-Net.pth"
    window = ImageSegmentationApp(model_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
