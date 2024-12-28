import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from model/STSU.STSU import UNetWithAttention
from dataset.Unet_plus_dataset import MyDataset
import tqdm
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 3 

net = UNetWithAttention(num_classes=num_classes).to(device)


weight_path = 'STSU.pth'
net.load_state_dict(torch.load(weight_path))  
net.eval() 


input_image_folder = 'dataset' 
result_dir = 'predict'  
os.makedirs(result_dir, exist_ok=True)


dataset = MyDataset(input_image_folder, input_image_folder, num_classes=num_classes)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  


with torch.no_grad():  
    for idx, (image, _) in enumerate(tqdm.tqdm(data_loader)): 
        image = image.to(device) 
        out_image = net(image)  
        preds = torch.argmax(out_image, dim=1).cpu().numpy().squeeze() 

        
        image_name = os.path.basename(dataset.images_fps[idx]).split('.')[0] 

        
        predicted_img = Image.fromarray((preds * 127).astype(np.uint8))

        
        output_path = os.path.join(result_dir, image_name + '_pred.png')
        predicted_img.save(output_path)
        print(f"Prediction saved to {output_path}")  

print('Inference completed, grayscale results saved.') 
