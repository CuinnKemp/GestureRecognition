from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import pygame
import pyautogui

pygame.font.init()
pygame.mixer.init()

device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WIDTH, HEIGHT = pyautogui.size()
WIDTH, HEIGHT = WIDTH//(10/9), HEIGHT//(10/9)
WIN = pygame.display.set_mode((WIDTH, HEIGHT)) 
pygame.display.set_caption("Gesture AI") 
FPS = 50
Red = (255,0,0)
FONT = pygame.font.SysFont('avenir', 100)

model = torch.load(os.path.join('data/model'), torch.device('cpu'))
model.eval()

data_transforms = {
    'run': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])

                  for x in ['run']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=False, num_workers=0)
              for x in ['run']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['run']}
class_names = image_datasets['run'].classes

print(device)

model.to(device)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, num_images=6):
	was_training = model.training
	model.eval()
	images_so_far = 0
	fig = plt.figure()

	with torch.no_grad():
	    for i, (inputs, labels) in enumerate(dataloaders['run']):
	        inputs = inputs.to(device)
	        labels = labels.to(device)

	        outputs = model(inputs)
	        _, preds = torch.max(outputs, 1)

	        # for j in range(inputs.size()[0]):
	        images_so_far += 1
	        ax = plt.subplot(num_images//2, 2, images_so_far)
	        ax.axis('off')
	        ax.set_title(f'predicted: {class_names[preds[0]]}')

	        
	        print(f'predicted: {class_names[preds[0]]}')
	        # imshow((inputs.cpu().data[0]))

	        if images_so_far == num_images:
	            model.train(mode=was_training)

	    model.train(mode=was_training)

	return class_names[preds[0]]

vid = cv2.VideoCapture(0)

# data_dir = 'data/run/YES/'
data_dir2 = 'data/run/NO/'
# directory = os.path.join(data_dir)
directory2 = os.path.join(data_dir2)

framecount = 1
counter = 0

def main():
	clock = pygame.time.Clock()
	counter = 0
	output = 'Looking...'
	run = True
	while run:
		clock.tick(FPS)
		for event in pygame.event.get(): #If the game is quit, close the window
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()

		ret, frame = vid.read()
		frame = cv2.resize(frame,(50,50))
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		cv2.imwrite((directory2 + 'image' + '.png'), frame)
		# print(framecount)
		
		# framecount += 1

		print(output)
		if counter == 10:
			output = visualize_model(model)
			counter = 0

		ret2, frame2 = vid.read()
		frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

		variable = pygame.transform.rotate(pygame.pixelcopy.make_surface(frame2), 270)
		variable2 = pygame.transform.scale(variable, (WIDTH, HEIGHT))

		WIN.blit(variable2, (0,0))

		draw_text = FONT.render(f'predicted: {output}', 1, (0,255,0))
		WIN.blit(draw_text, (WIDTH//2 - draw_text.get_width()//2, HEIGHT//2 - draw_text.get_height()//2))

		pygame.display.update()
		counter += 1

	
		# time.sleep(1)

main()

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

