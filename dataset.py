import torch
import time
import math
import os
import shutil
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import cv2
import numpy as np

class_to_idx = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9, 'diningtable':10,
                'dog':11, 'horse':12, 'motorbike':13, 'person':14, 'pottedplant':15,
                'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19
                }
idx_to_class = {i:c for c, i in class_to_idx.items()}

def get_pascal_voc2007_data(image_root, split='train'):
  check_file = os.path.join(image_root, "extracted.txt")
  download = not os.path.exists(check_file)

  train_dataset = datasets.VOCDetection(image_root, year='2007', image_set=split,
                                    download=download)

  open(check_file, 'a').close()
  
  return train_dataset

def voc_collate_fn(batch_lst, reshape_size=224):
    preprocess = transforms.Compose([transforms.Resize((reshape_size, reshape_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])])
    batch_size = len(batch_lst)
    img_batch = torch.zeros(batch_size, 3, reshape_size, reshape_size)
    max_num_box = max(len(batch_lst[i][1]['annotation']['object']) for i in range(batch_size))
    box_batch = torch.Tensor(batch_size, max_num_box, 5).fill_(-1.)
    w_list = []
    h_list = []
    img_id_list = []
    
    for i in range(batch_size):
        img, ann = batch_lst[i]
        w_list.append(img.size[0])
        h_list.append(img.size[1])
        img_id_list.append(ann['annotation']['filename'])
        img_batch[i] = preprocess(img)
        all_bbox = ann['annotation']['object']
        if type(all_bbox) == dict:
            all_bbox = [all_bbox]
        for bbox_idx, one_bbox in enumerate(all_bbox):
            bbox = one_bbox['bndbox']
            obj_cls = one_bbox['name']
            box_batch[i][bbox_idx] = torch.Tensor([float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax']), 
                                                   class_to_idx[obj_cls]])
            
    h_batch = torch.tensor(h_list)
    w_batch = torch.tensor(w_list)
    
    return img_batch, box_batch, w_batch, h_batch, img_id_list
    
def pascal_voc2007_loader(dataset, batch_size, num_workers=0):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                              num_workers=num_workers, collate_fn=voc_collate_fn)
    
    return train_loader

def data_visualizer(img, idx_to_class, bbox = None, pred = None):
  img_copy = np.array(img).astype('uint8')

  if bbox is not None:
    for bbox_idx in range(bbox.shape[0]):
      one_bbox = bbox[bbox_idx][:4]
      cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2], one_bbox[3]), (255, 0, 0), 2)

      if bbox.shape[1] > 4:
        obj_cls = idx_to_class[bbox[bbox_idx][4].item()]
        
        cv2.putText(img_copy, '%s' % (obj_cls),
                  (one_bbox[0], one_bbox[1]+15),
                  cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

  if pred is not None:
    for bbox_idx in range(pred.shape[0]):
      one_bbox = pred[bbox_idx][:4]
      cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                  one_bbox[3]), (0, 255, 0), 2)
      
      if pred.shape[1] > 4: # if class and conf score info provided
        obj_cls = idx_to_class[pred[bbox_idx][4].item()]
        conf_score = pred[bbox_idx][5].item()
        cv2.putText(img_copy, '%s, %.2f' % (obj_cls, conf_score),
                    (one_bbox[0], one_bbox[1]+15),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

  plt.imshow(img_copy)
  plt.axis('off')
  plt.show()

def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with elements in the range [0, 1]

    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr


def visualize_dataset(X_data, y_data, samples_per_class, class_list):
    """
    Make a grid-shape image to plot

    Inputs:
    - X_data: set of [batch, 3, width, height] data
    - y_data: paired label of X_data in [batch] shape
    - samples_per_class: number of samples want to present
    - class_list: list of class names; eg,
      ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    Outputs:
    - An grid-image that visualize samples_per_class number of samples per class
    """
    img_half_width = X_data.shape[2] // 2
    samples = []
    for y, cls in enumerate(class_list):
        tx = -4
        ty = (img_half_width * 2 + 2) * y + (img_half_width + 2)
        plt.text(tx, ty, cls, ha="right")
        idxs = (y_data == y).nonzero().view(-1)
        for i in range(samples_per_class):
            idx = idxs[random.randrange(idxs.shape[0])].item()
            samples.append(X_data[idx])

    img = make_grid(samples, nrow=samples_per_class)
    return tensor_to_image(img)

def detection_visualizer(img, idx_to_class, bbox=None, pred=None):
    """
    Data visualizer on the original image. Support both GT box input and proposal input.
    
    Input:
    - img: PIL Image input
    - idx_to_class: Mapping from the index (0-19) to the class name
    - bbox: GT bbox (in red, optional), a tensor of shape Nx5, where N is
            the number of GT boxes, 5 indicates (x_tl, y_tl, x_br, y_br, class)
    - pred: Predicted bbox (in green, optional), a tensor of shape N'x6, where
            N' is the number of predicted boxes, 6 indicates
            (x_tl, y_tl, x_br, y_br, class, object confidence score)
    """

    img_copy = np.array(img).astype('uint8')

    if bbox is not None:
        for bbox_idx in range(bbox.shape[0]):
            one_bbox = bbox[bbox_idx][:4]
            cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                        one_bbox[3]), (255, 0, 0), 2)
            if bbox.shape[1] > 4: # if class info provided
                obj_cls = idx_to_class[bbox[bbox_idx][4].item()]
                print(obj_cls)
                cv2.putText(img_copy, '%s' % (obj_cls),
                          (one_bbox[0], one_bbox[1]+15),
                          cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    if pred is not None:
        for bbox_idx in range(pred.shape[0]):
            one_bbox = pred[bbox_idx][:4]
            cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                        one_bbox[3]), (0, 255, 0), 2)
            
            if pred.shape[1] > 4: # if class and conf score info provided
                obj_cls = idx_to_class[pred[bbox_idx][4].item()]
                conf_score = pred[bbox_idx][5].item()
                cv2.putText(img_copy, '%s, %.2f' % (obj_cls, conf_score),
                            (one_bbox[0], one_bbox[1]+15),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()
