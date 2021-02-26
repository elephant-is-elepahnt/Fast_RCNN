import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(__file__))
import RPNSolver


def GenerateAnchor(anc, grid):
    B, H, W, _ = grid.shape
    A, _ = anc.shape
    
    anchors = torch.zeros((B, A, H, W, 4), device = grid.device, dtype = grid.dtype)
    for i in range(A):
        anchors[:, i, :, :, 0] = grid[:, :, :, 0] - anc[i, 0]/2
        anchors[:, i, :, :, 1] = grid[:, :, :, 1] - anc[i, 1]/2        
        anchors[:, i, :, :, 2] = grid[:, :, :, 0] + anc[i, 0]/2
        anchors[:, i, :, :, 3] = grid[:, :, :, 1] + anc[i, 1]/2
        
    return anchors

def GenerateGrid(batch_size, w_amap = 7, h_amap = 7, dtype = torch.float32, device = 'cuda'):
    w_range = torch.arange(0, w_amap, dtype = dtype, device = device) + 0.5
    h_range = torch.arange(0, h_amap, dtype = dtype, device = device) + 0.5
    
    w_grid_idx = w_range.unsqueeze(0).repeat(h_amap, 1)
    h_grid_idx = h_range.unsqueeze(1).repeat(1, w_amap)
    grid = torch.stack([w_grid_idx, h_grid_idx], dim = -1)
    #만든 gird를 batchsize만큼 늘려준다.
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    return grid

def GenerateProposal(anchors, offsets):
    """
    anchors : tx, ty, bx, by
    offsets : dx, dy, dw, dh
    """
    B, A, H, W, _ = anchors.shape
    b_proposals = torch.zeros(((B, A, H, W, 4)), dtype = anchors.dtype, device = anchors.device)
    proposals = torch.zeros_like(b_proposals, dtype = anchors.dtype, device = anchors.device)
    
    height = anchors[:, :, :, :, 3] - anchors[:, :, :, :, 1]
    width = anchors[:, :, :, :, 2] - anchors[:, :, :, :, 0]
    ctr_x = anchors[:, :, :, :, 0] + width * 0.5
    ctr_y = anchors[:, :, :, :, 1] + height * 0.5
    
    b_proposals[:, :, :, :, 0] = ctr_x + width*offsets[:, :, :, :, 0]
    b_proposals[:, :, :, :, 1] = ctr_y + width*offsets[:, :, :, :, 1]
    b_proposals[:, :, :, :, 2] = width * torch.exp(offsets[:, :, :, :, 2])
    b_proposals[:, :, :, :, 3] = height * torch.exp(offsets[:, :, :, :, 3])
    
    new_tx = b_proposals[:, :, :, :, 0] - b_proposals[:, :, :, :, 2] * 0.5
    new_ty = b_proposals[:, :, :, :, 1] - b_proposals[:, :, :, :, 3] * 0.5
    new_bx = b_proposals[:, :, :, :, 0] + b_proposals[:, :, :, :, 2] * 0.5
    new_by = b_proposals[:, :, :, :, 1] + b_proposals[:, :, :, :, 3] * 0.5
    
    proposals[:, :, :, :, 0] = new_tx
    proposals[:, :, :, :, 1] = new_ty
    proposals[:, :, :, :, 2] = new_bx
    proposals[:, :, :, :, 3] = new_by
    
    return proposals

def IoU(proposals, bboxes):
    B, A, H, W, _ = proposals.size()
    _, N, _ = bboxes.size()
    iou_mat = torch.zeros(B, A*H*W, N, device = proposals.device, dtype = proposals.dtype)
    proposals = proposals.view(B, A*H*W, -1)
    
    Area_of_Proposal = ((proposals[:, :, 0]-proposals[:, :, 2])*\
                        (proposals[:, :, 1]-proposals[:, :, 3]))
    Area_of_Proposal = Area_of_Proposal.view(B, A*H*W, -1).squeeze(2)
    # print(Area_of_Proposal.size()) #(B, A*H*W, 1)
  
    Area_of_BBox = ((bboxes[:, :, 0]-bboxes[:, :, 2])*(bboxes[:, :, 1]-bboxes[:, :, 3]))
    
    #broadcasting을 이용 -> proposal 전체를 bbox에 다 넣어버려야하니깐 bbox의 차원을 높게 잡는다.
    #이렇게하면 bbox의 차원이 더 높이니 broadcasting으로 각각의 bbox에 대해 모든 proposal의 계산이 된다.
    tl = torch.max(proposals[:, :, :2].unsqueeze(2), bboxes[:, :, :2].unsqueeze(1))
    br = torch.min(proposals[:, :, 2:4].unsqueeze(2), bboxes[:, :, 2:4].unsqueeze(1))
    
    Area_of_Intersection = torch.prod(br-tl, axis = 3) * (tl<br).all(dim = 3)
  
    iou_mat = torch.div(Area_of_Intersection, Area_of_Proposal.unsqueeze(2)+Area_of_BBox.unsqueeze(1)-Area_of_Intersection)
   
    return iou_mat

class ProposalModule(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_anchors = 9, drop_ratio = 0.3):
        super(ProposalModule, self).__init__()
        
        self.num_anchors = num_anchors
        
        self.pred_layer = nn.Sequential(
                            nn.Conv2d(in_dim, hidden_dim, 3, padding = 1),
                            nn.Dropout2d(drop_ratio),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dim, self.num_anchors*6, 1))
        
    def _extract_anchor_data(self, anchor_data, anchor_idx):
        #D는 데이터
        B,A,D,H,W = anchor_data.shape
        anchor_data = anchor_data.permute(0,1,3,4,2).contiguous().view(-1, D)
        extracted_anchors = anchor_data[anchor_idx]
        return extracted_anchors
    
    def forward(self, features, pos_anchor_coord=None, pos_anchor_idx = None, neg_anchor_idx = None):
        if pos_anchor_coord is None or pos_anchor_idx is None or neg_anchor_idx is None:
            mode = 'eval'
        else:
            mode = 'train'
        
        B, _, H, W = features.size()
        out = self.pred_layer(features)
        out = out.view(B, self.num_anchors, 6, 7, 7)     
        
        if mode == 'train':
            M, _ = pos_anchor_coord.size()
            pos = self._extract_anchor_data(out, pos_anchor_idx)
            neg = self._extract_anchor_data(out, neg_anchor_idx)
            offsets = pos[:, 2:]
            conf_scores = torch.cat((pos[:, :2], neg[:, :2]))
            proposals = GenerateProposal(pos_anchor_coord.view(M, 1, 1, 1, 4), offsets.view(M, 1, 1, 1, 4))
            proposals = proposals.squeeze()
            
            return conf_scores, offsets, proposals
            
        elif mode == 'eval':
            conf_scores = out[:, :, :2, :, :]
            offsets = out[:, :, 2:, :, :]
            
            return conf_scores, offsets
        
def ConfScoreRegression(conf_scores, batch_size):
    M = conf_scores.shape[0] // 2
    GT_conf_scores = torch.zeros_like(conf_scores)
    #positive 영역에서 첫번째는 True 
    GT_conf_scores[:M, 0] = 1.
    #negative 영역에서 두번째는 negative -> 결국 True
    GT_conf_scores[M:, 1] = 1.
    
    conf_score_loss = nn.functional.binary_cross_entropy_with_logits(conf_scores, GT_conf_scores, reduction = 'sum') * 1. /batch_size
    
    return conf_score_loss

def BboxRegression(offsets, GT_offsets, batch_size):
    bbox_reg_loss = nn.functional.smooth_l1_loss(offsets, GT_offsets, reduction = 'sum') * 1. /batch_size
    
    return bbox_reg_loss

def ReferenceOnActivatiedAnchors(anchors, bboxes, grid, iou_mat, pos_thresh=0.7, neg_thresh=0.3):
    """
    positive anchor -> IoU >0.7 or highest
    
    negative anchor -> IoU lower than neg_thrsh
    """
    B, A, h_amap, w_amap, _ = anchors.size()
    N = bboxes.shape[1]
    
    #activated/positive anchors
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim = -1) 
    max_iou_per_box = iou_mat.max(dim = 1, keepdim = True)[0]
    activated_anc_mask = (iou_mat == max_iou_per_box) & (max_iou_per_box > 0)
    activated_anc_mask |= (iou_mat > pos_thresh)
    #multiple GT boxes걸리면 제일 큰 iou가진 box선택
    activated_anc_mask = activated_anc_mask.max(dim=-1)[0]
    activated_anc_ind = torch.nonzero(activated_anc_mask.view(-1)).squeeze(-1)
    
    #GT conf scores
    GT_conf_scores = max_iou_per_anc[activated_anc_mask]
    
    #GT class
    box_cls = bboxes[:, :, 4].view(B, 1, N).expand((B, A*h_amap*w_amap, N))
    GT_class = torch.gather(box_cls.to(torch.int64), -1, max_iou_per_anc.to(torch.int64).unsqueeze(-1)).squeeze(-1) #M
    GT_class = GT_class[activated_anc_mask].long()

    
    bboxes_expand = bboxes[:, :, :4].view(B, 1, N, 4).expand((B, A*h_amap*w_amap, N, 4))
    bboxes = torch.gather(bboxes_expand, -2, max_iou_per_anc_ind.unsqueeze(-1).unsqueeze(-1).expand(B, A*h_amap*w_amap, 1, 4)).view(-1, 4)
    bboxes = bboxes[activated_anc_ind]
    
    print('number of pos proposals : ', activated_anc_ind.shape[0])
    
    activated_anc_coord = anchors.view(-1, 4)[activated_anc_ind]
    
    #GT offsets
    wh_offsets = torch.log((bboxes[:, 2:4] - bboxes[:, :2])/(activated_anc_coord[:, 2:4] - activated_anc_coord[:, :2]))
    xy_offsets = (bboxes[:, :2] + bboxes[:, 2:4] - activated_anc_coord[:, :2] - activated_anc_coord[:, 2:4])/2.
    xy_offsets /= (activated_anc_coord[:, 2:4] - activated_anc_coord[:, :2])
    
    GT_offsets = torch.cat((xy_offsets, wh_offsets), dim = -1)
    
    #negative anchors
    negative_anc_mask = (max_iou_per_anc < neg_thresh)
    negative_anc_ind = torch.nonzero(negative_anc_mask.view(-1)).squeeze(-1)
    negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (activated_anc_ind.shape[0],))]
    negative_anc_coord = anchors.view(-1, 4)[negative_anc_ind.view(-1)]
    
    return activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, activated_anc_coord, negative_anc_coord

class FeatureExtractor(nn.Module):
  def __init__(self, reshaped_size = 224, pooling = False, verbose = False):
    super().__init__()

    from torchvision import models
    from torchsummary import summary

    self.mobilenet = models.mobilenet_v2(pretrained= True)
    #마지막 부분은 제외
    self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])

    if pooling:
      self.mobilenet.add_module('LastAvgPool', nn.AvgPool2d(math.ceil(reshape_size/32.)))
    
    for i in self.mobilenet.named_parameters():
      i[1].required_grad = True

    if verbose:
      summary(self.mobilenet.cuda(), (3, reshaped_size, reshaped_size))
    
  def forward(self, img, verbose = False):
    num_img = img.shape[0]

    img_prepro = img

    feat = []
    process_batch = 500
    for b in range(math.ceil(num_img/process_batch)):
      feat.append(self.mobilenet(img_prepro[b*process_batch:(b+1)*process_batch]).squeeze(-1).squeeze(-1))
      feat = torch.cat(feat)

    if verbose:
      print('Output feature shape:', feat.shape)

    return feat    

class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]])
        self.feat_extractor = FeatureExtractor()
        self.prop_module = ProposalModule(1280, num_anchors = self.anchor_list.shape[0])
        
    def forward(self, images, bboxes, output_mode = 'loss'):
        #weights to mulityply to each loss
        w_conf = 1
        w_reg = 5
        
        B, _, _, _ = images.size()
        _,N,_ = bboxes.size()
        #1) image feature extracion
        feature = self.feat_extractor(images)
        #2) grid and anchor generation
        grid = GenerateGrid(B)
        anchors = GenerateAnchor(self.anchor_list, grid)
        anc_per_img = torch.prod(torch.tensor([anchors.shape[1:-1]]))
        #3) Compute IoU between anchors and GT boxes -> determine
        iou_mat = IoU(anchors, bboxes)
        pos_anchor_idx, negative_anchor_idx, GT_conf_scores, GT_offsets, GT_class, pos_anc_coord, _ = \
            ReferenceOnActivatiedAnchors(anchors, bboxes, grid, iou_mat)
        #4) Compute conf_scores, offsets, proposals through the region proposal module 
        conf_scores, offsets, proposals = self.prop_module(feature, pos_anc_coord, pos_anchor_idx, negative_anchor_idx)
        reg_loss = BboxRegression(offsets, GT_offsets, B)
        conf_loss = ConfScoreRegression(conf_scores, B)
        total_loss = w_conf * conf_loss + w_reg * reg_loss
        
        M, _ = conf_scores.size()
        conf_scores = conf_scores[:M//2, :]
        
        if output_mode == 'loss':
            return total_loss
        # #added
        elif output_mode == 'two_stage':
            return total_loss, conf_scores, proposals, feature, GT_class, pos_anchor_idx, anc_per_img, GT_offsets
        else:
            return total_loss, conf_scores, proposals, feature, GT_class, pos_anchor_idx, anc_per_img
        
    
    def inference(self, images, thresh = 0.5, nms_thresh = 0.7, mode = 'RPN'):
        final_conf_probs, final_proposals = [], []
        with torch.no_grad():
            features = self.feat_extractor(images)
            B, _, H, W = features.shape
            A = len(self.anchor_list)
            grid_list = GenerateGrid(B)
            self.anchor_list = self.anchor_list.to(features.dtype).to(features.device)
            anc_list = GenerateAnchor(self.anchor_list, grid_list)
            conf_scores, offsets = self.prop_module(features)
            conf_probs = torch.sigmoid(conf_scores[:, :, 0, :, :]).squeeze()
            offsets = offsets.permute(0,1,3,4,2).contiguous()
            proposals = GenerateProposal(anc_list, offsets)
            for b in range(B):
                index = torch.nonzero((conf_probs[b] > thresh), as_tuple = True)
                #pos만 모아둔거
                masked = conf_probs[b][index].view(-1) 
                s_proposal = proposals[b][index].view(-1, 4)
                left = torchvision.ops.nms(s_proposal, masked, nms_thresh)
                final_proposals.append(s_proposal[left])
                final_conf_probs.append(masked[left].unsqueeze(1))
                
        if mode == 'RPN':
            features = [torch.zeros_like(i) for i in final_conf_probs]
        return final_proposals, final_conf_probs, features

class TwoStageDetector(nn.Module):
  def __init__(self, in_dim = 1280, hidden_dim = 256, num_classes = 20, \
              roi_output_w = 2, roi_output_h = 2, drop_ratio = 0.3):
    super().__init__()

    self.num_classes = num_classes
    self.roi_output_h, self.roi_output_w = roi_output_h, roi_output_w

    #add
    self.num_anchors = 9

    self.rpn = RPN()
    self.cls_layer = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                nn.Dropout(drop_ratio),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, num_classes))  
    # add
    self.reg_layer = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                nn.Dropout(drop_ratio),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, 4))   
                
  def forward(self, images, bboxes):
    rpn_loss, conf_scores, proposals, features, GT_class, pos_anchor_idx, anc_per_img, GT_offsets = \
      self.rpn(images, bboxes, output_mode = 'two_stage')
    M, _ = proposals.size()
    boxes = torch.zeros((M, 5), device = proposals.device, dtype = proposals.dtype)
    boxes[:, 0] = (pos_anchor_idx//anc_per_img)  
    boxes[:, 1:5] = proposals

    roi_out = torchvision.ops.roi_align(features, boxes, (self.roi_output_w, self.roi_output_h))
    roi_pooled = torch.mean(roi_out, dim = (2,3))
    #added
    box_reg = self.reg_layer(roi_pooled)
    reg_loss = BboxRegression(box_reg, GT_offsets, images.shape[0])

    class_prob = self.cls_layer(roi_pooled)
    cls_loss = torch.nn.functional.cross_entropy(class_prob, GT_class, reduction = 'sum')*1./images.shape[0]
    total_loss = cls_loss + rpn_loss + reg_loss   

    return total_loss

  def inference(self, images, thresh = 0.5, nms_thresh=0.7):
    with torch.no_grad():
      final_proposals, final_conf_probs, features = self.rpn.inference(images, thresh, nms_thresh, mode = 'FasterRCNN')
      B, _, _, _ = features.size()
      roi_out = torchvision.ops.roi_align(features, final_proposals, (self.roi_output_w, self.roi_output_h))
      roi_pooled = torch.mean(roi_out, dim = (2,3))
      class_prob = self.cls_layer(roi_pooled)
      pred = torch.argmax(class_prob, dim = -1, keepdim = True)

      list_of_num = []
      for i in final_proposals:
        number = i.shape[0]
        list_of_num.append(number)
      final_class = torch.split(pred, list_of_num, dim = 0)

    return final_proposals, final_conf_probs, final_class
