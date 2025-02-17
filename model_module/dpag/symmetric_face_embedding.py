import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


class AdaptiveAvgPool2dIgnorePadding(nn.Module):
    def __init__(self, output_size, eps=1e-9):
        super().__init__()
        self.output_size = output_size
        self.eps = eps

    def forward(self, x, mask):
        pool_sum = F.adaptive_avg_pool2d(x, self.output_size)
        mask_sum = F.adaptive_avg_pool2d(mask.float(), self.output_size)
        pooled_result = pool_sum / (mask_sum + self.eps)
        return pooled_result


class SymmetricFaceEmbedding(nn.Module):
    def __init__(self, image_size, feat_size, feat_ch, ratio, dropout_rate):
        """
        Args:
            image_size, feat_size: should be int (height = width)
            feat_ch (int)
            ratio (Optional[float]): to calculate the threshold for special cases
        """
        super().__init__()
        self.image_size = image_size
        self.resize_rate = image_size // feat_size
        self.ratio = ratio
        self.final_ch = feat_ch
        self.feat_size = feat_size

        self.gap = AdaptiveAvgPool2dIgnorePadding(1)

        self.location_projector = nn.Sequential(
            nn.Linear(8, 4 * self.final_ch, bias=False),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * self.final_ch, 4 * self.final_ch, bias=False)
        )

    def extract_landmark_rectangle(self, landmark):
        """ get the rectangle of each landmark
        
        Args:
            landmark: (B, T, 10)
            
        Returns:
            final_rectangles (np.array): top-left and bottom-right of each landmark rectangle in feat (x, y)
        """
        landmark_rectangles = torch.zeros((*landmark.shape[:2], 4, 4), device=landmark.device)
        
        eyes_mid_x = ((landmark[:, :, 0] + landmark[:, :, 2]) / 2)  # (B, T)
        eyes_mid_y = landmark[:, :, 5]
        eyes_mid_x_res = (self.image_size - eyes_mid_x)
        eyes_min_size = torch.min(torch.concat([
            eyes_mid_x.unsqueeze(-1), 
            eyes_mid_y.unsqueeze(-1), 
            eyes_mid_x_res.unsqueeze(-1)
        ], dim=-1), dim=-1)[0]  # (B, T)

        landmark_rectangles[:, :, 0, :] = torch.concat([
            (eyes_mid_x - eyes_min_size).unsqueeze(-1), 
            (eyes_mid_y - eyes_min_size).unsqueeze(-1), 
            eyes_mid_x.unsqueeze(-1), 
            eyes_mid_y.unsqueeze(-1)
        ], dim=-1)
        
        landmark_rectangles[:, :, 1, :] = torch.concat([
            eyes_mid_x.unsqueeze(-1), 
            (eyes_mid_y - eyes_min_size).unsqueeze(-1), 
            (eyes_mid_x + eyes_min_size).unsqueeze(-1), 
            eyes_mid_y.unsqueeze(-1)
        ], dim=-1)
        
        mouth_mid_x = ((landmark[:, :, 6] + landmark[:, :, 8]) / 2)
        mouth_1 = (self.image_size - eyes_mid_y)
        mouth_2 = (self.image_size - mouth_mid_x)
        mouth_min_size = torch.min(torch.concat([
            mouth_mid_x.unsqueeze(-1), 
            mouth_1.unsqueeze(-1),
            mouth_2.unsqueeze(-1)
        ], dim=-1), dim=-1)[0]  # (B, T)

        landmark_rectangles[:, :, 2, :] = torch.concat([
            (mouth_mid_x - mouth_min_size).unsqueeze(-1), 
            eyes_mid_y.unsqueeze(-1), 
            mouth_mid_x.unsqueeze(-1), 
            (eyes_mid_y + mouth_min_size).unsqueeze(-1) 
        ], dim=-1)

        landmark_rectangles[:, :, 3, :] = torch.concat([
            mouth_mid_x.unsqueeze(-1), 
            eyes_mid_y.unsqueeze(-1),
            (mouth_mid_x + mouth_min_size).unsqueeze(-1), 
            (eyes_mid_y + mouth_min_size).unsqueeze(-1),
        ], dim=-1)

        # processing special cases
        if self.ratio is not None:
            special_case_bound = int(np.round(self.image_size * self.ratio))
            special_case_mask = (eyes_min_size <= special_case_bound) | (mouth_min_size <= special_case_bound)  # (B, T)
            special_case_mask = special_case_mask.unsqueeze(-1).unsqueeze(-1)
            fixed_rectangle = torch.tensor([
                [0, 0, self.image_size // 2, self.image_size // 2],
                [self.image_size // 2, 0, self.image_size, self.image_size // 2],
                [0, self.image_size // 2, self.image_size // 2, self.image_size],
                [self.image_size // 2, self.image_size // 2, self.image_size, self.image_size]
            ], device=landmark.device)

            landmark_rectangles = landmark_rectangles * ~special_case_mask + fixed_rectangle * special_case_mask
        
        # resize
        final_rectangles = torch.round(landmark_rectangles / self.resize_rate).int()

        return final_rectangles

    def get_landmark_mask(self, landmark, h, w):
        """Return mask of landmark rectangles.

        Args:
            landmark (torch.tensor): (B, T, 10) 

        Returns:
            landmark_mask: (B, T, n, 1, h, w)
        """
        rectangles = self.extract_landmark_rectangle(landmark)  # (B, T, 4, 4)
        x1 = rectangles[..., 0]  # (B, T, 4)
        y1 = rectangles[..., 1]
        x2 = rectangles[..., 2]
        y2 = rectangles[..., 3]
        
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

        grid_y = grid_y[None, None, :, :].to(landmark.device)  # (1, 1, H, W)
        grid_x = grid_x[None, None, :, :].to(landmark.device)

        mask = (grid_x >= x1[..., None, None]) & (grid_x < x2[..., None, None]) & \
            (grid_y >= y1[..., None, None]) & (grid_y < y2[..., None, None])

        return mask.unsqueeze(3)

    def forward(self, feat, landmarks):
        """ 
        Args:
            feat: (B, T, C, H, W) 
            landmarks (torch.tensor) (B, T, 10) 
        
        Returns:
            (B, T, N, D)  where D is node dimension
        """
        
        B, T, C, H, W = feat.shape
        N = 4
        feat = repeat(feat, 'b t c h w -> b t n c h w', n=N)
        landmark_mask = self.get_landmark_mask(landmarks, h=H, w=W)
        cropped_feat = feat * landmark_mask
        
        appearance_feat = self.gap(cropped_feat, landmark_mask).squeeze([-1, -2])

        locations = torch.concat([landmarks[:, :, :4], landmarks[:, :, 6:]], dim=-1)
        location_feat = self.location_projector(locations)  # (B, T, 4 * D)
        location_feat = location_feat.reshape(B, T, 4, self.final_ch)

        final_feat = appearance_feat + location_feat  # (B, T, 4, D)
       
        return final_feat
