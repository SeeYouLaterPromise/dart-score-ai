# utils/calibration_loss.py
import torch
import torch.nn as nn
import math

class GeometryConsistencyLoss(nn.Module):
    def __init__(self, lambda_angle=1.0, lambda_radius=0.5, lambda_symmetry=0.5):
        super().__init__()
        self.lambda_angle = lambda_angle
        self.lambda_radius = lambda_radius
        self.lambda_symmetry = lambda_symmetry

    def forward(self, pred_pts):  # pred_pts: [B, 4, 2]
        loss = 0
        loss += self.lambda_angle * self.angle_spacing_loss(pred_pts)
        loss += self.lambda_radius * self.radius_consistency_loss(pred_pts)
        loss += self.lambda_symmetry * self.square_symmetry_loss(pred_pts)
        return loss

    def angle_spacing_loss(self, pred_pts):
        center = pred_pts.mean(dim=1, keepdim=True)  # [B, 1, 2]
        relative = pred_pts - center
        theta = torch.atan2(relative[..., 1], relative[..., 0])  # [B, 4]
        theta = torch.sort(theta, dim=1)[0]
        d_theta = theta[:, 1:] - theta[:, :-1]
        d_theta = torch.cat([d_theta, 2 * math.pi - theta[:, -1:] + theta[:, :1]], dim=1)
        return ((d_theta - (2 * math.pi / 4)) ** 2).mean()

    def radius_consistency_loss(self, pred_pts):
        center = pred_pts.mean(dim=1, keepdim=True)
        dist = torch.norm(pred_pts - center, dim=-1)  # [B, 4]
        return ((dist - dist.mean(dim=1, keepdim=True)) ** 2).mean()

    def square_symmetry_loss(self, pred_pts):
        d1 = torch.norm(pred_pts[:, 0] - pred_pts[:, 2], dim=-1)
        d2 = torch.norm(pred_pts[:, 1] - pred_pts[:, 3], dim=-1)
        return ((d1 - d2) ** 2).mean()
