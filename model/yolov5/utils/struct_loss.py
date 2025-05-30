# utils/struct_loss.py
import torch
import math

def fill_missing_with_random(calib_xy_pred, has_pred):
    """
    any False in has_pred 用随机值(0~1)补，detach() 断梯度
    """
    B,_,_ = calib_xy_pred.shape
    rand  = torch.rand_like(calib_xy_pred)
    calib_xy_pred = torch.where(has_pred[...,None], 
                                calib_xy_pred, 
                                rand.detach())
    return calib_xy_pred


def extract_calib_centers(p, calib_cids=(0,1,2,3)):
    """
    p : list[(B,na,h,w,5+nc)]  —— 训练态输出
    返回:
        calib_xy  (B,4,2)    # 永远可导
    """
    device = p[0].device
    B      = p[0].shape[0]
    nc     = p[0].shape[-1] - 5

    # 拼所有预测到 (B,N,5+nc)
    outs = torch.cat([pi.view(B, -1, 5+nc) for pi in p], dim=1)
    obj  = outs[...,4].sigmoid()          # (B,N)
    cls  = outs[...,5:].sigmoid()         # (B,N,nc)
    score= obj.unsqueeze(-1) * cls        # (B,N,nc)

    xy_raw = outs[...,:2].sigmoid()*2 - .5    # 未解码中心 (B,N,2)
    B4xy = []
    for cid in calib_cids:                    # 0~3
        best = score[:,:,cid].argmax(1)       # (B,)
        # 取分最高的 anchor，不设阈值 -> 始终有梯度
        xy_c = xy_raw[torch.arange(B, device=device), best]  # (B,2)
        B4xy.append(xy_c)
    return torch.stack(B4xy, dim=1)           # (B,4,2)



def struct_loss_no_nums(calib_xy, eps=1e-6):
    """
    calib_xy : (B,4,2)  四校准点预测坐标
    -------------------------------------------------
    Loss =  radius_variance  +  angle_regularizer
    radius_variance   —— 四点应等距中心
    angle_regularizer —— 四点方位角 ≈ 0°, 90°, 180°, 270°
    """
    B = calib_xy.shape[0]
    centre = calib_xy.mean(1, keepdim=True)                # (B,1,2)
    vec    = calib_xy - centre                             # (B,4,2)
    radius = vec.norm(dim=-1)                              # (B,4)

    # (1) 让半径方差最小
    loss_rad = radius.var(dim=1).mean()

    # (2) 让 4 个向量两两正交（v_i·v_j = 0），对向 (i,j) = (0,2)(1,3) 应接近 -1
    # unit vectors
    v = vec / (radius[..., None] + eps)                    # (B,4,2)
    dot = torch.bmm(v, v.transpose(1,2))                   # (B,4,4)
    eye  = torch.eye(4, device=calib_xy.device)[None]      # (1,4,4)

    # 目标矩阵   [[ 1, 0, -1, 0],
    #            [ 0, 1,  0,-1],
    #            [-1, 0,  1, 0],
    #            [ 0,-1,  0, 1]]
    target = torch.tensor([[ 1, 0,-1, 0],
                           [ 0, 1, 0,-1],
                           [-1,0, 1, 0],
                           [ 0,-1,0, 1]],
                           device=calib_xy.device)[None]
    loss_ang = (dot - target).pow(2).mean()

    return loss_rad + loss_ang
