# losses_full.py
import torch
from losses import ComputeLoss
from utils.struct_loss import struct_loss_no_nums, extract_calib_centers, fill_missing_with_random
from utils.torch_utils import de_parallel
import torch.nn.functional as F

class ComputeLossFull(ComputeLoss):
    def __init__(self, model, autobalance=False):
        super().__init__(model, autobalance)
        self.w_struct = model.hyp.get('struct', 4.0)
        self.w_mse    = model.hyp.get('mse',    2.0)
        self.calib_ids = (0,1,2,3)               # 4 calib classes
        self.img_size  = de_parallel(model).img_size

    # ------------- 核心 -------------
    def __call__(self, p, targets, calib_gt=None):
        """
        p        : list[Tensor] 原始 3 层预测
        targets  : (n,6) 含所有类别 (calib+dart)
        calib_gt : (B,4,2) or None，少量有真值
        """
        # ① 先跑 YOLO 原损失（校准+飞镖一起）
        det_loss, det_items = super().__call__(p, targets)

        # ② 从 p 抽 4 calib 中心
        calib_pred, has_pred = extract_calib_centers(
                                p, self.img_size, self.calib_ids)
        calib_pred = fill_missing_with_random(calib_pred, has_pred)

        # ③ Struct loss
        struct_loss = struct_loss_no_nums(calib_pred)

        # ④ MSE 监督（可选）
        mse_loss = 0.
        if calib_gt is not None:
            mask = calib_gt.abs().sum((1,2)) > 0
            if mask.any():
                mse_loss = F.mse_loss(calib_pred[mask], calib_gt[mask])

        # ⑤ 汇总
        total = det_loss + self.w_struct*struct_loss + self.w_mse*mse_loss

        loss_items = torch.cat((det_items,
                                struct_loss.detach().unsqueeze(0),
                                torch.as_tensor([mse_loss], 
                                device=det_items.device)))
        return total, loss_items
