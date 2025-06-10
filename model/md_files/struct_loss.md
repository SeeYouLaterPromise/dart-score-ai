我们先尝试没有 8 个数字点坐标来初始化的方案：先尝试随机 4 个校准点，如何通过 YOLO 自带的 loss，以及我们提出的 struct loss 以及少量 MSE loss 的方案，训练看看是否能够收敛，请指导我实现。

可是，校准点也有 bbox，也有那三种损失，其实我们只要重写 compute loss 这个类，根据 class_id: 从 0 到 3 都是校准点，class_id=4 是飞镖点，就能区分开来。如果模型在 class_id0-3 有缺失，即校准点预测不全，我们需要用随机点来代替以便使用 struct_loss，对吧？

```
forward ──► ComputeLossFull
              │
              ├─ det_loss  (YOLO 原 3 项，对所有类别)
              ├─ struct_loss(calib_xy_pred)      ← 4 点结构
              └─ mse_loss   (有 GT 才算)

total_loss = det_loss + λ_struct·struct_loss + λ_mse·mse_loss
```

如果模型在 class_id0-3 有缺失，即校准点预测不全，用随机点来代替去没有梯度，是不是对训练没有效果？
