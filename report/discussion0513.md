我们使用 yolov5s 进行 16000 张飞镖图片进行微调训练飞镖点检测。但是实验`发现对于新的飞镖盘泛化能力`不好，比如学习到了底部字体的错误信息。是不是每换一个新的飞镖盘都要重新训练？这样效率太差了吧！现在我们打算加入新的数据集（新飞镖盘）进行微调，你觉得多少张合适？

# Calibration Point

> We chose to train a model to capture structural geometric rules by introducing a targeted loss function..

融合“扇区编号、圆形结构、参考点几何关系” → 学结构规律，不学图案偏差
现在的目标检测模型是不是都是学习图片表层特征，没有考虑结构规律？
那我们如何让 YOLO 学会结构规律，从而无论什么样式的飞镖盘都要适应？

```python
# 假设 pred_points 是预测的参考点 [x1, y1, x2, y2, x3, y3, x4, y4]
# 可以按 batch 循环处理
def geometry_consistency_loss(pred_points, ref_radius=300):
    # 1. 中心坐标（预设/均值）作为圆心
    cx = pred_points[:, :, 0::2].mean(dim=-1, keepdim=True)  # shape: [B, 1]
    cy = pred_points[:, :, 1::2].mean(dim=-1, keepdim=True)  # shape: [B, 1]

    # 2. 计算每个参考点的极坐标角度
    dx = pred_points[:, :, 0::2] - cx  # shape: [B, 4]
    dy = pred_points[:, :, 1::2] - cy
    theta = torch.atan2(dy, dx)  # angle in radians

    # 3. 计算角度差，理想情况下应该是 90 度间隔
    theta_sorted, _ = torch.sort(theta, dim=-1)
    d_theta = theta_sorted[:, 1:] - theta_sorted[:, :-1]  # shape: [B, 3]
    d_theta = torch.cat([d_theta, 2*np.pi - theta_sorted[:, -1:] + theta_sorted[:, 0:1]], dim=-1)

    # 4. 计算角度偏差（目标是每个角度间隔 ≈ π/2）
    target = (2*np.pi) / 4
    loss = ((d_theta - target)**2).mean()  # MSE 作为结构损失

    return loss
```

# Dart point

> 当前的 YOLO 检测任务是一个“点检测”任务，它只看到了局部的纹理/颜色区域，而没有建立起“这是一个飞镖整体”的概念。

现在的问题是：我们发现基于 yolov5s 微调飞镖数据集的泛化能力很差，好像会被任何类似于银色的东西视为飞镖点。我们认为这是由于教模型仅从视觉（图案、纹理等表层特征）学习的局限性，所以我们现在提出来的思路是先教会 YOLO 识别出飞镖这一个整体，然后再识别那个飞镖尖端。你觉得怎么样？应该如何开展工作让教会 YOLO 识别出飞镖这一个整体？需要自己收集数据集？需要单独拍摄飞镖还是扎在飞镖盘上再框选标注？

先训练 YOLO 从飞镖盘上框选出飞镖，这是第一波训练，其旨在让 YOLO 学习到飞镖的结构特征视觉知识。
第二波训练是飞镖尖端点（相当于更小的目标检测），最终用第二波训练好的 YOLO 模型进行飞镖点检测，你觉得这样可行吗？效果会好吗？

`model/crop.py`

> wzj: tip 被上面胖乎的区域遮住 看不到尖端的场景

单目：有的时候我们不一定能看见 tip，这也是单目的局限性。但是对于判断得分可以容忍。
