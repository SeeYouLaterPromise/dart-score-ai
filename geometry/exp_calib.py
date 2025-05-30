#########################################################
#  put this in e.g.  demo_calib_vis.py  and run
#########################################################
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------------------------
#  放在上一版脚本 optimise() 运行完、得到 traj 之后
#  ---------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from PIL import Image
import numpy as np
import os

import numpy as np

# --------------------------------------------------
# ① **全局随机**：四点均匀撒在整幅图里
# --------------------------------------------------
def rand_init_uniform(img_w, img_h, margin=20):
    """
    img_w, img_h : 图像宽高
    margin       : 距离边缘多少像素内不采样（防止落到看不见区域）
    return       : (4,2) numpy array
    """
    xs = np.random.uniform(margin, img_w - margin, size=4)
    ys = np.random.uniform(margin, img_h - margin, size=4)
    return np.stack([xs, ys], axis=1)              # (4,2)


# --------------------------------------------------
# ② **偏向有效象限**：在几何初始值附近加噪声
# --------------------------------------------------
def rand_init_near_pairs(numeric_pts,
                         sigma=60,
                         k=1.0):
    """
    numeric_pts : {id:(x,y)}  8 个数字点
    sigma       :  高斯噪声标准差 (px)。值越大 → 散得越远
    k           :  传给 init_calib() 的比例系数
    return      : (4,2) numpy array
    """
    base = init_calib(numeric_pts, k=k)            # 你的确定性函数
    noise = np.random.randn(*base.shape) * sigma
    return base + noise


def animate_trajectory(img_path, numeric_pts, traj,
                       save_gif='calib_iter.gif',
                       save_mp4=None, fps=25):
    """
    img_path   : 背景图片
    numeric_pts: {num:(x,y)} 八个数字点
    traj       : (T,4,2) numpy 数组，optimise() 的返回值
    save_gif   : 若给文件名则输出 GIF；填 None 不导出
    save_mp4   : 若给文件名则输出 MP4；填 None 不导出
    fps        : 帧率
    """
    # ------------ 画布 & 静态元素 -------------
    img = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img)
    ax.axis('off')

    # 数字点
    if calib_gt:
      gt=np.array(calib_gt)
      ax.scatter(gt[:,0],gt[:,1],c='lime',s=90,marker='*',label='GT')
      
    for n,(x,y) in numeric_pts.items():
        ax.scatter(x, y, c='cyan', s=40)
        ax.text(x+4, y-4, str(n), color='cyan', fontsize=9)

    # 动态元素：4 条轨迹 + 4 个红点
    paths = [ax.plot([],[], lw=1)[0] for _ in range(4)]
    dots  = ax.scatter([], [], c='red', s=90, marker='o')

    
    # ---------- 动画函数 ----------
    def init():
        dots.set_offsets(traj[0])                 # 起点
        for ln in paths: ln.set_data([], [])
        return paths + [dots]

    def update(frame):
        cur = traj[frame]                         # (4,2)
        dots.set_offsets(cur)
        for j in range(4):
            paths[j].set_data(traj[:frame+1,j,0],  # x 轨迹
                               traj[:frame+1,j,1]) # y 轨迹
        return paths + [dots]

    ani = FuncAnimation(fig, update,
                        frames=len(traj),
                        init_func=init,
                        interval=1000/fps,        # 毫秒
                        blit=True)

    # ------------ 保存 --------------
    if save_gif:
        print(f'⇢  Saving GIF  → {save_gif}')
        ani.save(save_gif,
                 writer=PillowWriter(fps=fps))

    if save_mp4:
        print(f'⇢  Saving MP4 → {save_mp4}')
        ani.save(save_mp4,
                 writer=FFMpegWriter(fps=fps,
                                     codec='libx264',
                                     bitrate=-1))

    plt.show()


# ========== 1. 手工输入 / 替换成你的坐标 =============
numeric_pts = {
    5:  (271,  48),   # <--  5 的中心像素 (x,y)
    20: (389,  23),   # <-- 20
    11: (25,  410),   # ...
    8:  (54,  526),
    3:  (416,  766),
    17: (528,  745),
    13: ( 769,  386),
    6:  ( 755,  275),
}
calib_gt = []
# calib_gt = [(348.9, 100.5), (104.8, 449.4), (452.6, 695.6), (697.8, 344.6)]          # [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]  ← 若暂时没有就留空
IMG_PATH = "data/darts_dataset/d1_02_04_2020/IMG_1081.JPG"
# =====================================================

# ---------- 2. 先验映射 -------------------------------
pair_cfg = [((5,20),'down'), ((11,8),'right'),
            ((3,17),'up'),   ((13,6),'left')]
ori_vec = {'up':np.array([0,-1]),'down':np.array([0,1]),
           'left':np.array([-1,0]),'right':np.array([1,0])}

def init_calib(pts,k=1.0):
    c=[]
    for (a,b),dirn in pair_cfg:
        pa,pb=np.array(pts[a]),np.array(pts[b])
        m=(pa+pb)/2; r=k*np.linalg.norm(pb-pa)
        c.append(m+r*ori_vec[dirn])
    return np.stack(c)

# ---------- 3. 几何 loss + 手写梯度 ---------------------
def geom_loss_grad(C,pts):
    g=np.zeros_like(C); L_dir=0
    pairs=[(5,20,0,'down'),(11,8,1,'right'),
           (3,17,2,'up'),  (13,6,3,'left')]
    for a,b,j,exp in pairs:
        pa,pb=np.array(pts[a]),np.array(pts[b]); pc=C[j]
        n=np.array([pb[1]-pa[1],-(pb[0]-pa[0])])
        n=n/np.linalg.norm(n)
        if n.dot(ori_vec[exp])<0: n=-n
        d=(pc-(pa+pb)/2).dot(n)
        if d<0: L_dir+=d**2; g[j]+=2*d*n
    centre=np.mean(list(pts.values()),axis=0)
    r=np.linalg.norm(C-centre,axis=1); var=np.var(r)
    r_bar=np.mean(r)
    for i in range(4):
        if r[i]>0: g[i]+=2*(r[i]-r_bar)*(C[i]-centre)/r[i]
    return L_dir+var,g

def optimise(pts,C0,gt=None,iters=50,lr=5e-2,w=5.0):
    C=C0.copy(); traj=[C.copy()]
    for _ in range(iters):
        L,g=geom_loss_grad(C,pts)
        if gt:
            diff=C-np.array(gt); L+=w*np.sum(diff**2); g+=2*w*diff
        C-=lr*g; traj.append(C.copy())
    return np.array(traj)

# ---------- 4. 运行 + 可视化 ---------------------------
# 例 1：完全随机
# img = Image.open(IMG_PATH)
# W, H = img.size
# C0 = rand_init_uniform(W, H, margin=30)
C0 = init_calib(numeric_pts,k=2.0)
traj = optimise(numeric_pts,C0,calib_gt)

# === 调用示例 =============================================
# 假设：
#   IMG_PATH   = "/path/your_image.png"
#   numeric_pts = {...}         # 8 个数字点
#   traj        = optimise(...) # shape (T,4,2)
# ----------------------------------------------------------
animate_trajectory(IMG_PATH,
                   numeric_pts,
                   traj,
                   save_gif=None,   # 改成 None 可不导出
                   save_mp4="geometry/calib_iter4.mp4",               # 或 'calib_iter.mp4'
                   fps=25)


# img = Image.open(IMG_PATH)
# fig,ax=plt.subplots(figsize=(6,6)); ax.imshow(img); ax.axis('off')
# # 数字点
# for n,(x,y) in numeric_pts.items():
#     ax.scatter(x,y,c='cyan',s=40); ax.text(x+4,y-4,str(n),color='cyan',fontsize=9)
# # 轨迹
# for j in range(4): ax.plot(traj[:,j,0],traj[:,j,1],lw=1)
# ax.scatter(C0[:,0],C0[:,1],c='yellow',s=90,marker='x',label='init')
# ax.scatter(traj[-1,:,0],traj[-1,:,1],c='red',s=90,marker='o',label='final')
# if calib_gt:
#     gt=np.array(calib_gt); ax.scatter(gt[:,0],gt[:,1],c='lime',s=90,marker='*',label='GT')
# ax.legend(loc='lower right'); plt.show()
