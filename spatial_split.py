#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
空间分割模块 - 避免数据泄露

核心思想：
1. 将图像分成网格区域
2. 整个区域分配给训练或测试（不是单个像素）
3. 训练区和测试区之间留出间隔带（gap）
4. 确保测试集中的patches与训练集完全不重叠

作者: CV-SSM-PriorNet Team
"""

import numpy as np
from sklearn.utils import shuffle


def spatial_train_test_split(X, Y, train_ratio=0.1, gap_size=2, block_size=32, random_state=42):
    """
    空间分割：将图像分成块，按块分配到训练/测试集
    
    参数:
        X: 原始图像数据 (H, W, C)
        Y: 标签图 (H, W)
        train_ratio: 训练集比例
        gap_size: 训练区和测试区之间的间隔（像素数）
        block_size: 每个块的大小
        random_state: 随机种子
    
    返回:
        train_mask: 训练区域掩码 (H, W)
        test_mask: 测试区域掩码 (H, W)
    """
    np.random.seed(random_state)
    
    H, W = Y.shape
    
    # 计算块数量
    n_blocks_h = H // block_size
    n_blocks_w = W // block_size
    total_blocks = n_blocks_h * n_blocks_w
    
    # 随机选择训练块
    n_train_blocks = max(1, int(total_blocks * train_ratio))
    all_blocks = list(range(total_blocks))
    np.random.shuffle(all_blocks)
    train_blocks = set(all_blocks[:n_train_blocks])
    
    # 创建掩码
    train_mask = np.zeros((H, W), dtype=bool)
    test_mask = np.zeros((H, W), dtype=bool)
    
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block_idx = i * n_blocks_w + j
            
            # 块的边界
            h_start = i * block_size
            h_end = min((i + 1) * block_size, H)
            w_start = j * block_size
            w_end = min((j + 1) * block_size, W)
            
            if block_idx in train_blocks:
                # 训练块：内部区域（留出边缘作为gap）
                h_start_inner = h_start + gap_size
                h_end_inner = h_end - gap_size
                w_start_inner = w_start + gap_size
                w_end_inner = w_end - gap_size
                
                if h_start_inner < h_end_inner and w_start_inner < w_end_inner:
                    train_mask[h_start_inner:h_end_inner, w_start_inner:w_end_inner] = True
            else:
                # 测试块：内部区域
                h_start_inner = h_start + gap_size
                h_end_inner = h_end - gap_size
                w_start_inner = w_start + gap_size
                w_end_inner = w_end - gap_size
                
                if h_start_inner < h_end_inner and w_start_inner < w_end_inner:
                    test_mask[h_start_inner:h_end_inner, w_start_inner:w_end_inner] = True
    
    # 只保留有标签的像素
    labeled_mask = Y > 0
    train_mask = train_mask & labeled_mask
    test_mask = test_mask & labeled_mask
    
    print(f"  空间分割完成:")
    print(f"    块大小: {block_size}×{block_size}")
    print(f"    间隔带: {gap_size} 像素")
    print(f"    训练块数: {n_train_blocks}/{total_blocks}")
    print(f"    训练像素: {train_mask.sum():,}")
    print(f"    测试像素: {test_mask.sum():,}")
    
    return train_mask, test_mask


def create_patches_with_spatial_split(X, Y, windowSize=13, train_ratio=0.1, 
                                       gap_size=None, block_size=32, random_state=42):
    """
    使用空间分割创建patches，避免数据泄露
    
    参数:
        X: 原始图像数据 (H, W, C)
        Y: 标签图 (H, W)
        windowSize: patch窗口大小
        train_ratio: 训练集比例
        gap_size: 间隔带大小（默认为windowSize）
        block_size: 块大小
        random_state: 随机种子
    
    返回:
        X_train, X_test, Y_train, Y_test
    """
    if gap_size is None:
        gap_size = windowSize  # 间隔至少为窗口大小，确保无重叠
    
    H, W, C = X.shape
    margin = windowSize // 2
    
    # 获取空间分割掩码
    train_mask, test_mask = spatial_train_test_split(
        X, Y, train_ratio=train_ratio, 
        gap_size=gap_size, block_size=block_size,
        random_state=random_state
    )
    
    # 零填充
    X_padded = np.zeros((H + 2*margin, W + 2*margin, C), dtype=X.dtype)
    X_padded[margin:H+margin, margin:W+margin, :] = X
    
    # 提取训练patches
    train_rows, train_cols = np.where(train_mask)
    n_train = len(train_rows)
    X_train = np.zeros((n_train, windowSize, windowSize, C), dtype=X.dtype)
    Y_train = np.zeros(n_train, dtype=np.int32)
    
    for i, (r, c) in enumerate(zip(train_rows, train_cols)):
        X_train[i] = X_padded[r:r+windowSize, c:c+windowSize, :]
        Y_train[i] = Y[r, c] - 1  # 标签从0开始
    
    # 提取测试patches
    test_rows, test_cols = np.where(test_mask)
    n_test = len(test_rows)
    X_test = np.zeros((n_test, windowSize, windowSize, C), dtype=X.dtype)
    Y_test = np.zeros(n_test, dtype=np.int32)
    
    for i, (r, c) in enumerate(zip(test_rows, test_cols)):
        X_test[i] = X_padded[r:r+windowSize, c:c+windowSize, :]
        Y_test[i] = Y[r, c] - 1
    
    # 打乱顺序
    X_train, Y_train = shuffle(X_train, Y_train, random_state=random_state)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=random_state)
    
    print(f"  ✓ Patches创建完成 (无数据泄露)")
    print(f"    训练集: {X_train.shape}")
    print(f"    测试集: {X_test.shape}")
    
    return X_train, X_test, Y_train, Y_test


def visualize_spatial_split(Y, train_mask, test_mask, save_path=None):
    """
    可视化空间分割结果
    """
    import matplotlib.pyplot as plt
    
    # 创建可视化图
    vis = np.zeros((*Y.shape, 3), dtype=np.uint8)
    vis[train_mask] = [0, 255, 0]    # 绿色：训练区
    vis[test_mask] = [255, 0, 0]     # 红色：测试区
    vis[(Y > 0) & (~train_mask) & (~test_mask)] = [128, 128, 128]  # 灰色：间隔带
    
    plt.figure(figsize=(12, 8))
    plt.imshow(vis)
    plt.title('Spatial Split Visualization\nGreen=Train, Red=Test, Gray=Gap')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  空间分割可视化已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    # 测试
    print("测试空间分割...")
    
    # 模拟数据
    H, W, C = 750, 1024, 6
    X = np.random.rand(H, W, C).astype(np.complex64)
    Y = np.random.randint(0, 16, (H, W))
    
    X_train, X_test, Y_train, Y_test = create_patches_with_spatial_split(
        X, Y, windowSize=13, train_ratio=0.1
    )
    
    print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print("✓ 测试通过!")

