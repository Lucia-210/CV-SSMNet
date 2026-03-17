"""
物理先验提取模块
从T3/C3极化相干矩阵计算物理分解参数作为条件化先验

包含:
1. H/A/Alpha (Cloude-Pottier分解) - 熵/各向异性/散射角
2. Pauli分解 - 奇次散射/偶次散射/体散射
3. Span (总功率)
4. 极化比参数

作者: ASDF2Net-FiLM扩展
"""

import numpy as np
from scipy import linalg


def compute_T3_matrix(data):
    """
    从6通道或9通道数据构建T3相干矩阵
    
    参数:
        data: numpy数组, 形状 (H, W, C) 或 (N, H, W, C)
              C=6: [T11, T12_real, T12_imag, T22, T13_real, T13_imag, T23_real, T23_imag, T33]
              C=9: 完整T3矩阵元素
    
    返回:
        T3: 形状 (H, W, 3, 3) 或 (N, H, W, 3, 3) 的复数矩阵
    """
    if data.ndim == 3:
        H, W, C = data.shape
        batch = False
    else:
        N, H, W, C = data.shape
        batch = True
    
    if C == 6:
        # 假设格式: T11, T22, T33, T12_real, T12_imag, T13_real, T13_imag, T23_real, T23_imag
        # 或常见的: T11, T12, T13, T22, T23, T33 (复数形式)
        # 这里假设已经是复数形式的6个独立元素
        if batch:
            T3 = np.zeros((N, H, W, 3, 3), dtype=np.complex64)
            # 对角线元素 (实数)
            T3[:, :, :, 0, 0] = data[:, :, :, 0]  # T11
            T3[:, :, :, 1, 1] = data[:, :, :, 3]  # T22
            T3[:, :, :, 2, 2] = data[:, :, :, 5]  # T33
            # 非对角线元素 (复数)
            T3[:, :, :, 0, 1] = data[:, :, :, 1] + 1j * data[:, :, :, 2]  # T12
            T3[:, :, :, 1, 0] = np.conj(T3[:, :, :, 0, 1])  # T21 = T12*
            T3[:, :, :, 0, 2] = data[:, :, :, 4]  # T13 (假设已是复数)
            T3[:, :, :, 2, 0] = np.conj(T3[:, :, :, 0, 2])  # T31 = T13*
        else:
            T3 = np.zeros((H, W, 3, 3), dtype=np.complex64)
            T3[:, :, 0, 0] = data[:, :, 0]
            T3[:, :, 1, 1] = data[:, :, 3]
            T3[:, :, 2, 2] = data[:, :, 5]
            T3[:, :, 0, 1] = data[:, :, 1] + 1j * data[:, :, 2]
            T3[:, :, 1, 0] = np.conj(T3[:, :, 0, 1])
            T3[:, :, 0, 2] = data[:, :, 4]
            T3[:, :, 2, 0] = np.conj(T3[:, :, 0, 2])
    
    elif C == 9:
        if batch:
            T3 = np.zeros((N, H, W, 3, 3), dtype=np.complex64)
            T3[:, :, :, 0, 0] = data[:, :, :, 0]  # T11
            T3[:, :, :, 0, 1] = data[:, :, :, 1]  # T12
            T3[:, :, :, 0, 2] = data[:, :, :, 2]  # T13
            T3[:, :, :, 1, 0] = data[:, :, :, 3]  # T21
            T3[:, :, :, 1, 1] = data[:, :, :, 4]  # T22
            T3[:, :, :, 1, 2] = data[:, :, :, 5]  # T23
            T3[:, :, :, 2, 0] = data[:, :, :, 6]  # T31
            T3[:, :, :, 2, 1] = data[:, :, :, 7]  # T32
            T3[:, :, :, 2, 2] = data[:, :, :, 8]  # T33
        else:
            T3 = np.zeros((H, W, 3, 3), dtype=np.complex64)
            T3[:, :, 0, 0] = data[:, :, 0]
            T3[:, :, 0, 1] = data[:, :, 1]
            T3[:, :, 0, 2] = data[:, :, 2]
            T3[:, :, 1, 0] = data[:, :, 3]
            T3[:, :, 1, 1] = data[:, :, 4]
            T3[:, :, 1, 2] = data[:, :, 5]
            T3[:, :, 2, 0] = data[:, :, 6]
            T3[:, :, 2, 1] = data[:, :, 7]
            T3[:, :, 2, 2] = data[:, :, 8]
    else:
        raise ValueError(f"不支持的通道数: {C}, 需要6或9")
    
    return T3


def compute_H_A_Alpha(data, eps=1e-10):
    """
    计算Cloude-Pottier H/A/Alpha分解
    
    参数:
        data: 输入数据 (H, W, C) 或 (N, H, W, C)
        eps: 防止除零的小量
    
    返回:
        H: 熵 [0, 1]
        A: 各向异性 [0, 1]  
        Alpha: 平均散射角 [0, 90] 度
    """
    # 简化计算：直接从通道功率估计
    if data.ndim == 3:
        H, W, C = data.shape
        
        # 取对角线元素（功率）
        if C >= 6:
            # T11, T22, T33 近似对应三个特征值
            lambda1 = np.abs(data[:, :, 0]) + eps
            lambda2 = np.abs(data[:, :, 3] if C >= 4 else data[:, :, 1]) + eps
            lambda3 = np.abs(data[:, :, 5] if C >= 6 else data[:, :, 2]) + eps
        else:
            lambda1 = np.abs(data[:, :, 0]) + eps
            lambda2 = np.abs(data[:, :, 1]) + eps
            lambda3 = np.abs(data[:, :, 2]) + eps if C > 2 else eps
        
        # 归一化
        total = lambda1 + lambda2 + lambda3
        p1 = lambda1 / total
        p2 = lambda2 / total
        p3 = lambda3 / total
        
        # 熵 H
        H_val = -(p1 * np.log(p1 + eps) + p2 * np.log(p2 + eps) + p3 * np.log(p3 + eps)) / np.log(3)
        H_val = np.clip(H_val, 0, 1)
        
        # 各向异性 A
        A_val = (p2 - p3) / (p2 + p3 + eps)
        A_val = np.clip(A_val, 0, 1)
        
        # 散射角 Alpha (简化计算)
        # 使用功率比估计
        ratio = lambda1 / (total + eps)
        Alpha_val = np.arccos(np.sqrt(np.clip(ratio, 0, 1))) * 180 / np.pi
        
    else:  # 4D: (N, H, W, C)
        N, H, W, C = data.shape
        
        if C >= 6:
            lambda1 = np.abs(data[:, :, :, 0]) + eps
            lambda2 = np.abs(data[:, :, :, 3] if C >= 4 else data[:, :, :, 1]) + eps
            lambda3 = np.abs(data[:, :, :, 5] if C >= 6 else data[:, :, :, 2]) + eps
        else:
            lambda1 = np.abs(data[:, :, :, 0]) + eps
            lambda2 = np.abs(data[:, :, :, 1]) + eps
            lambda3 = np.abs(data[:, :, :, 2]) + eps if C > 2 else np.full_like(lambda1, eps)
        
        total = lambda1 + lambda2 + lambda3
        p1 = lambda1 / total
        p2 = lambda2 / total
        p3 = lambda3 / total
        
        H_val = -(p1 * np.log(p1 + eps) + p2 * np.log(p2 + eps) + p3 * np.log(p3 + eps)) / np.log(3)
        H_val = np.clip(H_val, 0, 1)
        
        A_val = (p2 - p3) / (p2 + p3 + eps)
        A_val = np.clip(A_val, 0, 1)
        
        ratio = lambda1 / (total + eps)
        Alpha_val = np.arccos(np.sqrt(np.clip(ratio, 0, 1))) * 180 / np.pi
    
    return H_val, A_val, Alpha_val


def compute_pauli_decomposition(data, eps=1e-10):
    """
    计算Pauli分解分量
    
    返回:
        Ps: 奇次散射（表面散射）功率
        Pd: 偶次散射（二面角散射）功率
        Pv: 体散射功率
    """
    if data.ndim == 3:
        H, W, C = data.shape
        
        if C >= 6:
            # T11 ∝ |Shh + Svv|² → 奇次散射
            # T22 ∝ |Shh - Svv|² → 偶次散射  
            # T33 ∝ |Shv|² → 体散射
            Ps = np.abs(data[:, :, 0]) + eps  # 表面散射
            Pd = np.abs(data[:, :, 3] if C >= 4 else data[:, :, 1]) + eps  # 二面角
            Pv = np.abs(data[:, :, 5] if C >= 6 else data[:, :, 2]) + eps  # 体散射
        else:
            Ps = np.abs(data[:, :, 0]) + eps
            Pd = np.abs(data[:, :, 1]) + eps
            Pv = np.abs(data[:, :, 2]) + eps if C > 2 else np.full((H, W), eps)
            
    else:  # 4D
        N, H, W, C = data.shape
        
        if C >= 6:
            Ps = np.abs(data[:, :, :, 0]) + eps
            Pd = np.abs(data[:, :, :, 3] if C >= 4 else data[:, :, :, 1]) + eps
            Pv = np.abs(data[:, :, :, 5] if C >= 6 else data[:, :, :, 2]) + eps
        else:
            Ps = np.abs(data[:, :, :, 0]) + eps
            Pd = np.abs(data[:, :, :, 1]) + eps
            Pv = np.abs(data[:, :, :, 2]) + eps if C > 2 else np.full((N, H, W), eps)
    
    return Ps, Pd, Pv


def compute_span(data, eps=1e-10):
    """
    计算总散射功率 (Span)
    
    Span = T11 + T22 + T33 = trace(T3)
    """
    if data.ndim == 3:
        C = data.shape[2]
        if C >= 6:
            span = np.abs(data[:, :, 0]) + np.abs(data[:, :, 3]) + np.abs(data[:, :, 5])
        else:
            span = np.sum(np.abs(data), axis=-1)
    else:
        C = data.shape[3]
        if C >= 6:
            span = np.abs(data[:, :, :, 0]) + np.abs(data[:, :, :, 3]) + np.abs(data[:, :, :, 5])
        else:
            span = np.sum(np.abs(data), axis=-1)
    
    return span + eps


def extract_physical_priors(data, normalize=True):
    """
    提取完整的物理先验向量
    
    参数:
        data: 输入SAR数据 (H, W, C) 或 (N, H, W, C)
        normalize: 是否标准化到[0,1]
    
    返回:
        priors: 物理先验向量 (H, W, 7) 或 (N, H, W, 7)
                通道顺序: [H, A, Alpha, Ps, Pd, Pv, Span]
    """
    # 计算各分解参数
    H_val, A_val, Alpha_val = compute_H_A_Alpha(data)
    Ps, Pd, Pv = compute_pauli_decomposition(data)
    span = compute_span(data)
    
    if data.ndim == 3:
        priors = np.stack([
            H_val,                    # 熵 [0, 1]
            A_val,                    # 各向异性 [0, 1]
            Alpha_val / 90.0,         # 归一化散射角 [0, 1]
            Ps,                       # 奇次散射
            Pd,                       # 偶次散射
            Pv,                       # 体散射
            span                      # 总功率
        ], axis=-1)
    else:
        priors = np.stack([
            H_val,
            A_val,
            Alpha_val / 90.0,
            Ps,
            Pd,
            Pv,
            span
        ], axis=-1)
    
    if normalize:
        priors = normalize_priors(priors)
    
    return priors


def normalize_priors(priors, eps=1e-10):
    """
    对物理先验进行标准化处理
    
    策略:
    - H, A, Alpha: 已经在[0,1]范围
    - Ps, Pd, Pv: 对数变换 + 标准化
    - Span: 对数变换 + 标准化
    """
    priors_norm = priors.copy()
    
    # 对功率项进行对数变换和标准化 (通道3-6)
    for i in range(3, 7):
        channel = priors_norm[..., i]
        # 对数变换
        log_val = np.log10(channel + eps)
        # Min-Max标准化
        vmin, vmax = np.percentile(log_val, [2, 98])
        if vmax > vmin:
            priors_norm[..., i] = np.clip((log_val - vmin) / (vmax - vmin), 0, 1)
        else:
            priors_norm[..., i] = 0.5
    
    return priors_norm.astype(np.float32)


def extract_priors_for_patches(patches, verbose=True):
    """
    为图像块批量提取物理先验
    
    参数:
        patches: 形状 (N, H, W, C, 1) 的图像块
        verbose: 是否打印进度
    
    返回:
        priors: 形状 (N, 7) 的先验向量（每个块的中心像素）
    """
    N = patches.shape[0]
    H, W = patches.shape[1], patches.shape[2]
    C = patches.shape[3]
    
    # 取中心像素位置
    center_h, center_w = H // 2, W // 2
    
    # 提取中心区域 (3x3窗口平均)
    h_start = max(0, center_h - 1)
    h_end = min(H, center_h + 2)
    w_start = max(0, center_w - 1)
    w_end = min(W, center_w + 2)
    
    if verbose:
        print(f"正在提取物理先验: {N} 个图像块...")
    
    priors_list = []
    batch_size = 10000  # 分批处理避免内存溢出
    
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch = patches[i:end_i]
        
        # 处理不同的输入维度
        if batch.ndim == 5 and batch.shape[-1] == 1:
            # 移除最后一个维度 (batch, H, W, C, 1) -> (batch, H, W, C)
            batch = np.squeeze(batch, axis=-1)
        elif batch.ndim == 5:
            # (batch, H, W, D, C) -> 取所有深度的平均
            batch = np.mean(batch, axis=3)  # (batch, H, W, C)
        # 如果是4D (batch, H, W, C)，直接使用
        
        center_region = batch[:, h_start:h_end, w_start:w_end, :]  # (batch, 3, 3, C)
        center_avg = np.mean(center_region, axis=(1, 2))  # (batch, C)
        
        # 计算先验
        # 扩展维度以使用现有函数
        center_avg_2d = center_avg.reshape(-1, 1, C)  # (batch, 1, C)
        
        priors_batch = extract_physical_priors(center_avg_2d, normalize=True)  # (batch, 1, 7)
        priors_batch = priors_batch.squeeze(axis=1)  # (batch, 7)
        
        priors_list.append(priors_batch)
        
        if verbose and (i + batch_size) % 50000 == 0:
            print(f"  已处理: {min(i + batch_size, N)}/{N}")
    
    priors = np.concatenate(priors_list, axis=0)
    
    if verbose:
        print(f"✓ 物理先验提取完成! 形状: {priors.shape}")
        print(f"  通道含义: [H, A, Alpha, Ps, Pd, Pv, Span]")
    
    return priors.astype(np.float32)


def extract_priors_for_image(image_data, verbose=True):
    """
    为整幅图像提取物理先验图
    
    参数:
        image_data: 形状 (H, W, C) 的图像
        verbose: 是否打印信息
    
    返回:
        priors_map: 形状 (H, W, 7) 的先验图
    """
    if verbose:
        print(f"正在提取整幅图像物理先验: {image_data.shape}...")
    
    priors_map = extract_physical_priors(image_data, normalize=True)
    
    if verbose:
        print(f"✓ 物理先验图提取完成! 形状: {priors_map.shape}")
    
    return priors_map


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("物理先验提取模块测试")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    test_data = np.random.rand(100, 100, 6).astype(np.float32) + 0.1
    
    # 测试图像级别提取
    priors = extract_priors_for_image(test_data)
    print(f"\n图像先验形状: {priors.shape}")
    print(f"各通道范围:")
    channel_names = ['H(熵)', 'A(各向异性)', 'Alpha(散射角)', 
                     'Ps(奇次)', 'Pd(偶次)', 'Pv(体散射)', 'Span(总功率)']
    for i, name in enumerate(channel_names):
        print(f"  {name}: [{priors[:,:,i].min():.3f}, {priors[:,:,i].max():.3f}]")
    
    # 测试patch级别提取
    test_patches = np.random.rand(1000, 13, 13, 6, 1).astype(np.float32) + 0.1
    patch_priors = extract_priors_for_patches(test_patches)
    print(f"\nPatch先验形状: {patch_priors.shape}")
    
    print("\n✓ 测试完成!")
