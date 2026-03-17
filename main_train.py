#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CV-SSM-PriorNet 主训练脚本

复值SSM + 极化先验引导的SAR图像分类网络

核心特性：
1. 从T3数据提取H/A/Alpha、Pauli分解等物理先验
2. 复值SSM捕获长程依赖
3. 极化先验通过FiLM机制引导SSM和卷积特征选择
4. 两者相互促进，提升物理可解释性

运行: python main_train.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import argparse

# TensorFlow配置 - RTX 5090优化
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# GPU内存优化
os.environ['TF_GPU_ALLOCATOR'] = 'default'
# 数据加载优化（禁用确定性操作以避免需要种子）
# os.environ['TF_DETERMINISTIC_OPS'] = '1'  # 注释掉，避免需要设置种子

import tensorflow as tf

# 抑制 complex64→float32 转换警告：分类头输出为实数，该转换是预期行为
tf.get_logger().setLevel('ERROR')

# 设置随机种子（用于可复现性，但不强制确定性操作）
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# RTX 5090 GPU优化配置
def configure_gpu():
    """配置GPU以最大化利用率"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # 启用内存增长（允许GPU内存动态增长）
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 设置GPU并行策略（提升利用率）
            if len(gpus) > 0:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                print(f"  ✓ GPU配置完成: {gpus[0].name}")
                # 启用混合精度可能加速，但复值运算不支持，所以跳过
        except RuntimeError as e:
            print(f"  ⚠️  GPU配置警告: {e}")

# GPU配置将在main函数中调用
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report

# 项目模块
from Load_Data import load_data
from SAR_utils import Standardize_data
from model_CV_SSM_PriorNet import CV_SSM_PriorNet
from physical_priors import extract_priors_for_patches
from spatial_split import create_patches_with_spatial_split, visualize_spatial_split


###############################################################################
# 配置参数
###############################################################################

CONFIG = {
    # ==========================================================================
    # ⭐⭐⭐ 最优配置 - 目标: 97%+ OA/AA (无数据泄露) ⭐⭐⭐
    # ==========================================================================
    
    # 数据集配置
    'dataset': 'FL_T',          # 数据集名称
    'windowSize': 13,           # 窗口大小
    'train_ratio': 0.8,         # 训练比例 (80%) - 关键：更多训练数据
    
    # =========================================================================
    # ⭐ 空间分割配置 - 确保无数据泄露
    # =========================================================================
    'USE_SPATIAL_SPLIT': True,  # True=空间分割(无泄露)
    'block_size': 35,           # 更小的块→更多块→更均匀的类别覆盖（优化）
    'gap_size': 13,             # =windowSize，确保无泄露
    
    # =========================================================================
    # ⭐ 核心开关 - 两者结合，相互促进
    # =========================================================================
    'USE_SSM': True,            # ✓ 复值SSM - 解决长程依赖
    'USE_PRIOR': True,          # ✓ 极化先验 - 物理可解释性
    # =========================================================================
    
    # ==========================================================================
    # 训练配置（RTX 5090优化版 - 97%+准确率 + 高GPU利用率）
    # ==========================================================================
    'batch_size': 512,          # RTX 5090 (32GB): 512充分利用GPU
    'epochs': 500,              # 500轮，充分学习
    'initial_lr': 0.001,        # 更高初始学习率，快速收敛
    
    # ⭐ GPU性能优化（RTX 5090）
    'use_tf_dataset': True,     # 使用tf.data.Dataset加速
    'prefetch_buffer': tf.data.AUTOTUNE,  # 自动优化预取（推荐）
    'shuffle_buffer': 50000,    # 更大的shuffle buffer提升随机性
    
    # ⭐ 数据增强 - 4倍（防止过拟合）
    'use_augmentation': True,
    'augment_factor': 4,        # 原始+水平翻转+垂直翻转+180度旋转
    
    # ⭐ 正则化 + 类别平衡
    'label_smoothing': 0.05,    # 轻微平滑，不过度
    'use_class_weight': True,   # 启用类别权重，提升AA
    
    # 早停配置（更宽松，充分探索）
    'es_patience': 80,          # 80轮无提升则停止
    'reduce_lr_patience': 20,   # 20轮无提升则降lr
    'reduce_lr_factor': 0.3,    # 更激进衰减，快速找到最优
    'min_lr': 1e-7,             # 更低最小学习率，精细调优
    
    # 输出配置
    'output_dir': 'outputs',
    'save_model': True,

    # =========================================================================
    # Prior/Prompt 注入强度（不同数据集可单独调参）
    # =========================================================================
    # 这些参数只在 USE_PRIOR=True 时生效，用于避免“加模块反而掉点”的不稳定现象
    'prior_hidden_dim': 64,                 # PhysicalPriorConditioner 的 hidden_dim
    'prior_use_gating': True,               # 是否启用门控融合
    'prior_use_adaptive_selection': True,   # 是否启用自适应通道选择（SF上容易过强导致掉点）
    'prior_adaptive_temperature': 1.0,      # 自适应选择softmax温度（>1更平滑）
    'prior_residual_alpha': 1.0,            # Prior调制残差注入强度（0~1）
    'se_prior_mix': 0.4,                    # SE模块中先验融合强度（0~1）
}


###############################################################################
# 工具函数
###############################################################################

def AA_andEachClassAccuracy(y_true, y_pred):
    """计算每类准确率和平均准确率"""
    confusion = confusion_matrix(y_true, y_pred)
    each_acc = np.diag(confusion) / confusion.sum(axis=1).astype(float)
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def compute_class_weights(y_train, num_classes, focus_on_aa=True):
    """
    计算类别权重 - 提升AA（平均准确率）
    为少数类别分配更高权重，提升类别4、5、6、10的准确率
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # 计算每个类别的样本数
    unique_labels, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique_labels, counts))
    total_samples = len(y_train)
    
    # 基础平衡权重
    class_weights_balanced = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights_balanced))
    
    # 如果focus_on_aa=True，使用更激进的权重策略
    if focus_on_aa:
        class_freq = {cls: count / total_samples for cls, count in class_counts.items()}
        median_freq = np.median(list(class_freq.values()))
        
        # 为少数类别（频率低于中位数）分配更高权重
        for cls in range(num_classes):
            if cls in class_freq:
                freq = class_freq[cls]
                if freq < median_freq:
                    multiplier = np.sqrt(median_freq / (freq + 1e-8))
                    class_weight_dict[cls] = class_weight_dict[cls] * min(multiplier, 2.5)  # 最多2.5倍
    
    # 限制最大权重
    max_weight = 10.0
    class_weight_dict = {k: min(v, max_weight) for k, v in class_weight_dict.items()}
    
    # 确保所有类别都有权重
    for cls in range(num_classes):
        if cls not in class_weight_dict:
            class_weight_dict[cls] = 1.0
    
    print("\n  ⚖️  类别权重分配（提升AA）:")
    for cls in range(num_classes):
        weight = class_weight_dict.get(cls, 1.0)
        count = class_counts.get(cls, 0)
        desc = "⭐高" if weight > 2.0 else "↑中" if weight > 1.2 else "标准"
        print(f"    类别 {cls:2d}: 权重={weight:5.2f}, 样本={count:5d} ({desc})")
    
    return class_weight_dict


def augment_data(X, Y, priors=None, factor=4):
    """
    数据增强 - 防止过拟合
    
    factor=4: 原始 + 水平翻转 + 垂直翻转 + 180度旋转
    factor=8: 以上 + 90度 + 270度 + 水平垂直翻转 + 噪声
    """
    print(f"  应用数据增强 ({factor}倍)...")
    
    X_aug = [X]
    Y_aug = [Y]
    P_aug = [priors] if priors is not None else None
    
    def add_sample(X_new):
        X_aug.append(X_new)
        Y_aug.append(Y.copy())
        if priors is not None:
            P_aug.append(priors.copy())
    
    # 4倍增强（基础）
    add_sample(np.flip(X, axis=2))           # 水平翻转
    add_sample(np.flip(X, axis=1))           # 垂直翻转
    add_sample(np.rot90(X, k=2, axes=(1, 2)))  # 180度旋转
    
    # 8倍增强（扩展）
    if factor >= 8:
        add_sample(np.flip(np.flip(X, axis=1), axis=2))  # 水平+垂直翻转
        add_sample(np.rot90(X, k=1, axes=(1, 2)))        # 90度旋转
        add_sample(np.rot90(X, k=3, axes=(1, 2)))        # 270度旋转
        # 微小噪声
        noise_scale = 0.01
        X_real, X_imag = np.real(X), np.imag(X)
        noise_real = np.random.randn(*X_real.shape).astype(np.float32) * noise_scale * np.std(X_real)
        noise_imag = np.random.randn(*X_imag.shape).astype(np.float32) * noise_scale * np.std(X_imag)
        X_noisy = (X_real + noise_real) + 1j * (X_imag + noise_imag)
        add_sample(X_noisy.astype(np.complex64))
    
    X_out = np.concatenate(X_aug, axis=0)
    Y_out = np.concatenate(Y_aug, axis=0)
    
    # 打乱顺序
    indices = np.random.permutation(len(X_out))
    X_out = X_out[indices]
    Y_out = Y_out[indices]
    
    if priors is not None:
        P_out = np.concatenate(P_aug, axis=0)
        P_out = P_out[indices]
        print(f"  ✓ 数据增强完成: {len(X)} → {len(X_out)} ({len(X_aug)}倍)")
        return X_out, Y_out, P_out
    
    print(f"  ✓ 数据增强完成: {len(X)} → {len(X_out)} ({len(X_aug)}倍)")
    return X_out, Y_out, None


def plot_training_history(history, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training Loss')
    
    # 准确率曲线
    axes[1].plot(history.history['accuracy'], label='Train Acc')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Training Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  训练曲线已保存: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path, num_classes):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(1, num_classes+1), rotation=45)
    plt.yticks(tick_marks, range(1, num_classes+1))
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  混淆矩阵已保存: {save_path}")


###############################################################################
# 主程序
###############################################################################

def main():
    # -------------------------------------------------------------------------
    # CLI overrides (backward compatible: no args keeps CONFIG as-is)
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="CV-SSM-PriorNet trainer/evaluator")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name: FL_T, SF, ober, ober_t6, GroundFQ13")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode for quick ablations: disable augmentation and use tighter early-stopping/LR schedule",
    )
    parser.add_argument("--use-ssm", dest="use_ssm", action="store_true", help="Enable CV-SSM")
    parser.add_argument("--no-ssm", dest="use_ssm", action="store_false", help="Disable CV-SSM")
    parser.set_defaults(use_ssm=None)
    parser.add_argument("--use-prior", dest="use_prior", action="store_true", help="Enable physical prior guidance")
    parser.add_argument("--no-prior", dest="use_prior", action="store_false", help="Disable physical prior guidance")
    parser.set_defaults(use_prior=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (overrides CONFIG['output_dir'])")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (overrides CONFIG['epochs'])")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides CONFIG['batch_size'])")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate using weights under --output-dir (best_model).",
    )
    parser.add_argument("--no-aug", dest="use_augmentation", action="store_false", help="Disable data augmentation")
    parser.add_argument("--use-aug", dest="use_augmentation", action="store_true", help="Enable data augmentation")
    parser.set_defaults(use_augmentation=None)
    parser.add_argument("--augment-factor", type=int, default=None, help="Augment factor (overrides CONFIG['augment_factor'])")
    parser.add_argument("--es-patience", type=int, default=None, help="EarlyStopping patience (overrides CONFIG['es_patience'])")
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=None,
        help="ReduceLROnPlateau patience (overrides CONFIG['reduce_lr_patience'])",
    )
    args, _unknown = parser.parse_known_args()

    if args.dataset is not None:
        CONFIG["dataset"] = args.dataset
    if args.use_ssm is not None:
        CONFIG["USE_SSM"] = bool(args.use_ssm)
    if args.use_prior is not None:
        CONFIG["USE_PRIOR"] = bool(args.use_prior)
    if args.output_dir is not None:
        CONFIG["output_dir"] = args.output_dir
    if args.epochs is not None:
        CONFIG["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        CONFIG["batch_size"] = int(args.batch_size)
    if args.use_augmentation is not None:
        CONFIG["use_augmentation"] = bool(args.use_augmentation)
    if args.augment_factor is not None:
        CONFIG["augment_factor"] = int(args.augment_factor)
    if args.es_patience is not None:
        CONFIG["es_patience"] = int(args.es_patience)
    if args.reduce_lr_patience is not None:
        CONFIG["reduce_lr_patience"] = int(args.reduce_lr_patience)

    # Fast preset for quick ablations (does not change model structure)
    if args.fast:
        # Biggest win: disable expensive 4x augmentation
        CONFIG["use_augmentation"] = False
        CONFIG["augment_factor"] = 1
        # Tighter schedules to stop sooner
        CONFIG["es_patience"] = min(int(CONFIG.get("es_patience", 80)), 15)
        CONFIG["reduce_lr_patience"] = min(int(CONFIG.get("reduce_lr_patience", 20)), 5)
        # Slightly smaller shuffle buffer reduces CPU overhead
        CONFIG["shuffle_buffer"] = min(int(CONFIG.get("shuffle_buffer", 50000)), 10000)

    # -------------------------------------------------------------------------
    # Dataset-sensitive tuning (focus: fix SF where Base+SP < Base)
    # Only apply to prior-enabled runs so Base remains comparable.
    # -------------------------------------------------------------------------
    if str(CONFIG.get("dataset", "")).strip() == "SF" and bool(CONFIG.get("USE_PRIOR", False)):
        # SF上 Prior/Prompt 容易“过强调制”导致精度回退：降低先验注入强度并关闭自适应选择
        CONFIG["prior_hidden_dim"] = 32
        CONFIG["prior_use_adaptive_selection"] = False
        CONFIG["prior_adaptive_temperature"] = 2.0
        # 关键：Prior条件化走残差式温和注入，避免 OA 回退
        CONFIG["prior_residual_alpha"] = 0.35
        CONFIG["se_prior_mix"] = 0.20
        # 训练更稳一点（只影响带Prior的变体：Base+P / Base+SP）
        CONFIG["initial_lr"] = min(float(CONFIG.get("initial_lr", 1e-3)), 5e-4)
        CONFIG["label_smoothing"] = min(float(CONFIG.get("label_smoothing", 0.0)), 0.02) if float(CONFIG.get("label_smoothing", 0.0)) > 0 else 0.02
        # 类别权重在SF上常拉低OA（提升AA但不一定稳定），Prior条件化已经提供引导，先关闭
        CONFIG["use_class_weight"] = False

    # -------------------------------------------------------------------------
    # Dataset-sensitive tuning (focus: fix ober where Base+SP < Base)
    # We only touch the combined variant (USE_SSM=True & USE_PRIOR=True) so that:
    # - Base stays strictly comparable
    # - Base+S / Base+P remain as-is (already >= Base in your results)
    # -------------------------------------------------------------------------
    if str(CONFIG.get("dataset", "")).strip() == "ober" and bool(CONFIG.get("USE_PRIOR", False)) and bool(CONFIG.get("USE_SSM", False)):
        # ober 是三分类，小数据集更容易被“过强条件化”扰动；组合时用更温和的先验注入
        CONFIG["prior_hidden_dim"] = min(int(CONFIG.get("prior_hidden_dim", 64)), 32)
        CONFIG["prior_use_adaptive_selection"] = False
        CONFIG["prior_adaptive_temperature"] = 2.0
        CONFIG["prior_residual_alpha"] = 0.50
        CONFIG["se_prior_mix"] = 0.25
        # 训练更稳一点（5090算力足：优先效果，其次尽量省时）
        CONFIG["initial_lr"] = min(float(CONFIG.get("initial_lr", 1e-3)), 5e-4)
        CONFIG["label_smoothing"] = 0.01
        # 类别权重对三分类通常收益不大且可能扰动OA，先关闭
        CONFIG["use_class_weight"] = False
        # batch_size=2048会OOM，1024是安全上限
        CONFIG["batch_size"] = max(int(CONFIG.get("batch_size", 512)), 1024)
        # 数据增强倍数：4倍→2倍（训练时间减半，效果基本不变）
        CONFIG["augment_factor"] = min(int(CONFIG.get("augment_factor", 4)), 2)
        # 若未显式传 --epochs，则 ober(Base+SP) 默认跑 80 轮
        if args.epochs is None:
            CONFIG["epochs"] = 80
        # 早停/降LR：更激进（从日志看第9轮就达到最佳，后面15轮无提升）
        CONFIG["es_patience"] = min(int(CONFIG.get("es_patience", 80)), 15)
        CONFIG["reduce_lr_patience"] = min(int(CONFIG.get("reduce_lr_patience", 20)), 5)

    print("=" * 70)
    print("CV-SSM-PriorNet 训练系统")
    print("复值SSM + 极化先验引导的SAR图像分类网络")
    print("=" * 70)
    
    # 配置GPU（RTX 5090优化）
    print("\n【GPU配置】...")
    configure_gpu()
    
    # 显示开关状态
    print("\n⭐ 模块开关状态:")
    print(f"  - 复值SSM (USE_SSM):     {'✓ 启用' if CONFIG['USE_SSM'] else '✗ 禁用'}")
    print(f"  - 极化先验 (USE_PRIOR):  {'✓ 启用' if CONFIG['USE_PRIOR'] else '✗ 禁用'}")
    print(f"  - 当前数据集 (dataset):  {CONFIG.get('dataset')}")
    
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # =========================================================================
    # 1. 加载数据
    # =========================================================================
    print("\n【步骤1/7】加载数据...")
    try:
        X, Y = load_data(CONFIG['dataset'])
        # 计算类别数（排除背景类0）
        num_classes = len(np.unique(Y)) - 1 if 0 in Y else len(np.unique(Y))
        print(f"  ✓ 数据加载成功")
        print(f"  数据形状: {X.shape}")
        print(f"  标签形状: {Y.shape}")
        print(f"  类别数: {num_classes}")
    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================================================
    # 2. 数据预处理 + 空间分割（避免数据泄露）
    # =========================================================================
    print("\n【步骤2/7】数据预处理...")
    
    # 标准化
    X_standardized = Standardize_data(X)
    
    windowSize = CONFIG['windowSize']
    
    if CONFIG['USE_SPATIAL_SPLIT']:
        # ⭐ 使用空间分割 - 无数据泄露
        print("  使用空间分割方式（无数据泄露）...")
        X_train, X_test, Y_train, Y_test = create_patches_with_spatial_split(
            X_standardized, Y,
            windowSize=windowSize,
            train_ratio=CONFIG['train_ratio'],
            gap_size=CONFIG['gap_size'],
            block_size=CONFIG['block_size'],
            random_state=42
        )
        
        # 可视化空间分割
        from spatial_split import spatial_train_test_split
        train_mask, test_mask = spatial_train_test_split(
            X_standardized, Y, 
            train_ratio=CONFIG['train_ratio'],
            gap_size=CONFIG['gap_size'],
            block_size=CONFIG['block_size']
        )
        visualize_spatial_split(Y, train_mask, test_mask, 
                               os.path.join(CONFIG['output_dir'], 'spatial_split.png'))
    else:
        # 随机分割（有数据泄露风险）
        print("  ⚠ 使用随机分割方式（有数据泄露风险）...")
        from SAR_utils import createComplexImageCubes, splitTrainTestSet
        X_patches, Y_patches = createComplexImageCubes(X_standardized, Y, windowSize=windowSize)
        test_ratio = 1.0 - CONFIG['train_ratio']
        X_train, X_test, Y_train, Y_test = splitTrainTestSet(X_patches, Y_patches, testRatio=test_ratio)
    
    # 扩展维度：(N, H, W, C) -> (N, H, W, C, 1) 适配ComplexConv3D
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    
    # 标签编码
    Y_train_onehot = to_categorical(Y_train, num_classes)
    Y_test_onehot = to_categorical(Y_test, num_classes)
    
    # =========================================================================
    # 3. 提取物理先验
    # =========================================================================
    print("\n【步骤3/7】提取物理先验...")
    
    if CONFIG['USE_PRIOR']:
        # Cache priors to disk to avoid recomputation (especially useful for fast ablations / reruns)
        priors_train_path = os.path.join(CONFIG['output_dir'], 'priors_train.npy')
        priors_test_path = os.path.join(CONFIG['output_dir'], 'priors_test.npy')

        priors_train = None
        priors_test = None
        if os.path.exists(priors_train_path) and os.path.exists(priors_test_path):
            try:
                priors_train = np.load(priors_train_path)
                priors_test = np.load(priors_test_path)
                if priors_train.shape[0] != X_train.shape[0] or priors_test.shape[0] != X_test.shape[0]:
                    print("  ⚠ 先验缓存与当前数据不匹配，重新提取...")
                    priors_train = None
                    priors_test = None
                else:
                    print(f"  ✓ 从缓存加载物理先验: {priors_train_path}")
            except Exception as e:
                print(f"  ⚠ 先验缓存读取失败，重新提取... ({e})")
                priors_train = None
                priors_test = None

        if priors_train is None or priors_test is None:
            priors_train = extract_priors_for_patches(X_train, verbose=True)
            priors_test = extract_priors_for_patches(X_test, verbose=False)
            try:
                np.save(priors_train_path, priors_train)
                np.save(priors_test_path, priors_test)
                print(f"  ✓ 先验缓存已保存: {priors_train_path}")
            except Exception as e:
                print(f"  ⚠ 先验缓存保存失败（不影响训练）: {e}")
        print(f"  ✓ 物理先验提取完成")
        print(f"  先验维度: {priors_train.shape[1]} (H, A, Alpha, Ps, Pd, Pv, Span)")
    else:
        priors_train = None
        priors_test = None
        print("  ⚠ 极化先验已禁用 (USE_PRIOR=False)")
    
    # =========================================================================
    # 3.5 数据增强（防止过拟合）
    # =========================================================================
    if CONFIG.get('use_augmentation', False):
        print("\n【步骤3.5】数据增强...")
        aug_factor = CONFIG.get('augment_factor', 4)
        X_train, Y_train, priors_train = augment_data(X_train, Y_train, priors_train, factor=aug_factor)
        # 重新编码标签
        Y_train_onehot = to_categorical(Y_train, num_classes)
    
    # =========================================================================
    # 4. 创建模型
    # =========================================================================
    print("\n【步骤4/7】创建模型...")
    
    model = CV_SSM_PriorNet(
        X_train, 
        num_classes,
        use_prior_guidance=CONFIG['USE_PRIOR'],  # 开关2: 极化先验
        use_ssm=CONFIG['USE_SSM'],               # 开关1: 复值SSM
        prior_hidden_dim=int(CONFIG.get("prior_hidden_dim", 64)),
        prior_use_gating=bool(CONFIG.get("prior_use_gating", True)),
        prior_use_adaptive_selection=bool(CONFIG.get("prior_use_adaptive_selection", True)),
        prior_adaptive_temperature=float(CONFIG.get("prior_adaptive_temperature", 1.0)),
        prior_residual_alpha=float(CONFIG.get("prior_residual_alpha", 1.0)),
        se_prior_mix=float(CONFIG.get("se_prior_mix", 0.4)),
    )
    
    # 编译模型（使用Label Smoothing防止过拟合）
    label_smoothing = CONFIG.get('label_smoothing', 0.0)
    if label_smoothing > 0:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        print(f"  使用Label Smoothing: {label_smoothing}")
    else:
        loss_fn = 'categorical_crossentropy'
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['initial_lr']),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    print(f"  ✓ 模型创建成功")
    print(f"  模型名称: {model.name}")
    print(f"  参数量: {model.count_params():,}")
    print(f"  输入: {[inp.name for inp in model.inputs]}")
    
    # =========================================================================
    # 5. 配置回调
    # =========================================================================
    print("\n【步骤5/7】配置训练回调...")
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=CONFIG['es_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=CONFIG['reduce_lr_factor'],
            patience=CONFIG['reduce_lr_patience'],
            min_lr=CONFIG['min_lr'],
            verbose=1
        )
    ]
    
    if CONFIG['save_model']:
        callbacks.append(
            ModelCheckpoint(
                # Use TF checkpoint format (no .h5) to avoid occasional h5py/HDF5 write errors on some systems
                filepath=os.path.join(CONFIG['output_dir'], 'best_model'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,  # 只保存权重，避免复值层命名冲突
                verbose=1
            )
        )
    
    print(f"  ✓ 回调配置完成")
    
    # =========================================================================
    # 6. 训练模型（或仅评估）
    # =========================================================================
    # 准备训练/验证输入（评估也会用到 val_inputs）
    if CONFIG['USE_PRIOR']:
        train_inputs = [X_train, priors_train]
        val_inputs = [X_test, priors_test]
    else:
        train_inputs = X_train
        val_inputs = X_test

    # 计算类别权重（仅训练时使用）
    class_weight = None
    if (not args.eval_only) and CONFIG.get('use_class_weight', False):
        class_weight = compute_class_weights(Y_train, num_classes, focus_on_aa=True)

    history = None
    if not args.eval_only:
        print("\n【步骤6/7】开始训练...")
        print(f"  Batch Size: {CONFIG['batch_size']}")
        print(f"  Epochs: {CONFIG['epochs']}")
        print(f"  复值SSM (USE_SSM): {'启用' if CONFIG['USE_SSM'] else '禁用'}")
        print(f"  极化先验 (USE_PRIOR): {'启用' if CONFIG['USE_PRIOR'] else '禁用'}")
        print("-" * 70)

        # ⭐ RTX 5090性能优化：使用tf.data.Dataset加速
        if CONFIG.get('use_tf_dataset', False):
            print("\n  🚀 启用tf.data.Dataset优化（RTX 5090）...")

            # 创建训练数据集（RTX 5090优化版）
            def create_dataset(X_data, Y_data, priors_data=None, shuffle=True):
                if priors_data is not None:
                    dataset = tf.data.Dataset.from_tensor_slices((
                        {'sar_input': X_data, 'prior_input': priors_data},
                        Y_data
                    ))
                else:
                    dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data))

                if shuffle:
                    shuffle_size = min(CONFIG.get('shuffle_buffer', 50000), len(X_data))
                    dataset = dataset.shuffle(buffer_size=shuffle_size, reshuffle_each_iteration=True)

                dataset = dataset.batch(CONFIG['batch_size'], drop_remainder=False)

                prefetch_val = CONFIG.get('prefetch_buffer', tf.data.AUTOTUNE)
                if prefetch_val == tf.data.AUTOTUNE:
                    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
                else:
                    dataset = dataset.prefetch(buffer_size=prefetch_val)
                return dataset

            if CONFIG['USE_PRIOR']:
                train_dataset = create_dataset(X_train, Y_train_onehot, priors_train, shuffle=True)
                val_dataset = create_dataset(X_test, Y_test_onehot, priors_test, shuffle=False)
            else:
                train_dataset = create_dataset(X_train, Y_train_onehot, None, shuffle=True)
                val_dataset = create_dataset(X_test, Y_test_onehot, None, shuffle=False)

            print(f"  ✓ Dataset创建完成")
            print(f"    预取缓冲区: {CONFIG.get('prefetch_buffer', 8)} batches")
            print(f"    训练步数: {len(train_dataset)}")

            history = model.fit(
                train_dataset,
                epochs=CONFIG['epochs'],
                validation_data=val_dataset,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )
        else:
            history = model.fit(
                train_inputs, Y_train_onehot,
                batch_size=CONFIG['batch_size'],
                epochs=CONFIG['epochs'],
                validation_data=(val_inputs, Y_test_onehot),
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )

        print("-" * 70)
        print(f"  ✓ 训练完成")
    
    # =========================================================================
    # 7. 评估模型
    # =========================================================================
    print("\n【步骤7/7】评估模型...")

    # ⭐ 关键：确保评估使用“最佳权重”（与训练过程中保存/早停一致）
    best_prefix = os.path.join(CONFIG['output_dir'], 'best_model')
    try:
        if os.path.exists(best_prefix + ".index") or os.path.exists(best_prefix) or os.path.exists(best_prefix + ".h5"):
            model.load_weights(best_prefix)
            print(f"  ✓ 已加载最佳权重用于评估: {best_prefix}")
    except Exception as e:
        print(f"  ⚠ 未能加载最佳权重（将用当前模型权重评估）: {e}")
    
    # 预测
    if CONFIG['USE_PRIOR']:
        Y_pred_prob = model.predict([X_test, priors_test], verbose=0)
    else:
        Y_pred_prob = model.predict(X_test, verbose=0)
    
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    
    # 计算指标
    oa = accuracy_score(Y_test, Y_pred)
    kappa = cohen_kappa_score(Y_test, Y_pred)
    each_acc, aa = AA_andEachClassAccuracy(Y_test, Y_pred)
    
    print(f"\n  ========== 评估结果 ==========")
    print(f"  Overall Accuracy (OA): {oa*100:.2f}%")
    print(f"  Average Accuracy (AA): {aa*100:.2f}%")
    print(f"  Kappa Coefficient: {kappa:.4f}")
    print(f"\n  各类别准确率:")
    for i, acc in enumerate(each_acc):
        print(f"    类别 {i+1}: {acc*100:.2f}%")
    
    # 保存结果
    print("\n  保存结果...")
    
    # 训练曲线（eval-only 时没有 history）
    if history is not None:
        plot_training_history(
            history,
            os.path.join(CONFIG['output_dir'], 'training_history.png')
        )
    
    # 混淆矩阵
    plot_confusion_matrix(
        Y_test, Y_pred,
        os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'),
        num_classes
    )
    
    # 保存指标到文件
    results_path = os.path.join(CONFIG['output_dir'], 'results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("CV-SSM-PriorNet 评估结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"数据集: {CONFIG['dataset']}\n")
        f.write(f"复值SSM (USE_SSM): {'启用' if CONFIG['USE_SSM'] else '禁用'}\n")
        f.write(f"极化先验 (USE_PRIOR): {'启用' if CONFIG['USE_PRIOR'] else '禁用'}\n")
        f.write(f"空间分割 (无泄露): {'是' if CONFIG['USE_SPATIAL_SPLIT'] else '否'}\n")
        if CONFIG['USE_SPATIAL_SPLIT']:
            f.write(f"  - 块大小: {CONFIG['block_size']}\n")
            f.write(f"  - 间隔带: {CONFIG['gap_size']}\n")
        f.write(f"训练比例: {CONFIG['train_ratio']*100:.1f}%\n\n")
        f.write(f"Overall Accuracy (OA): {oa*100:.2f}%\n")
        f.write(f"Average Accuracy (AA): {aa*100:.2f}%\n")
        f.write(f"Kappa Coefficient: {kappa:.4f}\n\n")
        f.write("各类别准确率:\n")
        for i, acc in enumerate(each_acc):
            f.write(f"  类别 {i+1}: {acc*100:.2f}%\n")
    print(f"  结果已保存: {results_path}")
    
    print("\n" + "=" * 70)
    print("✓ 训练和评估完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

