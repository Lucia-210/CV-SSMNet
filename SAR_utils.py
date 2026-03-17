# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:21:37 2022

@author: malkhatib
"""
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
from operator import truediv
import random 
from sklearn.utils import shuffle




def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def spatial_train_test_split(X, y, windowSize=13, train_ratio=0.1, buffer_size=None, random_state=42):
    """
    空间分割方法 - 消除数据泄露
    
    将图像划分为网格块，整块分配给训练或测试集，
    并在训练区和测试区之间留出间隔带(buffer)避免重叠。
    
    参数:
        X: 原始图像数据 (H, W, C)
        y: 标签图 (H, W)
        windowSize: patch窗口大小
        train_ratio: 训练数据比例
        buffer_size: 间隔带大小，默认为windowSize
        random_state: 随机种子
    
    返回:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    
    if buffer_size is None:
        buffer_size = windowSize  # 间隔带至少等于窗口大小，确保无重叠
    
    H, W, C = X.shape
    margin = windowSize // 2
    
    # 定义网格块大小（比窗口大，减少边界效应）
    block_size = max(50, windowSize * 3)  # 每个块至少50x50或3倍窗口
    
    # 计算网格
    n_blocks_h = H // block_size
    n_blocks_w = W // block_size
    
    if n_blocks_h < 2 or n_blocks_w < 2:
        print("  ⚠ 图像太小，使用条带分割")
        return _strip_split(X, y, windowSize, train_ratio, buffer_size, margin)
    
    # 创建块索引
    block_indices = [(i, j) for i in range(n_blocks_h) for j in range(n_blocks_w)]
    np.random.shuffle(block_indices)
    
    # 分配块到训练/测试
    n_train_blocks = max(1, int(len(block_indices) * train_ratio))
    train_blocks = set(block_indices[:n_train_blocks])
    test_blocks = set(block_indices[n_train_blocks:])
    
    print(f"  空间分割: {n_blocks_h}x{n_blocks_w}={len(block_indices)} 块")
    print(f"  训练块: {len(train_blocks)}, 测试块: {len(test_blocks)}")
    print(f"  间隔带大小: {buffer_size} 像素")
    
    # Pad图像
    X_padded = padWithZeros(X, margin=margin)
    
    train_patches = []
    train_labels = []
    test_patches = []
    test_labels = []
    
    # 遍历每个像素
    for r in range(margin, H + margin):
        for c in range(margin, W + margin):
            orig_r, orig_c = r - margin, c - margin
            label = y[orig_r, orig_c]
            
            if label == 0:  # 跳过背景
                continue
            
            # 确定该像素属于哪个块
            block_i = min(orig_r // block_size, n_blocks_h - 1)
            block_j = min(orig_c // block_size, n_blocks_w - 1)
            block_idx = (block_i, block_j)
            
            # 检查是否在间隔带内（块边界附近）
            in_buffer = _is_in_buffer(orig_r, orig_c, block_size, buffer_size, 
                                       train_blocks, test_blocks, n_blocks_h, n_blocks_w)
            
            if in_buffer:
                continue  # 跳过间隔带内的像素
            
            # 提取patch
            patch = X_padded[r - margin:r + margin + 1, c - margin:c + margin + 1]
            
            if block_idx in train_blocks:
                train_patches.append(patch)
                train_labels.append(label - 1)  # 标签从0开始
            else:
                test_patches.append(patch)
                test_labels.append(label - 1)
    
    X_train = np.array(train_patches, dtype=np.complex64)
    y_train = np.array(train_labels)
    X_test = np.array(test_patches, dtype=np.complex64)
    y_test = np.array(test_labels)
    
    print(f"  训练样本: {len(y_train)}, 测试样本: {len(y_test)}")
    print(f"  间隔带丢弃: {(H*W - len(y_train) - len(y_test) - np.sum(y==0))} 像素")
    
    return X_train, X_test, y_train, y_test


def _is_in_buffer(r, c, block_size, buffer_size, train_blocks, test_blocks, n_blocks_h, n_blocks_w):
    """检查像素是否在间隔带内（训练块和测试块的边界）"""
    block_i = min(r // block_size, n_blocks_h - 1)
    block_j = min(c // block_size, n_blocks_w - 1)
    current_block = (block_i, block_j)
    
    # 当前块在训练集还是测试集
    current_in_train = current_block in train_blocks
    
    # 检查相邻块
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = block_i + di, block_j + dj
            if 0 <= ni < n_blocks_h and 0 <= nj < n_blocks_w:
                neighbor_block = (ni, nj)
                neighbor_in_train = neighbor_block in train_blocks
                
                # 如果当前块和相邻块分属不同集合
                if current_in_train != neighbor_in_train:
                    # 计算到块边界的距离
                    block_r_start = block_i * block_size
                    block_r_end = (block_i + 1) * block_size
                    block_c_start = block_j * block_size
                    block_c_end = (block_j + 1) * block_size
                    
                    dist_to_edge = min(
                        r - block_r_start,
                        block_r_end - r - 1,
                        c - block_c_start,
                        block_c_end - c - 1
                    )
                    
                    if dist_to_edge < buffer_size // 2:
                        return True
    return False


def _strip_split(X, y, windowSize, train_ratio, buffer_size, margin):
    """条带分割 - 用于小图像"""
    H, W, C = X.shape
    
    # 垂直分割为条带
    train_end = int(W * train_ratio)
    buffer_end = train_end + buffer_size
    
    X_padded = padWithZeros(X, margin=margin)
    
    train_patches, train_labels = [], []
    test_patches, test_labels = [], []
    
    for r in range(margin, H + margin):
        for c in range(margin, W + margin):
            orig_r, orig_c = r - margin, c - margin
            label = y[orig_r, orig_c]
            
            if label == 0:
                continue
            
            patch = X_padded[r - margin:r + margin + 1, c - margin:c + margin + 1]
            
            if orig_c < train_end:
                train_patches.append(patch)
                train_labels.append(label - 1)
            elif orig_c >= buffer_end:
                test_patches.append(patch)
                test_labels.append(label - 1)
            # buffer区域内的像素被丢弃
    
    return (np.array(train_patches, dtype=np.complex64), 
            np.array(test_patches, dtype=np.complex64),
            np.array(train_labels), 
            np.array(test_labels))


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]),dtype=('complex64'))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createComplexImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=('complex64'))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def target(name):
    if name == 'FL_T':
        target_names = ['Unassigned', 'Water', 'Forest', 'Lucerne', 'Grass', 'Rapeseed',
                        'Beet', 'Potatoes', 'Peas', 'Stem Beans', 'Bare Soil', 'Wheat', 'Wheat 2', 
                        'Wheat 3', 'Barley', 'Buildings']
    elif name == 'SF':
        target_names = ['Unassigned', 'Bare Soil', 'Mountain', 'Water', 'Urban', 'Vegetation']

    elif name == 'ober':
        target_names = ['Unassigned', 'Build-Up Areas', 'Wood Land', 'Open Areas']
        
    return target_names 
    
def num_classes(dataset):
    if dataset == 'FL_T':
        output_units = 15
    elif dataset == 'SF':
        output_units = 5
    elif dataset == 'ober':
        output_units = 3
    
    return output_units




def Patch(data,height_index,width_index, PATCH_SIZE):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch

def getTrainTestSplit(X_cmplx, X_rgb, y, pxls_num):
    if type(pxls_num) != list:
        pxls_num = [pxls_num]*len(np.unique(y))
        
    if len(np.unique(y)) != len(pxls_num):
        print("length of pixels list doen't match the number of classes in the dataset")
        return
    else:
        xTrain_cmplx = []
        xTrain_rgb = []
        yTrain = []
        
        xTest_cmplx  = []
        xTest_rgb  = []
        yTest  = []
        for i in range(len(np.unique(y))):
            if pxls_num[i] > len(y[y==i]):
                print("Number of training pixles is larger than total class pixels")
                return
            else:
                random.seed(321) #optional to reproduce the data
                samples = random.sample(range(len(y[y==i])), pxls_num[i])
                xTrain_cmplx.extend(X_cmplx[y==i][samples])
                xTrain_rgb.extend(X_rgb[y==i][samples])
                yTrain.extend(y[y==i][samples])
                
                tmp1 = list(X_cmplx[y==i])
                tmp2 = list(X_rgb[y==i])
                tmp3 = list(y[y==i])
                for ele in sorted(samples, reverse = True):
                    del tmp1[ele]
                    del tmp2[ele]
                    del tmp3[ele]

                xTest_cmplx.extend(tmp1)
                xTest_rgb.extend(tmp2)
                yTest.extend(tmp3)
     
  
    xTrain_cmplx, xTrain_rgb, yTrain = shuffle(xTrain_cmplx, xTrain_rgb, yTrain, random_state=321)  
    xTest_cmplx, xTest_rgb, yTest = shuffle(xTest_cmplx, xTest_rgb, yTest, random_state=345)
    
    #xTrain_rgb, yTrain = shuffle(xTrain_rgb, yTrain, random_state=321)  
    #xTest_rgb, yTest = shuffle(xTest_rgb, yTest, random_state=345)
    
    
    
    xTrain_cmplx = np.array(xTrain_cmplx)
    xTrain_rgb = np.array(xTrain_rgb)
    yTrain = np.array(yTrain)
    
    xTest_cmplx = np.array(xTest_cmplx)
    xTest_rgb = np.array(xTest_rgb)
    yTest = np.array(yTest)
    
      
    return xTrain_cmplx, xTrain_rgb, yTrain, xTest_cmplx, xTest_rgb, yTest
        
        
    
try:
    import cvnn.layers as complex_layers
    CVNN_AVAILABLE = True
except ImportError:
    # 允许在未安装 cvnn 的环境中导入本模块（例如只运行可视化/数据处理）。
    # 依赖复值神经网络层的函数会在调用时给出明确错误提示。
    complex_layers = None
    CVNN_AVAILABLE = False
def cmplx_SE_Block(xin, se_ratio = 8):
    if not CVNN_AVAILABLE or complex_layers is None:
        raise ImportError(
            "缺少依赖：cvnn。当前环境无法使用 cmplx_SE_Block（复值SE块）。"
            "请安装 cvnn，或在不需要复值网络层的场景下避免调用该函数。"
        )
    # Squeeze Path
    xin_gap =  GlobalCmplxAveragePooling2D(xin)
    sqz = complex_layers.ComplexDense(xin.shape[-1]//se_ratio, activation='cart_relu')(xin_gap)
    
    # Excitation Path
    excite1 = complex_layers.ComplexDense(xin.shape[-1], activation='cart_sigmoid')(sqz)
    
    out = tf.keras.layers.multiply([xin, excite1])
    
    return out
    
   

import tensorflow as tf
def GlobalCmplxAveragePooling2D(inputs):
    inputs_r = tf.math.real(inputs)
    inputs_i = tf.math.imag(inputs)
    
    output_r = tf.keras.layers.GlobalAveragePooling2D()(inputs_r)
    output_i = tf.keras.layers.GlobalAveragePooling2D()(inputs_i)
    
    if inputs.dtype == 'complex' or inputs.dtype == 'complex64':
           output = tf.complex(output_r, output_i)
    else:
           output = output_r
    
    return output




def Standardize_data(X):
    new_X = np.zeros(X.shape, dtype=(X.dtype))
    _,_,c = X.shape
    for i in range(c):
        new_X[:,:,i] = (X[:,:,i] - np.mean(X[:,:,i])) / np.std(X[:,:,i])
        
    return new_X
        
        









