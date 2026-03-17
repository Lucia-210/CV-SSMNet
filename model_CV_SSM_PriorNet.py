#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CV-SSM-PriorNet: 复值SSM + 极化先验引导的SAR图像分类网络

核心创新：
1. 复值SSM → 解决长程依赖与各向异性
2. 极化先验提示 → 增强物理可解释性与特征选择
3. 两者相互促进：
   - SSM捕获长程依赖后，极化先验引导网络关注物理有意义的特征
   - 极化先验通过物理知识引导SSM和卷积特征的选择

设计思路：
- 虽然SSM增强了长程建模能力，但网络仍可能关注统计相关但物理无意义的模式
- 因此，我们引入极化先验提示，通过H/A/Alpha、Pauli分解等物理参数引导特征选择

作者: CV-SSM-PriorNet Team
"""

import os
import numpy as np
import tensorflow as tf
import cvnn.layers as complex_layers
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input, Layer, BatchNormalization

# 导入复值SSM层
from complex_ssm_layer import (
    ComplexSSMLayer,
    LightweightComplexSSMLayer,
    construct_multi_directional_sequences,
)
# 导入极化先验提示模块
from film_layers import (
    PhysicalPriorConditioner,
    PriorEncoder,
    FiLMGenerator,
    ComplexFiLM,
    AdaptiveChannelSelection,
)


###############################################################################
# 辅助函数
###############################################################################

def GlobalCmplxAveragePooling3D(inputs):
    """复值3D全局平均池化"""
    inputs_r = tf.math.real(inputs)
    inputs_i = tf.math.imag(inputs)
    
    output_r = tf.keras.layers.GlobalAveragePooling3D()(inputs_r)
    output_i = tf.keras.layers.GlobalAveragePooling3D()(inputs_i)
    
    if inputs.dtype == 'complex' or inputs.dtype == 'complex64' or inputs.dtype == 'complex128':
        output = tf.complex(output_r, output_i)
    
    return output


def cmplx_SE_Block_3D(xin, se_ratio=8):
    """
    标准复值SE Block（不使用物理先验）
    """
    xin = tf.transpose(xin, perm=[0, 1, 2, 4, 3])
    xin_gap = GlobalCmplxAveragePooling3D(xin)
    sqz = complex_layers.ComplexDense(xin.shape[-1]//se_ratio, activation='cart_relu')(xin_gap)
    excite1 = complex_layers.ComplexDense(xin.shape[-1], activation='cart_sigmoid')(sqz)
    out = tf.keras.layers.multiply([xin, excite1])
    out = tf.transpose(out, perm=[0, 1, 2, 4, 3])
    return out


def cmplx_SE_Block_3D_with_prior(xin, priors=None, name='se_block', prior_mix=0.4):
    """
    带物理先验引导的复值SE Block
    
    ⭐ 核心改进：将物理先验融入SE的激励路径，动态调整通道注意力权重
    这样可以让网络根据物理知识选择重要的通道，增强物理可解释性
    
    参数:
        xin: 输入特征 (batch, H, W, D, C) - complex
        se_ratio: SE压缩比
        priors: 物理先验 (batch, 7) - float32
        name: 层名称
        prior_mix: 先验融合强度，范围[0,1]。
            - 0: 不使用先验（退化为普通SE路径）
            - 1: 最大程度使用先验门控
    """
    xin = tf.transpose(xin, perm=[0, 1, 2, 4, 3])
    xin_gap = GlobalCmplxAveragePooling3D(xin)
    
    channels = int(xin.shape[-1]) if xin.shape[-1] is not None else tf.shape(xin)[-1]
    reduced_channels = max(int(channels) // se_ratio, 1) if isinstance(channels, int) else channels // se_ratio
    
    sqz = complex_layers.ComplexDense(reduced_channels, activation='cart_relu', name=f'{name}_squeeze')(xin_gap)
    excite = complex_layers.ComplexDense(channels, activation='cart_sigmoid', name=f'{name}_excite')(sqz)
    

    if priors is not None:
        prior_encoder = Dense(64, activation='relu', name=f'{name}_prior_enc')  # 增强：32→64
        prior_gate = Dense(channels, activation='sigmoid', name=f'{name}_prior_gate')
        
        prior_feat = prior_encoder(priors)
        prior_weight = prior_gate(prior_feat)
        
        excite_real = tf.math.real(excite)
        excite_imag = tf.math.imag(excite)
        prior_weight = tf.cast(prior_weight, excite_real.dtype)
        
        # 加权融合：(1-prior_mix) * SE + prior_mix * PriorGate（先验强度可调，便于不同数据集稳定收益）
        prior_mix = float(np.clip(prior_mix, 0.0, 1.0))
        excite_real = excite_real * ((1.0 - prior_mix) + prior_mix * prior_weight)
        excite_imag = excite_imag * ((1.0 - prior_mix) + prior_mix * prior_weight)
        excite = tf.complex(excite_real, excite_imag)
    
    excite_expanded = tf.expand_dims(excite, axis=1)
    excite_expanded = tf.expand_dims(excite_expanded, axis=1)
    excite_expanded = tf.expand_dims(excite_expanded, axis=1)
    
    out = tf.keras.layers.multiply([xin, excite_expanded])
    out = tf.transpose(out, perm=[0, 1, 2, 3])
    return out


###############################################################################
# 主模型
###############################################################################

def CV_SSM_PriorNet(
    X_cmplx,
    num_classes,
    use_prior_guidance=False
    use_ssm=True,
    # ---- Prior conditioning knobs (dataset-sensitive) ----
    prior_hidden_dim=64,
    prior_use_gating=True,
    prior_use_adaptive_selection=True,
    prior_adaptive_temperature=1.0,
    prior_residual_alpha=1.0,
    se_prior_mix=0.4,
):
    """
    CV-SSM-PriorNet: 复值SSM + 极化先验引导的SAR图像分类网络
    
    参数:
        X_cmplx: 输入数据形状参考 (用于确定输入维度)
        num_classes: 分类类别数
        use_prior_guidance: 是否使用极化先验引导（默认True）
        use_ssm: 是否使用复值SSM（默认True）
    
    返回:
        model: Keras模型
            - 如果use_prior_guidance=True: 双输入 [SAR数据, 物理先验(7维)]
            - 如果use_prior_guidance=False: 单输入 [SAR数据]
    
    核心设计：
    1. 复值SSM捕获长程依赖与各向异性
    2. 极化先验通过FiLM机制引导SSM和卷积特征的选择
    3. 两者相互促进，提升模型的物理可解释性和分类性能
    """
    # 环境变量控制
    use_cvssm = use_ssm and os.environ.get("DISABLE_CVSSM", "0") != "1"
    use_light_ssm = use_ssm and os.environ.get("DISABLE_LIGHT_SSM", "0") != "1"

    # =========================================================================
    # 输入层
    # =========================================================================
    cmplx_inputs = complex_layers.complex_input(shape=(X_cmplx.shape[1:]), name='sar_input')
    
    if use_prior_guidance:
        # 物理先验输入（7维：H, A, Alpha, Ps, Pd, Pv, Span）
        prior_inputs = Input(shape=(7,), dtype='float32')
    
    # =========================================================================
    # 三条分支卷积路径
    # =========================================================================
    # Shallow Path (浅层路径)
    c0 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(3,3,3), 
                                       padding="same", name='shallow_conv')(cmplx_inputs)
    
    # Mid Path (中层路径)
    c1 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(3,3,3), 
                                       padding="same", name='mid_conv1')(cmplx_inputs)
    c1 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(3,3,3), 
                                       padding="same", name='mid_conv2')(c1)

    # Deep Path (深层路径)
    c2 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(3,3,3), 
                                       padding="same", name='deep_conv1')(cmplx_inputs)
    c2 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(9,9,9), 
                                       padding="same", name='deep_conv2')(c2)
    c2 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(6,6,6), 
                                       padding="same", name='deep_conv3')(c2)

    # =========================================================================
    # CV-SSM + 极化先验引导 (核心创新)
    # =========================================================================
    if use_cvssm:
        def _apply_branch_cvssm_with_prior(branch_feat, name_prefix: str, priors=None):
            """
            应用SSM增强，并使用极化先验引导特征选择
            
            设计思路：
            1. SSM捕获长程依赖关系
            2. 极化先验通过FiLM条件化引导SSM输出，选择物理有意义的特征
            """
            channels = int(branch_feat.shape[-1]) if branch_feat.shape[-1] is not None else 16
            
            # Step 1: SSM捕获长程依赖
            seqs = construct_multi_directional_sequences(branch_feat, directions=['row', 'col'])
            ssm_outs = []
            for dir_name, seq in seqs:
                ssm = ComplexSSMLayer(d_model=channels,
                                      d_state=max(16, channels),
                                      spectral_radius=0.73,
                                      use_low_rank=True, 
                                      name=f"{name_prefix}_cvssm_{dir_name}")
                y = ssm(seq)
                shp = tf.shape(branch_feat)
                b, h, w, d = shp[0], shp[1], shp[2], shp[3]
                y_spatial = tf.reshape(y, [b, h, w, d, channels])
                ssm_outs.append(y_spatial)
            fused = tf.add_n(ssm_outs) / float(len(ssm_outs))
            
            # Step 2: ⭐ 极化先验引导SSM特征选择（增强版）
            if use_prior_guidance and priors is not None:
                prior_conditioner = PhysicalPriorConditioner(
                    num_channels=channels,
                    hidden_dim=int(prior_hidden_dim),
                    use_gating=bool(prior_use_gating),
                    use_adaptive_selection=bool(prior_use_adaptive_selection),
                    adaptive_temperature=float(prior_adaptive_temperature),
                    residual_alpha=float(prior_residual_alpha),
                    name=f"{name_prefix}_ssm_prior_cond",
                )
                fused = prior_conditioner([fused, priors])
            
            # Step 3: 门控残差连接
            gap_real = tf.reduce_mean(tf.math.real(branch_feat), axis=[1, 2, 3], keepdims=True)
            gap_imag = tf.reduce_mean(tf.math.imag(branch_feat), axis=[1, 2, 3], keepdims=True)
            gap = tf.complex(gap_real, gap_imag)
            gate = complex_layers.ComplexDense(channels, activation='cart_sigmoid', 
                                               name=f"{name_prefix}_cvssm_gate")(gap)
            gate = tf.reshape(gate, [-1, 1, 1, 1, channels])
            gated_ssm = gate * fused
            enhanced = tf.keras.layers.Add(name=f"{name_prefix}_cvssm_res")([branch_feat, gated_ssm])
            return enhanced

        if use_prior_guidance:
            c0 = _apply_branch_cvssm_with_prior(c0, "branch_shallow", prior_inputs)
            c1 = _apply_branch_cvssm_with_prior(c1, "branch_mid", prior_inputs)
            c2 = _apply_branch_cvssm_with_prior(c2, "branch_deep", prior_inputs)
        else:
            c0 = _apply_branch_cvssm_with_prior(c0, "branch_shallow", None)
            c1 = _apply_branch_cvssm_with_prior(c1, "branch_mid", None)
            c2 = _apply_branch_cvssm_with_prior(c2, "branch_deep", None)
    
    # =========================================================================
    # 特征拼接 + 极化先验引导
    # =========================================================================
    features_concat = tf.concat([c0, c1, c2], axis=4, name='features_concat')
    
    if use_prior_guidance:
        total_channels = int(features_concat.shape[-1]) if features_concat.shape[-1] is not None else 48
        concat_conditioner = PhysicalPriorConditioner(
            num_channels=total_channels,
            hidden_dim=int(prior_hidden_dim),
            use_gating=bool(prior_use_gating),
            use_adaptive_selection=bool(prior_use_adaptive_selection),
            adaptive_temperature=float(prior_adaptive_temperature),
            residual_alpha=float(prior_residual_alpha),
            name="concat_prior_cond"
        )
        features_concat = concat_conditioner([features_concat, prior_inputs])
    
    # =========================================================================
    # SE注意力块 + 物理先验引导
    # =========================================================================
    if use_prior_guidance:
        se = cmplx_SE_Block_3D_with_prior(
            features_concat,
            se_ratio=8,
            priors=prior_inputs,
            name='se_block_1',
            prior_mix=float(se_prior_mix),
        )
    else:
        se = cmplx_SE_Block_3D(features_concat, se_ratio=8)
    
    # =========================================================================
    # 轻量级CV-SSM + 极化先验引导
    # =========================================================================
    if use_light_ssm:
        total_channels = int(se.shape[-1]) if se.shape[-1] is not None else 48
        lw_ssm = LightweightComplexSSMLayer(
            d_model=total_channels,
            d_state=max(8, total_channels // 2),
            spectral_radius=0.95,
            name="lightweight_cvssm"
        )
        se_lw = lw_ssm(se)
        
        if use_prior_guidance:
            lw_conditioner = PhysicalPriorConditioner(
                num_channels=total_channels,
                hidden_dim=int(prior_hidden_dim),
                use_gating=bool(prior_use_gating),
                # 轻量路径默认关闭自适应选择，避免过强稀疏化导致不稳定
                use_adaptive_selection=False,
                adaptive_temperature=float(prior_adaptive_temperature),
                residual_alpha=float(prior_residual_alpha),
                name="lightweight_ssm_prior_cond"
            )
            se_lw = lw_conditioner([se_lw, prior_inputs])
        
        gap_real = tf.reduce_mean(tf.math.real(se), axis=[1, 2, 3], keepdims=True)
        gap_imag = tf.reduce_mean(tf.math.imag(se), axis=[1, 2, 3], keepdims=True)
        gap = tf.complex(gap_real, gap_imag)
        gate = complex_layers.ComplexDense(total_channels, activation='cart_sigmoid',
                                           name="lightweight_cvssm_gate")(gap)
        gate = tf.reshape(gate, [-1, 1, 1, 1, total_channels])
        gated_lw = gate * se_lw
        se = tf.keras.layers.Add(name="lightweight_cvssm_res")([se, gated_lw])
    
    # 后续SE块
    if use_prior_guidance:
        se = cmplx_SE_Block_3D_with_prior(
            se, se_ratio=8, priors=prior_inputs, name='se_block_2', prior_mix=float(se_prior_mix)
        )
        se = cmplx_SE_Block_3D_with_prior(
            se, se_ratio=8, priors=prior_inputs, name='se_block_3', prior_mix=float(se_prior_mix)
        )
    else:
        se = cmplx_SE_Block_3D(se, se_ratio=8)
        se = cmplx_SE_Block_3D(se, se_ratio=8)

    # =========================================================================
    # 分类头
    # =========================================================================
    features_flat = complex_layers.ComplexFlatten(name='flatten')(se)
    
    c3 = complex_layers.ComplexDense(128, activation='cart_relu', name='dense_1')(features_flat)
    c3 = complex_layers.ComplexDropout(0.25, name='dropout_1')(c3)
    c4 = complex_layers.ComplexDense(64, activation='cart_relu', name='dense_2')(c3)
    c4 = complex_layers.ComplexDropout(0.25, name='dropout_2')(c4)
    
    predict = complex_layers.ComplexDense(num_classes, activation="softmax_real_with_abs", name='output')(c4)

    # =========================================================================
    # 构建模型
    # =========================================================================
    # 根据开关状态生成模型名称
    name_parts = ['CV']
    if use_ssm:
        name_parts.append('SSM')
    if use_prior_guidance:
        name_parts.append('Prior')
    if not use_ssm and not use_prior_guidance:
        name_parts.append('Base')
    model_name = '_'.join(name_parts) + 'Net'
    
    if use_prior_guidance:
        model = tf.keras.Model(inputs=[cmplx_inputs, prior_inputs], outputs=predict, 
                               name=model_name)
    else:
        model = tf.keras.Model(inputs=[cmplx_inputs], outputs=predict, 
                               name=model_name)
    
    return model


###############################################################################
# 向后兼容的别名
###############################################################################

def ASDF2Net(X_cmplx, num_classes, use_prior_guidance=True):
    """向后兼容的别名"""
    return CV_SSM_PriorNet(X_cmplx, num_classes, use_prior_guidance=use_prior_guidance)


###############################################################################
# 测试代码
###############################################################################

if __name__ == "__main__":
    print("=" * 70)
    print("CV-SSM-PriorNet 模型测试")
    print("复值SSM + 极化先验引导的SAR图像分类网络")
    print("=" * 70)
    
    batch_size = 4
    H, W, D, C = 13, 13, 9, 1
    num_classes = 15
    
    X_test = np.random.rand(batch_size, H, W, D, C).astype(np.float32) + 0.1
    priors_test = np.random.rand(batch_size, 7).astype(np.float32)
    
    print(f"\n输入SAR数据形状: {X_test.shape}")
    print(f"物理先验形状: {priors_test.shape}")
    
    # 测试使用极化先验的模型
    print("\n创建 CV-SSM-PriorNet (使用极化先验引导)...")
    model = CV_SSM_PriorNet(X_test, num_classes, use_prior_guidance=True)
    print(f"✓ 模型创建成功")
    print(f"  输入: {[inp.name for inp in model.inputs]}")
    print(f"  参数量: {model.count_params():,}")
    
    output = model.predict([X_test, priors_test], verbose=0)
    print(f"  输出形状: {output.shape}")
    print(f"  ✓ 前向传播成功")
    
    print("\n" + "=" * 70)
    print("✓ CV-SSM-PriorNet 测试通过!")
    print("=" * 70)

