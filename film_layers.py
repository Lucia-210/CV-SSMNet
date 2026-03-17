"""
FiLM (Feature-wise Linear Modulation) 条件化层
用于将物理先验注入神经网络特征

核心思想:
- 从物理先验生成缩放(gamma)和平移(beta)参数
- 对特征进行仿射变换: y = gamma * x + beta
- 支持复值特征的实部/虚部分别调制

参考文献:
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018

作者: ASDF2Net-FiLM扩展
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class PriorEncoder(layers.Layer):
    """
    物理先验编码器
    将7维物理先验向量编码为条件向量
    """
    def __init__(self, hidden_dim=32, output_dim=64, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.dense1 = layers.Dense(self.hidden_dim, activation='relu', name='prior_enc_1')
        self.bn1 = layers.BatchNormalization(name='prior_bn_1')
        self.dropout1 = layers.Dropout(self.dropout_rate)
        
        self.dense2 = layers.Dense(self.hidden_dim * 2, activation='relu', name='prior_enc_2')
        self.bn2 = layers.BatchNormalization(name='prior_bn_2')
        self.dropout2 = layers.Dropout(self.dropout_rate)
        
        self.dense_out = layers.Dense(self.output_dim, activation='tanh', name='prior_enc_out')
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        return self.dense_out(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


class FiLMGenerator(layers.Layer):
    """
    FiLM参数生成器
    从条件向量生成gamma和beta参数
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        
    def build(self, input_shape):
        condition_dim = input_shape[-1]
        
        # Gamma生成器 (缩放参数)
        self.gamma_dense = layers.Dense(
            self.num_channels, 
            activation='sigmoid',  # 输出范围 [0, 2] 通过后处理
            kernel_initializer='zeros',
            bias_initializer='ones',
            name='gamma_gen'
        )
        
        # Beta生成器 (平移参数)
        self.beta_dense = layers.Dense(
            self.num_channels,
            activation='tanh',  # 输出范围 [-1, 1]
            kernel_initializer='zeros',
            bias_initializer='zeros',
            name='beta_gen'
        )
        
        super().build(input_shape)
        
    def call(self, condition):
        # Gamma: [0.5, 1.5] 范围，初始化接近1
        gamma = self.gamma_dense(condition) + 0.5
        
        # Beta: [-0.5, 0.5] 范围，初始化为0
        beta = self.beta_dense(condition) * 0.5
        
        return gamma, beta
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_channels': self.num_channels})
        return config


class ComplexFiLM(layers.Layer):
    """
    复值FiLM层
    对复值特征的实部和虚部分别应用FiLM调制
    
    公式:
        real_out = gamma_r * real_in + beta_r
        imag_out = gamma_i * imag_in + beta_i
        output = complex(real_out, imag_out)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        # input_shape 是列表: [features_shape, gamma_shape, beta_shape]
        super().build(input_shape)
        
    def call(self, inputs):
        """
        参数:
            inputs: [features, gamma, beta]
                - features: 复值特征 (batch, ..., channels)
                - gamma: 缩放参数 (batch, channels)
                - beta: 平移参数 (batch, channels)
        """
        features, gamma, beta = inputs
        
        # 分离实部和虚部
        if features.dtype in [tf.complex64, tf.complex128]:
            real_part = tf.math.real(features)
            imag_part = tf.math.imag(features)
            is_complex = True
        else:
            real_part = features
            imag_part = None
            is_complex = False
        
        # 扩展gamma和beta的维度以匹配特征
        # features: (batch, H, W, D, C) 或 (batch, C)
        feature_rank = len(features.shape)
        
        if feature_rank == 5:  # 3D卷积特征
            gamma = tf.reshape(gamma, [-1, 1, 1, 1, gamma.shape[-1]])
            beta = tf.reshape(beta, [-1, 1, 1, 1, beta.shape[-1]])
        elif feature_rank == 4:  # 2D卷积特征
            gamma = tf.reshape(gamma, [-1, 1, 1, gamma.shape[-1]])
            beta = tf.reshape(beta, [-1, 1, 1, beta.shape[-1]])
        elif feature_rank == 3:  # 序列特征
            gamma = tf.reshape(gamma, [-1, 1, gamma.shape[-1]])
            beta = tf.reshape(beta, [-1, 1, beta.shape[-1]])
        # rank == 2: 已经是正确形状
        
        # 应用FiLM调制
        real_out = gamma * real_part + beta
        
        if is_complex:
            imag_out = gamma * imag_part + beta
            output = tf.complex(real_out, imag_out)
        else:
            output = real_out
            
        return output


class GatedFusion(layers.Layer):
    """
    门控融合层
    使用物理先验条件化的门控机制融合特征
    
    公式:
        gate = sigmoid(W_g * condition)
        output = gate * features + (1 - gate) * prior_features
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        
    def build(self, input_shape):
        self.gate_dense = layers.Dense(
            self.num_channels,
            activation='sigmoid',
            kernel_initializer='glorot_uniform',
            bias_initializer=tf.constant_initializer(0.5),  # 初始化为0.5
            name='gate'
        )
        super().build(input_shape)
        
    def call(self, inputs):
        """
        参数:
            inputs: [features, condition]
        """
        features, condition = inputs
        
        # 生成门控权重
        gate = self.gate_dense(condition)
        
        # 扩展维度
        feature_rank = len(features.shape)
        if feature_rank == 5:
            gate = tf.reshape(gate, [-1, 1, 1, 1, gate.shape[-1]])
        elif feature_rank == 4:
            gate = tf.reshape(gate, [-1, 1, 1, gate.shape[-1]])
        elif feature_rank == 3:
            gate = tf.reshape(gate, [-1, 1, gate.shape[-1]])
        
        # 门控调制
        if features.dtype in [tf.complex64, tf.complex128]:
            real_part = tf.math.real(features)
            imag_part = tf.math.imag(features)
            
            # 使用门控权重调制
            gate_float = tf.cast(gate, real_part.dtype)
            real_out = gate_float * real_part
            imag_out = gate_float * imag_part
            
            output = tf.complex(real_out, imag_out)
        else:
            output = gate * features
            
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_channels': self.num_channels})
        return config


class AdaptiveChannelSelection(layers.Layer):
    """
    自适应通道选择层
    根据物理先验动态选择重要通道，抑制冗余通道
    
    解决问题: 降低热力图中的"全局亮带"与通道冗余
    """
    def __init__(self, num_channels, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.temperature = temperature
        
    def build(self, input_shape):
        # 通道重要性评分网络
        self.importance_net = keras.Sequential([
            layers.Dense(self.num_channels, activation='relu'),
            layers.Dense(self.num_channels, activation=None)
        ], name='importance_net')
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """
        参数:
            inputs: [features, condition]
        """
        features, condition = inputs
        
        # 计算通道重要性分数
        importance_logits = self.importance_net(condition)
        
        # 使用带温度的softmax
        importance_weights = tf.nn.softmax(importance_logits / self.temperature, axis=-1)
        
        # 添加稀疏性正则化 (训练时)
        if training:
            # L1稀疏性惩罚
            sparsity_loss = tf.reduce_mean(tf.abs(importance_weights))
            self.add_loss(0.01 * sparsity_loss)
        
        # 扩展维度并应用权重
        feature_rank = len(features.shape)
        if feature_rank == 5:
            weights = tf.reshape(importance_weights, [-1, 1, 1, 1, self.num_channels])
        elif feature_rank == 4:
            weights = tf.reshape(importance_weights, [-1, 1, 1, self.num_channels])
        elif feature_rank == 3:
            weights = tf.reshape(weights, [-1, 1, self.num_channels])
        else:
            weights = importance_weights
        
        # 通道加权
        if features.dtype in [tf.complex64, tf.complex128]:
            real_part = tf.math.real(features)
            imag_part = tf.math.imag(features)
            
            weights_float = tf.cast(weights, real_part.dtype)
            real_out = weights_float * real_part * self.num_channels  # 保持能量
            imag_out = weights_float * imag_part * self.num_channels
            
            output = tf.complex(real_out, imag_out)
        else:
            output = weights * features * self.num_channels
            
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_channels': self.num_channels,
            'temperature': self.temperature
        })
        return config


class PhysicalPriorConditioner(layers.Layer):
    """
    完整的物理先验条件化模块
    整合编码器、FiLM生成器和调制
    """
    def __init__(self, 
                 num_channels,
                 prior_dim=7,
                 hidden_dim=32,
                 use_gating=True,
                 use_adaptive_selection=True,
                 adaptive_temperature=1.0,
                 residual_alpha=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.prior_dim = prior_dim
        self.hidden_dim = hidden_dim
        self.use_gating = use_gating
        self.use_adaptive_selection = use_adaptive_selection
        self.adaptive_temperature = adaptive_temperature
        # residual_alpha: 0=不改变特征；1=完全使用调制结果；介于其间=温和注入（SF更稳定）
        self.residual_alpha = residual_alpha
        
    def build(self, input_shape):
        # 先验编码器
        self.encoder = PriorEncoder(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim * 2,
            name='prior_encoder'
        )
        
        # FiLM参数生成器
        self.film_gen = FiLMGenerator(
            num_channels=self.num_channels,
            name='film_generator'
        )
        
        # FiLM调制层
        self.film_layer = ComplexFiLM(name='film_modulation')
        
        # 可选: 门控融合
        if self.use_gating:
            self.gating = GatedFusion(
                num_channels=self.num_channels,
                name='gated_fusion'
            )
        
        # 可选: 自适应通道选择
        if self.use_adaptive_selection:
            self.channel_selection = AdaptiveChannelSelection(
                num_channels=self.num_channels,
                temperature=float(self.adaptive_temperature),
                name='adaptive_channel_selection'
            )
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """
        参数:
            inputs: [features, priors]
                - features: 网络特征 (batch, ...)
                - priors: 物理先验 (batch, 7)
        返回:
            conditioned_features: 条件化后的特征
        """
        features, priors = inputs
        
        # 编码物理先验
        condition = self.encoder(priors, training=training)
        
        # 生成FiLM参数
        gamma, beta = self.film_gen(condition)
        
        # 应用FiLM调制
        modulated = self.film_layer([features, gamma, beta])
        
        # 可选: 门控融合
        if self.use_gating:
            modulated = self.gating([modulated, condition])
        
        # 可选: 自适应通道选择
        if self.use_adaptive_selection:
            modulated = self.channel_selection([modulated, condition], training=training)

        # ⭐ 残差式注入：避免“先验调制过强”导致性能回退
        # output = features + alpha * (modulated - features)
        alpha = float(self.residual_alpha)
        if alpha >= 0.999:
            return modulated
        if alpha <= 0.0:
            return features
        return features + alpha * (modulated - features)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_channels': self.num_channels,
            'prior_dim': self.prior_dim,
            'hidden_dim': self.hidden_dim,
            'use_gating': self.use_gating,
            'use_adaptive_selection': self.use_adaptive_selection,
            'adaptive_temperature': self.adaptive_temperature,
            'residual_alpha': self.residual_alpha,
        })
        return config


# ============================================================================
# 便捷函数
# ============================================================================

def create_prior_conditioner(num_channels, use_gating=True, use_adaptive=True):
    """
    创建物理先验条件化模块的便捷函数
    """
    return PhysicalPriorConditioner(
        num_channels=num_channels,
        use_gating=use_gating,
        use_adaptive_selection=use_adaptive
    )


def apply_film_conditioning(features, priors, num_channels, training=None):
    """
    应用FiLM条件化的便捷函数（函数式API）
    """
    conditioner = PhysicalPriorConditioner(num_channels=num_channels)
    return conditioner([features, priors], training=training)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FiLM层测试")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 4
    H, W, D, C = 13, 13, 9, 48
    prior_dim = 7
    
    # 模拟复值特征
    features_real = tf.random.normal([batch_size, H, W, D, C])
    features_imag = tf.random.normal([batch_size, H, W, D, C])
    features = tf.complex(features_real, features_imag)
    
    # 模拟物理先验
    priors = tf.random.uniform([batch_size, prior_dim], 0, 1)
    
    print(f"输入特征形状: {features.shape}")
    print(f"物理先验形状: {priors.shape}")
    
    # 测试条件化模块
    conditioner = PhysicalPriorConditioner(
        num_channels=C,
        use_gating=True,
        use_adaptive_selection=True
    )
    
    output = conditioner([features, priors], training=True)
    
    print(f"输出特征形状: {output.shape}")
    print(f"输出类型: {output.dtype}")
    
    # 验证形状一致
    assert output.shape == features.shape, "形状不匹配!"
    print("\n✓ FiLM层测试通过!")
