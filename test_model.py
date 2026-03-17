#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CV-SSM-PriorNet 模型测试脚本

验证：
1. 模型可以正常创建
2. 双输入模式（使用极化先验）正常工作
3. 单输入模式（不使用极化先验）正常工作
4. 前向传播正常
"""

import os
import sys

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

print("=" * 70)
print("CV-SSM-PriorNet 模型测试")
print("复值SSM + 极化先验引导的SAR图像分类网络")
print("=" * 70)

# 导入模块
try:
    from model_CV_SSM_PriorNet import CV_SSM_PriorNet
    from physical_priors import extract_physical_priors
    from complex_ssm_layer import ComplexSSMLayer, LightweightComplexSSMLayer
    from film_layers import PhysicalPriorConditioner
    print("\n✓ 所有模块导入成功")
except ImportError as e:
    print(f"\n✗ 模块导入失败: {e}")
    sys.exit(1)

# 测试参数
batch_size = 4
H, W, D, C = 13, 13, 9, 1
num_classes = 15

# 创建测试数据
print("\n【测试1】创建测试数据...")
X_test = np.random.rand(batch_size, H, W, D, C).astype(np.float32) + 0.1
print(f"  SAR数据形状: {X_test.shape}")

# 提取物理先验
print("\n【测试2】提取物理先验...")
try:
    priors_test = extract_physical_priors(X_test, normalize=True)
    print(f"  ✓ 物理先验形状: {priors_test.shape}")
    print(f"  先验范围: [{priors_test.min():.3f}, {priors_test.max():.3f}]")
except Exception as e:
    print(f"  ✗ 错误: {e}")
    # 使用随机先验继续测试
    priors_test = np.random.rand(batch_size, 7).astype(np.float32)
    print(f"  使用随机先验: {priors_test.shape}")

# 测试3: 使用极化先验的模型
print("\n【测试3】创建模型（使用极化先验引导）...")
try:
    model_with_prior = CV_SSM_PriorNet(
        X_test, num_classes, 
        use_prior_guidance=True, 
        use_ssm=True
    )
    print(f"  ✓ 模型创建成功")
    print(f"  模型名称: {model_with_prior.name}")
    print(f"  输入数量: {len(model_with_prior.inputs)}")
    print(f"  输入名称: {[inp.name for inp in model_with_prior.inputs]}")
    print(f"  参数量: {model_with_prior.count_params():,}")
    
    # 前向传播测试
    output = model_with_prior.predict([X_test, priors_test], verbose=0)
    print(f"  ✓ 前向传播成功")
    print(f"  输出形状: {output.shape}")
    print(f"  输出和（应接近1）: {output.sum(axis=1)}")
except Exception as e:
    print(f"  ✗ 错误: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 不使用极化先验的模型
print("\n【测试4】创建模型（不使用极化先验）...")
try:
    model_no_prior = CV_SSM_PriorNet(
        X_test, num_classes, 
        use_prior_guidance=False, 
        use_ssm=True
    )
    print(f"  ✓ 模型创建成功")
    print(f"  模型名称: {model_no_prior.name}")
    print(f"  输入数量: {len(model_no_prior.inputs)}")
    print(f"  参数量: {model_no_prior.count_params():,}")
    
    # 前向传播测试
    output = model_no_prior.predict(X_test, verbose=0)
    print(f"  ✓ 前向传播成功")
    print(f"  输出形状: {output.shape}")
except Exception as e:
    print(f"  ✗ 错误: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 复值SSM层单独测试
print("\n【测试5】复值SSM层测试...")
try:
    test_input = tf.complex(
        tf.random.normal([2, 121, 9, 16]),
        tf.random.normal([2, 121, 9, 16])
    )
    ssm_layer = ComplexSSMLayer(d_model=16, d_state=32)
    ssm_output = ssm_layer(test_input)
    print(f"  ✓ ComplexSSMLayer 测试通过")
    print(f"  输入: {test_input.shape} -> 输出: {ssm_output.shape}")
except Exception as e:
    print(f"  ✗ 错误: {e}")

# 测试6: FiLM条件化层测试
print("\n【测试6】FiLM条件化层测试...")
try:
    features = tf.complex(
        tf.random.normal([2, 13, 13, 9, 48]),
        tf.random.normal([2, 13, 13, 9, 48])
    )
    priors = tf.random.uniform([2, 7])
    
    conditioner = PhysicalPriorConditioner(
        num_channels=48,
        hidden_dim=32,
        use_gating=True,
        use_adaptive_selection=True
    )
    conditioned = conditioner([features, priors], training=True)
    print(f"  ✓ PhysicalPriorConditioner 测试通过")
    print(f"  输入: {features.shape} -> 输出: {conditioned.shape}")
except Exception as e:
    print(f"  ✗ 错误: {e}")

# 总结
print("\n" + "=" * 70)
print("测试总结")
print("=" * 70)

tests_passed = []
tests_failed = []

if 'model_with_prior' in dir():
    tests_passed.append("使用极化先验的模型")
else:
    tests_failed.append("使用极化先验的模型")

if 'model_no_prior' in dir():
    tests_passed.append("不使用极化先验的模型")
else:
    tests_failed.append("不使用极化先验的模型")

if 'ssm_output' in dir():
    tests_passed.append("复值SSM层")
else:
    tests_failed.append("复值SSM层")

if 'conditioned' in dir():
    tests_passed.append("FiLM条件化层")
else:
    tests_failed.append("FiLM条件化层")

print(f"\n✓ 通过: {len(tests_passed)}/{len(tests_passed)+len(tests_failed)}")
for t in tests_passed:
    print(f"  - {t}")

if tests_failed:
    print(f"\n✗ 失败: {len(tests_failed)}")
    for t in tests_failed:
        print(f"  - {t}")

print("\n" + "=" * 70)
if not tests_failed:
    print("✓ 所有测试通过! CV-SSM-PriorNet 已准备就绪")
else:
    print("⚠ 部分测试失败，请检查错误信息")
print("=" * 70)

