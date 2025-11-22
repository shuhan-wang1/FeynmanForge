"""
测试反粒子识别和可视化

验证：
1. 环境能否正确解析 e_bar (正电子)
2. 守恒定律是否正确计算反粒子的电荷
3. 可视化是否从右往左绘制反粒子
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feynman_env import FeynmanDiagramEnv
from physics_engine import PhysicsConstants
import json


def test_antiparticle_parsing():
    """测试反粒子解析"""
    print("=" * 80)
    print("测试 1: 反粒子解析")
    print("=" * 80)
    
    # 创建环境：e + e_bar -> mu + mu_bar
    env = FeynmanDiagramEnv(
        initial_state=['e', 'e_bar'],
        final_state=['mu', 'mu_bar'],
        max_vertices=10,
        max_steps=50
    )
    
    state, info = env.reset()
    
    print("\n初态粒子:")
    for i, edge in enumerate(env.edges[:2]):  # 前两个是初态
        print(f"  {i}: particle_id='{edge['particle_id']}', is_anti={edge['is_anti']}")
        charge = env._get_charge(edge)
        lepton = env._get_lepton(edge)
        print(f"      电荷={charge:+.1f}, 轻子数={lepton:+.1f}")
    
    print("\n末态粒子:")
    for i, edge in enumerate(env.edges[2:4], start=2):  # 后两个是末态
        print(f"  {i}: particle_id='{edge['particle_id']}', is_anti={edge['is_anti']}")
        charge = env._get_charge(edge)
        lepton = env._get_lepton(edge)
        print(f"      电荷={charge:+.1f}, 轻子数={lepton:+.1f}")
    
    # 验证守恒
    total_charge_in = sum(env._get_charge(edge) for edge in env.edges[:2])
    total_charge_out = sum(env._get_charge(edge) for edge in env.edges[2:4])
    total_lepton_in = sum(env._get_lepton(edge) for edge in env.edges[:2])
    total_lepton_out = sum(env._get_lepton(edge) for edge in env.edges[2:4])
    
    print(f"\n总电荷: 初态={total_charge_in:+.1f}, 末态={total_charge_out:+.1f}")
    print(f"总轻子数: 初态={total_lepton_in:+.1f}, 末态={total_lepton_out:+.1f}")
    
    if abs(total_charge_in - total_charge_out) < 1e-6:
        print("✅ 电荷守恒！")
    else:
        print("❌ 电荷不守恒！")
    
    if abs(total_lepton_in - total_lepton_out) < 1e-6:
        print("✅ 轻子数守恒！")
    else:
        print("❌ 轻子数不守恒！")
    
    return env


def test_antiparticle_visualization(env):
    """测试反粒子可视化方向"""
    print("\n" + "=" * 80)
    print("测试 2: 反粒子可视化方向")
    print("=" * 80)
    
    # 获取导出的 JSON
    shapes = env.get_diagram_json()
    
    print(f"\n导出了 {len(shapes)} 个形状")
    
    # 由于初始环境没有连接的边，我们手动创建一个测试场景
    # 添加一条从左往右的正电子线（应该被标记为反粒子）
    print("\n检查初态外部线:")
    
    for i, edge in enumerate(env.edges[:2]):
        source_v = env.vertices[edge['source']] if edge['source'] is not None else None
        target_v = env.vertices[edge['target']] if edge['target'] is not None else None
        
        print(f"\nEdge {i}: {edge['particle_id']}")
        print(f"  is_anti: {edge['is_anti']}")
        
        if source_v:
            print(f"  source: V{edge['source']} at ({source_v['x']:.0f}, {source_v['y']:.0f})")
        if target_v:
            print(f"  target: V{edge['target']} at ({target_v['x']:.0f}, {target_v['y']:.0f})")
        
        # 检查如果这条线被导出，方向是否正确
        if source_v and target_v:
            if edge['is_anti']:
                expected_direction = "从右往左"
                # 对于反粒子，p1 应该在 p2 右边
                if source_v['x'] > target_v['x']:
                    print(f"  ✅ 方向正确: {expected_direction}")
                else:
                    print(f"  ⚠️  反粒子应该{expected_direction}，但实际从左往右")
            else:
                expected_direction = "从左往右"
                if source_v['x'] < target_v['x']:
                    print(f"  ✅ 方向正确: {expected_direction}")
                else:
                    print(f"  ⚠️  正常粒子应该{expected_direction}，但实际从右往左")


def test_charge_reversal():
    """测试反粒子的电荷反转"""
    print("\n" + "=" * 80)
    print("测试 3: 反粒子电荷反转")
    print("=" * 80)
    
    test_cases = [
        ('e', False, -1.0, "电子"),
        ('e', True, +1.0, "正电子"),
        ('mu', False, -1.0, "缪子"),
        ('mu', True, +1.0, "反缪子"),
        ('u', False, +2/3, "上夸克"),
        ('u', True, -2/3, "反上夸克"),
    ]
    
    print("\n粒子电荷测试:")
    for particle_id, is_anti, expected_charge, name in test_cases:
        # 模拟一个 edge
        edge = {
            'particle_id': particle_id,
            'is_anti': is_anti
        }
        
        env = FeynmanDiagramEnv(['e'], ['e'], max_vertices=10, max_steps=50)
        charge = env._get_charge(edge)
        
        status = "✅" if abs(charge - expected_charge) < 1e-6 else "❌"
        print(f"  {status} {name:12s}: 期望={expected_charge:+.2f}, 实际={charge:+.2f}")


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "反粒子识别和可视化测试" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # 测试 1: 解析
    env = test_antiparticle_parsing()
    
    # 测试 2: 可视化
    test_antiparticle_visualization(env)
    
    # 测试 3: 电荷反转
    test_charge_reversal()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    print("\n💡 使用建议:")
    print("   训练命令: python train.py --reaction 'e+e_bar->mu+mu_bar'")
    print("   这将创建一个物理上可行的 e⁺e⁻ 湮灭过程")
    print()


if __name__ == '__main__':
    main()
