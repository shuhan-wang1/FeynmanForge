"""
修复可视化问题的诊断和解决脚本

运行此脚本来检查并修复常见问题：
1. diagrams 文件夹缺失
2. JSON 文件格式错误
3. 文件权限问题
"""

import os
import json
from datetime import datetime


def check_and_fix_visualization():
    """检查并修复可视化相关的问题"""
    
    print("=" * 60)
    print("🔧 Feynman-GCPN 可视化诊断工具")
    print("=" * 60)
    
    issues_found = []
    fixes_applied = []
    
    # 检查 1: diagrams 文件夹是否存在
    print("\n1️⃣  检查 diagrams 文件夹...")
    if not os.path.exists('diagrams'):
        issues_found.append("diagrams 文件夹不存在")
        os.makedirs('diagrams', exist_ok=True)
        fixes_applied.append("✅ 创建了 diagrams 文件夹")
    else:
        print("   ✅ diagrams 文件夹存在")
    
    # 检查 2: 创建初始 JSON 文件
    print("\n2️⃣  检查 JSON 文件...")
    
    test_diagram = {
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "episode": 0,
            "reward": 0.0,
            "initial_state": ["e", "e"],
            "final_state": ["mu", "mu"]
        },
        "shapes": []
    }
    
    files_to_create = [
        'diagrams/current_best.json',
        'diagrams/current_diagram.json'
    ]
    
    for filepath in files_to_create:
        if not os.path.exists(filepath):
            issues_found.append(f"{filepath} 不存在")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(test_diagram, f, indent=2, ensure_ascii=False)
            fixes_applied.append(f"✅ 创建了 {filepath}")
        else:
            # 验证 JSON 格式
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   ✅ {filepath} 格式正确")
            except Exception as e:
                issues_found.append(f"{filepath} 格式错误: {e}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(test_diagram, f, indent=2, ensure_ascii=False)
                fixes_applied.append(f"✅ 修复了 {filepath}")
    
    # 检查 3: 文件权限
    print("\n3️⃣  检查文件权限...")
    for filepath in files_to_create:
        if os.path.exists(filepath):
            if os.access(filepath, os.W_OK):
                print(f"   ✅ {filepath} 可写")
            else:
                issues_found.append(f"{filepath} 没有写入权限")
    
    # 检查 4: training_viz.html 配置
    print("\n4️⃣  检查 training_viz.html 配置...")
    
    viz_file = '../training_viz.html'
    if os.path.exists(viz_file):
        print(f"   ✅ {viz_file} 存在")
        print(f"   💡 在浏览器中打开: file://{os.path.abspath(viz_file)}")
    else:
        issues_found.append(f"{viz_file} 不存在")
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 诊断总结")
    print("=" * 60)
    
    if issues_found:
        print(f"\n⚠️  发现 {len(issues_found)} 个问题:")
        for issue in issues_found:
            print(f"   - {issue}")
    else:
        print("\n✅ 没有发现问题！")
    
    if fixes_applied:
        print(f"\n🔧 应用了 {len(fixes_applied)} 个修复:")
        for fix in fixes_applied:
            print(f"   {fix}")
    
    print("\n" + "=" * 60)
    print("📝 使用说明")
    print("=" * 60)
    print("""
1. 启动训练:
   python train.py --reaction "e+e->mu+mu"

2. 打开可视化 (在浏览器中):
   - 方法 1: 直接打开 training_viz.html
   - 方法 2: 使用 Live Server (VS Code 插件)
   
3. 如果仍然看不到图表:
   - 检查浏览器控制台 (F12) 查看错误
   - 确认训练正在运行
   - 等待至少 10 秒让第一个 episode 完成
   - 手动刷新浏览器 (Ctrl+F5)

4. 检查 JSON 文件:
   打开 diagrams/current_best.json 查看内容
    """)
    
    print("=" * 60)
    
    return len(issues_found) == 0


if __name__ == '__main__':
    success = check_and_fix_visualization()
    
    if success:
        print("\n🎉 所有检查通过！可以开始训练了。")
    else:
        print("\n⚠️  发现一些问题，但已尝试修复。请重新运行此脚本验证。")
