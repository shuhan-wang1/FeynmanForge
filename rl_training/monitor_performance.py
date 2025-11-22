"""
实时监控训练性能 - CPU 和 GPU 利用率

在训练运行时在另一个终端运行此脚本：
python monitor_performance.py
"""

import time
import psutil
import torch
from datetime import datetime


def format_bytes(bytes):
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def monitor_performance(interval=2):
    """
    实时监控性能指标
    
    Args:
        interval: 更新间隔（秒）
    """
    print("=" * 80)
    print("🔍 Feynman-GCPN 性能监控器")
    print("=" * 80)
    print("按 Ctrl+C 停止监控\n")
    
    has_gpu = torch.cuda.is_available()
    
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # CPU 监控
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # 内存监控
            memory = psutil.virtual_memory()
            
            # 进程监控（查找 Python 进程）
            python_procs = []
            for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_info']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_procs.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # 清屏（Windows）
            print("\033[2J\033[H", end="")
            
            # 显示标题
            print("=" * 80)
            print(f"📊 性能监控 - {timestamp}")
            print("=" * 80)
            
            # CPU 信息
            print(f"\n🖥️  CPU 信息")
            print(f"   核心数: {cpu_count}")
            print(f"   总体利用率: {cpu_percent:5.1f}%")
            if cpu_freq:
                print(f"   当前频率: {cpu_freq.current:.0f} MHz")
            
            # 每核心利用率
            cpu_percents = psutil.cpu_percent(interval=0.1, percpu=True)
            print(f"\n   每核心利用率:")
            for i, percent in enumerate(cpu_percents):
                bar_length = int(percent / 2)  # 最大 50 字符
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"   Core {i:2d}: [{bar}] {percent:5.1f}%")
            
            # 内存信息
            print(f"\n💾 内存信息")
            print(f"   总内存: {format_bytes(memory.total)}")
            print(f"   已用: {format_bytes(memory.used)} ({memory.percent:.1f}%)")
            print(f"   可用: {format_bytes(memory.available)}")
            
            # Python 进程
            if python_procs:
                print(f"\n🐍 Python 进程 (共 {len(python_procs)} 个)")
                total_cpu = 0
                total_mem = 0
                for proc in python_procs[:5]:  # 只显示前 5 个
                    try:
                        cpu = proc.cpu_percent()
                        mem = proc.memory_info().rss
                        total_cpu += cpu
                        total_mem += mem
                        print(f"   PID {proc.pid}: CPU {cpu:5.1f}% | MEM {format_bytes(mem)}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                print(f"   总计: CPU {total_cpu:5.1f}% | MEM {format_bytes(total_mem)}")
            
            # GPU 信息
            if has_gpu:
                print(f"\n🎮 GPU 信息")
                for i in range(torch.cuda.device_count()):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                    
                    # 内存使用
                    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    
                    mem_percent = (mem_reserved / mem_total) * 100
                    bar_length = int(mem_percent / 2)
                    bar = "█" * bar_length + "░" * (50 - bar_length)
                    
                    print(f"   内存使用: [{bar}] {mem_percent:5.1f}%")
                    print(f"   分配: {mem_allocated:.2f} GB / 保留: {mem_reserved:.2f} GB / 总计: {mem_total:.2f} GB")
                    
                    # 尝试获取 GPU 利用率（需要 nvidia-ml-py3）
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        
                        print(f"   GPU 利用率: {utilization.gpu}%")
                        print(f"   显存利用率: {utilization.memory}%")
                        print(f"   温度: {temperature}°C")
                        pynvml.nvmlShutdown()
                    except ImportError:
                        print(f"   💡 安装 nvidia-ml-py3 可查看详细 GPU 指标:")
                        print(f"      pip install nvidia-ml-py3")
                    except Exception as e:
                        pass
            else:
                print(f"\n🎮 GPU: 未检测到 CUDA 设备")
            
            print("\n" + "=" * 80)
            print(f"下次更新: {interval} 秒后 | 按 Ctrl+C 停止")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ 监控已停止")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="性能监控器")
    parser.add_argument('--interval', type=int, default=2, help='更新间隔（秒）')
    args = parser.parse_args()
    
    monitor_performance(interval=args.interval)
