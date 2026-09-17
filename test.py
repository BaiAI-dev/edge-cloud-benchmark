import time
import os
import requests
import statistics

# ================= 1. 配置区域 =================
# 从环境变量读取（请确保在终端 export 了这三个 Key）
# 提示：阿里云百炼的 API URL 需要替换为你们专属的地域地址！
MODELS = [
    {
        "name": "DeepSeek (深度推理)",
        "api_url": "https://api.deepseek.com/chat/completions",
        "api_key": os.environ.get("DEEPSEEK_API_KEY"),
        "model_id": "deepseek-chat"
    },
    {
        "name": "智谱 GLM-4-Flash (极速响应)",
        "api_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "api_key": os.environ.get("ZHIPU_API_KEY"),
        "model_id": "glm-4-flash"
    },
    {
        "name": "通义 Qwen-Turbo (高性价比)",
        # 注意：请将下面的 URL 替换为你阿里云百炼控制台里显示的“OpenAI兼容地址”
        "api_url": "https://ws-s8v0o45wei5767al.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions", 
        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
        "model_id": "qwen-turbo"
    }
]

# 测试用例设计
TEST_CASES = {
    "端侧控制指令": ["下一首", "音量调大", "暂停播放", "接听电话"],
    "云端推理指令": ["OpenComm2怎么开机？", "耳机的防水等级是多少？", "今天适合跑步吗？"]
}

# ================= 2. 端侧模拟逻辑 =================
def device_side_simulation(text):
    """
    模拟智能耳机（端侧）的物理约束：
    - 算力极弱：只能用极轻量的规则匹配，无大模型。
    - 响应极快：本地中断处理，延迟极低。
    - 成本极低：不依赖网络，0 Token消耗。
    """
    start = time.time()
    # 模拟耳机芯片5毫秒的极速响应
    time.sleep(0.005) 
    
    device_keywords = ["下一首", "音量", "暂停", "接听", "播放"]
    if any(kw in text for kw in device_keywords):
        return {
            "status": "device",
            "latency": time.time() - start,
            "tokens": 0
        }
    return None

# ================= 3. 云端真实 API 调用 =================
def cloud_side_call(text, model_config):
    """
    真实调用云端大模型 API，记录端到端延迟和 Token 消耗。
    """
    start = time.time()
    
    headers = {
        "Authorization": f"Bearer {model_config['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_config["model_id"],
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.0 # 降低随机性，保证测试稳定
    }
    
    try:
        # 设置30秒超时，防止网络阻塞
        resp = requests.post(model_config["api_url"], headers=headers, json=payload, timeout=30)
        latency = time.time() - start
        
        if resp.status_code == 200:
            data = resp.json()
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return {"status": "success", "latency": latency, "tokens": tokens}
        else:
            return {"status": "error", "msg": resp.text, "latency": latency, "tokens": 0}
    except Exception as e:
        return {"status": "error", "msg": str(e), "latency": time.time() - start, "tokens": 0}

# ================= 4. 核心评估流程 =================
def run_benchmark():
    print("="*70)
    print("📊 可穿戴设备端云协同与多模型选型评估报告")
    print("="*70)
    
    # ---------- 测试端侧 ----------
    print("\n[一、 端侧本地处理模拟] (模拟耳机MCU芯片)")
    device_latencies = []
    for text in TEST_CASES["端侧控制指令"]:
        res = device_side_simulation(text)
        if res:
            device_latencies.append(res["latency"])
            print(f"  ✓ 指令: '{text}' -> 延迟: {res['latency']*1000:.2f} ms, Token: {res['tokens']}")

    # ---------- 测试云端 ----------
    print("\n[二、 云端大模型真实调用对比]")
    cloud_summary = {}
    
    for model in MODELS:
        print(f"\n  ▶ 正在测试模型: {model['name']} ...")
        model_latencies = []
        model_tokens = []
        
        for text in TEST_CASES["云端推理指令"]:
            if not model["api_key"]:
                print(f"    ✗ 错误: 未找到 {model['name']} 的 API Key，跳过该模型")
                break
                
            print(f"    请求: '{text}' ...", end="")
            res = cloud_side_call(text, model)
            
            if res["status"] == "success":
                model_latencies.append(res["latency"])
                model_tokens.append(res["tokens"])
                print(f" 完成 | 延迟: {res['latency']:.2f}s | Token: {res['tokens']}")
            else:
                print(f" 失败 | 错误: {res['msg'][:50]}...")
        
        if model_latencies:
            cloud_summary[model["name"]] = {
                "avg_latency": statistics.mean(model_latencies),
                "avg_tokens": statistics.mean(model_tokens) if model_tokens else 0
            }

    # ---------- 汇总报告 ----------
    print("\n" + "="*70)
    print("📈 核心结论与架构建议")
    print("="*70)
    
    if device_latencies:
        avg_device = statistics.mean(device_latencies) * 1000
        print(f"1. 端侧平均延迟: {avg_device:.2f} ms (0 Token)")
        print("   建议: 高频控制指令必须在端侧拦截，绝不上云。")
    
    if cloud_summary:
        print("\n2. 云端模型横评:")
        for name, data in cloud_summary.items():
            print(f"   - {name}: 平均延迟 {data['avg_latency']:.2f}s, 平均消耗 {data['avg_tokens']:.1f} Token")
        
        print("\n3. 综合架构建议:")
        print("   对于需要深度推理的场景，选择逻辑强的模型；对于简单问答，选择响应快的模型。")
        print("   结合端侧分流，可极大降低云端 Token 成本，并提升耳机用户的响应体验。")

if __name__ == "__main__":
    run_benchmark()