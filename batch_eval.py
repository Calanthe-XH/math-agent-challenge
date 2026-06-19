# -*- coding: utf-8 -*-
import os
import re
import json
import time
import requests
from datasets import load_dataset

# ==================== 配置 ====================
API_TOKEN = "sk-LhBF4HSnuGV9eYt19KpzXTnpAxVOUp6x1d8WnBuTBeQ4mZBB"    # 替换为真实的 Token
TEST_SIZE = 10                         # 测试题目数量
HF_ENDPOINT = "https://hf-mirror.com"  # 镜像源
# =============================================

os.environ['HF_ENDPOINT'] = HF_ENDPOINT

# -------------------- 模拟客户端 --------------------
class MockClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://chat.intern-ai.org.cn/api/v1/chat/completions"

    def chat(self, messages, temperature, max_tokens):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "intern-s1",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=240)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content

# -------------------- 答案标准化函数（最终修复版） --------------------
def normalize_answer(ans):
    if ans is None:
        return ""
    ans = ans.strip()
    # 移除 LaTeX 转义符号（如 \$ 变为空）
    ans = ans.replace("\\$", "")
    # 移除普通美元符号
    ans = ans.replace("$", "")
    # 移除所有空格
    ans = ans.replace(" ", "")
    # 将小数形式转为整数（如 0.0 -> 0, 3.0 -> 3）
    ans = re.sub(r'(\d+)\.0', r'\1', ans)
    # 统一括号
    ans = ans.replace("（", "(").replace("）", ")")
    # 替换中文分隔符
    ans = ans.replace("或", ",").replace("和", ",")
    # 去除尾部标点
    ans = ans.rstrip(",.；;")
    return ans

# -------------------- 导入你的智能体 --------------------
from user_agent import ReasoningAgent

# -------------------- 加载数据集 --------------------
print("正在加载 MATH 数据集 (train split)...")
dataset = load_dataset("qwedsacf/competition_math", split="train")
print(f"数据集加载成功，共 {len(dataset)} 道题。")
total = min(TEST_SIZE, len(dataset))

# -------------------- 初始化智能体 --------------------
client = MockClient(API_TOKEN)
agent = ReasoningAgent(client)

# -------------------- 评测循环 --------------------
results = []
correct = 0

for idx in range(total):
    sample = dataset[idx]
    problem = sample["problem"]
    true_solution = sample["solution"]
    
    # 提取标准答案（取最后一个 \boxed{}）
    boxed_matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', true_solution)
    if not boxed_matches:
        print(f"⚠️ 第 {idx+1} 题无法提取标准答案，跳过")
        continue
    true_answer = boxed_matches[-1].strip()
    
    print(f"\n[{idx+1}/{total}] 题目: {problem[:80]}...")
    
    # 调用智能体求解
    try:
        result = agent.solve(problem, {"idx": idx})
        pred_answer = result["final_response"]
        trace = result.get("trace", [])
    except Exception as e:
        print(f"  ❌ 智能体执行失败: {e}")
        pred_answer = "ERROR"
        trace = []
    
    # 标准化后比较
    is_correct = (normalize_answer(pred_answer) == normalize_answer(true_answer))
    if is_correct:
        correct += 1
    
    results.append({
        "idx": idx,
        "problem": problem,
        "true_answer": true_answer,
        "predicted_answer": pred_answer,
        "is_correct": is_correct,
        "trace": trace
    })
    
    print(f"  标准答案: {true_answer}")
    print(f"  模型答案: {pred_answer}")
    print(f"  结果: {'✅ 正确' if is_correct else '❌ 错误'}")
    
    time.sleep(1)

# -------------------- 保存与统计 --------------------
with open("batch_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

accuracy = correct / total * 100 if total > 0 else 0
print("\n" + "="*50)
print(f"评测完成！共测试 {total} 题，正确 {correct} 题")
print(f"正确率: {accuracy:.1f}%")
print(f"详细结果已保存到 batch_eval_results.json")
print("="*50)