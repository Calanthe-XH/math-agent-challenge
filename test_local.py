# -*- coding: utf-8 -*-
import requests
import json
from user_agent import ReasoningAgent

# ---------- 配置 ----------
API_TOKEN = "sk-LhBF4HSnuGV9eYt19KpzXTnpAxVOUp6x1d8WnBuTBeQ4mZBB"   # 请替换为真实 Token
# -------------------------

class MockClient:
    """
    模拟比赛平台提供的 client 对象。
    直接使用 requests 发送 HTTP 请求，避免 openai 库的编码问题。
    """
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
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # 打印原始输出，方便调试
        print("\n" + "="*50)
        print("模型原始输出：")
        print(content)
        print("="*50 + "\n")
        return content


if __name__ == "__main__":
    # 创建模拟客户端和智能体
    client = MockClient(API_TOKEN)
    agent = ReasoningAgent(client)

    # 测试题目（你可以换成任何题目）
    test_problem = "求方程 x^2 - 5x + 6 = 0 的解。"

    print("开始解题...")
    result = agent.solve(test_problem, {"idx": 0})
    print(f"最终答案: {result['final_response']}")