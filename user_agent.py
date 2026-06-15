import re

class ReasoningAgent:
    def __init__(self, client, *args, **kwargs):
        self.client = client

    def solve(self, problem: str, metadata: dict) -> dict:
        prompt = f"""你是一个数学解题专家。请一步步推理并解决下面的数学问题。

【重要】最终答案必须严格放在一个单独的 \\boxed{{}} 中，例如 \\boxed{{42}}。如果答案有多个值，请放在同一个 \\boxed{{}} 内，例如 \\boxed{{2,3}}。

问题：{problem}
"""
        response = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
        )
        model_output = response

        final_answer = self._extract_answer(model_output)
        if final_answer is None:
            final_answer = "未提取到答案"

        return {"final_response": final_answer, "trace": [model_output]}

    def _extract_answer(self, text: str):
        if not text:
            return None

        # 1. 提取所有 \boxed{...} 中的内容
        pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
        matches = re.findall(pattern, text)
        if matches:
            # 如果只有一个答案，直接返回；多个答案用逗号连接
            if len(matches) == 1:
                return matches[0].strip()
            else:
                return ",".join(m.strip() for m in matches)

        # 2. 回退：匹配 "答案是：" 或 "答案：" 后面的内容
        pattern2 = r'答案[是为：:]\s*(.+)'
        match2 = re.search(pattern2, text)
        if match2:
            return match2.group(1).strip()

        # 3. 回退：取最后一行（可能答案单独一行）
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            if re.search(r'[\d\+\-\*\/\=\(\)\[\]\{\}]', last_line):
                return last_line

        return None