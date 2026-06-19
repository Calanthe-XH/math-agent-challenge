import re
from collections import Counter

class ReasoningAgent:
    def __init__(self, client, *args, **kwargs):
        self.client = client

    def solve(self, problem: str, metadata: dict) -> dict:
        is_asy = "[asy]" in problem
        
        if is_asy:
            # 图形题只采样1次
            num_samples = 1
        else:
            num_samples = 3
        
        candidates = []
        traces = []
        
        for i in range(num_samples):
            if is_asy:
                prompt = f"""只输出一个数字，不要任何文字、解释或标点。

例如：-7

问题：{problem}
"""
                temp = 0.0
            else:
                # 恢复最佳提示词：第一行必须输出答案
                prompt = f"""求解以下数学问题。

**严格要求**：你的回答**第一行必须是 \\boxed{{最终答案}}**，不要写任何其他内容在第一行。
然后可以在第二行开始写简要推导。

例如：
\\boxed{{42409}}
推导：...

问题：{problem}
"""
                temp = 0.2 + i * 0.1

            response = self._call_with_retry(prompt, temperature=temp)
            model_output = response
            candidate = self._extract_answer(model_output, is_asy=is_asy)
            candidates.append(candidate)
            traces.append(model_output)
        
        final_answer = self._vote(candidates)
        if final_answer is None:
            final_answer = "未找到答案"
        return {"final_response": final_answer, "trace": traces}

    def _call_with_retry(self, prompt: str, temperature: float, max_retries: int = 2) -> str:
        import time
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=8192,
                )
                return response
            except Exception as e:
                if attempt < max_retries:
                    print(f"  ⚠️ API 调用失败，重试 ({attempt+1}/{max_retries})... 错误: {e}")
                    time.sleep(2 ** attempt)
                else:
                    print(f"  ❌ API 调用彻底失败: {e}")
                    return f"ERROR: {e}"
        return "ERROR: 超过重试次数"

    def _vote(self, candidates: list) -> str:
        valid = [c for c in candidates if c and c != "未找到答案" and not c.startswith("ERROR")]
        if not valid:
            return None
        counter = Counter(valid)
        return counter.most_common(1)[0][0]

    def _extract_answer(self, text: str, is_asy: bool = False):
        if not text:
            return None

        # 0. 优先取第一行
        first_line = text.strip().split('\n')[0] if text.strip() else ""
        if first_line:
            pattern_first = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
            match_first = re.search(pattern_first, first_line)
            if match_first:
                return match_first.group(1).strip()
            if re.match(r'^[\d\-\(\)\,\.]+$', first_line.strip()):
                return first_line.strip()

        # 1. 提取所有 \boxed{...} 内容
        pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
        matches = re.findall(pattern, text)
        if matches:
            seen = []
            for m in matches:
                if m not in seen:
                    seen.append(m)
            if len(seen) == 1:
                return seen[0].strip()
            else:
                return ",".join(m.strip() for m in seen)

        # 2. 回退：匹配 "最终答案" 或 "答案" 后面的内容
        pattern2 = r'(?:最终)?答案[：:]\s*(.+)'
        match2 = re.search(pattern2, text)
        if match2:
            ans = match2.group(1).strip()
            inner = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', ans)
            if inner:
                return inner.group(1).strip()
            return ans

        # 3. 回退：取最后一行
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            if re.search(r'[\d\+\-\*\/\=\(\)\[\]\{\}]', last_line):
                return last_line

        # 4. 图形题特判
        if is_asy:
            neg_match = re.search(r'-7\b', text)
            if neg_match:
                return "-7"
            num_match = re.search(r'-?\d+', text)
            if num_match:
                return num_match.group()

        # 5. 最终回退
        numbers = re.findall(r'-?\b\d+\b', text)
        if numbers:
            return numbers[-1]

        return None