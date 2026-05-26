import os
import json
import requests
from openai import OpenAI

BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    "https://genaiapi.shanghaitech.edu.cn/api/v1/start"
).rstrip("/")

API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.environ.get("OPENAI_MODEL", "qwen-instruct")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

def dump(title, obj):
    print(f"\n===== {title} =====")
    try:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    except Exception:
        print(obj)

def test_openai_sdk(extra_body=None):
    print("\n===== OpenAI SDK test =====")
    print("BASE_URL =", BASE_URL)
    print("MODEL =", MODEL)
    print("extra_body =", extra_body)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "不要输出思考过程，不要输出 <think> 标签，只输出最终答案。",
                },
                {
                    "role": "user",
                    "content": "/no_think\n只回答两个字：成功",
                },
            ],
            temperature=0,
            max_tokens=100,
            extra_body=extra_body or {},
        )

        dump("FULL RESPONSE", resp.model_dump())

        choice = resp.choices[0]
        msg = choice.message

        print("\nfinish_reason =", choice.finish_reason)
        print("response.model =", resp.model)
        print("message.content =", repr(msg.content))

        if hasattr(msg, "reasoning_content"):
            print("message.reasoning_content =", repr(msg.reasoning_content))

    except Exception as e:
        print("SDK 请求失败：", type(e).__name__, str(e))

def test_raw_http(extra_body=None):
    print("\n===== Raw HTTP test =====")

    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "不要输出思考过程，不要输出 <think> 标签，只输出最终答案。",
            },
            {
                "role": "user",
                "content": "/no_think\n只回答两个字：成功",
            },
        ],
        "temperature": 0,
        "max_tokens": 100,
    }

    if extra_body:
        payload.update(extra_body)

    print("POST", url)

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        print("HTTP status =", r.status_code)
        print("Raw text:")
        print(r.text[:5000])

        try:
            dump("JSON", r.json())
        except Exception:
            pass

    except Exception as e:
        print("Raw HTTP 请求失败：", type(e).__name__, str(e))

if __name__ == "__main__":
    # 1. 不带 thinking 参数
    test_openai_sdk()

    # 2. Qwen / DashScope 常见关闭 thinking 参数
    test_openai_sdk({"enable_thinking": False})

    # 3. vLLM / SGLang 常见写法
    test_openai_sdk({"chat_template_kwargs": {"enable_thinking": False}})

    # 4. 原始 HTTP，看服务端到底返回了什么
    test_raw_http({"enable_thinking": False})
