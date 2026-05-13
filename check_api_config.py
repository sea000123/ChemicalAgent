import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


DEFAULT_CONFIG_PATH = Path("./scripts/startup.json")


def mask_key(key: str) -> str:
    if not key:
        return "[MISSING]"
    if len(key) <= 10:
        return key[:2] + "***"
    return key[:6] + "..." + key[-4:]


def load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"[WARN] Config not found: {path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[OK] Loaded config: {path}")
        return config
    except Exception as e:
        print(f"[ERROR] Failed to load config: {path}")
        print(repr(e))
        return {}


def find_value(obj: Any, possible_keys: set[str]) -> Optional[Any]:
    """
    Recursively find a value by key names in nested dict/list config.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in possible_keys and v not in (None, ""):
                return v

        for v in obj.values():
            found = find_value(v, possible_keys)
            if found not in (None, ""):
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = find_value(item, possible_keys)
            if found not in (None, ""):
                return found

    return None


def normalize_base_url(base_url: str) -> str:
    """
    Convert any of these:
      https://api.openai.com/v1/chat/completions
      https://api.openai.com/v1/models
      https://genaiapi.xxx/api/v1/start/chat/completions

    into:
      https://api.openai.com/v1
      https://genaiapi.xxx/api/v1/start
    """
    base_url = str(base_url).strip().rstrip("/")

    suffixes = [
        "/chat/completions",
        "/v1/chat/completions",
        "/models",
        "/v1/models",
    ]

    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)].rstrip("/")
                changed = True

    return base_url


def get_config_values(config: Dict[str, Any]) -> Tuple[Optional[str], str, str]:
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or find_value(config, {"api_key", "apikey", "key", "openai_api_key", "deepseek_api_key"})
    )

    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or find_value(
            config,
            {
                "base_url",
                "baseurl",
                "api_base",
                "api_base_url",
                "openai_base_url",
                "deepseek_base_url",
                "url",
            },
        )
        or "https://api.openai.com/v1"
    )

    model = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or find_value(
            config,
            {
                "model",
                "model_name",
                "modelname",
                "llm_model",
                "openai_model",
                "deepseek_model",
                "vision_model",
                "multimodal_model",
                "filter_image_model",
            },
        )
        or "qwen-instruct"
    )

    return str(api_key) if api_key else None, normalize_base_url(str(base_url)), str(model)


def print_current_config(api_key: Optional[str], base_url: str, model: str) -> None:
    print("\n========== Current API Config ==========")
    print("API key :", mask_key(api_key or ""))
    print("Base URL:", base_url)
    print("Model   :", model)
    print("========================================\n")


def safe_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return None


def extract_error(data: Any) -> Optional[str]:
    """
    Supports several response styles:
      {"error": {"message": "...", "code": 404}}
      {"success": false, "message": "...", "code": 500}
      {"message": "..."}
    """
    if not isinstance(data, dict):
        return None

    if data.get("success") is False:
        message = data.get("message", "Unknown API error")
        code = data.get("code")
        return f"{message} | code={code}"

    if "error" in data:
        err = data.get("error")
        if isinstance(err, dict):
            message = err.get("message", "Unknown API error")
            code = err.get("code")
            err_type = err.get("type")
            return f"{message} | type={err_type} | code={code}"
        return str(err)

    return None


def looks_like_chat_success(data: Any) -> bool:
    if not isinstance(data, dict):
        return False

    if data.get("success") is False:
        return False

    if "error" in data:
        return False

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False

    first = choices[0]
    if not isinstance(first, dict):
        return False

    message = first.get("message")
    if not isinstance(message, dict):
        return False

    return "content" in message


def get_assistant_content(data: Any) -> str:
    try:
        return str(data["choices"][0]["message"].get("content", ""))
    except Exception:
        return ""


def classify_failure(status_code: int, data: Any, raw_text: str) -> str:
    message = ""

    if isinstance(data, dict):
        message = extract_error(data) or str(data.get("message", ""))
    else:
        message = raw_text

    lower = message.lower()

    if status_code in (401, 403):
        return "结论：认证失败。API key 无效、过期，或没有权限。"

    if "invalid subscription key" in lower or "wrong api endpoint" in lower:
        return (
            "结论：API key 和 base_url 不匹配，或者 key 没有该 endpoint 的权限。\n"
            "如果你用 OpenAI 官方 key，base_url 应该是 https://api.openai.com/v1。\n"
            "如果你用学校 genaiapi，就必须使用学校平台给的 key。"
        )

    if "model" in lower and ("does not exist" in lower or "not found" in lower):
        return "结论：model 名称错误，当前平台找不到这个模型。"

    if "not a multimodal model" in lower:
        return "结论：当前模型不是多模态模型，不能处理图片输入。"

    if "unsupported" in lower and ("image" in lower or "vision" in lower):
        return "结论：当前 endpoint 或模型不支持图片输入。"

    if status_code == 404:
        return "结论：endpoint 或 base_url 可能错误，也可能是模型名错误。"

    if status_code == 429:
        return "结论：触发限流或额度不足。"

    if status_code >= 500:
        return "结论：服务端错误，或中转平台返回了内部错误。"

    return "结论：API 返回了非预期错误，需要看上面的 Text 进一步判断。"


def post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int = 90,
) -> Tuple[Optional[requests.Response], Any]:
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        data = safe_json(response)
        return response, data
    except Exception as e:
        print("[ERROR] Request failed:")
        print(repr(e))
        return None, None


def test_models_endpoint(base_url: str, headers: Dict[str, str]) -> None:
    print("[TEST 1] Checking /models endpoint ...")

    models_url = f"{base_url}/models"

    try:
        response = requests.get(models_url, headers=headers, timeout=30)
        data = safe_json(response)

        print("URL   :", models_url)
        print("Status:", response.status_code)
        print("Text  :", response.text[:1200])

        err = extract_error(data)
        if err:
            print("\n[WARN] /models did not return a model list.")
            print("Reason:", err)
            print("这不一定影响 chat/completions，因为有些中转平台不支持 GET /models。\n")
            return

        if response.ok and isinstance(data, dict) and isinstance(data.get("data"), list):
            models = [m.get("id") for m in data["data"] if isinstance(m, dict) and m.get("id")]
            if models:
                print("\n[OK] /models returned available models:")
                for m in models:
                    print(" -", m)
                print()
                return

        print("\n[WARN] /models response was not recognized as a standard model list.\n")

    except Exception as e:
        print("[WARN] /models request failed:")
        print(repr(e))
        print("这不一定影响 chat/completions。\n")


def test_text_chat(base_url: str, headers: Dict[str, str], model: str) -> bool:
    print("[TEST 2] Checking text chat completion ...")

    chat_url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with only: OK",
            }
        ],
        "stream": False,
    }

    response, data = post_json(chat_url, headers, payload)

    if response is None:
        return False

    print("URL   :", chat_url)
    print("Status:", response.status_code)
    print("Text  :", response.text[:2000])

    if looks_like_chat_success(data):
        content = get_assistant_content(data)
        print("\n[OK] Text chat completion succeeded.")
        print("Assistant content:", repr(content))
        print()
        return True

    err = extract_error(data)
    print("\n[FAILED] Text chat completion failed.")
    if err:
        print("Error :", err)
    print(classify_failure(response.status_code, data, response.text))
    print()
    return False


def guess_mime_type(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime:
        return mime

    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"

    return "application/octet-stream"


def image_to_data_url(image_path: Path) -> str:
    mime = guess_mime_type(image_path)
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def find_default_image() -> Optional[Path]:
    candidates = []

    for folder in [
        Path("./Results/Images"),
        Path("./results/images"),
        Path("./images"),
        Path("."),
    ]:
        if folder.exists():
            for pattern in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
                candidates.extend(folder.glob(pattern))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: str(p).lower())
    return candidates[0]


def test_image_chat(
    base_url: str,
    headers: Dict[str, str],
    model: str,
    image_path: Optional[Path],
) -> bool:
    print("[TEST 3] Checking image chat completion ...")

    if image_path is None:
        image_path = find_default_image()

    if image_path is None:
        print("[SKIP] No image found.")
        print("用法示例：python check_api_config.py --image ./Results/Images/xxx.png")
        print()
        return False

    image_path = image_path.expanduser().resolve()

    if not image_path.exists():
        print(f"[FAILED] Image file does not exist: {image_path}")
        print()
        return False

    try:
        data_url = image_to_data_url(image_path)
    except Exception as e:
        print("[FAILED] Could not read or encode image.")
        print(repr(e))
        print()
        return False

    print("Image :", image_path)
    print("MIME  :", guess_mime_type(image_path))

    chat_url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Look at this image. Reply with one short English sentence "
                            "describing what is visible. If you can read the image, start with: IMAGE_OK."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        "stream": False,
    }

    response, data = post_json(chat_url, headers, payload)

    if response is None:
        return False

    print("URL   :", chat_url)
    print("Status:", response.status_code)
    print("Text  :", response.text[:2500])

    if looks_like_chat_success(data):
        content = get_assistant_content(data)
        print("\n[OK] Image chat completion succeeded.")
        print("Assistant content:", repr(content))
        print("结论：这个 model 至少可以接受当前格式的图片输入。")
        print()
        return True

    err = extract_error(data)
    print("\n[FAILED] Image chat completion failed.")
    if err:
        print("Error :", err)
    print(classify_failure(response.status_code, data, response.text))
    print()
    return False


def print_final_summary(text_ok: bool, image_ok: bool) -> None:
    print("========== Final Summary ==========")

    if text_ok and image_ok:
        print("[PASS] 文字和图片都测试成功。")
        print("结论：API key、base_url、model、图片输入格式基本都没问题。")
        print("如果 dataraider 仍报错，重点检查 dataraider 是否读取了同一个配置。")

    elif text_ok and not image_ok:
        print("[PARTIAL] 文字测试成功，但图片测试失败。")
        print("结论：API key / base_url / 文本模型调用大概率没问题。")
        print("问题通常是：当前 model 不是多模态模型，或平台不支持这种图片输入格式。")
        print("dataraider 的 Filtering relevant images 阶段需要多模态模型。")

    elif not text_ok:
        print("[FAIL] 文字测试失败。")
        print("结论：先不要跑 dataraider。请先修复 API key、base_url 或 model 配置。")

    print("===================================")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to startup.json. Default: ./scripts/startup.json",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to an image file for multimodal test. If omitted, the script tries ./Results/Images first.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip GET /models test.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json_config(config_path)

    api_key, base_url, model = get_config_values(config)
    print_current_config(api_key, base_url, model)

    if not api_key:
        print("[ERROR] API key is missing.")
        print("请设置 OPENAI_API_KEY，或在 startup.json 里配置 api_key。")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if not args.skip_models:
        test_models_endpoint(base_url, headers)

    text_ok = test_text_chat(base_url, headers, model)

    image_path = Path(args.image) if args.image else None
    image_ok = test_image_chat(base_url, headers, model, image_path)

    print_final_summary(text_ok, image_ok)


if __name__ == "__main__":
    main()
