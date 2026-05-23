# check openai api
import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


import requests


DEFAULT_CONFIG_PATH = Path("./scripts/startup.json")
DEFAULT_DOTENV_PATHS = [
    Path(".env"),
    Path("./scripts/.env"),
]


def mask_key(key: str) -> str:
    if not key:
        return "[MISSING]"
    if len(key) <= 10:
        return key[:2] + "***"
    return key[:6] + "..." + key[-4:]


def load_dotenv_file(path: Path) -> None:
    """
    Minimal .env loader without requiring python-dotenv.

    It does NOT override existing environment variables.
    Supports simple lines like:
      OPENAI_API_KEY=xxx
      LLM_BASE_URL=https://...
      LLM_DEFAULT_HEADERS_JSON={"X-Test":"abc"}
    """
    if not path.exists():
        return

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            # Remove optional quotes.
            if (
                len(value) >= 2
                and (
                    (value[0] == value[-1] == '"')
                    or (value[0] == value[-1] == "'")
                )
            ):
                value = value[1:-1]

            os.environ.setdefault(key, value)

        print(f"[OK] Loaded env file: {path}")

    except Exception as e:
        print(f"[WARN] Failed to load env file: {path}")
        print(repr(e))


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
    Normalize API root.

    OpenAI standard:
      https://api.openai.com/v1/chat/completions
      -> https://api.openai.com/v1

    ShanghaiTech GenAI preferred:
      https://genaiapi.shanghaitech.edu.cn/api/v1
      + chat_path=start
      -> https://genaiapi.shanghaitech.edu.cn/api/v1/start

    If user accidentally sets:
      https://genaiapi.shanghaitech.edu.cn/api/v1/start
    we keep it, and build_url() will avoid duplicate /start.
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


def build_url(base_url: str, path: str) -> str:
    """
    Join base_url and path safely.

    If:
      base_url = https://.../api/v1/start
      path = start

    Then return:
      https://.../api/v1/start

    Not:
      https://.../api/v1/start/start
    """
    base_url = str(base_url).strip().rstrip("/")
    path = str(path or "").strip().strip("/")

    if not path:
        return base_url

    if base_url.endswith("/" + path) or base_url == path:
        return base_url

    return f"{base_url}/{path}"


def get_first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return None


def parse_extra_headers(value: Any) -> Dict[str, str]:
    """
    Parse extra headers from env/config.

    Example:
      LLM_DEFAULT_HEADERS_JSON={"X-Example-Header":"east-us-2-gpt-4.1-mini"}
    """
    if value in (None, ""):
        return {}

    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}

    if isinstance(value, str):
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            print("[WARN] LLM_DEFAULT_HEADERS_JSON is not a JSON object.")
            return {}
        except Exception as e:
            print("[WARN] Failed to parse LLM_DEFAULT_HEADERS_JSON.")
            print(repr(e))
            return {}

    return {}


def guess_default_chat_path(base_url: str, used_llm_base_url: bool) -> str:
    """
    ShanghaiTech GenAI uses:
      POST {LLM_BASE_URL}/start

    OpenAI-compatible standard usually uses:
      POST {OPENAI_BASE_URL}/chat/completions
    """
    lower = base_url.lower()

    if used_llm_base_url:
        return "start"

    if "genaiapi.shanghaitech.edu.cn" in lower:
        return "start"

    return "chat/completions"


def get_config_values(
    config: Dict[str, Any],
) -> Tuple[Optional[str], str, str, str, Optional[str], Dict[str, str]]:
    """
    Returns:
      api_key, base_url, chat_path, model, fallback_model, extra_headers
    """

    api_key = (
        get_first_env(
            "OPENAI_API_KEY_PRIMARY",
            "OPENAI_API_KEY",
            "LLM_API_KEY",
            "DEEPSEEK_API_KEY",
        )
        or find_value(
            config,
            {
                "api_key",
                "apikey",
                "key",
                "openai_api_key",
                "llm_api_key",
                "deepseek_api_key",
                "openai_api_key_primary",
            },
        )
    )

    llm_base_url = get_first_env("LLM_BASE_URL")
    openai_base_url = get_first_env("OPENAI_BASE_URL", "DEEPSEEK_BASE_URL")

    config_llm_base_url = find_value(
        config,
        {
            "llm_base_url",
            "base_url",
            "baseurl",
            "api_base",
            "api_base_url",
            "openai_base_url",
            "deepseek_base_url",
            "url",
        },
    )

    raw_base_url = llm_base_url or openai_base_url or config_llm_base_url or "https://api.openai.com/v1"
    used_llm_base_url = bool(llm_base_url) or (
        isinstance(config_llm_base_url, str)
        and "genaiapi.shanghaitech.edu.cn" in config_llm_base_url.lower()
    )

    base_url = normalize_base_url(str(raw_base_url))

    chat_path = (
        get_first_env("LLM_CHAT_PATH", "OPENAI_CHAT_PATH")
        or find_value(
            config,
            {
                "llm_chat_path",
                "openai_chat_path",
                "chat_path",
                "chat_endpoint",
                "endpoint_path",
            },
        )
        or guess_default_chat_path(base_url, used_llm_base_url)
    )

    model = (
        get_first_env("LLM_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL")
        or find_value(
            config,
            {
                "llm_model",
                "model",
                "model_name",
                "modelname",
                "openai_model",
                "deepseek_model",
                "vision_model",
                "multimodal_model",
                "filter_image_model",
            },
        )
        or "GPT-4.1-mini"
    )

    fallback_model = (
        get_first_env("LLM_FALLBACK_MODEL", "OPENAI_FALLBACK_MODEL")
        or find_value(
            config,
            {
                "llm_fallback_model",
                "fallback_model",
                "openai_fallback_model",
            },
        )
    )

    extra_headers_raw = (
        get_first_env("LLM_DEFAULT_HEADERS_JSON", "OPENAI_DEFAULT_HEADERS_JSON")
        or find_value(
            config,
            {
                "llm_default_headers_json",
                "openai_default_headers_json",
                "default_headers",
                "headers",
            },
        )
    )

    extra_headers = parse_extra_headers(extra_headers_raw)

    return (
        str(api_key) if api_key else None,
        base_url,
        str(chat_path).strip().strip("/"),
        str(model),
        str(fallback_model) if fallback_model else None,
        extra_headers,
    )


def build_headers(api_key: str, extra_headers: Dict[str, str]) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Extra headers can add routing headers or override Authorization if needed.
    headers.update(extra_headers)

    return headers


def print_current_config(
    api_key: Optional[str],
    base_url: str,
    chat_path: str,
    model: str,
    fallback_model: Optional[str],
    extra_headers: Dict[str, str],
) -> None:
    chat_url = build_url(base_url, chat_path)

    print("\n========== Current API Config ==========")
    print("API key       :", mask_key(api_key or ""))
    print("Base URL      :", base_url)
    print("Chat path     :", chat_path)
    print("Chat URL      :", chat_url)
    print("Model         :", model)
    print("Fallback model:", fallback_model or "[NONE]")

    if extra_headers:
        print("Extra headers :", ", ".join(sorted(extra_headers.keys())))
    else:
        print("Extra headers : [NONE]")

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

    # Some gateways return {"message": "..."} even without success=false.
    if "message" in data and "choices" not in data and "result" not in data:
        return str(data.get("message"))

    return None


def unwrap_result(data: Any) -> Any:
    """
    Some gateway APIs return:
      {"success": true, "result": {...OpenAI style...}}

    Others return OpenAI response directly:
      {"choices": [...]}
    """
    if isinstance(data, dict) and data.get("success") is True and "result" in data:
        return data.get("result")
    return data


def looks_like_chat_success(data: Any) -> bool:
    data = unwrap_result(data)

    if isinstance(data, str) and data.strip():
        return True

    if not isinstance(data, dict):
        return False

    if data.get("success") is False:
        return False

    if "error" in data:
        return False

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and "content" in message:
                return True

            # Some APIs may use text instead of message.content.
            if "text" in first:
                return True

    # Some APIs may directly return content/result.
    if isinstance(data.get("content"), str):
        return True

    if isinstance(data.get("message"), str) and data.get("message").strip():
        return True

    return False


def get_assistant_content(data: Any) -> str:
    data = unwrap_result(data)

    if isinstance(data, str):
        return data

    if not isinstance(data, dict):
        return ""

    try:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content

                text = first.get("text")
                if isinstance(text, str):
                    return text
    except Exception:
        pass

    for key in ["content", "message", "result"]:
        value = data.get(key)
        if isinstance(value, str):
            return value

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
            "结论：API key、鉴权 header 或 endpoint 不匹配。\n"
            "对上海科技大学 GenAI，通常应使用：\n"
            "  LLM_BASE_URL=https://genaiapi.shanghaitech.edu.cn/api/v1\n"
            "  LLM_CHAT_PATH=start\n"
            "实际请求应为：POST https://genaiapi.shanghaitech.edu.cn/api/v1/start"
        )

    if "model" in lower and ("does not exist" in lower or "not found" in lower):
        return "结论：model 名称错误，当前平台找不到这个模型。"

    if "not a multimodal model" in lower:
        return "结论：当前模型不是多模态模型，不能处理图片输入。"

    if "unsupported" in lower and ("image" in lower or "vision" in lower):
        return "结论：当前 endpoint 或模型不支持图片输入。"

    if "method" in lower and ("not support" in lower or "不支持" in lower):
        return "结论：HTTP 方法或 endpoint 路径不符合平台要求。"

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

    models_url = build_url(base_url, "models")

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
            print("这不一定影响 chat，因为有些中转平台不支持 GET /models。\n")
            return

        result = unwrap_result(data)

        if response.ok and isinstance(result, dict) and isinstance(result.get("data"), list):
            models = [
                m.get("id")
                for m in result["data"]
                if isinstance(m, dict) and m.get("id")
            ]
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
        print("这不一定影响 chat。\n")


def make_text_payload(model: str) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with only: OK",
            }
        ],
        "stream": False,
    }


def test_text_chat(
    base_url: str,
    chat_path: str,
    headers: Dict[str, str],
    model: str,
) -> bool:
    print("[TEST 2] Checking text chat completion ...")

    chat_url = build_url(base_url, chat_path)
    payload = make_text_payload(model)

    response, data = post_json(chat_url, headers, payload)

    if response is None:
        return False

    print("URL   :", chat_url)
    print("Status:", response.status_code)
    print("Text  :", response.text[:2000])

    if response.ok and looks_like_chat_success(data):
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


def make_image_payload(model: str, image_path: Path) -> Dict[str, Any]:
    data_url = image_to_data_url(image_path)

    return {
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


def test_image_chat(
    base_url: str,
    chat_path: str,
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
        payload = make_image_payload(model, image_path)
    except Exception as e:
        print("[FAILED] Could not read or encode image.")
        print(repr(e))
        print()
        return False

    print("Image :", image_path)
    print("MIME  :", guess_mime_type(image_path))

    chat_url = build_url(base_url, chat_path)

    response, data = post_json(chat_url, headers, payload)

    if response is None:
        return False

    print("URL   :", chat_url)
    print("Status:", response.status_code)
    print("Text  :", response.text[:2500])

    if response.ok and looks_like_chat_success(data):
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
        print("结论：API key、base_url、chat_path、model、图片输入格式基本都没问题。")
        print("如果 dataraider 仍报错，重点检查 dataraider 是否读取了同一套配置。")

    elif text_ok and not image_ok:
        print("[PARTIAL] 文字测试成功，但图片测试失败。")
        print("结论：API key / base_url / chat_path / 文本模型调用大概率没问题。")
        print("问题通常是：当前 model 不是多模态模型，或平台不支持这种图片输入格式。")
        print("dataraider 的 Filtering relevant images 阶段需要多模态模型。")

    elif not text_ok:
        print("[FAIL] 文字测试失败。")
        print("结论：先不要跑 dataraider。请先修复 API key、base_url、chat_path 或 model 配置。")

    print("===================================")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to startup.json. Default: ./scripts/startup.json",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Optional .env file path. If omitted, tries ./.env and ./scripts/.env.",
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
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Skip image chat completion test.",
    )
    args = parser.parse_args()

    # Load .env first so env variables can override JSON config.
    if args.env:
        load_dotenv_file(Path(args.env))
    else:
        for env_path in DEFAULT_DOTENV_PATHS:
            load_dotenv_file(env_path)

    config_path = Path(args.config)
    config = load_json_config(config_path)

    (
        api_key,
        base_url,
        chat_path,
        model,
        fallback_model,
        extra_headers,
    ) = get_config_values(config)

    print_current_config(
        api_key=api_key,
        base_url=base_url,
        chat_path=chat_path,
        model=model,
        fallback_model=fallback_model,
        extra_headers=extra_headers,
    )

    if not api_key:
        print("[ERROR] API key is missing.")
        print("请设置 OPENAI_API_KEY 或 LLM_API_KEY，或在 startup.json 里配置 api_key。")
        return

    headers = build_headers(api_key, extra_headers)

    if not args.skip_models:
        test_models_endpoint(base_url, headers)

    text_ok = test_text_chat(base_url, chat_path, headers, model)

    # If primary model failed and fallback model is set, try fallback for text.
    if not text_ok and fallback_model and fallback_model != model:
        print("[TEST 2B] Retrying text chat with fallback model ...")
        text_ok = test_text_chat(base_url, chat_path, headers, fallback_model)

        if text_ok:
            print(f"[OK] Fallback model succeeded: {fallback_model}")
            model = fallback_model

    if args.skip_image:
        image_ok = False
        print("[SKIP] Image test skipped by --skip-image.")
        print()
    else:
        image_path = Path(args.image) if args.image else None
        image_ok = test_image_chat(base_url, chat_path, headers, model, image_path)

        # If image failed and fallback model is set, try fallback for image.
        if not image_ok and fallback_model and fallback_model != model:
            print("[TEST 3B] Retrying image chat with fallback model ...")
            image_ok = test_image_chat(
                base_url,
                chat_path,
                headers,
                fallback_model,
                image_path,
            )

    print_final_summary(text_ok, image_ok)


if __name__ == "__main__":
    main()
