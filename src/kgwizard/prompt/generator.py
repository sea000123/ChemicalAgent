# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Any, Union

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from .builder import (
    HEADER_PATH,
    INSTRUCTIONS_PATH,
    TAIL_PATH,
    apply_substitutions,
    build_guidelines,
)

# client = OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get(
        "OPENAI_BASE_URL",
        "https://genaiapi.shanghaitech.edu.cn/api/v1/start"
    ),
)


def build_prompt(
    s: str
) -> dict[str, str]:
    """
    Construct a prompt dictionary from a given string.

    :param s: The prompt content.
    :type s: str
    :return: A dictionary containing the role (`"user"`) and prompt content.
    :rtype: dict[str, str]
    """
    return {
        "role": "user"
        , "content": s
    }


def build_prompt_from_react(
    react_str: str
    , header_path: Union[None, str, Path] = HEADER_PATH
    , instructions_path: Union[None, str, Path] = INSTRUCTIONS_PATH
    , tail_path: Union[None, str, Path] = TAIL_PATH
    , **kwargs
) -> dict[str,str]:
    """
    Construct a prompt by applying template substitutions using a React-style JSON string.

    :param react_str: The React-style JSON string used for substitutions.
    :type react_str: str
    :param kwargs: Additional keyword arguments for template substitution.
    :type kwargs: dict[str, str]
    :return: A structured prompt dictionary.
    :rtype: dict[str, str]
    """
    return build_prompt(
        str(apply_substitutions(
            build_guidelines(
                header_path=header_path
                , instructions_path=instructions_path
                , tail_path=tail_path
            )
            , **{"json": react_str
              , **kwargs
              }
        ))
    )


def build_prompt_from_react_file(
    path: Union[str, Path]
    , header_path: Union[None, str, Path] = HEADER_PATH
    , instructions_path: Union[None, str, Path] = INSTRUCTIONS_PATH
    , tail_path: Union[None, str, Path] = TAIL_PATH
    , **kwargs
) -> dict[str, str]:
    """
    Read a React-style JSON file and construct a prompt with applied substitutions.

    :param path: The path to the JSON file.
    :type path: Path | str
    :param kwargs: Additional keyword arguments for template substitution.
    :type kwargs: dict[str, str]
    :return: A structured prompt dictionary.
    :rtype: dict[str, str]
    """
    with open(path, 'r', encoding='utf-8') as f:
        return build_prompt_from_react(
            f.read()
            , header_path=header_path
            , instructions_path=instructions_path
            , tail_path=tail_path
            , **kwargs
        )


def _extract_assistant_text(chat_completion) -> str:
    """
    Extract assistant output from standard OpenAI content field,
    or from non-standard fields used by the ShanghaiTech gateway.
    """
    msg = chat_completion.choices[0].message

    # Standard OpenAI-compatible field
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    # ShanghaiTech gateway seems to put output here
    reasoning = getattr(msg, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    # Some OpenAI SDK versions store unknown fields in model_extra
    model_extra = getattr(msg, "model_extra", None) or {}
    for key in ("reasoning", "reasoning_content"):
        value = model_extra.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Last fallback: inspect dumped dict
    try:
        dumped = msg.model_dump()
        for key in ("content", "reasoning", "reasoning_content"):
            value = dumped.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        pass

    return ""


def get_response(
    messages: list[dict[str, Any]]
) -> dict[str, str]:
    """
    Send a list of messages to the OpenAI-compatible API and retrieve the assistant's response.
    """
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=os.environ.get("OPENAI_MODEL", "qwen-instruct"),
        temperature=0,
        max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "4096")),
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
    )

    content = _extract_assistant_text(chat_completion)

    if not content:
        raise RuntimeError(
            "Empty assistant response. Full response: "
            + str(chat_completion.model_dump())
        )

    return {
        "role": "assistant",
        "content": content,
    }

