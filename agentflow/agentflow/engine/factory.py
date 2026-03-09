from typing import Any
import os


def create_llm_engine(model_string: str, use_cache: bool = False, is_multimodal: bool = False, **kwargs) -> Any:
    print(f"Creating LLM engine for model: {model_string}")
    """
    Factory function to create appropriate LLM engine instance.

    For supported models and model_string examples, see:
    https://github.com/lupantech/AgentFlow/blob/main/assets/doc/llm_engine.md

    - Uses kwargs.get() instead of setdefault
    - Only passes supported parameters to each backend
    - Handles frequency_penalty, presence_penalty, repetition_penalty per backend
    - External parameters (temperature, top_p) are respected if provided
    """
    original_model_string = model_string

    print(f"creating llm engine {model_string} with: is_multimodal: {is_multimodal}, kwargs: {kwargs}")

    model_string_lower = model_string.lower()

    # === Forge (OpenAI-compatible router) ===
    if model_string_lower.startswith("forge-"):
        from .openai import ChatOpenAI

        if "FORGE_API_KEY" not in os.environ:
            raise ValueError("Please set the FORGE_API_KEY environment variable.")

        model_string = model_string[len("forge-"):]
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "presence_penalty": kwargs.get("presence_penalty", 0.5),
            "api_key": os.getenv("FORGE_API_KEY"),
            "base_url": os.getenv("FORGE_API_BASE") or "https://api.forge.tensorblock.co/v1",
        }
        return ChatOpenAI(**config)

    # === Azure OpenAI ===
    if "azure" in model_string:
        from .azure import ChatAzureOpenAI
        model_string = model_string.replace("azure-", "")

        # Azure supports: temperature, top_p, frequency_penalty, presence_penalty
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "presence_penalty": kwargs.get("presence_penalty", 0.5),
        }
        return ChatAzureOpenAI(**config)

    # === OpenAI (GPT) ===
    elif any(x in model_string for x in ["gpt", "o1", "o3", "o4"]):
        from .openai import ChatOpenAI
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "presence_penalty": kwargs.get("presence_penalty", 0.5),
        }
        return ChatOpenAI(**config)

    # === DashScope (Qwen) ===
    elif "dashscope" in model_string:
        from .dashscope import ChatDashScope
        # DashScope uses temperature, top_p — but not frequency/presence_penalty
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }
        return ChatDashScope(**config)

    # === Anthropic (Claude) ===
    elif "claude" in model_string:
        from .anthropic import ChatAnthropic

        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")

        # Anthropic supports: temperature, top_p, top_k — NOT frequency/presence_penalty
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),  # optional
        }
        return ChatAnthropic(**config)

    # === DeepSeek ===
    elif any(x in model_string for x in ["deepseek-chat", "deepseek-reasoner"]):
        from .deepseek import ChatDeepseek

        # DeepSeek uses repetition_penalty, not frequency/presence
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
        }
        return ChatDeepseek(**config)

    # === Gemini ===
    elif "gemini" in model_string:
        print("gemini model found")
        from .gemini import ChatGemini
        # Gemini uses repetition_penalty
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
        }
        return ChatGemini(**config)

    # === Grok (xAI) ===
    elif "grok" in model_string:
        from .xai import ChatGrok
        if "GROK_API_KEY" not in os.environ:
            raise ValueError("Please set the GROK_API_KEY environment variable.")

        # Assume Grok uses repetition_penalty
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
        }
        return ChatGrok(**config)

    # === vLLM ===
    elif "vllm" in model_string:
        from .vllm import ChatVLLM

        model_string = model_string.replace("vllm-", "")
        config = {
            "model_string": model_string,
            "base_url": kwargs.get("base_url", "http://localhost:8000/v1"), # TODO: check the RL training initialized port and name
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "frequency_penalty": kwargs.get("frequency_penalty", 1.2),
            "max_model_len": kwargs.get("max_model_len", 15200),
            "max_seq_len_to_capture": kwargs.get("max_seq_len_to_capture", 15200),
        }
        print("serving ")
        return ChatVLLM(**config)

    # === LiteLLM ===
    elif "litellm" in model_string:
        from .litellm import ChatLiteLLM

        model_string = model_string.replace("litellm-", "")
        # LiteLLM supports frequency/presence_penalty as routing params
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "presence_penalty": kwargs.get("presence_penalty", 0.5),
        }
        return ChatLiteLLM(**config)

    # === Together AI ===
    elif "together" in model_string:
        from .together import ChatTogether

        if "TOGETHER_API_KEY" not in os.environ:
            raise ValueError("Please set the TOGETHER_API_KEY environment variable.")

        model_string = model_string.replace("together-", "")
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
        }
        return ChatTogether(**config)

    # === Ollama ===
    elif "ollama" in model_string:
        from .ollama import ChatOllama

        model_string = model_string.replace("ollama-", "")
        config = {
            "model_string": model_string,
            "use_cache": use_cache,
            "is_multimodal": is_multimodal,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
        }
        return ChatOllama(**config)

    else:
        raise ValueError(
            f"Engine {original_model_string} not supported. "
            "If you are using Azure OpenAI models, please ensure the model string has the prefix 'azure-'. "
            "For Together models, use 'together-'. For VLLM models, use 'vllm-'. For LiteLLM models, use 'litellm-'. "
            "For Ollama models, use 'ollama-'. "
            "For other custom engines, you can edit the factory.py file and add its interface file. "
            "Your pull request will be warmly welcomed!"
        )
