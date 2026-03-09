# API Keys Setup Guide

This guide provides detailed instructions on how to obtain API keys for all LLM providers and tools used in AgentFlow.

---

## 1. OpenAI API Key

**Purpose**: Access OpenAI's language models (GPT-5, GPT-4o, etc.), used in AgentFlow for judging answer correctness.

**How to obtain**:
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up/Log in to your account
3. Navigate to [API Keys page](https://platform.openai.com/api-keys)
4. Click "Create new secret key" to generate a new API key

**Available models**: [OpenAI Models Documentation](https://platform.openai.com/docs/models)

**Common model names**: `gpt-4o`,`gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`

---

## 2. Google API Key

**Purpose**: used in AgentFlow for Google Search tool.

**How to obtain**:
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Log in with your Google account
3. Click "Get API key"
4. Create or select a Google Cloud project
5. Copy the generated API key

---

## 3. DashScope API Key (Alibaba Cloud)

**Purpose**: Access Alibaba Cloud's Qwen (Tongyi Qianwen) model series. In AgentFlow, we use DashScope to call **Qwen-2.5-7B-Instruct** as the LLM engine for agents (except planner) and tools.

**How to obtain**:
1. Visit [Alibaba Cloud DashScope Console](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.609055efmpUjqR&tab=model#/model-market)
2. Log in with your Alibaba Cloud account
3. Navigate to [API-KEY Management](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.609055efmpUjqR&tab=model#/api-key)
4. Create a new API key

**Official guide**: [Get API Key](https://help.aliyun.com/zh/model-studio/get-api-key) (you may need to translate the page to English using your browser's translation feature)

**Available models**: [DashScope Model Documentation](https://help.aliyun.com/zh/dashscope/developer-reference/model-square)

**Common model names**: `qwen-turbo`, `qwen-plus`, `qwen-max`, `qwen2.5-7b-instruct`, `qwen2.5-72b-instruct`

> **Note**: For international users, we recommend using [Together AI](#4-together-api-key---recommended-for-international-users) to access Qwen-2.5-7B-Instruct model. Alternatively, you can serve the model locally using vLLM.

---

## 4. Together API Key - Recommended for More International Users

**Purpose**: Access open-source models on TogetherAI platform, including Qwen, Llama, Mixtral, etc.

**Recommended for**: International users who want to access Qwen-2.5-7B-Instruct and other open-source models with better global network connectivity.

**How to obtain**:
1. Visit [Together.ai](https://www.together.ai/)
2. Sign up/Log in to your account
3. Navigate to [Settings > API Keys](https://api.together.xyz/settings/api-keys)
4. Create a new API key

**Available models**: [Together Models Documentation](https://docs.together.ai/docs/inference-models)

**Common model names**:
- Qwen models: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`
- Other models: `meta-llama/Llama-3-70b-chat-hf`, `mistralai/Mixtral-8x7B-Instruct-v0.1`

> **Important Note for Qwen Models**: Together AI offers both Turbo (quantized) and standard (non-quantized) versions of Qwen models. For best performance and accuracy, we recommend using the **non-quantized versions** (e.g., `Qwen/Qwen2.5-7B-Instruct` instead of `Qwen/Qwen2.5-7B-Instruct-Turbo`). The Turbo versions are faster but may have reduced quality due to quantization.

---

## 5. Forge API Key

**Purpose**: Access Forge's OpenAI-compatible router with model strings like `OpenAI/gpt-4o-mini`.

**Environment variables**:
- `FORGE_API_KEY`
- `FORGE_API_BASE` (optional, defaults to `https://api.forge.tensorblock.co/v1`)


## Important Notes

1. **Security**: All API keys are sensitive information - never expose them or commit to public repositories
2. **Costs**: Most API services are paid - understand pricing before use
3. **Quotas**: Some services have free tiers or rate limits - monitor your usage
4. **Environment Variables**: Copy `.env.template` to `.env` and fill in your actual API keys
5. **Regional Recommendations**:
   - **China users**: Use DashScope for Qwen models
   - **International users**: Use Together AI for Qwen models

---

## Quick Reference Table

| API Key | Purpose | Sign Up Link | Documentation |
|---------|---------|--------------|---------------|
| OPENAI_API_KEY | OpenAI GPT models | [platform.openai.com](https://platform.openai.com/) | [Docs](https://platform.openai.com/docs/models) |
| GOOGLE_API_KEY | Gemini models | [aistudio.google.com](https://aistudio.google.com/) | [Docs](https://ai.google.dev/gemini-api/docs/models/gemini) |
| DASHSCOPE_API_KEY | Qwen models | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/) | [Docs](https://help.aliyun.com/zh/dashscope/developer-reference/model-square) |
| TOGETHER_API_KEY | Qwen & open-source models | [together.ai](https://www.together.ai/) | [Docs](https://docs.together.ai/docs/inference-models) |
| FORGE_API_KEY | Forge router | [forge.tensorblock.co](https://forge.tensorblock.co) | [Docs](https://forge.tensorblock.co) |
