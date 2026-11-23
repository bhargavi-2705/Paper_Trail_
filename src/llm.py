import os


def get_llm(cfg):
    l = cfg["llm"]
    temp = l.get("temperature", 0)
    if l["provider"] == "anthropic":
        from langchain_anthropic import ChatAnthropic   # reads ANTHROPIC_API_KEY
        return ChatAnthropic(model=l["model"], max_tokens=l["max_tokens"], temperature=temp)
    if l["provider"] == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=l["model"], max_tokens=l["max_tokens"], temperature=temp,
            base_url=l.get("base_url"),
            api_key=os.environ.get("OPENAI_API_KEY", "none"),
            reasoning_effort=l.get("reasoning_effort"),   # None = leave to the provider default
            timeout=l.get("timeout", 120),
            max_retries=l.get("max_retries", 6),          # 429s: wait and retry instead of crashing
        )
    raise ValueError(f"unknown provider: {l['provider']}")
