"""
Shared Configuration Module - Multi-Provider
============================================

Centralizes environment variable loading, validation, and LangChain client
initialization.

Supports:
- OpenAI for chat and embeddings
- AWS Bedrock for chat and embeddings

Provider selection:
- LLM_PROVIDER=openai | bedrock
- EMBEDDINGS_PROVIDER=openai | bedrock

Examples:
    from config import load_config, get_llm, get_embeddings

    config = load_config()
    llm = get_llm()
    embeddings = get_embeddings()
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


ProviderName = Literal["openai", "bedrock"]

SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "bedrock")

DEFAULT_LLM_PROVIDER: ProviderName = "openai"
DEFAULT_EMBEDDINGS_PROVIDER: ProviderName = "openai"

DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-small"

DEFAULT_BEDROCK_CHAT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
DEFAULT_BEDROCK_EMBEDDINGS_MODEL = "amazon.titan-embed-text-v2:0"


class ConfigError(Exception):
    """Raised when the application configuration is invalid."""

    def __init__(
        self,
        message: str,
        *,
        var_name: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.var_name = var_name
        self.provider = provider
        super().__init__(message)


@dataclass(frozen=True)
class AppConfig:
    """Resolved application configuration."""

    llm_provider: ProviderName
    embeddings_provider: ProviderName

    openai_api_key: Optional[str]
    aws_region: Optional[str]

    default_temperature: float

    openai_chat_model: str
    openai_embeddings_model: str

    bedrock_chat_model_id: str
    bedrock_embeddings_model_id: str

    langsmith_api_key: Optional[str]
    langsmith_project: str

    google_api_key: Optional[str]
    deepeval_api_key: Optional[str]

    postgres_url: Optional[str]
    redis_url: Optional[str]


def load_config(env_file: Optional[str | Path] = None) -> AppConfig:
    """Load environment variables and validate the selected providers.

    Args:
        env_file: Optional explicit path to a .env file. If omitted, the function
                  looks for .env in the same directory as this file.

    Returns:
        AppConfig: Resolved and validated application configuration.

    Raises:
        ConfigError: If any required environment variable is missing or invalid.
    """
    _load_dotenv(env_file)

    llm_provider = _get_provider("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    embeddings_provider = _get_provider("EMBEDDINGS_PROVIDER", DEFAULT_EMBEDDINGS_PROVIDER)

    _validate_provider_config(llm_provider)
    _validate_provider_config(embeddings_provider)

    config = AppConfig(
        llm_provider=llm_provider,
        embeddings_provider=embeddings_provider,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        aws_region=os.environ.get("AWS_REGION"),
        default_temperature=_get_float_env("DEFAULT_TEMPERATURE", 0.7),
        openai_chat_model=os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL),
        openai_embeddings_model=os.environ.get(
            "OPENAI_EMBEDDINGS_MODEL",
            DEFAULT_OPENAI_EMBEDDINGS_MODEL,
        ),
        bedrock_chat_model_id=os.environ.get(
            "BEDROCK_MODEL_ID",
            DEFAULT_BEDROCK_CHAT_MODEL,
        ),
        bedrock_embeddings_model_id=os.environ.get(
            "BEDROCK_EMBEDDINGS_MODEL_ID",
            DEFAULT_BEDROCK_EMBEDDINGS_MODEL,
        ),
        langsmith_api_key=os.environ.get("LANGSMITH_API_KEY"),
        langsmith_project=os.environ.get("LANGSMITH_PROJECT", "langchain-study"),
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        deepeval_api_key=os.environ.get("DEEPEVAL_API_KEY"),
        postgres_url=os.environ.get("POSTGRES_URL"),
        redis_url=os.environ.get("REDIS_URL"),
    )

    warnings.warn(
        (
            f"Using LLM provider '{config.llm_provider}' "
            f"and embeddings provider '{config.embeddings_provider}'."
        ),
        stacklevel=2,
    )

    return config


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    provider: Optional[ProviderName] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a configured LangChain chat model.

    Args:
        model: Optional model name/model ID override.
        temperature: Optional temperature override.
        provider: Optional provider override. Defaults to LLM_PROVIDER.
        **kwargs: Extra provider-specific arguments passed to the LangChain client.

    Returns:
        BaseChatModel: A ready-to-use chat model.
    """
    load_config()

    selected_provider = provider or _get_provider("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    _validate_provider_config(selected_provider)

    resolved_temperature = (
        temperature
        if temperature is not None
        else _get_float_env("DEFAULT_TEMPERATURE", 0.7)
    )

    factory = _LLM_FACTORIES[selected_provider]
    return factory(model=model, temperature=resolved_temperature, **kwargs)


def get_embeddings(
    provider: Optional[ProviderName] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Embeddings:
    """Create a configured LangChain embeddings model.

    Args:
        provider: Optional provider override. Defaults to EMBEDDINGS_PROVIDER.
        model: Optional embeddings model override.
        **kwargs: Extra provider-specific arguments passed to the LangChain client.

    Returns:
        Embeddings: A ready-to-use embeddings model.
    """
    load_config()

    selected_provider = provider or _get_provider(
        "EMBEDDINGS_PROVIDER",
        DEFAULT_EMBEDDINGS_PROVIDER,
    )
    _validate_provider_config(selected_provider)

    factory = _EMBEDDINGS_FACTORIES[selected_provider]
    return factory(model=model, **kwargs)


def validate_env_vars(required_vars: list[str]) -> None:
    """Validate an explicit list of required environment variables.

    This function is useful for modules that require additional variables
    beyond the LLM/embeddings providers.
    """
    for var_name in required_vars:
        if not os.environ.get(var_name):
            raise ConfigError(
                (
                    f"Environment variable '{var_name}' is required. "
                    "Set it in your .env file or system environment."
                ),
                var_name=var_name,
            )


def _load_dotenv(env_file: Optional[str | Path]) -> None:
    if env_file is not None:
        env_path = Path(env_file).expanduser().resolve()
    else:
        env_path = Path(__file__).resolve().parent / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        return

    warnings.warn(
        f".env file not found at {env_path}. Falling back to system environment variables.",
        stacklevel=2,
    )
    load_dotenv()


def _get_provider(env_var: str, default: ProviderName) -> ProviderName:
    provider = os.environ.get(env_var, default).lower().strip()

    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            (
                f"Unsupported provider '{provider}' in {env_var}. "
                f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
            ),
            var_name=env_var,
            provider=provider,
        )

    return provider  # type: ignore[return-value]


def _validate_provider_config(provider: ProviderName) -> None:
    if provider == "openai":
        _require_env("OPENAI_API_KEY", provider)
        return

    if provider == "bedrock":
        _require_env("AWS_REGION", provider)
        return

    raise ConfigError(f"Unsupported provider '{provider}'.", provider=provider)


def _require_env(var_name: str, provider: ProviderName) -> None:
    if not os.environ.get(var_name):
        raise ConfigError(
            (
                f"Environment variable '{var_name}' is required for provider '{provider}'. "
                "Set it in your .env file or system environment."
            ),
            var_name=var_name,
            provider=provider,
        )


def _get_float_env(var_name: str, default: float) -> float:
    raw_value = os.environ.get(var_name)

    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable '{var_name}' must be a valid float. Received: {raw_value!r}.",
            var_name=var_name,
        ) from exc


def _create_openai_llm(
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL),
        temperature=temperature,
        **kwargs,
    )


def _create_bedrock_llm(
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    try:
        from langchain_aws import ChatBedrock
    except ImportError as exc:
        raise ConfigError(
            "Package 'langchain-aws' is not installed. Run: pip install langchain-aws boto3",
            provider="bedrock",
        ) from exc

    model_kwargs = kwargs.pop("model_kwargs", {})
    model_kwargs.setdefault("temperature", temperature)

    try:
        return ChatBedrock(
            model_id=model or os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_CHAT_MODEL),
            region_name=os.environ["AWS_REGION"],
            model_kwargs=model_kwargs,
            **kwargs,
        )
    except Exception as exc:
        if _looks_like_aws_credentials_error(exc):
            raise ConfigError(
                (
                    "AWS credentials are not configured for provider 'bedrock'. "
                    "Configure AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, "
                    "AWS SSO, or an IAM role."
                ),
                provider="bedrock",
            ) from exc

        raise


def _create_openai_embeddings(
    model: Optional[str] = None,
    **kwargs: Any,
) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=model or os.environ.get(
            "OPENAI_EMBEDDINGS_MODEL",
            DEFAULT_OPENAI_EMBEDDINGS_MODEL,
        ),
        **kwargs,
    )


def _create_bedrock_embeddings(
    model: Optional[str] = None,
    **kwargs: Any,
) -> Embeddings:
    try:
        from langchain_aws import BedrockEmbeddings
    except ImportError as exc:
        raise ConfigError(
            "Package 'langchain-aws' is not installed. Run: pip install langchain-aws boto3",
            provider="bedrock",
        ) from exc

    return BedrockEmbeddings(
        model_id=model or os.environ.get(
            "BEDROCK_EMBEDDINGS_MODEL_ID",
            DEFAULT_BEDROCK_EMBEDDINGS_MODEL,
        ),
        region_name=os.environ["AWS_REGION"],
        **kwargs,
    )


def _looks_like_aws_credentials_error(exc: Exception) -> bool:
    text = str(exc).lower()
    class_name = exc.__class__.__name__.lower()

    credential_markers = (
        "credential",
        "nocredentials",
        "unable to locate credentials",
        "the security token included in the request is invalid",
        "unrecognizedclientexception",
    )

    return any(marker in text or marker in class_name for marker in credential_markers)


_LLM_FACTORIES: dict[
    ProviderName,
    Callable[..., BaseChatModel],
] = {
    "openai": _create_openai_llm,
    "bedrock": _create_bedrock_llm,
}

_EMBEDDINGS_FACTORIES: dict[
    ProviderName,
    Callable[..., Embeddings],
] = {
    "openai": _create_openai_embeddings,
    "bedrock": _create_bedrock_embeddings,
}
