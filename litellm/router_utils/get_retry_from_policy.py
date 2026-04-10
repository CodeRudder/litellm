"""
Get num retries for an exception.

- Account for retry policy by exception type.
"""

from typing import Dict, Optional, Union

import httpx

from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.types.router import RetryPolicy
from litellm._logging import verbose_router_logger


def get_num_retries_from_retry_policy(
    exception: Exception,
    retry_policy: Optional[Union[RetryPolicy, dict]] = None,
    model_group: Optional[str] = None,
    model_group_retry_policy: Optional[Dict[str, RetryPolicy]] = None,
):
    """
    BadRequestErrorRetries: Optional[int] = None
    AuthenticationErrorRetries: Optional[int] = None
    TimeoutErrorRetries: Optional[int] = None
    RateLimitErrorRetries: Optional[int] = None
    ContentPolicyViolationErrorRetries: Optional[int] = None
    """
    # if we can find the exception then in the retry policy -> return the number of retries

    if (
        model_group_retry_policy is not None
        and model_group is not None
        and model_group in model_group_retry_policy
    ):
        retry_policy = model_group_retry_policy.get(model_group, None)  # type: ignore

    if retry_policy is None:
        return None
    if isinstance(retry_policy, dict):
        retry_policy = RetryPolicy(**retry_policy)

    if (
        isinstance(exception, AuthenticationError)
        and retry_policy.AuthenticationErrorRetries is not None
    ):
        num = retry_policy.AuthenticationErrorRetries
        verbose_router_logger.debug(f"RetryPolicy: AuthenticationError -> {num} retries, model_group={model_group}")
        return num
    if isinstance(exception, Timeout) and retry_policy.TimeoutErrorRetries is not None:
        num = retry_policy.TimeoutErrorRetries
        verbose_router_logger.debug(f"RetryPolicy: Timeout -> {num} retries, model_group={model_group}")
        return num
    if (
        isinstance(exception, RateLimitError)
        and retry_policy.RateLimitErrorRetries is not None
    ):
        num = retry_policy.RateLimitErrorRetries
        verbose_router_logger.debug(f"RetryPolicy: RateLimitError -> {num} retries, model_group={model_group}")
        return num
    if (
        isinstance(exception, ContentPolicyViolationError)
        and retry_policy.ContentPolicyViolationErrorRetries is not None
    ):
        num = retry_policy.ContentPolicyViolationErrorRetries
        verbose_router_logger.debug(f"RetryPolicy: ContentPolicyViolationError -> {num} retries, model_group={model_group}")
        return num
    # 检查 httpx 的 timeout 异常
    if isinstance(exception, httpx.ReadTimeout) and retry_policy.TimeoutErrorRetries is not None:
        num = retry_policy.TimeoutErrorRetries
        verbose_router_logger.debug(f"RetryPolicy: httpx.ReadTimeout -> {num} retries, model_group={model_group}")
        return num
    if isinstance(exception, httpx.TimeoutException) and retry_policy.TimeoutErrorRetries is not None:
        num = retry_policy.TimeoutErrorRetries
        verbose_router_logger.debug(f"RetryPolicy: httpx.TimeoutException -> {num} retries, model_group={model_group}")
        return num
    if (
        isinstance(exception, BadRequestError)
        and retry_policy.BadRequestErrorRetries is not None
    ):
        num = retry_policy.BadRequestErrorRetries
        verbose_router_logger.debug(f"RetryPolicy: BadRequestError -> {num} retries, model_group={model_group}")
        return num
    if (
        isinstance(exception, InternalServerError)
        and retry_policy.InternalServerErrorRetries is not None
    ):
        num = retry_policy.InternalServerErrorRetries
        verbose_router_logger.debug(f"RetryPolicy: InternalServerError -> {num} retries, model_group={model_group}")
        return num
    if (
        isinstance(exception, ServiceUnavailableError)
        and retry_policy.ServiceUnavailableErrorRetries is not None
    ):
        num = retry_policy.ServiceUnavailableErrorRetries
        verbose_router_logger.debug(f"RetryPolicy: ServiceUnavailableError -> {num} retries, model_group={model_group}")
        return num


def reset_retry_policy() -> RetryPolicy:
    return RetryPolicy()
