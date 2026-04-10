import asyncio
import json
from datetime import datetime
from typing import List, Optional, Tuple

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.utils import StandardPassThroughResponseObject

from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
)
from .llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from .success_handler import PassThroughEndpointLogging


class PassThroughStreamingHandler:
    @staticmethod
    async def chunk_processor(
        response: httpx.Response,
        request_body: Optional[dict],
        litellm_logging_obj: LiteLLMLoggingObj,
        endpoint_type: EndpointType,
        start_time: datetime,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        async_client: Optional[httpx.AsyncClient] = None,
        url: Optional[str] = None,
        headers: Optional[dict] = None,
        max_retries: Optional[int] = None,
    ):
        """
        - Yields chunks from the response
        - Collect non-empty chunks for post-processing (logging)
        - Inject cost into chunks if include_cost_in_streaming_usage is enabled
        - Detect streaming errors and trigger inline retry (Anthropic endpoint only)

        Retry behaviour:
        - On error, the failed attempt's raw_bytes are discarded (not logged).
        - On success after retry, only the successful attempt's chunks are logged.
        - If all retries are exhausted, a standard Anthropic error event is sent to
          the client so it receives a well-formed error instead of an empty stream.
        """
        try:
            if max_retries is None:
                max_retries = getattr(litellm, "num_retries", None) or litellm.DEFAULT_MAX_RETRIES

            # Extract model name for cost injection
            model_name = PassThroughStreamingHandler._extract_model_for_cost_injection(
                request_body=request_body,
                url_route=url_route,
                endpoint_type=endpoint_type,
                litellm_logging_obj=litellm_logging_obj,
            )

            retry_count = 0
            current_response = response

            while True:
                raw_bytes: List[bytes] = []  # Reset per attempt; only successful attempt is logged
                stream_error_detected = False
                error_message = None
                has_content = False  # Track whether any real Anthropic content was received

                async for chunk in current_response.aiter_bytes():
                    raw_bytes.append(chunk)

                    if endpoint_type == EndpointType.ANTHROPIC:
                        # Check for explicit Anthropic error event
                        error_detected, error_msg = PassThroughStreamingHandler._detect_anthropic_error_in_chunk(chunk)
                        if error_detected:
                            stream_error_detected = True
                            error_message = error_msg
                            verbose_proxy_logger.warning(
                                f"Detected streaming error in Anthropic response: {error_msg}, "
                                f"retry_count={retry_count}/{max_retries}"
                            )
                            break  # Do NOT yield this error chunk to the client

                        # Track whether real content has been received
                        if not has_content:
                            has_content = PassThroughStreamingHandler._chunk_has_content(chunk)

                        # ZhipuAI 1302: returns only data:[DONE] with no preceding content.
                        # Intercept before yield so client gets nothing, then retry.
                        if not has_content and b"[DONE]" in chunk:
                            stream_error_detected = True
                            error_message = "empty stream (data:[DONE] with no content, possible rate limit)"
                            verbose_proxy_logger.warning(
                                f"Empty Anthropic stream: received [DONE] without any content, "
                                f"retry_count={retry_count}/{max_retries}. "
                                f"Possible ZhipuAI rate limit (1302)."
                            )
                            break  # Do NOT yield [DONE] to client

                    if (
                        getattr(litellm, "include_cost_in_streaming_usage", False)
                        and model_name
                    ):
                        if endpoint_type == EndpointType.VERTEX_AI:
                            if "streamRawPredict" in url_route or "rawPredict" in url_route:
                                modified_chunk = ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(
                                    chunk, model_name
                                )
                                if modified_chunk is not None:
                                    chunk = modified_chunk
                        elif endpoint_type == EndpointType.ANTHROPIC:
                            modified_chunk = ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(
                                chunk, model_name
                            )
                            if modified_chunk is not None:
                                chunk = modified_chunk

                    yield chunk

                # Decide whether to retry
                can_retry = (
                    stream_error_detected
                    and retry_count < max_retries
                    and async_client is not None
                    and url is not None
                )

                if can_retry:
                    retry_count += 1
                    verbose_proxy_logger.info(
                        f"Retrying streaming request due to error: {error_message}, "
                        f"attempt {retry_count}/{max_retries}"
                    )
                    req = async_client.build_request(
                        "POST",
                        url,
                        json=request_body,
                        headers=headers,
                    )
                    current_response = await async_client.send(req, stream=True)
                    current_response.raise_for_status()
                    continue

                # Retries exhausted but last attempt still had an error:
                # send a standard Anthropic error event so the client gets a
                # well-formed response instead of an empty stream.
                if stream_error_detected and endpoint_type == EndpointType.ANTHROPIC:
                    verbose_proxy_logger.error(
                        f"All {max_retries} retries exhausted for streaming request. "
                        f"Last error: {error_message}"
                    )
                    error_event = (
                        "event: error\n"
                        "data: " + json.dumps({
                            "type": "error",
                            "error": {
                                "type": "server_error",
                                "message": error_message or "upstream error after retries",
                            }
                        }) + "\n\n"
                    )
                    yield error_event.encode("utf-8")

                break  # Normal exit (success or exhausted retries)

            # Post-processing: log only the last (successful) attempt's chunks
            end_time = datetime.now()
            asyncio.create_task(
                PassThroughStreamingHandler._route_streaming_logging_to_handler(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body or {},
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    raw_bytes=raw_bytes,
                    end_time=end_time,
                )
            )
        except (asyncio.TimeoutError, httpx.TimeoutException) as e:
            verbose_proxy_logger.warning(f"Timeout error in chunk_processor: {str(e)}")
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"Error in chunk_processor: {str(e)}")
            raise

    @staticmethod
    def _detect_anthropic_error_in_chunk(chunk: bytes) -> Tuple[bool, Optional[str]]:
        """
        Detect if a chunk contains a standard Anthropic error event:
            data: {"type":"error","error":{...}}

        Silent empty-stream detection (ZhipuAI 1302 — stream returns only
        data:[DONE] with no content) is handled in chunk_processor via
        has_content tracking, not here.
        """
        try:
            chunk_str = chunk.decode("utf-8", errors="ignore")
            for line in chunk_str.split("\n"):
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                try:
                    data = json.loads(payload)
                    if data.get("type") == "error":
                        error_info = data.get("error", {})
                        msg = error_info.get("message", str(data))
                        return True, msg
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            pass
        return False, None

    @staticmethod
    def _chunk_has_content(chunk: bytes) -> bool:
        """
        Returns True if a chunk contains actual Anthropic message content.

        Uses a blacklist of known non-content event types so that future
        Anthropic event types are not accidentally treated as empty.
        Returns False only when every data: line is a known control event
        or cannot be parsed.
        """
        # Known non-content control events
        _CONTROL_TYPES = {"ping", "error"}
        # [DONE] is OpenAI-style; in Anthropic streams it signals an empty/error stream
        _CONTROL_PAYLOADS = {"[DONE]"}

        try:
            chunk_str = chunk.decode("utf-8", errors="ignore")
            for line in chunk_str.split("\n"):
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload in _CONTROL_PAYLOADS:
                    continue
                try:
                    data = json.loads(payload)
                    if data.get("type") not in _CONTROL_TYPES:
                        return True  # Any non-control JSON event counts as content
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            pass
        return False

    @staticmethod
    async def _route_streaming_logging_to_handler(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        raw_bytes: List[bytes],
        end_time: datetime,
        model: Optional[str] = None,
    ):
        """
        Route the logging for the collected chunks to the appropriate handler

        Supported endpoint types:
        - Anthropic
        - Vertex AI
        - OpenAI
        """
        try:
            all_chunks = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(
                raw_bytes
            )
            standard_logging_response_object: Optional[
                PassThroughEndpointLoggingResultValues
            ] = None
            kwargs: dict = {}
            if endpoint_type == EndpointType.ANTHROPIC:
                anthropic_passthrough_logging_handler_result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                )
                standard_logging_response_object = (
                    anthropic_passthrough_logging_handler_result["result"]
                )
                kwargs = anthropic_passthrough_logging_handler_result["kwargs"]
            elif endpoint_type == EndpointType.VERTEX_AI:
                vertex_passthrough_logging_handler_result = VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                    model=model,
                )
                standard_logging_response_object = (
                    vertex_passthrough_logging_handler_result["result"]
                )
                kwargs = vertex_passthrough_logging_handler_result["kwargs"]
            elif endpoint_type == EndpointType.OPENAI:
                openai_passthrough_logging_handler_result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                )
                standard_logging_response_object = (
                    openai_passthrough_logging_handler_result["result"]
                )
                kwargs = openai_passthrough_logging_handler_result["kwargs"]

            if standard_logging_response_object is None:
                standard_logging_response_object = StandardPassThroughResponseObject(
                    response=f"cannot parse chunks to standard response object. Chunks={all_chunks}"
                )
            await litellm_logging_obj.async_success_handler(
                result=standard_logging_response_object,
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
                **kwargs,
            )
            if (
                litellm_logging_obj._should_run_sync_callbacks_for_async_calls()
                is False
            ):
                return

            executor.submit(
                litellm_logging_obj.success_handler,
                result=standard_logging_response_object,
                end_time=end_time,
                cache_hit=False,
                start_time=start_time,
                **kwargs,
            )
        except Exception as e:
            import traceback
            verbose_proxy_logger.error(
                f"Error in _route_streaming_logging_to_handler: {str(e)}\n{traceback.format_exc()}"
            )

    @staticmethod
    def _extract_model_for_cost_injection(
        request_body: Optional[dict],
        url_route: str,
        endpoint_type: EndpointType,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> Optional[str]:
        """
        Extract model name for cost injection from various sources.
        """
        if request_body:
            model = request_body.get("model")
            if model:
                return model

        if hasattr(litellm_logging_obj, "model_call_details"):
            model = litellm_logging_obj.model_call_details.get("model")
            if model:
                return model

        if endpoint_type == EndpointType.VERTEX_AI:
            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)
            if model and model != "unknown":
                return model

        return None

    @staticmethod
    def _convert_raw_bytes_to_str_lines(raw_bytes: List[bytes]) -> List[str]:
        """
        Converts a list of raw bytes into a list of string lines, similar to aiter_lines()

        Args:
            raw_bytes: List of bytes chunks from aiter.bytes()

        Returns:
            List of string lines, with each line being a complete data: {} chunk
        """
        combined_str = b"".join(raw_bytes).decode("utf-8")
        lines = [line.strip() for line in combined_str.split("\n") if line.strip()]
        return lines
