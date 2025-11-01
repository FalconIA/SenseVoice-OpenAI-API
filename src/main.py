# -*- coding: utf-8 -*-
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import BinaryIO, List, Literal, Optional

import ffmpeg
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from pydantic import BaseModel

# 配置日志
LOG_DIR = os.getenv("LOG_DIR", "./logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 创建日志目录
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# 生成日志文件名
log_filename = os.path.join(LOG_DIR, "sensevoice.log")

# 获取 uvicorn 的日志记录器
logger = logging.getLogger("uvicorn")

SAMPLE_RATE = 16000

# 模型基础路径
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "./models/")

# 模型名称
MODEL_NAME = "iic/SenseVoiceSmall"
VAD_MODEL_NAME = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

# 模型路径（如果未设置，则使用基础路径拼接模型名称）
model_path = os.getenv("MODEL_PATH")
if not model_path:
    model_path = os.path.join(MODEL_BASE_PATH, MODEL_NAME)

vad_path = os.getenv("VAD_PATH")
if not vad_path:
    vad_path = os.path.join(MODEL_BASE_PATH, VAD_MODEL_NAME)

# 支持任意时长音频输入
vad_enable = os.getenv("VAD_ENABLE", "false").lower() in ("true", "1", "yes")

# 推理方式
device_type = os.getenv("DEVICE_TYPE", "cpu")

# 设置用于 CPU 内部操作并行性的线程数
cpu_num = int(os.getenv("CPU_NUM", "4"))

# 识别语言
language = os.getenv("LANGUAGE", "zh")

# 批处理大小
batch_size = int(os.getenv("BATCH_SIZE", "64"))

# 使用 ITN（逆文本规范化）
use_itn = os.getenv("USE_ITN", "false").lower() in ("true", "1", "yes")

# 服务器配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9991"))

# 全局模型变量
model = None


# OpenAI 协议模型定义


# 错误响应模型
class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# 音频转录接口模型
class AudioTranscriptionResponse(BaseModel):
    text: str


# Chat Completions 接口模型
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "SenseVoiceSmall"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: dict
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型，关闭时清理资源"""
    global model

    # 启动时执行
    logger.info("=" * 50)
    logger.info(f"服务地址: {HOST}")
    logger.info(f"服务端口: {PORT}")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"VAD 路径: {vad_path}")
    logger.info(f"启用 VAD: {vad_enable}")
    logger.info(f"设备类型: {device_type}")
    logger.info(f"CPU 线程: {cpu_num}")
    logger.info(f"识别语言: {language}")
    logger.info(f"批处理数: {batch_size}")
    logger.info(f"使用 ITN: {use_itn}")
    logger.info("=" * 50)

    logger.info("开始加载 SenseVoice 模型...")
    if vad_enable:
        # 准确预测
        logger.info("使用 VAD 模式加载模型（准确预测）")
        model = AutoModel(
            model=model_path,
            vad_model=vad_path,
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=False,
            device=device_type,
            ncpu=cpu_num,
            disable_update=True,
        )
    else:
        # 快速预测
        logger.info("使用快速预测模式加载模型")
        model = AutoModel(
            model=model_path,
            trust_remote_code=False,
            device=device_type,
            ncpu=cpu_num,
            disable_update=True,
        )
    logger.info("模型加载完成")

    yield  # 应用运行期间

    # 关闭时执行 - 清理模型资源
    if model is not None:
        logger.info("开始清理模型资源...")
        try:
            # 释放模型资源
            del model

            # 如果使用 GPU，清理 GPU 缓存
            if device_type != "cpu":
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("GPU 缓存已清理")
                except ImportError:
                    pass

            logger.info("模型资源清理完成")
        except Exception as e:
            logger.error(f"清理模型资源时出错: {e}")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    """根路径 - API 信息"""
    return {
        "message": "SenseVoice OpenAI API",
        "version": "1.0.0",
        "endpoints": {
            "audio_transcriptions": "/v1/audio/transcriptions",
            "chat_completions": "/v1/chat/completions",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """健康检查端点 - 参考 Spring Boot Actuator 结构"""
    # 检查模型状态
    model_status = "UP" if model is not None else "DOWN"

    # 整体状态：所有组件都 UP 才是 UP
    overall_status = model_status

    return {
        "status": overall_status,
        "components": {
            "model": {
                "status": model_status,
                "details": {
                    "loaded": model is not None,
                    "path": model_path,
                    "device": device_type,
                },
            },
            "vad": {
                "status": model_status if vad_enable else "NOT_CONFIGURED",
                "details": {
                    "enabled": vad_enable,
                    "path": vad_path if vad_enable else None,
                },
            },
            "service": {
                "status": "UP",
                "details": {
                    "host": HOST,
                    "port": PORT,
                    "language": language,
                    "batch_size": batch_size,
                    "use_itn": use_itn,
                },
            },
        },
    }


@app.post("/v1/audio/transcriptions", response_model=AudioTranscriptionResponse)
async def transcriptions(file: UploadFile = File(...)):
    """
    OpenAI 兼容的音频转录接口
    将音频文件转录为文本
    """
    try:
        # 参数校验：检查文件是否存在
        if not file:
            logger.warning("请求缺少文件参数")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Missing required parameter: 'file'",
                        "type": "invalid_request_error",
                        "param": "file",
                        "code": "missing_required_parameter",
                    }
                },
            )

        # 参数校验：检查文件名
        if not file.filename:
            logger.warning("文件名为空")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Invalid file: filename is required",
                        "type": "invalid_request_error",
                        "param": "file",
                        "code": "invalid_file",
                    }
                },
            )

        # 参数校验：检查文件扩展名
        allowed_extensions = {
            ".mp3",
            ".mp4",
            ".mpeg",
            ".mpga",
            ".m4a",
            ".wav",
            ".webm",
            ".flac",
            ".ogg",
            ".opus",
        }
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            logger.warning(f"不支持的文件格式: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Invalid file format: '{file_ext}'. Supported formats: {', '.join(sorted(allowed_extensions))}",
                        "type": "invalid_request_error",
                        "param": "file",
                        "code": "unsupported_file_format",
                    }
                },
            )

        # 参数校验：检查文件大小（OpenAI 限制为 25MB）
        max_file_size = int(os.getenv("MAX_FILE_SIZE", str(25 * 1024 * 1024)))  # 25MB
        file.file.seek(0, 2)  # 移动到文件末尾
        file_size = file.file.tell()
        file.file.seek(0)  # 回到开头

        if file_size == 0:
            logger.warning("文件大小为 0")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Invalid file: file is empty",
                        "type": "invalid_request_error",
                        "param": "file",
                        "code": "invalid_file",
                    }
                },
            )

        if file_size > max_file_size:
            logger.warning(f"文件过大: {file_size} bytes (最大 {max_file_size} bytes)")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"File size exceeds maximum allowed size of {max_file_size / (1024 * 1024):.1f}MB",
                        "type": "invalid_request_error",
                        "param": "file",
                        "code": "file_too_large",
                    }
                },
            )

        logger.info(
            f"接收音频转录请求 - 文件名: {file.filename}, 内容类型: {file.content_type}, 大小: {file_size} bytes"
        )

        # 加载音频
        try:
            data = load_audio(file.file)
            logger.info(f"音频加载成功，数据形状: {data.shape}")
        except RuntimeError as e:
            logger.error(f"音频加载失败: {e}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Failed to process audio file: {str(e)}",
                        "type": "invalid_request_error",
                        "param": "file",
                        "code": "audio_processing_failed",
                    }
                },
            )

        # 模型推理
        try:
            res = model.generate(
                input=data,
                cache={},
                language=language,
                use_itn=use_itn,
                batch_size=batch_size,
            )

            if not res or len(res) == 0:
                raise ValueError("模型返回结果为空")

            result = rich_transcription_postprocess(res[0]["text"])
            logger.info(f"音频转录完成，结果: {result}")

            return AudioTranscriptionResponse(text=result)

        except Exception as e:
            logger.exception(f"模型推理失败: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": f"Transcription failed: {str(e)}",
                        "type": "server_error",
                        "param": None,
                        "code": "transcription_error",
                    }
                },
            )

    except HTTPException:
        # 重新抛出 HTTPException（已经格式化好的错误）
        raise
    except Exception as e:
        # 捕获所有其他未预期的错误
        logger.exception(f"未知错误: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Internal server error: {str(e)}",
                    "type": "server_error",
                    "param": None,
                    "code": "internal_error",
                }
            },
        )


def load_audio(file: BinaryIO, encode=True, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    out = b""
    if encode:
        try:
            logger.debug(f"开始使用 ffmpeg 处理音频，采样率: {sr}")
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd="ffmpeg",
                    capture_stdout=True,
                    capture_stderr=True,
                    input=file.read(),
                )
            )
            logger.debug(f"ffmpeg 处理完成，输出大小: {len(out)} bytes")
        except ffmpeg.Error as e:
            logger.error(f"ffmpeg 处理音频失败: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI 兼容的 Chat Completions 接口
    支持流式和非流式响应，用于 OneAPI 渠道测试
    """
    logger.info(f"接收 Chat Completions 请求 - 模型: {request.model}, 流式: {request.stream}")

    # 生成唯一的请求 ID
    request_id = f"chat-{int(time.time() * 1000)}"
    created_time = int(time.time())

    # 响应消息内容
    response_content = "OK"

    if request.stream:
        # 流式响应
        logger.info("返回流式响应")

        async def generate_stream():
            # 单个 chunk 返回完整内容
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={"role": "assistant", "content": response_content},
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

            # 结束标记
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # 非流式响应
        logger.info("返回非流式响应")
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=len(str(request.messages)),
                completion_tokens=len(response_content),
                total_tokens=len(str(request.messages)) + len(response_content),
            ),
        )
        return response


if __name__ == "__main__":
    # 配置 uvicorn 日志，添加文件输出
    import copy

    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)

    # 添加文件处理器配置
    log_config["handlers"]["file"] = {
        "class": "logging.handlers.TimedRotatingFileHandler",
        "filename": log_filename,
        "when": "midnight",  # 每天午夜轮转
        "interval": 1,  # 轮转间隔为 1 天
        "backupCount": 30,  # 保留 30 天的日志
        "formatter": "default",
        "encoding": "utf-8",
    }

    # 让 uvicorn 和 uvicorn.access 的日志同时输出到控制台和文件
    log_config["loggers"]["uvicorn"]["handlers"] = ["default", "file"]
    log_config["loggers"]["uvicorn.access"]["handlers"] = ["access", "file"]

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL.lower(),
        log_config=log_config,
    )
