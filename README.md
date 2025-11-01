# SenseVoice-OpenAI-API

基于 SenseVoice 的 FunASR 版本封装的 RESTful API 服务，完全兼容 OpenAI API 协议。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

## ✨ 功能特性

- 🎤 **高质量语音识别** - 基于阿里达摩院 SenseVoice 模型
- 🎯 **完全兼容 OpenAI API** - 支持 OpenAI SDK 和 OneAPI 等第三方平台
- 🐳 **Docker 支持** - 提供完整的 Docker 和 Docker Compose 部署方案
- ⚙️ **灵活配置** - 所有配置项均支持环境变量

## 📋 目录结构

```
SenseVoice-OpenAI-API/
├── src/                            # 源代码
│   └── main.py                     # 主程序
├── tests/                          # 测试文件
│   ├── audio/                      # 测试音频文件
│   │   ├── asr_example_zh.wav
│   │   └── asr_example_zh_1.m4a
│   ├── __init__.py
│   └── test_smoke.py
├── logs/                           # 日志目录（自动创建）
├── models/                         # 模型存放目录
│   └── iic/
│       ├── SenseVoiceSmall/
│       └── speech_fsmn_vad_zh-cn-16k-common-pytorch/
├── Dockerfile                      # Docker 镜像构建文件
├── Makefile                        # Make 命令配置
├── pyproject.toml                  # 项目配置和依赖
├── uv.lock                         # 依赖锁定文件
└── README.md                       # 项目说明
```

## 🚀 快速开始

### 前置要求

- Python 3.11+
- ffmpeg（用于音频处理）

### 方式一：使用 Makefile（推荐）

项目提供了 Makefile 来简化开发和部署流程。

#### 1. 初始化项目

```bash
# 自动安装 uv 并同步依赖
make init
```

此命令会：
- 自动安装 `uv`（如果未安装）
- 使用 `uv` 同步所有依赖（从 `pyproject.toml` 和 `uv.lock`）

#### 2. 下载模型

```bash
# 下载 SenseVoice 模型
modelscope download --model iic/SenseVoiceSmall --local_dir ./models/iic/SenseVoiceSmall

# 下载 VAD 模型（可选，用于长音频处理）
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch --local_dir ./models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
```

#### 3. 启动服务

```bash
# 使用默认配置运行
make run
```

```bash
# 或使用自定义配置
export PORT=9991
export MODEL_BASE_PATH=./models
export VAD_ENABLE=true
make run
```

#### 其他 Makefile 命令

```bash
make fmt        # 格式化代码（black + isort）
make lint       # 代码检查（pflake8）
make coverage   # 运行测试并生成覆盖率报告
make build      # 构建项目
```

### 方式二：手动运行

如果不使用 Makefile，可以手动执行以下步骤：

#### 1. 安装 uv（如果未安装）

```bash
pip install uv
```

#### 2. 同步依赖

```bash
# 设置超时时间并同步依赖
export UV_HTTP_TIMEOUT=300  # Linux/Mac
set UV_HTTP_TIMEOUT=300     # Windows

uv sync --frozen --all-extras --no-install-project
```

#### 3. 下载模型

参考上面"方式一"中的模型下载步骤。

#### 4. 运行服务

```bash
# 使用 uv 运行
uv run python -m src.main
```

```bash
# 或直接使用 python（需要先激活虚拟环境）
python -m src.main
```

服务启动后访问：http://localhost:9991

### 方式三：Docker 部署

#### 1. 使用 Docker Compose（推荐）

创建 `docker-compose.yml`：

```yaml
services:
  sense-voice-api:
    image: falconia/sense-voice-openai-api:latest
    container_name: sense-voice-api
    restart: unless-stopped
    ports:
      - "9991:9991"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - PORT=9991
      - DEVICE_TYPE=cpu
      - VAD_ENABLE=true
      - LOG_LEVEL=INFO
```

启动服务：

```bash
docker-compose up -d
```

#### 2. 直接使用 Docker

```bash
# 构建镜像
docker build -t falconia/sense-voice-openai-api:latest .
```

```bash
# 运行容器
docker run -d \
  --name sensevoice-api \
  -p 9991:9991 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -e VAD_ENABLE=true \
  falconia/sense-voice-openai-api:latest
```

## 📖 API 接口文档

### 音频转录

#### POST `/v1/audio/transcriptions`

将音频文件转录为文本，完全兼容 OpenAI Whisper API。

**请求参数：**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| file | File | 是 | 音频文件 |

**支持的音频格式：**

`.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, `.webm`, `.flac`, `.ogg`, `.opus`

**文件大小限制：** 默认 25MB（可通过 `MAX_FILE_SIZE` 环境变量配置）

**请求示例：**

```bash
curl --request POST 'http://127.0.0.1:9991/v1/audio/transcriptions' \
  --header 'Content-Type: multipart/form-data' \
  --form 'file=@tests/audio/asr_example_zh.wav'
```

**响应示例：**

```json
{
  "text": "欢迎大家来体验达摩院推出的语音识别模型"
}
```

**错误响应示例：**

```json
{
  "error": {
    "message": "Invalid file format: '.txt'. Supported formats: .flac, .m4a, .mp3, .mp4, .mpeg, .mpga, .ogg, .opus, .wav, .webm",
    "type": "invalid_request_error",
    "param": "file",
    "code": "unsupported_file_format"
  }
}
```

## ⚙️ 环境变量配置

### 服务器配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `HOST` | `0.0.0.0` | 服务绑定地址 |
| `PORT` | `9991` | 服务监听端口 |

### 日志配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LOG_DIR` | `./logs` | 日志文件存放目录 |
| `LOG_LEVEL` | `INFO` | 日志级别（`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`） |

### 模型配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_BASE_PATH` | `./models` | 模型基础路径 |
| `MODEL_PATH` | 自动拼接 | SenseVoice 模型完整路径 |
| `VAD_PATH` | 自动拼接 | VAD 模型完整路径 |

> 💡 如果未设置 `MODEL_PATH` 或 `VAD_PATH`，系统会自动使用 `MODEL_BASE_PATH` + 模型名称拼接

### 推理配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DEVICE_TYPE` | `cpu` | 推理设备类型（cpu/cuda） |
| `CPU_NUM` | `4` | CPU 线程数 |
| `VAD_ENABLE` | `false` | 是否启用 VAD（支持长音频） |
| `LANGUAGE` | `zh` | 识别语言（zh/en/auto） |
| `BATCH_SIZE` | `64` | 批处理大小 |
| `USE_ITN` | `false` | 是否使用逆文本规范化 |

### 文件限制

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MAX_FILE_SIZE` | `26214400` | 最大文件大小（字节，默认 25MB） |

### 配置示例

#### 基础配置（快速预测）

```bash
export MODEL_BASE_PATH=./models
export DEVICE_TYPE=cpu
```

#### 高精度配置（启用 VAD）

```bash
export MODEL_BASE_PATH=./models
export DEVICE_TYPE=cpu
export CPU_NUM=8
export VAD_ENABLE=true
export USE_ITN=true
```

#### GPU 加速配置

```bash
export MODEL_BASE_PATH=./models
export DEVICE_TYPE=cuda:0
export BATCH_SIZE=128
```

## 🔗 集成 OneAPI

1. 在 OneAPI 管理面板中添加渠道
2. 选择类型：**OpenAI**
3. Base URL：`http://<HOST>:9991/v1`
4. 密钥：任意字符串
5. 模型：`SenseVoiceSmall`
6. 点击"测试"按钮验证连接

## 📊 性能优化建议

### CPU 优化

```bash
# 增加 CPU 线程数
export CPU_NUM=8

# 增加批处理大小
export BATCH_SIZE=128
```

### GPU 加速

```bash
# 启用 CUDA
export DEVICE_TYPE=cuda:0

# 增加批处理大小
export BATCH_SIZE=256
```

### 长音频处理

```bash
# 启用 VAD 分段处理
export VAD_ENABLE=true

# VAD 会自动将长音频分段处理，支持任意时长
```

## 🐛 故障排查

### 模型加载失败

**问题**：启动时提示模型加载失败

**解决方案**：
1. 检查模型路径是否正确
2. 确认模型文件已完整下载
3. 查看日志文件获取详细错误信息

```bash
# 查看日志
tail -f logs/sensevoice.log
```

### 音频处理失败

**问题**：音频转录时返回错误

**解决方案**：
1. 确认音频格式在支持列表中
2. 检查文件大小是否超过限制
3. 确认 ffmpeg 已正确安装

### 健康检查失败

**问题**：`/health` 接口返回 DOWN

**解决方案**：
1. 检查模型是否加载成功
2. 查看日志文件排查问题
3. 确认配置的模型路径正确

## 📝 开发指南

详细的开发命令请参考项目根目录的 `Makefile`。

### 初始化开发环境

```bash
# 安装依赖
make init
```

### 运行和测试

```bash
# 运行服务
make run

# 代码格式化
make fmt

# 代码检查
make lint

# 运行测试并生成覆盖率报告
make coverage

# 构建项目
make build
```

## 📄 License

MIT License

## 🙏 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 阿里达摩院语音识别框架
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 语音识别模型
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web 框架
- [OpenAI](https://platform.openai.com/docs/api-reference/audio/createTranscription) - API 协议规范
