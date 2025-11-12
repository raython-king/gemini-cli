# OpenAI 兼容 API 集成指南

本项目已支持使用 OpenAI 兼容的本地服务替代 Google Gemini API。

## 功能特性

- ✅ 完全兼容 OpenAI Chat Completions API 格式
- ✅ 支持流式响应
- ✅ 支持工具调用（Function Calling）
- ✅ 自动格式转换（Gemini ↔ OpenAI）
- ✅ 保持所有现有功能不变

## 使用方法

### 1. 设置环境变量

使用 OpenAI 兼容的本地服务需要设置以下环境变量：

```bash
# 必需：本地服务的 API 端点
export OPENAI_BASE_URL="http://localhost:8000/v1"

# 可选：API Key（某些本地服务可能需要）
export OPENAI_API_KEY="your-api-key-or-dummy"

# 可选：指定模型名称（默认为 gpt-3.5-turbo）
export OPENAI_MODEL="your-model-name"
```

### 2. 启动应用

启动应用时指定使用 OpenAI 兼容模式：

```bash
# 方式 1: 通过命令行参数（需要在 CLI 中添加支持）
gemini --auth-type openai-compatible

# 方式 2: 通过环境变量
export AUTH_TYPE=openai-compatible
gemini
```

## 配置示例

### 使用本地 LLaMA 服务

```bash
# 使用 llama.cpp 的 server
export OPENAI_BASE_URL="http://localhost:8080/v1"
export OPENAI_MODEL="llama-3-70b"
export OPENAI_API_KEY="dummy"

gemini --auth-type openai-compatible
```

### 使用 vLLM

```bash
# 使用 vLLM 服务
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL="meta-llama/Meta-Llama-3-70B"

gemini --auth-type openai-compatible
```

### 使用 Ollama

```bash
# 使用 Ollama（需要启用 OpenAI 兼容模式）
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_MODEL="llama3"

gemini --auth-type openai-compatible
```

### 使用 LocalAI

```bash
# 使用 LocalAI
export OPENAI_BASE_URL="http://localhost:8080/v1"
export OPENAI_MODEL="gpt-4"

gemini --auth-type openai-compatible
```

## 技术实现

### 架构设计

```
┌─────────────────────────────────────┐
│         GeminiClient                │
│     (保持不变)                       │
└───────────┬─────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│      ContentGenerator 接口          │
└───────────┬─────────────────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌────────────┐  ┌──────────────────────┐
│ GoogleGenAI│  │ OpenAIContentGenerator│ ← 新增
│  (原有)    │  │   (适配器)            │
└────────────┘  └──────────────────────┘
                        │
                        ▼
                ┌──────────────────┐
                │  OpenAI API      │
                │  (本地服务)       │
                └──────────────────┘
```

### 格式转换

#### Gemini → OpenAI

- `Content` → `Message`
- `Part.text` → `Message.content`
- `Part.functionCall` → `Message.tool_calls`
- `Part.functionResponse` → 独立的 `tool` 角色消息

#### OpenAI → Gemini

- `Message` → `Content`
- `Message.content` → `Part.text`
- `Message.tool_calls` → `Part.functionCall`
- 工具消息 → `Part.functionResponse`

### 工具调用支持

工具调用格式自动转换：

**Gemini 格式**:

```json
{
  "functionCall": {
    "id": "call_123",
    "name": "read_file",
    "args": { "path": "/tmp/file.txt" }
  }
}
```

**OpenAI 格式**:

```json
{
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\":\"/tmp/file.txt\"}"
      }
    }
  ]
}
```

## 限制和注意事项

### 当前限制

1. **嵌入（Embedding）不支持**: `embedContent` 方法会抛出错误
2. **Token 计数**: 使用简化的估算（4 字符 ≈ 1 token）
3. **思考模式**: 取决于本地模型是否支持

### 兼容性

本实现兼容所有遵循 OpenAI Chat Completions API 规范的服务，包括但不限于：

- ✅ vLLM
- ✅ llama.cpp server
- ✅ Ollama (OpenAI 兼容模式)
- ✅ LocalAI
- ✅ FastChat
- ✅ Text Generation Inference (TGI)
- ✅ 任何 OpenAI 兼容的推理服务

### 测试建议

1. **基本对话测试**:

   ```bash
   gemini
   > Hello, can you help me?
   ```

2. **工具调用测试**:

   ```bash
   gemini
   > Read the file at /tmp/test.txt
   ```

3. **流式响应测试**:
   ```bash
   gemini
   > Tell me a long story
   ```

## 故障排查

### 常见问题

#### 1. 连接错误

```
Error: OpenAI API error (Connection refused)
```

**解决方案**: 检查 `OPENAI_BASE_URL` 是否正确，确保本地服务正在运行。

#### 2. 工具调用失败

```
Error: Failed to parse function arguments
```

**解决方案**: 确保本地模型支持工具调用（Function
Calling），并且返回的 JSON 格式正确。

#### 3. 模型不存在

```
Error: Model 'xxx' not found
```

**解决方案**: 检查 `OPENAI_MODEL` 环境变量，确保模型名称与本地服务中的模型匹配。

### 调试模式

启用调试日志查看详细信息：

```bash
export DEBUG=1
export VERBOSE=true
gemini --auth-type openai-compatible
```

## 性能优化

### 建议配置

1. **使用本地 GPU**: 确保本地服务使用 GPU 加速
2. **调整上下文长度**: 根据模型支持设置合适的 `max_tokens`
3. **启用流式响应**: 提升用户体验

### 示例配置文件

创建 `.env` 文件：

```bash
# OpenAI 兼容配置
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_MODEL=meta-llama/Meta-Llama-3-70B
OPENAI_API_KEY=dummy

# 性能配置
MAX_TOKENS=4096
TEMPERATURE=0.7
```

## 开发者信息

### 关键文件

- `packages/core/src/core/openaiContentGenerator.ts` - OpenAI 适配器实现
- `packages/core/src/core/contentGenerator.ts` - 配置和工厂函数
- `packages/core/src/core/geminiChat.ts` - 聊天会话管理（无需修改）

### 扩展支持

如需添加其他本地服务支持，可以：

1. 创建新的 `ContentGenerator` 实现
2. 在 `createContentGenerator` 中添加分支
3. 添加相应的认证类型到 `AuthType` 枚举

## 示例场景

### 场景 1: 使用本地 LLaMA 模型进行代码审查

```bash
export OPENAI_BASE_URL="http://localhost:8080/v1"
export OPENAI_MODEL="codellama-70b"

gemini
> Review the code in src/main.ts and suggest improvements
```

### 场景 2: 使用本地模型生成文档

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL="mistral-large"

gemini
> Generate documentation for all functions in utils.ts
```

### 场景 3: 离线开发环境

```bash
# 完全离线，使用本地模型
export OPENAI_BASE_URL="http://localhost:8080/v1"
export OPENAI_MODEL="phi-3-medium"

gemini
> Help me refactor this function to be more efficient
```

## 贡献

欢迎提交 Issue 和 Pull Request 来改进 OpenAI 兼容性！

## 许可

遵循项目主许可证 Apache 2.0
