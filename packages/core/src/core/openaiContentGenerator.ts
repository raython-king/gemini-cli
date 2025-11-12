/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  EmbedContentResponse} from '@google/genai';
import {
  CountTokensResponse,
  GenerateContentResponse,
  type GenerateContentParameters,
  type CountTokensParameters,
  type EmbedContentParameters,
  type Content,
  type Part,
  type Tool,
  type FinishReason,
  type ContentListUnion,
  type ToolListUnion,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';

/**
 * OpenAI API compatible request/response types
 */
interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: string | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
  name?: string;
}

interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

interface OpenAITool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

interface OpenAIRequest {
  model: string;
  messages: OpenAIMessage[];
  tools?: OpenAITool[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream?: boolean;
}

interface OpenAIResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: OpenAIMessage;
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface OpenAIStreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
      tool_calls?: Array<{
        index?: number;
        id?: string;
        type?: 'function';
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    finish_reason: string | null;
  }>;
}

/**
 * ContentGenerator implementation that uses OpenAI-compatible API
 */
export class OpenAIContentGenerator implements ContentGenerator {
  private baseUrl: string;
  private apiKey: string;
  private defaultModel: string;

  constructor(config: { baseUrl: string; apiKey?: string; model?: string }) {
    this.baseUrl = config.baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.apiKey = config.apiKey || 'dummy-key';
    this.defaultModel = config.model || 'gpt-3.5-turbo';
  }

  /**
   * Convert Gemini Content format to OpenAI messages format
   */
  private geminiToOpenAI(
    contents: Content[],
    systemInstruction?: unknown,
  ): OpenAIMessage[] {
    const messages: OpenAIMessage[] = [];

    // Add system instruction if provided
    if (systemInstruction) {
      let systemContent = '';
      if (typeof systemInstruction === 'string') {
        systemContent = systemInstruction;
      } else if (Array.isArray(systemInstruction)) {
        systemContent = this.partsToText(systemInstruction as Part[]);
      } else if (
        typeof systemInstruction === 'object' &&
        systemInstruction !== null &&
        'parts' in systemInstruction
      ) {
        const parts = (systemInstruction as { parts?: Part[] }).parts;
        systemContent = this.partsToText(parts || []);
      } else if (
        typeof systemInstruction === 'object' &&
        systemInstruction !== null &&
        'text' in systemInstruction
      ) {
        const text = (systemInstruction as { text?: string }).text;
        systemContent = text || '';
      }

      if (systemContent) {
        messages.push({
          role: 'system',
          content: systemContent,
        });
      }
    }

    // Convert conversation history
    for (const content of contents) {
      const role = content.role === 'model' ? 'assistant' : content.role;

      // Check if this is a function response
      const functionResponseParts = content.parts?.filter(
        (p) => p.functionResponse,
      );
      if (functionResponseParts && functionResponseParts.length > 0) {
        // Convert function responses to tool messages
        for (const part of functionResponseParts) {
          if (part.functionResponse) {
            messages.push({
              role: 'tool',
              tool_call_id: part.functionResponse.id || 'unknown',
              name: part.functionResponse.name,
              content: JSON.stringify(part.functionResponse.response),
            });
          }
        }
        continue;
      }

      // Check if this has function calls
      const functionCallParts = content.parts?.filter((p) => p.functionCall);
      if (functionCallParts && functionCallParts.length > 0) {
        const toolCalls: OpenAIToolCall[] = functionCallParts
          .filter((part) => part.functionCall?.name)
          .map((part) => {
            const fc = part.functionCall!;
            return {
              id: fc.id || `call_${Date.now()}`,
              type: 'function' as const,
              function: {
                name: fc.name!,
                arguments: JSON.stringify(fc.args || {}),
              },
            };
          });

        const textContent = this.partsToText(
          content.parts?.filter((p) => p.text) || [],
        );

        messages.push({
          role: role as 'assistant',
          content: textContent || null,
          tool_calls: toolCalls,
        });
        continue;
      }

      // Regular text message
      const textContent = this.partsToText(content.parts || []);
      if (textContent) {
        messages.push({
          role: role as 'user' | 'assistant',
          content: textContent,
        });
      }
    }

    return messages;
  }

  /**
   * Convert parts array to text string
   */
  private partsToText(parts: Part[]): string {
    return parts
      .filter((p) => p.text && !p.thought)
      .map((p) => p.text)
      .join('\n')
      .trim();
  }

  /**
   * Convert Gemini tools to OpenAI tools format
   */
  private convertTools(geminiTools?: Tool[]): OpenAITool[] | undefined {
    if (!geminiTools || geminiTools.length === 0) {
      return undefined;
    }

    const openaiTools: OpenAITool[] = [];

    for (const tool of geminiTools) {
      if ('functionDeclarations' in tool && tool.functionDeclarations) {
        for (const func of tool.functionDeclarations) {
          if (func.name) {
            openaiTools.push({
              type: 'function',
              function: {
                name: func.name,
                description: func.description,
                parameters: func.parameters as Record<string, unknown>,
              },
            });
          }
        }
      }
    }

    return openaiTools.length > 0 ? openaiTools : undefined;
  }

  /**
   * Convert OpenAI response to Gemini format
   */
  private openaiToGemini(
    response: OpenAIResponse,
    model: string,
  ): GenerateContentResponse {
    const choice = response.choices[0];
    const message = choice?.message;

    const parts: Part[] = [];

    // Add text content
    if (message?.content) {
      parts.push({ text: message.content });
    }

    // Add function calls
    if (message?.tool_calls) {
      for (const toolCall of message.tool_calls) {
        if (toolCall.type === 'function') {
          let args: Record<string, unknown> = {};
          try {
            args = JSON.parse(toolCall.function.arguments);
          } catch (e) {
            console.error('Failed to parse function arguments:', e);
          }

          parts.push({
            functionCall: {
              id: toolCall.id,
              name: toolCall.function.name || 'unknown',
              args,
            },
          });
        }
      }
    }

    const finishReason = choice?.finish_reason?.toUpperCase() || 'STOP';

    const responseData = {
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason: finishReason as FinishReason,
          index: 0,
        },
      ],
      usageMetadata: response.usage
        ? {
            promptTokenCount: response.usage.prompt_tokens,
            candidatesTokenCount: response.usage.completion_tokens,
            totalTokenCount: response.usage.total_tokens,
          }
        : undefined,
      modelVersion: model,
    };

    return Object.setPrototypeOf(
      responseData,
      GenerateContentResponse.prototype,
    );
  }

  /**
   * Normalize contents to array
   */
  private normalizeContents(contents: ContentListUnion): Content[] {
    if (Array.isArray(contents)) {
      return contents.filter(
        (c): c is Content => typeof c === 'object' && c !== null && 'role' in c,
      );
    }
    if (typeof contents === 'string') {
      return [{ role: 'user', parts: [{ text: contents }] }];
    }
    if (
      typeof contents === 'object' &&
      contents !== null &&
      'role' in contents
    ) {
      return [contents as Content];
    }
    // Assume it's a Part, wrap it
    return [{ role: 'user', parts: [contents as Part] }];
  }

  /**
   * Normalize tools to array
   */
  private normalizeTools(tools?: ToolListUnion): Tool[] | undefined {
    if (!tools) return undefined;
    if (Array.isArray(tools)) {
      return tools.filter(
        (t): t is Tool => 'functionDeclarations' in t || 'codeExecution' in t,
      );
    }
    if ('functionDeclarations' in tools || 'codeExecution' in tools) {
      return [tools];
    }
    return undefined;
  }

  /**
   * Generate content using OpenAI-compatible API
   */
  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const contents = this.normalizeContents(request.contents);
    const messages = this.geminiToOpenAI(
      contents,
      request.config?.systemInstruction,
    );

    const openaiRequest: OpenAIRequest = {
      model: request.model || this.defaultModel,
      messages,
      tools: this.convertTools(this.normalizeTools(request.config?.tools)),
      temperature: request.config?.temperature,
      top_p: request.config?.topP,
      stream: false,
    };

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
        'X-Prompt-ID': userPromptId,
      },
      body: JSON.stringify(openaiRequest),
      signal: request.config?.abortSignal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API error (${response.status}): ${errorText}`);
    }

    const data: OpenAIResponse = await response.json();
    return this.openaiToGemini(data, request.model || this.defaultModel);
  }

  /**
   * Generate content stream using OpenAI-compatible API
   */
  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const contents = this.normalizeContents(request.contents);
    const messages = this.geminiToOpenAI(
      contents,
      request.config?.systemInstruction,
    );

    const openaiRequest: OpenAIRequest = {
      model: request.model || this.defaultModel,
      messages,
      tools: this.convertTools(this.normalizeTools(request.config?.tools)),
      temperature: request.config?.temperature,
      top_p: request.config?.topP,
      stream: true,
    };

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
        'X-Prompt-ID': userPromptId,
      },
      body: JSON.stringify(openaiRequest),
      signal: request.config?.abortSignal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API error (${response.status}): ${errorText}`);
    }

    const model = request.model || this.defaultModel;

    return this.processStream(response, model);
  }

  /**
   * Process SSE stream from OpenAI API
   */
  private async *processStream(
    response: Response,
    model: string,
  ): AsyncGenerator<GenerateContentResponse> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let accumulatedText = '';
    const accumulatedToolCalls: Map<
      number,
      { id?: string; name?: string; arguments: string }
    > = new Map();
    let finishReason: string | null = null;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim() || line.trim() === 'data: [DONE]') continue;

          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6);
            try {
              const chunk: OpenAIStreamChunk = JSON.parse(jsonStr);
              const delta = chunk.choices[0]?.delta;

              if (!delta) continue;

              // Accumulate text
              if (delta.content) {
                accumulatedText += delta.content;

                // Yield text chunk
                yield Object.setPrototypeOf(
                  {
                    candidates: [
                      {
                        content: {
                          role: 'model',
                          parts: [{ text: delta.content }],
                        },
                        index: 0,
                      },
                    ],
                    modelVersion: model,
                  },
                  GenerateContentResponse.prototype,
                );
              }

              // Accumulate tool calls
              if (delta.tool_calls) {
                for (const tc of delta.tool_calls) {
                  const index = tc.index ?? 0;
                  if (!accumulatedToolCalls.has(index)) {
                    accumulatedToolCalls.set(index, {
                      id: tc.id,
                      name: tc.function?.name,
                      arguments: '',
                    });
                  }

                  const accumulated = accumulatedToolCalls.get(index)!;
                  if (tc.id) accumulated.id = tc.id;
                  if (tc.function?.name) accumulated.name = tc.function.name;
                  if (tc.function?.arguments) {
                    accumulated.arguments += tc.function.arguments;
                  }
                }
              }

              // Check finish reason
              if (chunk.choices[0]?.finish_reason) {
                finishReason = chunk.choices[0].finish_reason;
              }
            } catch (e) {
              console.error('Failed to parse SSE chunk:', e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    // Yield final response with all accumulated data
    const parts: Part[] = [];

    if (accumulatedText) {
      parts.push({ text: accumulatedText });
    }

    if (accumulatedToolCalls.size > 0) {
      for (const [_, toolCall] of accumulatedToolCalls) {
        if (toolCall.name) {
          let args: Record<string, unknown> = {};
          try {
            args = JSON.parse(toolCall.arguments);
          } catch (e) {
            console.error('Failed to parse tool arguments:', e);
          }

          parts.push({
            functionCall: {
              id: toolCall.id || `call_${Date.now()}`,
              name: toolCall.name,
              args,
            },
          });
        }
      }
    }

    if (parts.length > 0 || finishReason) {
      const finalFinishReason = (finishReason?.toUpperCase() ||
        'STOP') as FinishReason;

      yield Object.setPrototypeOf(
        {
          candidates: [
            {
              content: {
                role: 'model',
                parts: parts.length > 0 ? parts : [{ text: '' }],
              },
              finishReason: finalFinishReason,
              index: 0,
            },
          ],
          modelVersion: model,
        },
        GenerateContentResponse.prototype,
      );
    }
  }

  /**
   * Count tokens (simplified implementation)
   */
  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Simplified token counting - estimate 4 chars per token
    const contents = this.normalizeContents(request.contents);
    const messages = this.geminiToOpenAI(contents);
    const text = messages.map((m) => m.content || '').join(' ');
    const estimatedTokens = Math.ceil(text.length / 4);

    return Object.setPrototypeOf(
      {
        totalTokens: estimatedTokens,
      },
      CountTokensResponse.prototype,
    );
  }

  /**
   * Embed content (not implemented for OpenAI-compatible local models)
   */
  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error(
      'Embedding is not supported for local OpenAI-compatible models',
    );
  }
}
