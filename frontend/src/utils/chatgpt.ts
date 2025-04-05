import { useMutation, useQueryClient } from '@tanstack/react-query';

// 后端API请求类型
export type BackendApiRequest = {
  query: string;
  session_id?: string;
  mode?: string;
};

// 后端API响应类型
export type BackendApiResponse = {
  answer: string;
  session_id: string;
};

// 为了兼容现有代码，保留OpenAI风格的类型定义
export type ChatCompletionRequest = {
  model: string;
  messages: {
    role: 'system' | 'user' | 'assistant';
    content: string;
  }[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  session_id?: string; // 添加session_id字段用于传递给后端
};

export type ChatCompletionResponse = {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    message: {
      role: 'assistant';
      content: string;
    };
    finish_reason: string;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

export type ChatCompletionChunk = {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    delta: {
      content?: string;
      role?: string;
    };
    finish_reason: string | null;
  }[];
};
// API function to send streaming requests to ChatGPT
export const streamChatGptRequest = async (
  requestData: ChatCompletionRequest,
  onChunk: (chunk: ChatCompletionChunk) => void,
  onError: (error: Error) => void,
  onComplete: () => void
): Promise<void> => {
  try {
    // 从ChatGPT风格的请求中提取最后一条用户消息作为查询
    const lastUserMessage = [...requestData.messages].reverse().find(msg => msg.role === 'user');
    
    if (!lastUserMessage) {
      throw new Error('No user message found in the request');
    }

    // 构建后端API请求
    const backendRequest: BackendApiRequest = {
      query: lastUserMessage.content,
      session_id: requestData.session_id,
      mode: 'local' // 默认使用本地模型
    };

    const response = await fetch(`${import.meta.env.VITE_API_URL}/api/gpt`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(backendRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP error! status: ${response.status}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    
    // 用于生成唯一ID的时间戳
    const timestamp = Date.now();
    let done = false;
    
    // 直接读取流
    while (!done) {
      const { value, done: doneReading } = await reader.read();
      done = doneReading;

      if (done) {
        onComplete();
        break;
      }

      // 解码这个块
      const chunk = decoder.decode(value, { stream: true });
      
      // 为了兼容现有前端代码，将文本块转换为ChatGPT风格的块
      const mockChunk: ChatCompletionChunk = {
        id: `chunk-${timestamp}-${Math.random().toString(36).substring(2, 9)}`,
        object: 'chat.completion.chunk',
        created: Date.now(),
        model: requestData.model || 'gemma3',
        choices: [
          {
            index: 0,
            delta: {
              content: chunk
            },
            finish_reason: null
          }
        ]
      };
      
      // 将这个模拟块传递给回调
      onChunk(mockChunk);
    }
  } catch (error) {
    onError(error instanceof Error ? error : new Error(String(error)));
  }
};

// Custom hook for streaming ChatGPT responses
export const useStreamChatGptMutation = () => {
  const queryClient = useQueryClient();
  
  return {
    streamChatCompletion: (
      requestData: ChatCompletionRequest,
      callbacks: {
        onChunk: (chunk: ChatCompletionChunk) => void;
        onError: (error: Error) => void;
        onComplete: () => void;
      }
    ) => {
      return streamChatGptRequest(
        requestData,
        callbacks.onChunk,
        callbacks.onError,
        () => {
          callbacks.onComplete();
          queryClient.invalidateQueries({ queryKey: ['chatHistory'] });
        }
      );
    }
  };
};

// Utility function to convert our app message format to OpenAI API format
export const convertMessagesToApiFormat = (
  messages: Array<{ role: 'user' | 'assistant'; content: string }>,
  model: string
): ChatCompletionRequest => {
  return {
    model,
    messages: messages.map(msg => ({
      role: msg.role,
      content: msg.content,
    })),
    temperature: 0.7,
  };
};
