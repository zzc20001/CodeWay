import { useMutation, useQueryClient } from '@tanstack/react-query';

// Types
export type ChatCompletionRequest = {
  model: string;
  messages: {
    role: 'system' | 'user' | 'assistant';
    content: string;
  }[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
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

// API function to send regular (non-streaming) requests to ChatGPT
export const sendChatGptRequest = async (
  requestData: ChatCompletionRequest
): Promise<ChatCompletionResponse> => {
  const response = await fetch(`${import.meta.env.VITE_OPENAI_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY || ''}`,
    },
    body: JSON.stringify({
      ...requestData,
      stream: false,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error?.message || 'Failed to get response from ChatGPT');
  }

  return response.json();
};

// API function to send streaming requests to ChatGPT
export const streamChatGptRequest = async (
  requestData: ChatCompletionRequest,
  onChunk: (chunk: ChatCompletionChunk) => void,
  onError: (error: Error) => void,
  onComplete: () => void
): Promise<void> => {
  try {
    const response = await fetch(`${import.meta.env.VITE_OPENAI_BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY || ''}`,
      },
      body: JSON.stringify({
        ...requestData,
        stream: true,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || 'Failed to get streaming response from ChatGPT');
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    
    let buffer = '';

    const readChunk = async (): Promise<void> => {
      const { done, value } = await reader.read();
      
      if (done) {
        onComplete();
        return;
      }
      
      // Decode the chunk and add it to our buffer
      buffer += decoder.decode(value, { stream: true });
      
      // Process each line in the buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // The last line might be incomplete
      
      for (const line of lines) {
        if (line.trim() === '') continue;
        
        // SSE format: lines starting with "data: "
        if (line.startsWith('data: ')) {
          const data = line.slice(6); // Remove "data: " prefix
          
          // The "data: [DONE]" message indicates the end of the stream
          if (data === '[DONE]') {
            onComplete();
            return;
          }
          
          try {
            const chunk = JSON.parse(data) as ChatCompletionChunk;
            onChunk(chunk);
          } catch (e) {
            console.error('Error parsing SSE chunk:', e);
          }
        }
      }
      
      // Continue reading
      return readChunk();
    };
    
    await readChunk();
  } catch (error) {
    onError(error instanceof Error ? error : new Error(String(error)));
  }
};

// Custom hook using React Query's useMutation for ChatGPT requests
export const useChatGptMutation = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: sendChatGptRequest,
    onSuccess: () => {
      // Example of using queryClient - you can customize this based on your needs
      queryClient.invalidateQueries({ queryKey: ['chatHistory'] });
    },
  });
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
