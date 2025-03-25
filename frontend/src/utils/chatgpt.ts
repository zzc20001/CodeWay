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

// API function to send requests to ChatGPT
export const sendChatGptRequest = async (
  requestData: ChatCompletionRequest
): Promise<ChatCompletionResponse> => {
  const response = await fetch(`${import.meta.env.VITE_OPENAI_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY || ''}`,
    },
    body: JSON.stringify(requestData),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error?.message || 'Failed to get response from ChatGPT');
  }

  return response.json();
};

// Custom hook using React Query's useMutation for ChatGPT requests
export const useChatGptMutation = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: sendChatGptRequest,
    onSuccess: () => {
      // Example of using queryClient - you can customize this based on your needs
      // This will invalidate and refetch any queries with the specified key
      queryClient.invalidateQueries({ queryKey: ['chatHistory'] });
    },
  });
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
