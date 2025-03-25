import { createFileRoute } from '@tanstack/react-router'
import { useState, useMemo, useRef } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select"
import { Send, DownloadCloud, Plus, Menu, Search } from "lucide-react"
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism'
import 'katex/dist/katex.min.css'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { useChatGptMutation, useStreamChatGptMutation, convertMessagesToApiFormat } from '@/utils/chatgpt'

// Create a query client
const queryClient = new QueryClient()

export const Route = createFileRoute('/')({
  component: () => (
    <QueryClientProvider client={queryClient}>
      <ReactQueryDevtools initialIsOpen={false} />
      <ChatGPT />
    </QueryClientProvider>
  ),
})

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

type Chat = {
  id: string;
  title: string;
  messages: Message[];
  timestamp: Date;
}

function ChatGPT() {
  const [chats, setChats] = useState<Chat[]>([
    {
      id: '1',
      title: '新对话',
      messages: [],
      timestamp: new Date()
    }
  ]);

  const [activeChat, setActiveChat] = useState<string>('1');
  const [inputValue, setInputValue] = useState<string>('');
  const [model, setModel] = useState<string>('gpt-4o');
  const [filterValue, setFilterValue] = useState<string>('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  
  // Use the ChatGPT mutation hooks
  const chatGptMutation = useChatGptMutation();
  const { streamChatCompletion } = useStreamChatGptMutation();
  
  // Reference to the current streaming message
  const streamingMessageRef = useRef<Message | null>(null);
  
  const activeMessages = chats.find(chat => chat.id === activeChat)?.messages || [];
  
  const filteredChats = useMemo(() => {
    if (!filterValue.trim()) return chats;
    return chats.filter(chat => 
      chat.title.toLowerCase().includes(filterValue.toLowerCase())
    );
  }, [chats, filterValue]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isStreaming) return;
    
    const newMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };
    
    // Update current chat with new message
    const updatedChats = chats.map(chat => {
      if (chat.id === activeChat) {
        // If this is the first message, update the chat title
        const updatedTitle = chat.messages.length === 0 ? 
          inputValue.substring(0, 20) + (inputValue.length > 20 ? '...' : '') : 
          chat.title;
        
        return {
          ...chat,
          title: updatedTitle,
          messages: [...chat.messages, newMessage]
        };
      }
      return chat;
    });
    
    setChats(updatedChats);
    setInputValue('');
    
    // Get the current chat after update
    const currentChat = updatedChats.find(chat => chat.id === activeChat);
    if (!currentChat) return;
    
    // Format messages for the API
    const apiMessages = convertMessagesToApiFormat(
      currentChat.messages,
      model
    );
    
    // Create an initial streaming response message
    const streamingMessage: Message = {
      id: Date.now().toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date()
    };
    
    // Store the reference to the current streaming message
    streamingMessageRef.current = streamingMessage;
    
    // Add the initial empty assistant message
    setChats(prevChats => prevChats.map(chat => {
      if (chat.id === activeChat) {
        return {
          ...chat,
          messages: [...chat.messages, streamingMessage]
        };
      }
      return chat;
    }));
    
    // Set streaming state to true
    setIsStreaming(true);
    
    // Send request to ChatGPT API with streaming
    try {
      await streamChatCompletion(
        apiMessages,
        {
          onChunk: (chunk) => {
            const content = chunk.choices[0]?.delta?.content || '';
            
            if (content) {
              // Update the streaming message with new content
              setChats(prevChats => 
                prevChats.map(chat => {
                  if (chat.id === activeChat) {
                    return {
                      ...chat,
                      messages: chat.messages.map(msg => {
                        if (msg.id === streamingMessageRef.current?.id) {
                          return {
                            ...msg,
                            content: msg.content + content
                          };
                        }
                        return msg;
                      })
                    };
                  }
                  return chat;
                })
              );
            }
          },
          onError: (error) => {
            // Update streaming message with error
            setChats(prevChats => 
              prevChats.map(chat => {
                if (chat.id === activeChat) {
                  return {
                    ...chat,
                    messages: chat.messages.map(msg => {
                      if (msg.id === streamingMessageRef.current?.id) {
                        return {
                          ...msg,
                          content: `Error: ${error.message || '请求失败，请稍后重试。'}`
                        };
                      }
                      return msg;
                    })
                  };
                }
                return chat;
              })
            );
            
            // Reset streaming state
            setIsStreaming(false);
            streamingMessageRef.current = null;
          },
          onComplete: () => {
            // Reset streaming state
            setIsStreaming(false);
            streamingMessageRef.current = null;
          }
        }
      );
    } catch (error) {
      console.error('Failed to stream response:', error);
      setIsStreaming(false);
      streamingMessageRef.current = null;
    }
  };

  const createNewChat = () => {
    const newChat: Chat = {
      id: Date.now().toString(),
      title: '新对话',
      messages: [],
      timestamp: new Date()
    };
    
    setChats([newChat, ...chats]);
    setActiveChat(newChat.id);
  };

  const exportChat = () => {
    const chat = chats.find(c => c.id === activeChat);
    if (!chat) return;
    
    const chatContent = chat.messages.map(msg => 
      `${msg.role === 'user' ? '用户' : 'GPT'}: ${msg.content}`
    ).join('\n\n');
    
    const blob = new Blob([chatContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${chat.title}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarOpen(prev => !prev);
    console.log("Toggling sidebar:", !sidebarOpen); // Debug log
  };

  return (
    <div className="flex h-full w-full relative">
      {/* Fixed Header - Always visible */}
      <div className="fixed top-0 left-0 z-30 flex items-center h-12 bg-background border-b w-full px-3">
        <Button variant="ghost" size="icon" className="mr-2" onClick={toggleSidebar}>
          <Menu size={18} />
        </Button>
        
        {/* Smooth transition for these buttons when sidebar collapses */}
        <div className={`flex items-center transition-all duration-300 ${sidebarOpen ? 'ml-[10.5rem]' : 'ml-0'}`}>
          <Button variant="ghost" size="icon" onClick={createNewChat} className="mr-2">
            <Plus size={18} />
          </Button>
          
          <Select
            value={model}
            onValueChange={setModel}
          >
            <SelectTrigger className="w-[180px] h-8">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gpt-4o">GPT-4o</SelectItem>
              <SelectItem value="qwen-qwq-32b">Qwen QwQ-32B</SelectItem>
            </SelectContent>
          </Select>
        </div>
        
        <div className="ml-auto">
          <Button variant="outline" size="icon" onClick={exportChat}>
            <DownloadCloud size={18} />
          </Button>
        </div>
      </div>

      {/* Left Sidebar - Position absolute */}
      <div 
        className={`fixed left-0 top-12 bottom-0 z-20 transition-all duration-300 border-r bg-background ${
          sidebarOpen ? 'w-[260px]' : 'w-0 overflow-hidden'
        }`}
      >
        {/* Search Input */}
        <div className="sticky top-0 z-20 bg-background w-full border-b p-3">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={filterValue}
              onChange={(e) => setFilterValue(e.target.value)}
              placeholder="搜索对话..."
              className="pl-8 h-8 text-sm"
            />
          </div>
        </div>
        
        {/* Chat List */}
        <div className="p-2 overflow-y-auto h-[calc(100%-56px)]">
          {filteredChats.length === 0 ? (
            <div className="text-center py-4 text-sm text-muted-foreground">
              没有找到匹配的对话
            </div>
          ) : (
            filteredChats.map(chat => (
              <div
                key={chat.id}
                className={`mb-1 cursor-pointer rounded-lg p-2 text-sm text-ellipsis text-nowrap ${
                  chat.id === activeChat ? 'bg-muted' : 'hover:bg-muted/50'
                }`}
                onClick={() => setActiveChat(chat.id)}
              >
                {chat.title}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Chat Area - Full width with padding */}
      <div className={`w-full pt-12 flex flex-col transition-all duration-300 ${
        sidebarOpen ? 'pl-[260px]' : 'pl-0'
      }`}>
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4">
          {activeMessages.map(message => (
            <div key={message.id} className="mb-4">
              {message.role === 'user' ? (
                <div className="ml-auto max-w-[80%]">
                  <div className="rounded-lg bg-muted p-3 markdown-content">
                    <div className="prose prose-headings:mt-2 prose-headings:mb-2 prose-headings:font-bold prose-p:my-1 max-w-none">
                      <Markdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                          code(props) {
                            const {children, className, ...rest} = props
                            const match = /language-(\w+)/.exec(className || '')
                            return match ? (
                              <SyntaxHighlighter
                                style={dracula}
                                language={match[1]}
                                customStyle={{ margin: 0 }}
                              >
                                {String(children).replace(/\n$/, '')}
                              </SyntaxHighlighter>
                            ) : (
                              <code {...rest} className={className}>
                                {children}
                              </code>
                            )
                          }
                        }}
                      >
                        {message.content}
                      </Markdown>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mr-auto max-w-[80%]">
                  <div className="rounded-lg p-3 markdown-content">
                    <div className="prose prose-headings:mt-2 prose-headings:mb-2 prose-headings:font-bold prose-p:my-1 max-w-none">
                      <Markdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                          code(props) {
                            const {children, className, ...rest} = props
                            const match = /language-(\w+)/.exec(className || '')
                            return match ? (
                              <SyntaxHighlighter
                                style={dracula}
                                language={match[1]}
                                customStyle={{ margin: 0 }}
                              >
                                {String(children).replace(/\n$/, '')}
                              </SyntaxHighlighter>
                            ) : (
                              <code {...rest} className={className}>
                                {children}
                              </code>
                            )
                          }
                        }}
                      >
                        {message.content}
                      </Markdown>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Input Area */}
        <div className="border-t p-4">
          <div className="relative">
            <Textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={isStreaming ? "等待AI回复中..." : "输入消息..."}
              className="pr-10 rounded-lg bg-muted resize-none min-h-[60px] max-h-[200px]"
              disabled={isStreaming}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
            />
            <Button 
              size="icon" 
              variant="ghost" 
              className="absolute right-2 bottom-2" 
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isStreaming}
            >
              <Send size={18} />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}