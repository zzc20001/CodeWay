import re
from typing import List, Tuple, Optional

from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema.output_parser import StrOutputParser
from langchain.tools.base import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import render_text_description
from pydantic import ValidationError
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler

from Agent.Action import Action
from Utils.CallbackHandlers import *
from Utils.LangfuseMonitor import LangfuseMonitor


class ReActAgent:
    """AutoGPT：基于Langchain实现"""

    @staticmethod
    def __format_thought_observation(thought: str, action: Action, observation: str) -> str:
        # 将全部JSON代码块替换为空
        ret = re.sub(r'```json(.*?)```', '', thought, flags=re.DOTALL)
        ret += "\n" + str(action) + "\n返回结果:\n" + observation
        return ret

    @staticmethod
    def __extract_json_action(text: str) -> str | None:
        # 匹配最后出现的JSON代码块
        json_pattern = re.compile(r'```json(.*?)```', re.DOTALL)
        matches = json_pattern.findall(text)
        if matches:
            last_json_str = matches[-1]
            return last_json_str
        return None

    def __init__(
            self,
            llm: BaseChatModel,
            tools: List[BaseTool],
            work_dir: str,
            main_prompt_file: str,
            max_thought_steps: Optional[int] = 10,
            callbacks: Optional[List[BaseCallbackHandler]] = None,
    ):
        self.llm = llm
        self.tools = tools
        self.work_dir = work_dir
        self.max_thought_steps = max_thought_steps
        self.callbacks = callbacks or []

        # OutputFixingParser： 如果输出格式不正确，尝试修复
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.robust_parser = OutputFixingParser.from_llm(
            parser=self.output_parser,
            llm=llm
        )

        self.main_prompt_file = main_prompt_file

        self.__init_prompt_templates()
        self.__init_chains()

        self.verbose_handler = ColoredPrintHandler(color=THOUGHT_COLOR)
        print("Tools:")
        print(self.tools)

    def __init_prompt_templates(self):
        with open(self.main_prompt_file, 'r', encoding='utf-8') as f:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template(f.read()),
                ]
            ).partial(
                work_dir=self.work_dir,
                tools=render_text_description(self.tools),
                tool_names=','.join([tool.name for tool in self.tools]),
                format_instructions=self.output_parser.get_format_instructions(),
            )

    def __init_chains(self):
        # 主流程的chain
        self.main_chain = (self.prompt | self.llm | StrOutputParser())

    def __find_tool(self, tool_name: str) -> Optional[BaseTool]:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def __step(self,
               task,
               short_term_memory,
               chat_history,
               verbose=False
               ) -> Tuple[Action, str]:

        """执行一步思考"""

        inputs = {
            "input": task,
            "agent_scratchpad": "\n".join(short_term_memory),
            "chat_history": chat_history.messages,
        }

        # 合并所有回调，包括主动传入的和基于verbose的
        all_callbacks = self.callbacks.copy()
        if verbose:
            all_callbacks.append(self.verbose_handler)
            
        config = {
            "callbacks": all_callbacks
        }
        response = ""
        chunk_received = False
        for s in self.main_chain.stream(inputs, config=config):
            response += s
            chunk_received = True
        
        # If no chunks were returned, handle the error gracefully
        if not chunk_received:
            return Action(name="FINISH", args={"response": "抱歉，处理请求时发生错误：模型未返回任何内容。可能是网络连接问题或模型服务暂时不可用。请稍后再试。"}), "No generation chunks were returned"
        # 提取JSON代码块
        json_action = self.__extract_json_action(response)
        # 带容错的解析
        action = self.robust_parser.parse(
            json_action if json_action else response
        )
        return action, response

    def __exec_action(self, action: Action) -> str:
        # 查找工具
        tool = self.__find_tool(action.name)
        if tool is None:
            observation = (
                f"Error: 找不到工具或指令 '{action.name}'. "
                f"请从提供的工具/指令列表中选择，请确保按对顶格式输出。"
            )
        else:
            try:
                # 执行工具
                observation = tool.run(action.args)
            except ValidationError as e:
                # 工具的入参异常
                observation = (
                    f"Validation Error in args: {str(e)}, args: {action.args}"
                )
            except Exception as e:
                # 工具执行异常
                observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"

        return observation

    def run(
            self,
            task: str,
            chat_history: ChatMessageHistory,
            verbose=False,
            session_id: Optional[str] = None,
            user_id: Optional[str] = None,
            callbacks: Optional[List[BaseCallbackHandler]] = None
    ) -> str:
        """
        运行智能体
        :param task: 用户任务
        :param chat_history: 对话上下文（长时记忆）
        :param verbose: 是否显示详细信息
        :param session_id: 可选的会话ID，用于Langfuse跟踪
        :param user_id: 可选的用户ID，用于Langfuse跟踪
        :param callbacks: 外部提供的回调处理程序列表
        """
        # 初始化短时记忆: 记录推理过程
        short_term_memory = []

        # 思考步数
        thought_step_count = 0

        reply = ""
        
        # 合并所有回调处理程序
        run_callbacks = self.callbacks.copy()
        
        # 添加外部提供的回调
        if callbacks:
            run_callbacks.extend(callbacks)
            
        # 如果提供了 session_id 或 user_id 但没有提供外部回调，尝试创建 Langfuse 回调
        if (session_id or user_id) and not callbacks:
            langfuse_monitor = LangfuseMonitor()
            langfuse_handler = langfuse_monitor.get_langchain_handler(user_id=user_id, session_id=session_id)
            if langfuse_handler:
                run_callbacks.append(langfuse_handler)

        # 开始逐步思考
        while thought_step_count < self.max_thought_steps:
            if verbose:
                self.verbose_handler.on_thought_start(thought_step_count)

            # 临时合并运行时的回调和实例的回调
            original_callbacks = self.callbacks
            self.callbacks = run_callbacks
            
            # 执行一步思考
            action, response = self.__step(
                task=task,
                short_term_memory=short_term_memory,
                chat_history=chat_history,
                verbose=verbose,
            )
            
            # 恢复原始回调
            self.callbacks = original_callbacks

            # 如果是结束指令，执行最后一步
            if action.name == "FINISH":
                reply = self.__exec_action(action)
                break

            # 执行动作
            observation = self.__exec_action(action)

            if verbose:
                self.verbose_handler.on_tool_end(observation)

            # 更新短时记忆
            short_term_memory.append(
                self.__format_thought_observation(
                    response, action, observation
                )
            )

            thought_step_count += 1

        if thought_step_count >= self.max_thought_steps:
            # 如果思考步数达到上限，返回错误信息
            reply = "抱歉，我没能完成您的任务。"

        # 更新长时记忆
        chat_history.add_user_message(task)
        chat_history.add_ai_message(reply)
        return reply
