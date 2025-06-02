import json
from dataclasses import dataclass

from flask import current_app
from injector import inject
from typing_extensions import Generator

from internal.core.agent.agents import FunctionCallAgent, ReACTAgent
from internal.core.agent.entities.agent_entity import AgentConfig
from internal.core.agent.entities.queue_entity import QueueEvent
from internal.core.language_model.entities.model_entity import ModelFeature
from internal.core.memory.token_buffer_memory import TokenBufferMemory
from internal.entity.app_entity import AppStatus
from internal.entity.conversation_entity import InvokeFrom, MessageStatus
from internal.entity.dataset_entity import RetrievalSource
from internal.exception import NotFoundException, ForbiddenException
from internal.model import Account, EndUser, Conversation, Message
from internal.schema.openapi_schema import OpenAPIChatReq
from pkg.response import Response
from pkg.sqlalchemy import SQLAlchemy
from .app_config_service import AppConfigService
from .app_service import AppService
from .base_service import BaseService
from .conversation_service import ConversationService
from .language_model_service import LanguageModelService
from .optimized_mcp_service import OptimizedMCPServiceWithFucCache
from .retrieval_service import RetrievalService


@inject
@dataclass
class OpenAPIService(BaseService):
    """开放API服务"""
    db: SQLAlchemy
    app_service: AppService
    retrieval_service: RetrievalService
    app_config_service: AppConfigService
    conversation_service: ConversationService
    language_model_service: LanguageModelService
    optimized_mcp_service: OptimizedMCPServiceWithFucCache

    def chat(self, req: OpenAPIChatReq, account: Account):
        """根据传递的请求+账号信息发起聊天对话，返回数据为块内容或者生成器"""
        # 1.判断当前应用是否属于当前账号
        app = self.app_service.get_app(req.app_id.data, account)

        # 2.判断当前应用是否已发布
        if app.status != AppStatus.PUBLISHED:
            raise NotFoundException("该应用不存在或未发布，请核实后重试")

        # 3.判断是否传递了终端用户id，如果传递了则检测终端用户关联的应用
        if req.end_user_id.data:
            end_user = self.get(EndUser, req.end_user_id.data)
            if not end_user or end_user.app_id != app.id:
                raise ForbiddenException("当前账号不存在或不属于该应用，请核实后重试")
        else:
            # 4.如果不存在则创建一个终端用户
            end_user = self.create(
                EndUser,
                **{"tenant_id": account.id, "app_id": app.id},
            )

        # 5.检测是否传递了会话id，如果传递了需要检测会话的归属信息
        if req.conversation_id.data:
            conversation = self.get(Conversation, req.conversation_id.data)
            if (
                    not conversation
                    or conversation.app_id != app.id
                    or conversation.invoke_from != InvokeFrom.SERVICE_API
                    or conversation.created_by != end_user.id
            ):
                raise ForbiddenException("该会话不存在，或者不属于该应用/终端用户/调用方式")
        else:
            # 6.如果不存在则创建会话信息
            conversation = self.create(Conversation, **{
                "app_id": app.id,
                "name": "New Conversation",
                "invoke_from": InvokeFrom.SERVICE_API,
                "created_by": end_user.id,
            })

        # 7.获取校验后的运行时配置
        app_config = self.app_config_service.get_app_config(app)

        # 8.新建一条消息记录
        message = self.create(Message, **{
            "app_id": app.id,
            "conversation_id": conversation.id,
            "invoke_from": InvokeFrom.SERVICE_API,
            "created_by": end_user.id,
            "query": req.query.data,
            "image_urls": req.image_urls.data,
            "status": MessageStatus.NORMAL,
        })

        # 9.从语言模型中根据模型配置获取模型实例
        llm = self.language_model_service.load_language_model(app_config.get("model_config", {}))

        # 10.实例化TokenBufferMemory用于提取短期记忆
        token_buffer_memory = TokenBufferMemory(
            db=self.db,
            conversation=conversation,
            model_instance=llm,
        )
        history = token_buffer_memory.get_history_prompt_messages(
            message_limit=app_config["dialog_round"],
        )

        # 11.将草稿配置中的tools转换成LangChain工具
        tools = self.app_config_service.get_langchain_tools_by_tools_config(app_config["tools"])

        # 8.将草稿配置中的mcp_tool
        if app_config["mcp_tools"]:
            print(app_config["mcp_tools"])
            mcp_tools = self.optimized_mcp_service.get_langchain_tools_by_mcp_tool_config(app_config["mcp_tools"])
            # mcp_tools = self.app_config_service.get_langchain_tools_by_mcp_tool_config(draft_app_config["mcp_tools"])
            for tool in mcp_tools:
                tools.append(tool)
        # 12.检测是否关联了知识库
        if app_config["datasets"]:
            # 13.构建LangChain知识库检索工具
            dataset_retrieval = self.retrieval_service.create_langchain_tool_from_search(
                flask_app=current_app._get_current_object(),
                dataset_ids=[dataset["id"] for dataset in app_config["datasets"]],
                account_id=account.id,
                retrival_source=RetrievalSource.APP,
                **app_config["retrieval_config"],
            )
            tools.append(dataset_retrieval)

        # 14.检测是否关联工作流，如果关联了工作流则将工作流构建成工具添加到tools中
        if app_config["workflows"]:
            workflow_tools = self.app_config_service.get_langchain_tools_by_workflow_ids(
                [workflow["id"] for workflow in app_config["workflows"]]
            )
            tools.extend(workflow_tools)

        # 14.根据LLM是否支持tool_call决定使用不同的Agent
        agent_class = FunctionCallAgent if ModelFeature.TOOL_CALL in llm.features else ReACTAgent
        agent = agent_class(
            llm=llm,
            agent_config=AgentConfig(
                user_id=account.id,
                invoke_from=InvokeFrom.DEBUGGER,
                preset_prompt=app_config["preset_prompt"],
                enable_long_term_memory=app_config["long_term_memory"]["enable"],
                tools=tools,
                review_config=app_config["review_config"],
            ),
        )

        # 15.定义智能体状态基础数据
        agent_state = {
            "messages": [llm.convert_to_human_message(req.query.data, req.image_urls.data)],
            "history": history,
            "long_term_memory": conversation.summary,
        }

        # 16.根据stream类型差异执行不同的代码
        if req.stream.data is True:
            agent_thoughts_dict = {}

            def handle_stream() -> Generator:
                """流式事件处理器，在Python只要在函数内部使用了yield关键字，那么这个函数的返回值类型肯定是生成器"""
                for agent_thought in agent.stream(agent_state):
                    # 提取thought以及answer
                    event_id = str(agent_thought.id)

                    # 将数据填充到agent_thought，便于存储到数据库服务中
                    if agent_thought.event != QueueEvent.PING:
                        # 除了agent_message数据为叠加，其他均为覆盖
                        if agent_thought.event == QueueEvent.AGENT_MESSAGE:
                            if event_id not in agent_thoughts_dict:
                                # 初始化智能体消息事件
                                agent_thoughts_dict[event_id] = agent_thought
                            else:
                                # 叠加智能体消息
                                agent_thoughts_dict[event_id] = agent_thoughts_dict[event_id].model_copy(update={
                                    "thought": agent_thoughts_dict[event_id].thought + agent_thought.thought,
                                    "answer": agent_thoughts_dict[event_id].answer + agent_thought.answer,
                                    "latency": agent_thought.latency,
                                })
                        else:
                            # 处理其他类型事件的消息
                            agent_thoughts_dict[event_id] = agent_thought
                    data = {
                        **agent_thought.model_dump(include={
                            "event", "thought", "observation", "tool", "tool_input", "answer", "latency",
                        }),
                        "id": event_id,
                        "end_user_id": str(end_user.id),
                        "conversation_id": str(conversation.id),
                        "message_id": str(message.id),
                        "task_id": str(agent_thought.task_id),
                    }
                    yield f"event: {agent_thought.event}\ndata:{json.dumps(data)}\n\n"

                # 22.将消息以及推理过程添加到数据库
                self.conversation_service.save_agent_thoughts(
                    account_id=account.id,
                    app_id=app.id,
                    app_config=app_config,
                    conversation_id=conversation.id,
                    message_id=message.id,
                    agent_thoughts=[agent_thought for agent_thought in agent_thoughts_dict.values()],
                )

            return handle_stream()

        # 17.块内容输出
        agent_result = agent.invoke(agent_state)

        # 18.将消息以及推理过程添加到数据库
        self.conversation_service.save_agent_thoughts(
            account_id=account.id,
            app_id=app.id,
            app_config=app_config,
            conversation_id=conversation.id,
            message_id=message.id,
            agent_thoughts=agent_result.agent_thoughts,
        )

        return Response(data={
            "id": str(message.id),
            "end_user_id": str(end_user.id),
            "conversation_id": str(conversation.id),
            "query": req.query.data,
            "image_urls": req.image_urls.data,
            "answer": agent_result.answer,
            "total_token_count": 0,
            "latency": agent_result.latency,
            "agent_thoughts": [{
                "id": str(agent_thought.id),
                "event": agent_thought.event,
                "thought": agent_thought.thought,
                "observation": agent_thought.observation,
                "tool": agent_thought.tool,
                "tool_input": agent_thought.tool_input,
                "latency": agent_thought.latency,
                "created_at": 0,
            } for agent_thought in agent_result.agent_thoughts]
        })
