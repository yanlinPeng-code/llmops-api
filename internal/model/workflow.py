from datetime import datetime

from sqlalchemy import (
    Column,
    UUID,
    String,
    Text,
    Boolean,
    DateTime,
    Float,
    text,
    PrimaryKeyConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB

from internal.extension.database_extension import db


class Workflow(db.Model):
    """工作流模型"""
    __tablename__ = "workflow"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_workflow_id"),
        Index("workflow_account_id_idx", "account_id"),
        Index("workflow_tool_call_name_idx", "tool_call_name"),
    )

    id = Column(UUID, nullable=False, server_default=text("uuid_generate_v4()"))
    account_id = Column(UUID, nullable=False)  # 创建账号id
    name = Column(String(255), nullable=False, server_default=text("''::character varying"))  # 工作流名字
    tool_call_name = Column(String(255), nullable=False, server_default=text("''::character varying"))  # 工作流工具调用名字
    icon = Column(String(255), nullable=False, server_default=text("''::character varying"))  # 工作流图标
    description = Column(Text, nullable=False, server_default=text("''::text"))  # 应用描述
    graph = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))  # 运行时配置
    draft_graph = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))  # 草稿图配置
    is_debug_passed = Column(Boolean, nullable=False, server_default=text("false"))  # 是否调试通过
    status = Column(String(255), nullable=False, server_default=text("''::character varying"))  # 工作流状态
    published_at = Column(DateTime, nullable=True)  # 发布时间
    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
        onupdate=datetime.now,
    )
    created_at = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP(0)"))


class WorkflowResult(db.Model):
    """工作流存储结果模型"""
    __tablename__ = "workflow_result"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_workflow_result_id"),
        Index("workflow_result_app_id_idx", "app_id"),
        Index("workflow_result_account_id_idx", "account_id"),
        Index("workflow_result_workflow_id_idx", "workflow_id"),
    )

    id = Column(UUID, nullable=False, server_default=text("uuid_generate_v4()"))  # 结果id
    app_id = Column(UUID, nullable=True)  # 工作流调用的应用id，如果为空则代表非应用调用
    account_id = Column(UUID, nullable=False)  # 创建账号id
    workflow_id = Column(UUID, nullable=False)  # 结果关联的工作流id
    graph = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))  # 运行时配置
    state = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))  # 工作流最终状态
    latency = Column(Float, nullable=False, server_default=text("0.0"))  # 消息的总耗时
    status = Column(String(255), nullable=False, server_default=text("''::character varying"))  # 运行状态
    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
        onupdate=datetime.now,
    )
    created_at = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP(0)"))
