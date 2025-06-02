from flask_wtf import FlaskForm
from marshmallow import Schema, fields, pre_dump
from wtforms.fields.simple import StringField
from wtforms.validators import DataRequired, Length, URL, Regexp, AnyOf, Optional

from internal.core.workflow.entities.workflow_entity import WORKFLOW_CONFIG_NAME_PATTERN
from internal.entity.workflow_entity import WorkflowStatus
from internal.lib.helper import datetime_to_timestamp
from internal.model import Workflow
from pkg.paginator import PaginatorReq


class CreateWorkflowReq(FlaskForm):
    name = StringField("name", validators=[
        Length(max=50, message="名称的长度不能超过50个字符"),
        DataRequired()])

    tool_call_name = StringField("tool_call_name", validators=[
        Length(max=50, message="工具调用名称的长度不能超过50个字符"),
        DataRequired(),
        Regexp(WORKFLOW_CONFIG_NAME_PATTERN, message="英文名称仅支持字母、数字和下划线，且以字母/下划线为开头")
    ])
    icon = StringField("icon", validators=[
        DataRequired(),
        URL(message="请输入正确的图标链接")

    ])
    description = StringField("description", validators=[
        Length(max=1024, message="描述的长度不能超过1024个字符"),
        DataRequired()
    ])


class UpdateWorkflowReq(FlaskForm):
    """创建工作流基础请求"""
    name = StringField("name", validators=[
        DataRequired("工作流名称不能为空"),
        Length(max=50, message="工作流名称长度不能超过50"),
    ])
    tool_call_name = StringField("tool_call_name", validators=[
        DataRequired("英文名称不能为空"),
        Length(max=50, message="英文名称不能超过50个字符"),
        Regexp(WORKFLOW_CONFIG_NAME_PATTERN, message="英文名称仅支持字母、数字和下划线，且以字母/下划线为开头")
    ])
    icon = StringField("icon", validators=[
        DataRequired("工作流图标不能为空"),
        URL(message="工作流图标必须是图片URL地址"),
    ])
    description = StringField("description", validators=[
        DataRequired("工作流描述不能为空"),
        Length(max=1024, message="工作流描述不能超过1024个字符")
    ])


class GetWorkflowResp(Schema):
    """获取工作流详情响应结构"""
    id = fields.UUID(dump_default="")
    name = fields.String(dump_default="")
    tool_call_name = fields.String(dump_default="")
    icon = fields.String(dump_default="")
    description = fields.String(dump_default="")
    status = fields.String(dump_default="")
    is_debug_passed = fields.Boolean(dump_default=False)
    node_count = fields.Integer(dump_default=0)
    published_at = fields.Integer(dump_default=0)
    updated_at = fields.Integer(dump_default=0)
    created_at = fields.Integer(dump_default=0)

    @pre_dump
    def process_data(self, data: Workflow, **kwargs):
        return {
            "id": data.id,
            "name": data.name,
            "tool_call_name": data.tool_call_name,
            "icon": data.icon,
            "description": data.description,
            "status": data.status,
            "is_debug_passed": data.is_debug_passed,
            "node_count": len(data.draft_graph.get("nodes", [])),
            "published_at": datetime_to_timestamp(data.published_at),
            "updated_at": datetime_to_timestamp(data.updated_at),
            "created_at": datetime_to_timestamp(data.created_at),
        }


class GetWorkflowsWithPageReq(PaginatorReq):
    """获取工作流分页列表数据请求结构"""
    status = StringField("status", default="", validators=[
        Optional(),
        AnyOf(WorkflowStatus.__members__.values(), message="工作流状态格式错误")
    ])
    search_word = StringField("search_word", default="", validators=[Optional()])


class GetWorkflowsWithPageResp(Schema):
    """获取工作流分页列表数据响应结构"""
    id = fields.UUID(dump_default="")
    name = fields.String(dump_default="")
    tool_call_name = fields.String(dump_default="")
    icon = fields.String(dump_default="")
    description = fields.String(dump_default="")
    status = fields.String(dump_default="")
    is_debug_passed = fields.Boolean(dump_default=False)
    node_count = fields.Integer(dump_default=0)
    published_at = fields.Integer(dump_default=0)
    updated_at = fields.Integer(dump_default=0)
    created_at = fields.Integer(dump_default=0)

    @pre_dump
    def process_data(self, data: Workflow, **kwargs):
        return {
            "id": data.id,
            "name": data.name,
            "tool_call_name": data.tool_call_name,
            "icon": data.icon,
            "description": data.description,
            "status": data.status,
            "is_debug_passed": data.is_debug_passed,
            "node_count": len(data.graph.get("nodes", [])),
            "published_at": datetime_to_timestamp(data.published_at),
            "updated_at": datetime_to_timestamp(data.updated_at),
            "created_at": datetime_to_timestamp(data.created_at),
        }
