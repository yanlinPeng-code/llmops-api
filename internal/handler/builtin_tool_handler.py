import io
from dataclasses import dataclass

from flask import send_file
from flask_login import login_required
from injector import inject

from internal.service import BuiltinToolService
from pkg.response import success_json


@inject
@dataclass
class BuiltinToolHandler:
    """内置工具处理器"""
    builtin_tool_service: BuiltinToolService

    @login_required
    def get_builtin_tools(self):
        """获取LLMOps所有内置工具信息+提供商信息"""
        builtin_tools = self.builtin_tool_service.get_builtin_tools()
        return success_json(builtin_tools)

    @login_required
    def get_provider_tool(self, provider_name: str, tool_name: str):
        """根据传递的提供商名字+工具名字获取指定工具的信息"""
        builtin_tool = self.builtin_tool_service.get_provider_tool(provider_name, tool_name)
        return success_json(builtin_tool)

    def get_provider_icon(self, provider_name: str):
        """根据传递的提供商获取icon图标流信息"""
        icon, mimetype = self.builtin_tool_service.get_provider_icon(provider_name)
        return send_file(io.BytesIO(icon), mimetype)

    @login_required
    def get_categories(self):
        """获取所有内置提供商的分类信息"""
        categories = self.builtin_tool_service.get_categories()
        return success_json(categories)
