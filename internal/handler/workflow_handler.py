from dataclasses import dataclass
from uuid import UUID

from flask import request
from flask_login import current_user, login_required
from injector import inject

from internal.schema.workflow_schema import CreateWorkflowReq, UpdateWorkflowReq, GetWorkflowResp, \
    GetWorkflowsWithPageReq, GetWorkflowsWithPageResp
from internal.service import WorkflowService
from pkg.paginator import PageModel
from pkg.response import validate_error_json, success_json, success_message, compact_generate_response


@inject
@dataclass
class WorkflowHandler:
    workflow_service: WorkflowService

    @login_required
    def create_workflow(self):
        """新增工作流"""
        # 1.提取请求并校验
        req = CreateWorkflowReq()
        if not req.validate():
            return validate_error_json(req.errors)

        # 2.调用服务创建工作流
        workflow = self.workflow_service.create_workflow(req, current_user)

        return success_json({"id": workflow.id})

    @login_required
    def delete_workflow(self, workflow_id: UUID):
        """根据传递的工作流id删除指定的工作流"""
        self.workflow_service.delete_workflow(workflow_id, current_user)
        return success_message("删除工作流成功")

    @login_required
    def update_workflow(self, workflow_id: UUID):
        """根据传递的工作流id获取工作流详情"""
        # 1.提取请求并校验
        req = UpdateWorkflowReq()
        print("*" * 100, req.data, "*" * 100)
        if not req.validate():
            return validate_error_json(req.errors)

        # 2.调用服务更新工作流数据
        self.workflow_service.update_workflow(workflow_id, current_user, **req.data)

        return success_message("修改工作流基础信息成功")

    @login_required
    def get_workflow(self, workflow_id: UUID):
        """根据传递的工作流id获取工作流详情"""
        workflow = self.workflow_service.get_workflow(workflow_id, current_user)
        resp = GetWorkflowResp()
        return success_json(resp.dump(workflow))

    @login_required
    def get_workflows_with_page(self):
        """获取当前登录账号下的工作流分页列表数据"""
        # 1.提取请求并校验
        req = GetWorkflowsWithPageReq(request.args)
        if not req.validate():
            return validate_error_json(req.errors)

        # 2.获取分页列表数据
        workflows, paginator = self.workflow_service.get_workflows_with_page(req, current_user)

        # 3.构建响应并返回
        resp = GetWorkflowsWithPageResp(many=True)

        return success_json(PageModel(list=resp.dump(workflows), paginator=paginator))

    @login_required
    def update_draft_graph(self, workflow_id: UUID):
        """根据传递的工作流id+请求信息更新工作流草稿图配置"""
        # 1.提取草稿图接口请求json数据
        draft_graph_dict = request.get_json(force=True, silent=True) or {
            "nodes": [],
            "edges": [],
        }

        # 2.调用服务更新工作流的草稿图配置
        self.workflow_service.update_draft_graph(workflow_id, draft_graph_dict, current_user)

        return success_message("更新工作流草稿配置成功")

    @login_required
    def get_draft_graph(self, workflow_id: UUID):
        """根据传递的工作流id获取该工作流的草稿配置信息"""
        draft_graph = self.workflow_service.get_draft_graph(workflow_id, current_user)
        return success_json(draft_graph)

    @login_required
    def debug_workflow(self, workflow_id: UUID):
        """根据传递的变量字典+工作流id调试指定的工作流"""
        # 1.提取用户传递的输入变量信息
        inputs = request.get_json(force=True, silent=True) or {}

        # 2.调用服务调试指定的API接口
        response = self.workflow_service.debug_workflow(workflow_id, inputs, current_user)

        return compact_generate_response(response)

    @login_required
    def publish_workflow(self, workflow_id: UUID):
        """根据传递的工作流id发布指定的工作流"""
        self.workflow_service.publish_workflow(workflow_id, current_user)
        return success_message("发布工作流成功")

    @login_required
    def cancel_publish_workflow(self, workflow_id: UUID):
        """根据传递的工作流id取消发布指定的工作流"""
        self.workflow_service.cancel_publish_workflow(workflow_id, current_user)
        return success_message("取消发布工作流成功")
