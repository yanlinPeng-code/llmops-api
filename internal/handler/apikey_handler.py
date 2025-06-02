#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/19 15:04
@Author  : thezehui@gmail.com
@File    : api_key_handler.py
"""
from dataclasses import dataclass
from uuid import UUID

from flask import request
from flask_login import login_required, current_user
from injector import inject

from internal.schema.apikey_schema import (
    CreateApiKeyReq,
    UpdateApiKeyReq,
    UpdateApiKeyIsActiveReq,
    GetApiKeysWithPageResp,
)
from internal.service import ApiKeyService
from pkg.paginator import PaginatorReq, PageModel
from pkg.response import validate_error_json, success_message, success_json


@inject
@dataclass
class ApiKeyHandler:
    """API秘钥处理器"""
    api_key_service: ApiKeyService

    @login_required
    def create_api_key(self):
        """创建API秘钥"""
        # 1.提取请求并校验
        req = CreateApiKeyReq()
        if not req.validate():
            return validate_error_json(req.errors)

        # 2.调用服务创建秘钥
        self.api_key_service.create_api_key(req, current_user)

        return success_message("创建API秘钥成功")

    @login_required
    def delete_api_key(self, api_key_id: UUID):
        """根据传递的id删除API秘钥"""
        self.api_key_service.delete_api_key(api_key_id, current_user)
        return success_message("删除API秘钥成功")

    @login_required
    def update_api_key(self, api_key_id: UUID):
        """根据传递的信息更新API秘钥"""
        # 1.提取请求并校验
        req = UpdateApiKeyReq()
        print(req.data)
        if not req.validate():
            return validate_error_json(req.errors)

        # 2.调用服务更新秘钥
        self.api_key_service.update_api_key(api_key_id, current_user, **req.data)

        return success_message("更新API秘钥成功")

    @login_required
    def update_api_key_is_active(self, api_key_id: UUID):
        """根据传递的信息更新API秘钥激活状态"""
        # 1.提取请求并校验
        req = UpdateApiKeyIsActiveReq()
        if not req.validate():
            return validate_error_json(req.errors)

        # 2.调用服务更新秘钥是否激活
        self.api_key_service.update_api_key(api_key_id, current_user, **req.data)

        return success_message("更新API秘钥激活状态成功")

    @login_required
    def get_api_keys_with_page(self):
        """获取当前登录账号的API秘钥分页列表信息"""
        # 1.提取请求并校验
        req = PaginatorReq(request.args)
        if not req.validate():
            return validate_error_json(req.errors)

        # 2.调用服务获取数据
        api_keys, paginator = self.api_key_service.get_api_keys_with_page(req, current_user)

        # 3.构建响应结构并返回
        resp = GetApiKeysWithPageResp(many=True)

        return success_json(PageModel(list=resp.dump(api_keys), paginator=paginator))
