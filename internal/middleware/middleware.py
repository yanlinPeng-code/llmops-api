from dataclasses import dataclass

from flask import Request
from injector import inject
from typing_extensions import Optional

from internal.exception import UnauthorizedException
from internal.model.account import Account
from internal.service import JwtService, AccountService, ApiKeyService


@inject
@dataclass
class Middleware:
    """应用中间件，可以重写request_loader与unauthorized_handler"""
    jwt_service: JwtService
    api_key_service: ApiKeyService
    account_service: AccountService

    def request_loader(self, request: Request) -> Optional[Account]:
        """登录管理器的请求加载器"""
        # 1.单独为llmops路由蓝图创建请求加载器
        if request.blueprint == "llmops":
            # 2.提取请求头headers中的信息
            # 2.校验access_token
            access_token = self._validate_credential(request)
            # 3.解析token信息得到用户信息并返回
            payload = self.jwt_service.parse_token(access_token)
            account_id = payload.get("sub")
            account = self.account_service.get_account(account_id)
            if not account:
                raise UnauthorizedException("用户不存在")
            return account
        elif request.blueprint == "openapi":
            # 4.校验获取api_key
            api_key = self._validate_credential(request)

            # 5. # 5.解析得到APi秘钥记录
            api_key_record = self.api_key_service.get_api_by_by_credential(api_key)

            if not api_key_record or not api_key_record.is_active:
                raise UnauthorizedException("该ApiKey不存在或已禁用")
            return api_key_record.account
        else:
            return None

    @classmethod
    def _validate_credential(cls, request: Request) -> str:
        """校验请求头中的凭证信息，涵盖access_token和api_key"""
        # 1.提取请求头headers中的信息
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise UnauthorizedException("该接口需要授权才能访问，请登录后尝试")
        # 2.请求信息中没有空格分隔符，则验证失败，Authorization: Bearer access_token
        if " " not in auth_header:
            raise UnauthorizedException("该接口需要授权才能访问，验证格式失败")
        # 4.分割授权信息，必须符合Bearer access_token
        auth_schema, credential = auth_header.split(None, 1)
        if auth_schema.lower() != "bearer":
            raise UnauthorizedException("该接口需要授权才能访问，验证格式失败")
        return credential
