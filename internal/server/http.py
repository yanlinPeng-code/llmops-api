#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/3/29 15:10
@Author  : thezehui@gmail.com
@File    : http.py
"""
import logging
import os

from flask import Flask
from flask_cors import CORS
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_weaviate import FlaskWeaviate

from config import Config
from internal.exception import CustomException
from internal.extension import logging_extension, redis_extension, celery_extension
from internal.middleware import Middleware
from internal.router import Router
from pkg.response import json, Response, HttpCode
from pkg.sqlalchemy import SQLAlchemy


class Http(Flask):
    """Http服务引擎"""

    preset_mcp_list: list = []

    def __init__(
            self,
            *args,
            conf: Config,
            db: SQLAlchemy,
            weaviate: FlaskWeaviate,
            migrate: Migrate,
            login_manager: LoginManager,
            # 中间件
            middleware: Middleware,
            router: Router,
            **kwargs,
    ):
        # 1.调用父类构造函数初始化
        super().__init__(*args, **kwargs)

        # 2.初始化应用配置
        self.config.from_object(conf)

        # 3.注册绑定异常错误处理
        self.register_error_handler(Exception, self._register_error_handler)

        # 4.初始化flask扩展
        db.init_app(self)
        weaviate.init_app(self)
        migrate.init_app(self, db, directory="internal/migration")
        redis_extension.init_app(self)
        celery_extension.init_app(self)
        logging_extension.init_app(self)
        login_manager.init_app(self)

        # 5.解决前后端跨域问题
        CORS(self, resources={
            r"/*": {
                "origins": "http://localhost:5173",
                "supports_credentials": True,
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Always include OPTIONS for preflight
                "allow_headers": ["Content-Type", "Authorization"],  # Add any custom headers your frontend sends
            }
        })
        # CORS(self, resources={
        #     r"/*": {
        #         "origins": "*",
        #         "supports_credentials": True,
        #         # "methods": ["GET", "POST"],
        #         # "allow_headers": ["Content-Type"],
        #     }
        # })
        # 6.注册应用中间件
        login_manager.request_loader(middleware.request_loader)

        # 5.注册应用路由
        router.register_router(self)

    def _register_error_handler(self, error: Exception):
        # 1.日志记录异常信息
        logging.error("An error occurred: %s", error, exc_info=True)

        # 2.异常信息是不是我们的自定义异常，如果是可以提取message和code等信息
        if isinstance(error, CustomException):
            return json(Response(
                code=error.code,
                message=error.message,
                data=error.data if error.data is not None else {},
            ))

        # 3.如果不是我们的自定义异常，则有可能是程序、数据库抛出的异常，也可以提取信息，设置为FAIL状态码
        if self.debug or os.getenv("FLASK_ENV") == "development":
            raise error
        else:
            return json(Response(
                code=HttpCode.FAIL,
                message=str(error),
                data={},
            ))
