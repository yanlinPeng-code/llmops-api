#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/8/13 21:57
@Author  : thezehui@gmail.com
@File    : logging_extension.py
"""
import logging
import os.path

from concurrent_log_handler import ConcurrentTimedRotatingFileHandler
from flask import Flask


def init_app(app: Flask):
    """日志记录器初始化"""
    # 1.设置日志存储的文件夹，如果不存在则创建
    # 根据不同的环境配置logging根处理器的日志级别
    logging.getLogger().setLevel(
        logging.DEBUG if app.debug or os.getenv("FLASK_ENV") == "development" else logging.WARNING
    )
    log_folder = os.path.join(os.getcwd(), "storage", "log")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # 2.定义日志的文件名
    log_file = os.path.join(log_folder, "app.log")

    # 3.设置日志的格式，并且让日志每天更新一次
    handler = ConcurrentTimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s]: %(message)s"
    )
    handler.setLevel(logging.DEBUG if app.debug or os.getenv("FLASK_ENV") == "development" else logging.WARNING)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    # 4.在开发环境下同时将日志输出到控制台
    if app.debug or os.getenv("FLASK_ENV") == "development":
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
