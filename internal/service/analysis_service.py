import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from injector import inject
from redis import Redis
from typing_extensions import Any

from internal.model import Account, App, Message
from pkg.sqlalchemy import SQLAlchemy
from .app_service import AppService
from .base_service import BaseService


@inject
@dataclass
class AnalysisService(BaseService):
    """统计分析服务"""
    db: SQLAlchemy
    redis_client: Redis
    app_service: AppService

    def get_app_analysis(self, app_id: UUID, account: Account) -> dict[str, Any]:
        """根据传递的应用id+账号获取指定应用的分析信息"""
        # 1.根据传递的应用id获取应用信息并校验权限
        app = self.app_service.get_app(app_id, account)

        # 2.获取当前时间、午夜时间、7天前时间、14天前的时间
        today = datetime.now()
        today_midnight = datetime.combine(today, datetime.min.time())
        seven_days_ago = today_midnight - timedelta(days=7)
        fourteen_days_ago = today_midnight - timedelta(days=14)

        # 3.计算统计分析数据的缓存键
        cache_key = f"{today.strftime('%Y_%m_%d')}:{str(app.id)}"

        # 4.从缓存中获取指定的结果，如果存在则直接返回
        try:
            if self.redis_client.exists(cache_key):
                # 5.解析数据并将数据返回
                app_analysis = self.redis_client.get(cache_key)
                return json.loads(app_analysis)
        except Exception:
            # 6.如果出错则什么都不处理，重新计算数据并更新缓存
            pass

        # 7.查询消息表最近7天，以及最近14-7天的数据（数据要求答案非空、并且只要需要计算的数据）
        seven_days_messages = self.get_messages_by_time_range(app, seven_days_ago, today_midnight)
        fourteen_days_messages = self.get_messages_by_time_range(app, fourteen_days_ago, seven_days_ago)

        # 8.计算5个概念指标，涵盖：全部会话数、激活用户数、平均会话互动数、Token输出速度、费用消耗
        seven_overview_indicators = self.calculate_overview_indicators_by_messages(seven_days_messages)
        fourteen_overview_indicators = self.calculate_overview_indicators_by_messages(fourteen_days_messages)

        # 9.统计环比数据
        pop = self.calculate_pop_by_overview_indicators(seven_overview_indicators, fourteen_overview_indicators)

        # 10.计算4个指标对应的趋势
        trend = self.calculate_trend_by_messages(today_midnight, 7, seven_days_messages)

        # 11.定义5个指标字段名称
        fields = [
            "total_messages", "active_accounts", "avg_of_conversation_messages",
            "token_output_rate", "cost_consumption",
        ]

        # 12.构建应用分析字典
        app_analysis = {
            **trend,
            **{
                field: {
                    "data": seven_overview_indicators.get(field),
                    "pop": pop.get(field),
                } for field in fields
            }
        }

        # 13.将数据存储到redis缓存中，并设置过期时间为1天
        self.redis_client.setex(cache_key, 24 * 60 * 60, json.dumps(app_analysis))

        return app_analysis

    def get_messages_by_time_range(self, app: App, start_at: datetime, end_at: datetime) -> list[Message]:
        """根据传递的时间段获取指定应用的消息会话数据"""
        return self.db.session.query(Message).with_entities(
            Message.id, Message.conversation_id, Message.created_by,
            Message.latency, Message.total_token_count, Message.total_price,
            Message.created_at,
        ).filter(
            Message.app_id == app.id,
            Message.created_at >= start_at,
            Message.created_at < end_at,
            Message.answer != "",
        ).all()

    @classmethod
    def calculate_overview_indicators_by_messages(cls, messages: list[Message]) -> dict[str, Any]:
        """根据传递的消息列表计算概览指标，涵盖全部会话数、激活用户数、平均会话互动数、Token输出速度、费用消耗"""
        # 1.计算全部会话数，使用消息总数来计算
        total_messages = len(messages)

        # 2.计算激活用户数，使用唯一用户数判断
        active_accounts = len({message.created_by for message in messages})

        # 3.平均会话互动数，使用消息总数/会话总数，涉及除法要做/0判断
        avg_of_conversation_messages = 0
        conversation_count = len({message.conversation_id for message in messages})
        if conversation_count != 0:
            avg_of_conversation_messages = total_messages / conversation_count

        # 4.Token输出速度，使用总token数/总耗时，涉及除法要做/0判断
        token_output_rate = 0
        latency_sum = sum(message.latency for message in messages)
        if latency_sum != 0:
            token_output_rate = sum(message.total_token_count for message in messages) / latency_sum

        # 5.计算费用消耗，使用总花费进行求和
        cost_consumption = sum(message.total_price for message in messages)

        # 6.返回数据，并且对于小数型数据，如果数值过小，需要转换成float，避免Python使用科学计数法进行展示
        return {
            "total_messages": total_messages,
            "active_accounts": active_accounts,
            "avg_of_conversation_messages": float(avg_of_conversation_messages),
            "token_output_rate": float(token_output_rate),
            "cost_consumption": float(cost_consumption),
        }

    @classmethod
    def calculate_pop_by_overview_indicators(
            cls, current_data: dict[str, Any], previous_data: dict[str, Any]
    ) -> dict[str, Any]:
        """根据传递的当前数据+相邻时期的数据计算环比增长"""
        # 1.初始化环比字典
        pop = {}

        # 2.定义需要计算环比的字段
        fields = [
            "total_messages", "active_accounts", "avg_of_conversation_messages",
            "token_output_rate", "cost_consumption",
        ]

        # 3.循环遍历字段计算环比增长
        for field in fields:
            current_value = current_data.get(field)
            previous_value = previous_data.get(field)

            # 4.确保上一时期的数据不为0，避免/0错误
            if previous_value != 0:
                pop[field] = float((current_value - previous_value) / previous_value)
            else:
                # 5.如果前一期数据为0，则直接设置环比为0
                pop[field] = 0

        return pop

    @classmethod
    def calculate_trend_by_messages(
            cls, end_at: datetime, days_ago: int, messages: list[Message],
    ) -> dict[str, Any]:
        """根据传递的结束时间、回退天数、消息列表计算对应指标的趋势数据"""
        # 1.重新计算end_at为午夜时间
        end_at = datetime.combine(end_at, datetime.min.time())

        # 2.定义初始数据
        total_messages_trend = {"x_axis": [], "y_axis": []}
        active_accounts_trend = {"x_axis": [], "y_axis": []}
        avg_of_conversation_messages_trend = {"x_axis": [], "y_axis": []}
        cost_consumption_trend = {"x_axis": [], "y_axis": []}

        # 3.循环遍历计算并提取数据
        for day in range(days_ago):
            # 4.计算起点和终点时间
            trend_start_at = end_at - timedelta(days_ago - day)
            trend_end_at = end_at - timedelta(days_ago - day - 1)

            # 5.计算全部会话趋势
            total_messages_trend_y_axis = len([
                message for message in messages
                if trend_start_at <= message.created_at < trend_end_at
            ])
            total_messages_trend["x_axis"].append(int(trend_start_at.timestamp()))
            total_messages_trend["y_axis"].append(total_messages_trend_y_axis)

            # 6.计算激活用户趋势数据
            active_accounts_trend_y_axis = len({
                message.created_by for message in messages
                if trend_start_at <= message.created_at < trend_end_at
            })
            active_accounts_trend["x_axis"].append(int(trend_start_at.timestamp()))
            active_accounts_trend["y_axis"].append(active_accounts_trend_y_axis)

            # 7.计算平均会话互动趋势
            avg_of_conversation_messages_trend_y_axis = 0
            conversation_count = len({
                message.conversation_id for message in messages
                if trend_start_at <= message.created_at < trend_end_at
            })
            if conversation_count != 0:
                avg_of_conversation_messages_trend_y_axis = total_messages_trend_y_axis / conversation_count
            avg_of_conversation_messages_trend["x_axis"].append(int(trend_start_at.timestamp()))
            avg_of_conversation_messages_trend["y_axis"].append(float(avg_of_conversation_messages_trend_y_axis))

            # 8.计算费用消耗趋势
            cost_consumption_trend_y_axis = sum(
                message.total_price for message in messages
                if trend_start_at <= message.created_at < trend_end_at
            )
            cost_consumption_trend["x_axis"].append(int(trend_start_at.timestamp()))
            cost_consumption_trend["y_axis"].append(float(cost_consumption_trend_y_axis))

        # 9.返回数据
        return {
            "total_messages_trend": total_messages_trend,
            "active_accounts_trend": active_accounts_trend,
            "avg_of_conversation_messages_trend": avg_of_conversation_messages_trend,
            "cost_consumption_trend": cost_consumption_trend,
        }
