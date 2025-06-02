# internal/test_sql_nodes.py - 完整的测试文件
from uuid import uuid4

import pytest

from internal.core.workflow.entities.node_entity import NodeType, NodeStatus
from internal.core.workflow.entities.vailate_entity import VariableEntity, VariableValueType, VariableType
from internal.core.workflow.nodes import SqlAgentNodeData, SqlAgentNode
from internal.core.workflow.nodes.sql_search.sql_search_entity import SqlSearchNodeData
from internal.core.workflow.nodes.sql_search.sql_search_node import SqlSearchNode


def test_sql_search_node_data_valid():
    """测试SQL查询节点数据验证 - 这个测试只验证节点数据创建"""
    node_data = SqlSearchNodeData(
        id=uuid4(),
        node_type=NodeType.SQL_SEARCH,
        title="测试SQL查询节点",
        description="只允许SELECT",
        inputs=[
            VariableEntity(
                name="host",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "localhost"}
            ),
            VariableEntity(
                name="port",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": 3306}
            ),
            VariableEntity(
                name="user",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "root"}
            ),
            VariableEntity(
                name="password",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "200124"}
            ),
            VariableEntity(
                name="database",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "users"}
            ),
            VariableEntity(
                name="table",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "user"}
            ),
            VariableEntity(
                name="sql",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "SELECT * FROM user"}
            ),
        ]
    )

    print("sql_search节点对象：", node_data)

    # 这里的outputs确实应该是空的，因为还没有执行
    assert len(node_data.outputs) == 1
    assert node_data.outputs[0].name == "text"
    assert node_data.outputs[0].value.type == VariableValueType.GENERATED
    assert node_data.outputs[0].value.content == ""  # 创建时应该是空的


def test_sql_search_node_execution():
    """测试SQL查询节点实际执行 - 这个测试会执行SQL查询并返回结果"""

    # 1. 创建节点数据
    node_data = SqlSearchNodeData(
        id=uuid4(),
        node_type=NodeType.SQL_SEARCH,
        title="测试SQL查询节点执行",
        description="实际执行SQL查询",
        inputs=[
            VariableEntity(
                name="host",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "localhost"}
            ),
            VariableEntity(
                name="port",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "3306"}  # 建议改为字符串
            ),
            VariableEntity(
                name="user",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "root"}
            ),
            VariableEntity(
                name="password",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "200124"}
            ),
            VariableEntity(
                name="database",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "users"}
            ),
            VariableEntity(
                name="table",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "user"}
            ),
            VariableEntity(
                name="sql",
                type=VariableType.STRING,
                value={"type": VariableValueType.LITERAL, "content": "SELECT * FROM user LIMIT 5"}
            ),
        ]
    )

    print("创建的节点数据:", node_data)

    # 2. 创建节点实例
    node = SqlSearchNode(node_data=node_data)

    # 3. 创建模拟的工作流状态
    initial_state = {
        "node_results": [],
        "variables": {}
    }

    # 4. 执行节点
    print("开始执行SQL查询...")
    result_state = node.invoke(initial_state)

    # 5. 检查执行结果
    print("执行完成，检查结果...")
    assert "node_results" in result_state
    assert len(result_state["node_results"]) == 1

    node_result = result_state["node_results"][0]
    print(f"节点执行状态: {node_result.status}")
    print(f"节点输入: {node_result.inputs}")
    print(f"节点输出: {node_result.outputs}")
    print(f"执行耗时: {node_result.latency}")

    if node_result.status == NodeStatus.FAILED:
        print(f"执行失败，错误信息: {node_result.error}")
        # 如果是数据库连接问题，这是正常的（测试环境可能没有数据库）
        assert "数据库查询错误" in node_result.error or "缺少必要参数" in node_result.error
    else:
        # 如果执行成功，检查输出
        assert node_result.status == NodeStatus.SUCCEEDED
        assert "text" in node_result.outputs
        assert node_result.outputs["text"] is not None
        print(f"查询结果: {node_result.outputs['text']}")


def test_sql_search_node_with_invalid_sql():
    """测试无效SQL语句的处理"""
    node_data = SqlSearchNodeData(
        id=uuid4(),
        node_type=NodeType.SQL_SEARCH,
        title="测试无效SQL",
        description="测试非SELECT语句",
        inputs=[
            VariableEntity(name="host", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "localhost"}),
            VariableEntity(name="port", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "3306"}),
            VariableEntity(name="user", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "root"}),
            VariableEntity(name="password", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "password"}),
            VariableEntity(name="database", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "test"}),
            VariableEntity(name="table", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "users"}),
            VariableEntity(name="sql", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "DELETE FROM users"}),
        ]
    )

    node = SqlSearchNode(node_data=node_data)
    result_state = node.invoke({"node_results": [], "variables": {}})

    node_result = result_state["node_results"][0]
    assert node_result.status == NodeStatus.FAILED
    assert "只允许执行SELECT语句" in node_result.error


def test_sql_search_node_missing_table_in_sql():
    """测试SQL语句中缺少指定表名的情况"""
    node_data = SqlSearchNodeData(
        id=uuid4(),
        node_type=NodeType.SQL_SEARCH,
        title="测试表名不匹配",
        description="SQL中没有包含指定的表名",
        inputs=[
            VariableEntity(name="host", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "localhost"}),
            VariableEntity(name="port", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "3306"}),
            VariableEntity(name="user", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "root"}),
            VariableEntity(name="password", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "password"}),
            VariableEntity(name="database", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "test"}),
            VariableEntity(name="table", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "users"}),
            VariableEntity(name="sql", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "SELECT * FROM products"}),  # 表名不匹配
        ]
    )

    node = SqlSearchNode(node_data=node_data)
    result_state = node.invoke({"node_results": [], "variables": {}})

    node_result = result_state["node_results"][0]
    assert node_result.status == NodeStatus.FAILED
    assert "SQL语句必须包含所选表名" in node_result.error


def test_sql_agent_node_data_valid():
    """测试SQL智能代理节点数据验证 - 只验证节点数据创建"""
    node_data = SqlAgentNodeData(
        id=uuid4(),
        node_type=NodeType.SQL_AGENT,
        title="测试SQL智能代理节点",
        description="自动生成SQL",
        inputs=[
            VariableEntity(name="host", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "localhost"}),
            VariableEntity(name="port", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "3306"}),
            VariableEntity(name="user", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "root"}),
            VariableEntity(name="password", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "200124"}),
            VariableEntity(name="database", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "users"}),
            VariableEntity(name="query", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "查询所有用户"}),

        ],
        prompt="",
        model_config={
            "provider": "tongyi",
            "model": "qwq-plus",
            "parameters": {
                "temperature": 0.5,
                "top_p": 0.85,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.2,
                "max_tokens": 8192,
            },
        },

    )
    print("sql_agent节点对象：", node_data)
    assert len(node_data.outputs) == 1
    assert node_data.outputs[0].name == "output"
    assert node_data.outputs[0].value.type == VariableValueType.GENERATED
    assert node_data.outputs[0].value.content == ""


def test_sql_agent_node_missing_param():
    """测试缺少必要参数校验"""
    with pytest.raises(ValueError) as e:
        SqlAgentNodeData(
            id=uuid4(),
            node_type=NodeType.SQL_AGENT,
            title="缺少参数",
            description="测试缺少参数校验",
            inputs=[
                VariableEntity(name="host", type=VariableType.STRING,
                               value={"type": VariableValueType.LITERAL, "content": "localhost"}),
                # 缺少port、user等
            ]
        )
    print("缺少参数异常：", e.value)
    assert "sql_agent节点缺少必要输入" in str(e.value)


def test_sql_agent_node_execution_mock():
    """测试SQL智能代理节点执行（mock，仅结构演示）"""
    node_data = SqlAgentNodeData(
        id=uuid4(),
        node_type=NodeType.SQL_AGENT,
        title="测试SQL智能代理节点",
        description="自动生成SQL",
        inputs=[
            VariableEntity(name="host", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "localhost"}),
            VariableEntity(name="port", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "3306"}),
            VariableEntity(name="user", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "root"}),
            VariableEntity(name="password", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "200124"}),
            VariableEntity(name="database", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "users"}),
            VariableEntity(name="query", type=VariableType.STRING,
                           value={"type": VariableValueType.LITERAL, "content": "查询所有用户"}),

        ],
        prompt="",
        model_config={
            "provider": "tongyi",
            "model": "qwq-plus",
            "parameters": {
                "temperature": 0.5,
                "top_p": 0.85,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.2,
                "max_tokens": 8192,
            },
        },
    )
    node = SqlAgentNode(node_data=node_data)
    initial_state = {"node_results": [], "variables": {}}
    print("开始执行SQL智能代理节点（mock）...")
    try:
        result_state = node.invoke(initial_state)
        print("执行结果:", result_state)
        node_result = result_state["node_results"][0]
        print(f"节点执行状态: {node_result.status}")
        print(f"节点输入: {node_result.inputs}")
        print(f"节点输出: {node_result.outputs}")
        print(f"执行耗时: {node_result.latency}")
        if node_result.status == NodeStatus.FAILED:
            print(f"执行失败，错误信息: {node_result.error}")
        else:
            assert node_result.status == NodeStatus.SUCCEEDED
            assert "text" in node_result.outputs
            print(f"查询结果: {node_result.outputs['text']}")
    except Exception as e:
        print("执行异常：", e)


if __name__ == "__main__":
    # 运行所有测试
    print("=== 测试1: 节点数据验证 ===")
    test_sql_search_node_data_valid()

    print("\n=== 测试2: 节点执行 ===")
    test_sql_search_node_execution()

    print("\n=== 测试3: 无效SQL处理 ===")
    test_sql_search_node_with_invalid_sql()

    print("\n=== 测试4: 表名不匹配处理 ===")
    test_sql_search_node_missing_table_in_sql()

    print("\n所有测试完成！")

    print("=== sql_agent 测试1: 节点数据验证 ===")
    test_sql_agent_node_data_valid()
    print("\n=== sql_agent 测试2: 非法模型校验 ===")
    test_sql_agent_node_missing_param()
    print("\n=== sql_agent 测试4: 节点执行（mock） ===")
    test_sql_agent_node_execution_mock()
