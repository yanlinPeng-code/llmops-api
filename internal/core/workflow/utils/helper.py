from typing_extensions import Any

from internal.core.workflow.entities.vailate_entity import VARIABLE_TYPE_MAP, VariableEntity, \
    VariableValueType, VARIABLE_TYPE_DEFAULT_VALUE_MAP
from internal.core.workflow.entities.workflow_entity import WorkFlowState


def extract_variables_from_state(variables: list[VariableEntity], state: WorkFlowState) -> dict[str, Any]:
    """从状态中提取变量映射值信息"""
    # 1.构建变量字典信息
    variables_dict = {}

    # 2.循环遍历输入变量实体
    for variable in variables:
        # 3.获取数据变量类型
        variable_type_cls = VARIABLE_TYPE_MAP.get(variable.type)

        # 4.判断数据是引用还是直接输入
        if variable.value.type == VariableValueType.LITERAL:
            variables_dict[variable.name] = variable_type_cls(variable.value.content)
        else:
            # 5.引用or生成数据类型，遍历节点获取数据
            for node_result in state["node_results"]:
                if node_result.node_data.id == variable.value.content.ref_node_id:
                    # 6.提取数据并完成数据强制转换
                    variables_dict[variable.name] = variable_type_cls(node_result.outputs.get(
                        variable.value.content.ref_var_name,
                        VARIABLE_TYPE_DEFAULT_VALUE_MAP.get(variable.type)
                    ))
    return variables_dict
