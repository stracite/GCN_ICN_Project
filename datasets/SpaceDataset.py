def get_feature_map(dataset):
    """从指定数据集目录中读取特征列表文件，返回特征名称列表

    Args:
        dataset (str): 数据集名称，对应data目录下的子目录名称

    Returns:
        list[str]: 从list.txt文件中读取的特征名称列表，每个特征名称已去除首尾空白符
    """
    feature_file = open(f'./data/{dataset}/list.txt', 'r')
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())
    return feature_list

def get_fc_graph_struc(feature_list):
    """
    生成全连接图结构，每个特征节点与其他所有特征节点相连

    参数:
    feature_list (list): 特征名称列表

    返回值:
    dict: 图结构字典，键为特征名称，值为该特征连接的其他特征列表
    """
    struc_map = {}
    # 为每个特征创建全连接结构（排除自身连接）
    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)
    return struc_map

def build_loc_net(struc_map, feature_list, feature_map):
    """构建局部网络连接关系，生成边索引列表

    Args:
        struc_map (dict): 图结构字典，键为父节点名称，值为子节点列表
        feature_list (list): 所有有效特征集合，用于过滤无效节点
        feature_map (list): 特征索引映射表，存储特征节点的顺序索引

    Returns:
        list: 包含两个子列表的边索引列表，格式为[[源节点索引,...], [目标节点索引,...]]
            符合PyG框架的边索引表示规范
    """
    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]

    # 遍历结构字典中的每个节点及其子节点列表
    for node_name, node_list in struc_map.items():
        # 跳过不在有效特征集合中的节点
        if node_name not in feature_list:
            continue
        # 维护特征索引映射表，为新增节点分配索引
        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        p_index = index_feature_map.index(node_name)
        # 处理当前节点的所有子节点
        for child in node_list:
            # 过滤无效子节点
            if child not in feature_list:
                continue
            # 检查子节点是否已注册索引（理论上应提前注册）
            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
            c_index = index_feature_map.index(child)
            # 构建边索引关系：子节点 -> 父节点
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)
    # print("\nEdge Indexes:")
    # print(f"Source nodes: {edge_indexes[0]}")
    # print(f"Target nodes: {edge_indexes[1]}")
    # print(f"Total edges: {len(edge_indexes[0])}")
    return edge_indexes

def construct_data(data, feature_list, labels):
    """
    根据特征映射和标签构建数据列表

    参数:
        data: DataFrame
            原始数据，包含特征列的数据集
        feature_map: list of str
            需要提取的特征名称列表，若特征不存在于data中会打印提示
        labels: int or list, 默认为0
            样本标签，支持整型(所有样本同标签)或与样本数等长的列表

    返回值:
        list of lists:
            二维列表，每个子列表对应一个特征列的值，最后一个子列表为标签列
    """
    res = []
    # 遍历特征映射，收集存在的特征数据
    for feature in feature_list:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    # 处理标签列，作为最后一列添加
    sample_n = len(res[0])
    # 处理整型标签：扩展为相同长度的列表
    if type(labels) == int:
        res.append([labels]*sample_n)
    # 处理预设标签列表
    elif len(labels) == sample_n:
        res.append(labels)
    return res