import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import cdist


def first_round_selection(
    embeddings, labels,
    num_mutants_per_round=16,
    first_round_method="pca", # 可选 "pca", "random", "explicit"
    explicit_variants=None, # 当first_round_method为"explicit"时，指定要选择的突变体列表
    random_seed=42
):
    """ active learning第一轮选择样本的方法，默认使用PCA方法选择与中心点距离最远的样本
    Args:
        embeddings (pd.DataFrame): DataFrame of embeddings, 所有样本的embedding，index为variant名称，embedding为具体的向量
        labels: pd.DataFrame, shape (num_samples, num_properties)，所有样本的标签
            example: variant	site	wildtype	mutation	mut_escape
                    N331V	331	N	V	0.003631406000000001
                    N331T	331	N	T	0.0038603619999999996
        num_mutants_per_round: int, 每轮选择的样本数量
        first_round_method: str, 第一轮选择样本的方法，默认使用PCA方法
        random_seed: int, 随机种子，仅当first_round_method为"random"时使用
    Returns:
        selected_variants: pd.DataFrame, shape (num_mutants_per_round, num_properties)，第一轮选择的样本的标签
        updated_labels0: pd.DataFrame, shape (num_samples, num_properties+1)，所有样本的标签，新增一列"selected"，标记是否被选择
    """  
    print("Total samples number:", len(labels))
    # 数据表包含variants列，如果为WT，则不是我们需要的突变体，需要排除
    variants = labels.variant[labels.variant != "WT"]
    print("Total variants number:", len(variants))

    if first_round_method == "pca":
        pca = PCA(n_components=10) # 选择前10个主成分

        # pca策略中，我们需要舍弃embedding中包含WT的成分
        if "WT" in embeddings.index:
            embeddings_no_wt = embeddings.drop(index="WT")
        else:
            embeddings_no_wt = embeddings.copy()
        pca_result = pca.fit_transform(embeddings_no_wt)

        # 指定seed
        if random_seed is not None:
            np.random.seed(
                random_seed
            )  

        # 将所有突变体在空间中分成 N 个簇（Cluster），并找到每个簇中最具代表性的那个点（Medoid）。
        from sklearn.cluster import KMedoids
        clusters = KMedoids(n_clusters=num_mutants_per_round, random_state=0).fit(pca_result)
        clusters_medoids = clusters.medoid_indices_

        selected_variants = labels.loc[embeddings_no_wt.index[clusters_medoids]]

    elif first_round_method == "random":
        if random_seed is not None:
            np.random.seed(random_seed)
        random_mutants = np.random.choice(
            variants, size=num_mutants_per_round, replace=False
        )
        selected_variants = labels.loc[random_mutants]
    elif first_round_method == "explicit" and explicit_variants is not None:
        # 直接指定要选择的突变体列表
        selected_variants = explicit_variants
    else:
        # 啥也没有，直接返回空
        selected_variants = pd.DataFrame()
    
    # 更新标签表，新增一列"selected"，标记是否被选择
    updated_labels = labels.copy()
    updated_labels["selected"] = updated_labels.variant.isin(selected_variants.variant)

    return selected_variants, updated_labels

def iterative_update(
    train_set, test_set,
    embeddings_pd, labels_pd,
    measure_variable, # 我们需要学习的参数
    sort_order="descending",  # 我们不一定要让参数越来越大，还可以让参数越来越小，甚至是随意指定排序函数
    regression_model_type = "random_forest",
    top_n=None,
    final_round=10,
    experimental=False
):
    """ active learning的迭代更新过程
    active learning的迭代更新过程包括以下步骤：
    1. 数据对齐与切分，训练集为已经在实验室中获得结果的序列，测试集可能是已经获得结果的序列，也可能是没有获得结果的序列
    2. 训练回归模型，使用训练集的embedding作为输入，测量结果作为输出，训练一个回归模型
    3. 预测与选择，使用训练好的回归模型对测试集的embedding进行预测
    4. 排序与评估，根据预测结果对测试集进行排序，选择top_n个样本进行实验验证，并评估模型的性能
    parameters:
        train_set: pd.DataFrame, shape (num_train_samples, num_properties)，训练集的标签，包含variant列和measure_variable列
        test_set: pd.DataFrame, shape (num_test_samples, num_properties)，测试集的标签，包含variant列和measure_variable列（如果有的话）
        embeddings_pd: pd.DataFrame, index + embedding ，所有样本的embedding，index为variant名称，embedding为具体的向量
        labels_pd: pd.DataFrame, shape (num_samples, num_properties)，所有样本的标签，包含variant列和measure_variable列
        measure_variable: str, 测量结果所在的列名，例如"mut_escape"
        regression_model_type: str, 回归模型的类型，支持"linear_regression", "random_forest", "xgboost", "knn", "gaussian_process", "mlp"
        top_n: int, 每轮选择的样本数量，如果为None，则选择所有样本
        final_round: int, 最终轮数，即迭代更新的总轮数
        experimental: bool, 是否进行实验验证，如果为True，则每轮选择的样本会被送去实验室验证，并获得真实测量结果；如果为False，则直接使用预测结果进行评估
    """
    