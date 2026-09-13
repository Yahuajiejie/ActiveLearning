""" filter.py  从当前目录下的fpbase_data/json文件中筛选出符合条件的数据，构造出适合esmc训练的数据集
需要调用的包：
- os: 用于文件和目录操作
- json: 用于处理JSON数据
- pandas: 用于数据处理和分析
筛选的条件：
- 基本条件：存在蛋白徐磊和荧光寿命信息
- 成为优质数据的附加条件：pubmed等数据库上可溯源的蛋白质
"""
import os
import json
import pandas as pd

def load_json_files(directory):
    """加载指定目录下的所有JSON文件"""
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                try:
                    json_data = json.load(file)
                    data.append(json_data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filepath}")
    return data
"""
json example:
{
  "uuid": "4Y1D1",
  "name": "(n1)StayGold",
  "slug": "n1staygold",
  "seq": "MVSTGEELFTGVVPFKFQLKGTINGKSFTVEGEGEGNSHEGSHKGKYVCTSGKLPMSWAALGTSFGYGMKYYTKYPSGLKNWFHEVMPEGFTYDRHIQYKGDGSIHAKHQHFMKNGTYHNIVEFTGQDFKENSPVLTGDMNVSLPNEVQHIPRDDGVECPVTLLYPLLSDKSKCVEAHQNTICKPLHNQPAPDVPYHWIRKQYTQSKDDTEERDHICQSETLEAHL",
  "ipg_id": null,
  "genbank": null,
  "uniprot": null,
  "pdb": [],
  "agg": "d",
  "switch_type": "b",
  "states": [
    {
      "slug": "n1staygold_default",
      "name": "default",
      "ex_max": 496,
      "em_max": 505,
      "ext_coeff": null,
      "qy": null,
      "pka": null,
      "maturation": null,
      "lifetime": null,
      "brightness": null
    }
  ],
  "transitions": [],
  "doi": "10.1038/s41587-022-01278-2"
}

"""
def flatten_record(record):
    """
    提取 record 中的所有非嵌套字段
    （即非 list 和 dict 的字段）
    """
    base = {}
    for k, v in record.items():
        if not isinstance(v, (list, dict)):
            base[k] = v
    return base


def filter_data(data, filter_lifetime = True):
    filtered_data = []
    high_quality_data = []
    flat_states = []

    for record in data:
        has_lifetime_info = False
        states = record.get('states', [])

        # 动态提取 record 非嵌套字段
        base_fields = flatten_record(record)

        flat_states = {}
        for state in states:
            if state.get('lifetime') is not None:
                has_lifetime_info = True

            # 合并：base（动态 record 字段） + state（动态）
            flat_states = {**flat_states, **state}

        final = {**base_fields,**flat_states}
        # 保留有 lifetime 的记录
        if filter_lifetime:
            if has_lifetime_info:
                filtered_data.append(final)
    
                # 动态判断有高质量信息
                if any(record.get(field) for field in ['pubmed','doi','genbank','uniprot']):
                    high_quality_data.append(final)
        else:
            filtered_data.append(final)
    
            # 动态判断有高质量信息
            if any(record.get(field) for field in ['pubmed','doi','genbank','uniprot']):
                high_quality_data.append(final)
    return filtered_data, high_quality_data

def save_filtered_data(filtered_data, output_filepath):
    """将筛选后的数据保存为适合训练的数据集，尽量是pandas的DataFrame格式"""
    df = pd.DataFrame(filtered_data)
    df.to_csv(output_filepath, index=False) # 保存为CSV文件


if __name__ == "__main__":
    input_directory = '/gpfs/share/home/2300012116/lifespanResearch/fpbase_data/json'
    output_filepath = '/gpfs/share/home/2300012116/lifespanResearch/original_data.csv'
    
    # 加载数据
    data = load_json_files(input_directory)
    
    # 筛选数据
    filtered_data, high_quality_indices = filter_data(data,filter_lifetime=False)
    
    # 保存筛选后的数据
    save_filtered_data(filtered_data, output_filepath)
    
    print(f"Total records loaded: {len(data)}")
    print(f"Records after filtering: {len(filtered_data)}")
    print(f"High quality records count: {len(high_quality_indices)}")