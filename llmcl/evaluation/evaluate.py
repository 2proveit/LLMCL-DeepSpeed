from typing import Dict
from metrics import (
    eval_20Minuten, eval_CStance, eval_FOMC, eval_MeetingBank, eval_numGLUE_cm, eval_numGLUE_ds, eval_Py150, eval_ScienceQA
)
import json, re
import pandas as pd
import numpy as np
from pathlib import Path
eval_fn = {
    "C-STANCE": eval_CStance,
    "FOMC": eval_FOMC,
    "MeetingBank": eval_MeetingBank,
    "Py150": eval_Py150,
    "ScienceQA": eval_ScienceQA,
    "NumGLUE-cm": eval_numGLUE_cm,
    "NumGLUE-ds": eval_numGLUE_ds,
    "20Minuten": eval_20Minuten,
}

ZERO_SHOT_PERFORMANCE = {
    "C-STANCE": 0.5475,
    "FOMC": 0.5826612903225806,
    # "MeetingBank": 0.10632118161853012,
    # "Py150": 0.030245,
    "ScienceQA": 0.352,
    "NumGLUE-cm": 0.8518518518518519,
    "NumGLUE-ds": 0.6,
    "20Minuten": 0.10889139813684952,
}

def get_train_infer_dataset_name(path:Path, names:list):
    path = str(path)
    names.append('MTL')
    pattern = r'(' + '|'.join(re.escape(name) for name in names) + r')_infer_(' + '|'.join(re.escape(name) for name in names) + r')'
    matches = re.search(pattern, path, re.DOTALL)
    if matches:
        return matches.group(1), matches.group(2)
    else:
        raise ValueError((f"Can not match any of {names} for {path}"))

def get_method(path:Path, methods:list):
    path = str(path)
    pattern = r'\b(' + '|'.join(re.escape(mtd) for mtd in methods) + r')\b'
    match = re.search(pattern, path)
    if match:
        return match.group(0).strip()
    else:
        raise ValueError(f"Can not match any of {methods} for {path}")
    
def calculate_metrics(results: Dict[str, Dict[str, float]], incremental_order: list):
    overall_accs = []
    triangle_performance = []
    bwts = []
    fws = []
    cl_method = list(results.keys())[0]
    if cl_method == 'MTL':
        return {
            'acc': np.mean(list(results[cl_method].values()))
        }
    assert len(results) == len(incremental_order) and all(len(res) == len(incremental_order) for res in results.values())
    len_tasks = len(incremental_order)
    for i in range(len_tasks):
        train_name = incremental_order[i]
        for j in range(len_tasks):
            test_name = incremental_order[j]
            if i == j:
                triangle_performance.append(results[train_name][test_name])
                bwts.append(results[incremental_order[-1]][test_name] - results[train_name][test_name])
            if i > 0:
                fws.append(results[incremental_order[i-1]][test_name] - ZERO_SHOT_PERFORMANCE[test_name])
            if i == len_tasks - 1:
                overall_accs.append(results[train_name][test_name])
    assert bwts[-1] == 0
    print(f"accs: {overall_accs}, tri:{triangle_performance}, bwts: {bwts}, fws: {fws}")
    return {
        "acc": sum(overall_accs)/len(overall_accs),
        "bwt": sum(bwts)/(len(bwts)-1),
        "fw": sum(fws)/len(fws)
    }

def main(json_path):
    json_path = Path(json_path)
    result = {}
    for file in json_path.rglob('*'):
        if not file.is_file() or not str(file.name).endswith('jsonl'):
            continue
        trained, infered  = get_train_infer_dataset_name(file, names=list(eval_fn.keys()))
        if trained not in result:
            result[trained] = {}
        
        eval_func = eval_fn[infered]
        task_results = file.open('r', encoding='utf-8').readlines()
        task_results = [json.loads(d) for d in task_results]
        response = [d['response'] for d in task_results]
        answer = [d['answer'] for d in task_results]
        
        performance = eval_func(response=response, answers=answer)
        result[trained][infered] = performance
    over_all_performance = calculate_metrics(result, incremental_order=list(ZERO_SHOT_PERFORMANCE.keys()))
    result['over_all_performance'] = over_all_performance

    print(json.dumps(result, indent=4))
    with open(json_path.joinpath('result.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
    
    df = pd.DataFrame(index=list(ZERO_SHOT_PERFORMANCE.keys()), dtype=float)
    for name in list(ZERO_SHOT_PERFORMANCE.keys()):
        df[name] = [-1] * len(list(ZERO_SHOT_PERFORMANCE.keys()))
    for train, test_set in result.items():
        if train == 'over_all_performance':
            continue
        for test, val in test_set.items():
            df.loc[train, test] = val
    df = df.round(4)
    df.to_csv(json_path.joinpath('result.csv'), index=True)
    
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)