from metrics import (
    eval_20Minuten, eval_CStance, eval_FOMC, eval_MeetingBank, eval_numGLUE_cm, eval_numGLUE_ds, eval_Py150, eval_ScienceQA
)
import json
from pathlib import Path
eval_fn = {
    "20Minuten": eval_20Minuten,
    "C-STANCE": eval_CStance,
    "FOMC": eval_FOMC,
    "MeetingBank": eval_MeetingBank,
    "NumGLUE-cm": eval_numGLUE_cm,
    "NumGLUE-ds": eval_numGLUE_ds,
    "Py150": eval_Py150,
    "ScienceQA": eval_ScienceQA
}

def main(json_path):
    json_path = Path(json_path)
    result = {}
    for file in json_path.iterdir():
        if not file.is_file():
            continue
        if 'train_' in str(file.parent):
            trained = str(file.parent).split('train_')[1]
        else:
            raise ValueError(f"'train_' not in {file.parent}") 
        if trained not in result:
            result[trained] = {}
        
        eval_func = eval_fn[file.name[:-6]]
        task_results = file.open('r', encoding='utf-8').readlines()
        task_results = [json.loads(d) for d in task_results]
        response = [d['response'] for d in task_results]
        answer = [d['answer'] for d in task_results]
        
        performance = eval_func(response=response, answers=answer)
        result[trained][file.name[:-6]] = performance
    
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    import fire
    fire.Fire(main)