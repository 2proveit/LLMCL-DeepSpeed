from typing import List
from rouge import Rouge
from fuzzywuzzy import fuzz
import re

def eval_CStance(response:List, answers:List):
    num_correct = 0
    for i, resp in enumerate(response):
        if resp[0] in ['A', 'B', 'C', 'D'] and resp[0] == answers[i]:
            num_correct += 1
        
    return num_correct / len(response)

def eval_FOMC(response:List, answers:List):
    num_correct = 0
    for i, resp in enumerate(response):
        if resp[0] in ['A', 'B', 'C', 'D'] and resp[0] == answers[i]:
            num_correct += 1
        
    return num_correct / len(response)

rouger = Rouge()
def eval_MeetingBank(response:List, answers:List):
    res = []
    for i,(resp, ans) in enumerate(zip(response, answers)):
        try:
            res.append(rouger.get_scores(hyps=resp, refs=ans)[0]['rouge-l']['p'])
        except:
            print(resp,'\n', ans)
    return sum(res) / len(res)



def postprocess(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code

def eval_Py150(response, answers:List):
    response = [postprocess(res) for res in response]
    answers = [postprocess(ans) for ans in answers]
    scores = 0
    for res, ans in zip(response, answers):
        if res == "" or ans == "":
            continue
        scores += fuzz.ratio(res, ans)
    avg_score = scores / len(response)
    return avg_score / 100



def eval_ScienceQA(response, answers):
    num_correct = 0
    for res, ans in zip(response, answers):
        if res[0] in ['A', 'B', 'C', 'D'] and ans[0] in ['A', 'B', 'C', 'D']:
            if res[0] == ans[0]:
                num_correct += 1
    
    return num_correct/len(response)


def eval_numGLUE_cm(response, answers):
    num_correct = 0
    for i,(res, ans) in enumerate(zip(response, answers)):
        matches = re.findall(r'\d+', res)
        try:
            if matches and (float(matches[-1]) - float(ans)) < 1e-3:
                num_correct += 1 
        except:
            print(f"Error with line:{i+1}, matches: {matches} and ans: {ans}")
        
    return num_correct/len(response)
        
def eval_numGLUE_ds(response, answers):
    num_correct = 0
    for res, ans in zip(response, answers):
        matches = re.findall(r'\d+\.?\d*', res)
        if matches and (float(matches[-1]) - float(ans)) < 1e-3:
            num_correct += 1 
    return num_correct/len(response)


def eval_20Minuten(response, answers):
    res = []
    for resp, ans in zip(response, answers):
        res.append(rouger.get_scores(hyps=resp, refs=ans)[0]['rouge-l']['p'])
    return sum(res) / len(res)