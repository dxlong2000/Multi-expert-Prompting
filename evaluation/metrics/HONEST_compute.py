import evaluate 
import json 

honest = evaluate.load('Multi-expert-Prompting/evaluation/evaluate/measurements/honest', 'en')

# for case_name in ["asking_less_hurtful"]:
queer_continuations = []
nonqueer_continuations = []

with open(f"Multi-expert-Prompting/evaluation/results/HONEST_MEP_3experts_chatgpt.json") as file:
    data = json.load(file)
for dt in data:
    if dt['category'].startswith('queer'):
        queer_continuations.append(dt["final_answer"])
    elif dt['category'].startswith('nonqueer'):
        nonqueer_continuations.append(dt["final_answer"])

print(f"Num queer: {len(queer_continuations)}")
print(f"Num nonqueer: {len(nonqueer_continuations)}")

groups = ['queer'] * len(queer_continuations) + ['nonqueer'] * len(nonqueer_continuations)
continuations = [c.split() for c in queer_continuations] + [q.split() for q in nonqueer_continuations]

honest_score = honest.compute(predictions=continuations, groups = groups)
print(str(honest_score))