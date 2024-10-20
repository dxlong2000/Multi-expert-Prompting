import evaluate 
import json 

model_name = "chatgpt" # or `mistral`
regard = evaluate.load('Multi-expert-Prompting/evaluation/evaluate/measurements/regard', 'compare')

male_continuations = []
with open(f"Multi-expert-Prompting/evaluation/results/BOLD_male_MEP_3experts_{model_name}_output.json") as file:
  data = json.load(file)
  for dt in data: male_continuations.append(dt["final_answer"])

female_continuations = []
with open(f"Multi-expert-Prompting/evaluation/results/BOLD_female_MEP_3experts_{model_name}_output.json") as file:
  data = json.load(file)
  for dt in data: female_continuations.append(dt["final_answer"])

scores = regard.compute(data = male_continuations, references= female_continuations)
final_score = 0
for score in scores["regard_difference"]:
  final_score += abs(scores["regard_difference"][score])
print(final_score)
