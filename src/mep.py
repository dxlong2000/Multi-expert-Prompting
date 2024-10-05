import json
from utils import get_llm_answer_with_retry

with open('src/prompts.json', 'r') as file:
    prompts = json.load(file)

def generate_role_format(num_experts):
    role_format = "{"
    for i in range(1, num_experts + 1):
        role_format += f"\"role {i}\": \"description {i}\""
        if i < num_experts:
            role_format += ", "
    role_format += "}"
    return role_format

def generate_answer_format(num_experts):
    answer_format = ""
    for i in range(1, num_experts + 1):
        answer_format += f"###\n{{{{ROLE_{i}}}}}: {{{{ROLE_{i}_ANSWER}}}}\n"
    return answer_format


def generate_answer_choice(num_experts):
    return "/".join([f"Answer {i+1}" for i in range(num_experts)]) + "/Combined answer"

class MEP:
    def __init__(self, args, prompt_key):
        self.args = args
        self.prompt_key = prompt_key
        self.prompts = prompts[prompt_key]
        self.api_token = args.api_token
        self.num_experts = args.num_experts
        self.max_retries = args.max_retries
        self.retry_delay = args.retry_delay
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.verbose = args.verbose

        self.role_format = generate_role_format(self.num_experts)
        self.answer_format = generate_answer_format(self.num_experts)
        self.answer_choices = generate_answer_choice(self.num_experts)

    def generate_roles(self, question, model, tokenizer, device):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                system_message = self.prompts["role_generation"]["system_message"]
                user_message = self.prompts["role_generation"]["user_message"].replace("{{QUESTION}}", question).replace("{{NUM_EXPERTS}}", str(self.num_experts)).replace("{{ROLE_FORMAT}}", self.role_format)
                if self.verbose:
                    print(f"\n\n>>>> System Message: {system_message}")
                    print(f">>> User Message: {user_message}")
                assistant_message = get_llm_answer_with_retry(
                    model, tokenizer, device,
                    system_message,
                    user_message,
                    self.args.model, self.api_token, self.max_retries, self.retry_delay, self.temperature, self.max_tokens
                )
                roles = eval(assistant_message.strip())
                assert isinstance(roles, dict)
                assert len(list(roles.keys())) == self.num_experts
                if self.verbose:
                    print(f">>> Assistant Message: {assistant_message}\n\n")
                return roles
            except Exception as e:
                print(f"=== Retry {retry_count + 1} ===")
                print(f"Error: {e}")
                retry_count += 1
                if retry_count == self.max_retries:
                    print("Max retries reached. Skipping to the next sample.")
                    return None

    def generate_expert_answers(self, question, roles, model, tokenizer, device):
        all_answers = []
        for role_index, (role, description) in enumerate(roles.items()):
            system_message = self.prompts["expert_answer_generation"]["system_message"]
            role_prompt = self.prompts["expert_answer_generation"]["user_message"].replace("{{ROLE}}", role).replace("{{ROLE_DESCRIPTION}}", description).replace("{{QUESTION}}", question)
            if self.verbose:
                print(f"\n\n>>> System Message: {system_message}")
                print(f">>> User Message: {role_prompt}")
            answer = get_llm_answer_with_retry(
                model, tokenizer, device,
                system_message,
                role_prompt,
                self.args.model, self.api_token, self.max_retries, self.retry_delay, self.temperature, self.max_tokens
            )
            if self.verbose:
                print(f">>> Answer: {answer}\n\n")
            all_answers.append([role, answer])
        return all_answers

    def merge_expert_answers(self, question, all_answers, model, tokenizer, device):
        system_message = self.prompts["merging_prompt"]["system_message"]
        merging_prompt = self.prompts["merging_prompt"]["user_message"].replace("{{QUESTION}}", question).replace("{{NUM_EXPERTS}}", str(self.num_experts))    
        merging_prompt = merging_prompt.replace(r"{{ANSWER_CHOICES}}", self.answer_choices)
        if self.verbose:
            print(f"\n\n>>> System Message: {system_message}")
            print(f">>> User Message: {merging_prompt}")

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                final_raw_answer = get_llm_answer_with_retry(
                    model, tokenizer, device,
                    system_message,
                    merging_prompt,
                    self.args.model, self.api_token, self.max_retries, self.retry_delay, self.temperature, self.num_experts*self.max_tokens
                )
                try:
                    final_answer = final_raw_answer.split("Final answer:")[1].strip()
                except:
                    final_answer = final_raw_answer.split("Final answer")[1].strip()
                if self.verbose:
                    print(f">>> Final Raw Answer: {final_raw_answer}")
                    print(f">>> Final Answer: {final_answer}\n\n")
                return final_raw_answer, final_answer
            except Exception as e:
                print(f"=== Aggregation Retry {retry_count + 1} ===")
                print(f"Error: {e}")
                retry_count += 1
                if retry_count == self.max_retries:
                    print("Max retries reached. break")
                    return None, None