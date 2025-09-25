import os
import json
from .gsm8k import GSM8KDataset
import warnings
import re
from .parser_helper import remove_boxed, last_boxed_only_string
from datasets import load_dataset
CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step and provide your final expression inside \\boxed"
    + "{}"
    + " tags without including an equals sign or the target number. For example: \\boxed{a + b * c}"
#     + """Respond in the following format:
# <reasoning>
# Your reasoning here
# </reasoning>
# <answer>
# \\boxed{...}
# </answer>"""
)

class CTDDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        train_data_path=None,
        num_examples=0,
        add_reasoning=True,
        system_prompt=CTD_SYSTEM_PROMPT,
        subsample=256,
    ):
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for Countdown dataset. Overriding num_examples to 0.")
        super().__init__(
            tokenizer,
            0,
            add_reasoning,
            system_prompt,
            subsample,
        )  # num_examples = always 0
        if train_data_path is not None:
            self.dataset = load_dataset(train_data_path, split="train", cache_dir="cache")

    def validate(self, generated_text, solution, question=None):
        numbers, target = solution
        try:
            equation = remove_boxed(last_boxed_only_string(generated_text))
        except:
            # Try to extract from answer tags
            answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
            if answer_match:
                equation = answer_match.group(1).strip()
            else:
                equation = generated_text
        # Replace LaTeX operators with Python operators
        equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")

        # Check for equation with equals sign and extract only the expression part
        equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
        if equation_match:
            equation = equation_match.group(1).strip()

        is_correct = False
        result = None

        def validate_equation(equation_str, available_numbers):
            """Validate that equation only uses available numbers and each number once."""
            try:
                numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
                available_numbers = sorted(available_numbers)
                numbers_in_eq = sorted(numbers_in_eq)
                return numbers_in_eq == available_numbers
            except:
                return False

        def evaluate_equation(equation_str):
            """Safely evaluate the arithmetic equation."""
            try:
                allowed_pattern = r"^[\d+\-*/().\s]+$"
                if not re.match(allowed_pattern, equation_str):
                    raise ValueError("Invalid characters in equation.")
                result = eval(equation_str.strip(), {"__builtins__": None}, {})
                return result
            except Exception:
                return float("Inf")

        # Validate and evaluate the equation
        is_valid = validate_equation(equation, numbers)
        if is_valid:
            result = evaluate_equation(equation)
            if target is not None and abs(result - target) < 1e-5:
                return True
        return False
    def load_test_dataset(self):
        self.dataset = []
        cur_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{cur_path}/dataset/countdown_cd3_test.jsonl", "r") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        print(len(self.dataset), "examples loaded")

    def __getitem__(self, idx):
        target = int(self.dataset[self.subsample[idx].item()]["output"])
        numbers_str = self.dataset[self.subsample[idx].item()]["input"]
        numbers = [int(num) for num in numbers_str.split(",")]
        question = f"Numbers: {numbers}\nTarget: {target}"
        prompt = self.create_prompt(question)
        return prompt, question, (numbers, target)