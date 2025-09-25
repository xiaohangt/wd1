import re
import pandas as pd
from .gsm8k import GSM8KDataset
from datasets import Dataset as HFDataset
import os
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle and output the final answer within \\boxed{}. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4 with no spaces or separators, representing your final solved grid.
"""


class SudokuDataset(GSM8KDataset):

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=SUDOKU_SYSTEM_PROMPT,
        subsample=256,
    ):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.sudoku_file_path = f"{cur_path}/dataset/4x4_test_sudoku.csv"
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)

    def load_test_dataset(self):
        """Load the Sudoku dataset from the CSV file."""
        df = pd.read_csv(self.sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
        # Convert pandas DataFrame to HuggingFace Dataset using from_pandas
        self.dataset = HFDataset.from_pandas(df)
        print("Loaded Testing Sudoku dataset with {} examples".format(len(self.dataset)))

    def format_sudoku_grid(self, sudoku_str):
        """Simplified function to format a sudoku string."""
        # Simply pass through the raw string as requested
        return sudoku_str

    def validate(self, solution_str, ground_truth=None, question=None):
        if len(question) == 16:
            puzzle_str = question
        else:
            match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
            if match:
                puzzle_str = match.group(1)
        empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
        empty_cells = len(empty_indices)
        if solution_str is None or len(solution_str) == 0:
            return 0, empty_cells, 0.0
        answer_parsed = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        if len(answer_parsed) > 0:
            solution_str = str(answer_parsed[-1])
        # Handle length issues
        if len(solution_str) < 16:
            # Pad with zeros if too short
            solution_str = solution_str + "0" * (16 - len(solution_str))
        elif len(solution_str) > 16:
            # Truncate if too long
            solution_str = solution_str[:16]

        assert len(puzzle_str) == 16
        # Count correct cells among originally empty cells
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        accuracy = correct_cells / empty_cells
        return correct_cells, empty_cells, accuracy

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        puzzle = self.dataset[self.subsample[idx].item()]["Puzzle"]
        solution = self.dataset[self.subsample[idx].item()]["Solution"]

        # Modified question format to reference the examples in the system prompt
        question = f"Solve the following Sudoku puzzle: {puzzle}\n"

        assert len(puzzle) == 16, f"Invalid puzzle length: {len(puzzle)}"

        prompt = self.create_prompt(question)
        return prompt, question, solution