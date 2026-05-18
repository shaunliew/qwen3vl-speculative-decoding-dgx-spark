"""
question_loader.py: Synthetic question loader for the Qwen3 speculative decoding benchmark.

Loads a bank of 20 synthetic text questions with known correct answers. Each question
is formatted as a multiple-choice prompt for Qwen3's chat template and returned as
(messages, expected_answer) tuples ready for the benchmark client.

The questions span math, science, history, logic, and language, diverse enough to
exercise different model reasoning paths, which is good for measuring speculation across
varied output distributions.

To swap in a different question set, implement load_formatted() returning
(messages, expected_answer) tuples and drop it in place of this module.
"""

# Synthetic question bank, 20 questions with known correct answers.
# Format: (question_text, [option_A, option_B, option_C, option_D], correct_letter)
QUESTIONS = [
    (
        "A train travels 120 km in 2 hours. What is its average speed in km/h?",
        ["40", "60", "80", "100"],
        "B",
    ),
    (
        "Which gas makes up the largest percentage of Earth's atmosphere?",
        ["Oxygen", "Nitrogen", "Carbon dioxide", "Argon"],
        "B",
    ),
    (
        "In which year did World War II end?",
        ["1943", "1944", "1945", "1946"],
        "C",
    ),
    (
        "What is 15% of 200?",
        ["20", "25", "30", "35"],
        "C",
    ),
    (
        "What is the capital city of France?",
        ["London", "Berlin", "Paris", "Rome"],
        "C",
    ),
    (
        "What is the chemical symbol for gold on the periodic table?",
        ["Gd", "Go", "Au", "Ag"],
        "C",
    ),
    (
        "All cats are mammals. All mammals are warm-blooded. "
        "Which conclusion MUST be true?",
        [
            "Some cats are not warm-blooded",
            "All warm-blooded animals are cats",
            "All cats are warm-blooded",
            "Mammals are not always warm-blooded",
        ],
        "C",
    ),
    (
        "What is the square root of 144?",
        ["10", "11", "12", "13"],
        "C",
    ),
    (
        "Who wrote the play 'Romeo and Juliet'?",
        ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
        "B",
    ),
    (
        "How many planets are currently recognised in our solar system?",
        ["7", "8", "9", "10"],
        "B",
    ),
    (
        "If x + 5 = 12, what is the value of x?",
        ["5", "6", "7", "8"],
        "C",
    ),
    (
        "Which river is traditionally listed as the longest in the world?",
        ["Amazon", "Mississippi", "Yangtze", "Nile"],
        "D",
    ),
    (
        "Approximately how fast does light travel in a vacuum?",
        ["3 × 10^6 m/s", "3 × 10^7 m/s", "3 × 10^8 m/s", "3 × 10^9 m/s"],
        "C",
    ),
    (
        "Who was the first President of the United States?",
        ["Abraham Lincoln", "Thomas Jefferson", "George Washington", "Benjamin Franklin"],
        "C",
    ),
    (
        "What is 7 × 8?",
        ["54", "56", "58", "62"],
        "B",
    ),
    (
        "What is the antonym (opposite in meaning) of the word 'ancient'?",
        ["Old", "Historic", "Modern", "Aged"],
        "C",
    ),
    (
        "Which organelle is known as the 'powerhouse of the cell'?",
        ["Nucleus", "Ribosome", "Mitochondria", "Golgi apparatus"],
        "C",
    ),
    (
        "It takes 5 machines 5 minutes to make 5 widgets. How long does it take "
        "100 machines to make 100 widgets?",
        ["1 minute", "5 minutes", "20 minutes", "100 minutes"],
        "B",
    ),
    (
        "A rectangle has length 8 cm and width 5 cm. What is its area?",
        ["13 cm²", "26 cm²", "40 cm²", "80 cm²"],
        "C",
    ),
    (
        "Which programming paradigm does Python primarily support?",
        [
            "Purely functional",
            "Purely object-oriented",
            "Multi-paradigm (procedural, OOP, functional)",
            "Assembly-level only",
        ],
        "C",
    ),
]


def format_sample(question_text, options, correct_letter):
    """
    Convert one synthetic question into (messages, expected_answer).

    messages, a list with one user turn, content is a plain string
    expected_answer, the correct letter (A/B/C/D)

    WHY ASK FOR LONG EXPLANATION:
    Speculative decoding only shows meaningful speedup over many decode steps.
    With 50-90 token outputs, there are too few speculation rounds (~200 total)
    to reliably measure the 1-2% differences between methods. With 200-400 token
    outputs, we get 1000+ speculation rounds and statistically meaningful
    acceptance rate estimates (plus or minus 2-3% confidence interval vs plus or
    minus 7% before). AngelSlim benchmarks Eagle3 at output_len=1024 for the
    same reason.
    Accuracy is still checked by extracting the first letter from the response.
    """
    # Build labelled choices: "A) 40\nB) 60\nC) 80\nD) 100"
    labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(
        f"{label}) {opt}"
        for label, opt in zip(labels, options)
    )

    prompt = (
        f"Question: {question_text}\n\n"
        f"{choices_text}\n\n"
        f"Answer with the letter (A, B, C, or D). Then walk through your complete reasoning step by step, explaining what information from the question led you to that answer, why the other options are incorrect, and any relevant background knowledge or formulas you applied. Aim for a thorough explanation of at least 4-6 sentences."
    )

    # Plain user message, no system prompt needed.
    # Thinking mode is disabled via enable_thinking=False in chat_template_kwargs
    # inside run_client.py (the official vLLM approach for Qwen3 reasoning models).
    messages = [{"role": "user", "content": prompt}]
    return messages, correct_letter


def load_formatted(max_samples=None):
    """
    Return all synthetic questions as (messages, expected_answer) tuples.

    max_samples: if set, only return the first N questions (useful for quick tests).
    """
    dataset = QUESTIONS if max_samples is None else QUESTIONS[:max_samples]
    print(f"Loaded {len(dataset)} synthetic questions")
    return [format_sample(q, opts, ans) for q, opts, ans in dataset]


if __name__ == "__main__":
    # Quick smoke test, run this to verify the loader works
    samples = load_formatted()
    messages, answer = samples[0]
    print(f"Expected answer: {answer}")
    print(f"\nPrompt:\n{messages[0]['content']}")
