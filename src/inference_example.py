import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

# Set model path
model_path = "Video-R1/Video-R1-7B"

# Set video path and question
video_path = "./src/example_video/video1.mp4"
question = "Which move motion in the video lose the system energy?"

# Choose the question type from 'multiple choice', 'numerical', 'OCR', 'free-form', 'regression'
problem_type = 'free-form'

# Initialize the LLM
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    max_model_len=81920,
    gpu_memory_utilization=0.8,
    limit_mm_per_prompt={"video": 1, "image": 1},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=1024,
)

# Load processor and tokenizer
processor = AutoProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

# Prompt template
QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

# Question type 
TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
}

# Construct multimodal message
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 200704, # max pixels for each frame
                "nframes": 32 # max frame number
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[problem_type]
            },
        ],
    }
]

# Convert to prompt string
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process video input
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

# Prepare vllm input
llm_inputs = [{
    "prompt": prompt,
    "multi_modal_data": {"video": video_inputs[0]},
    "mm_processor_kwargs": {key: val[0] for key, val in video_kwargs.items()},
}]

# Run inference
outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
output_text = outputs[0].outputs[0].text

print(output_text)

