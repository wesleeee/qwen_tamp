from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/home/chuziyuan/zju/TAMP/checkpoints/VLM-R1-Qwen2.5VL-3B-OVD-0321"

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
)

image_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": "What is the text in the illustrate?"},
        ],
    },
]


# For video input, you can pass following values instead:
# "type": "video",
# "video": "<video URL>",
video_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
            {"type": "text", "text": "请用表格总结一下视频中的商品特点"},
            {
                "type": "video",
                "video": "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4",
                "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28
            }
        ]
    },
]

# Here we use video messages as a demonstration
messages = video_messages

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs
if video_inputs is not None:
    mm_data["video"] = video_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)