## 파인튜닝이란 ? 
수십억개의 파라미터와 방대한 데이터로 학습된 거대 언어 모델(LLM)을 **우리가 풀고 싶은 특정 문제나 도메인에 맞게 추가 학습하여 최적화** 하는 과정을 의미합니다. 즉 범용적인 지능을 가진 모델을 특정 작업의 "전문가"로 만드는 과정입니다.

## 파인튜닝 vs 프롬프트 엔지니어링
LLM의 성능을 특정 작업에 맞게 끌어올리는 또 다른 방법으로 **프롬프트 엔지니어링**이 있습니다. 둘 다 강력한 기법이지만 접근방식과 결과물에서 근본적인 차이가 있습니다. 

프롬프트엔지니어링은 모델 자체는 전혀 변경하지 않고, 모델에 전달하는 **입력(프롬프트)를 정교하게 설계하여 원하는 결과물을 유도**하는 방식을 사용합니다. Zero-shot, Few-shot learning, CoT(Chain-of-Thought)등의 기법을 활용해서 **모델의 잠재력을 최대한 이끌어내는**게 목표입니다.
  * 장점: 빠르고, 비용이 들지 않으며 즉각적인 테스트와 수정이 가능합니다. 
  * 한계
    * 특정 작업을 위해서 매번 수십 줄에 달하는 장황한 지시사항과 예스를 프롬프트에 포함해야 합니다. 이는 API 호출 비용(토큰 수)을 증가시키고, 모델의 컨텍스트창의 한계를 시험합니다.  
    * 일관성 부족: 모델의 상태에 따라 혹은 미묘한 표현 차이에 따라 프롬프트를 무시하거나 결과물이 불안정해 질 수 있습니다.
    * 지식의 한계: 모델이 원래 학습하지 않은 새로운 지식이나 스타일을 가르칠 수 없습니다. 
물론 지식의 한계의 경우에는 **검색 증강 생성(RAG)** 을 이용해서 보완할 수 있습니다.

파인튜닝은 모델의 가중치(Weight)를 직접 업데이트해서, **새로운 지식이나 행동양식을 모델의 내부구조에 영구히 각인** 시키는 과정입니다. 이에 따라서 프롬프트 엔지니어링에 비해서 특징적인 장점을 가지게 됩니다.

 * 간결하고 효율적인 프롬프팅: 모델이 이미 특정 작업을 수행하는 방법을 알고 있기 때문에, 매번 긴 지시사항을 반복할 필요가 없습니다. 이를 통해 API 비용과 응답시간을 모두 단축 시킬 수 있습니다.
 * 일관성과 신뢰성: 원하는 결과물의 형식, 스타일, 말투가 모델의 핵심 로직에 통합되어 있기 때문에, 언제나 **예측 가능하고 일관된 결과물을 안정적으로 생성** 할 수 있습니다.   

이렇듯 파인튜닝의 분명한 장점이 있기는 하지만 현실은 "프롬프트 엔지니어링 과 RAG"를 주로 사용하고 있다. 그 이유는 **모델의 크기에서 오는 막대한 비용과 시간** 때문입니다. 하지만 270M 정도라면, 빠른시간에 파인튜닝을 하고 그 결과를 확인 할 수 있을 겁니다. 

## 개발환경 준비
파인튜닝에 필요한 라이브러리를 설치합니다. 
```bash
# Python 가상 환경 생성 및 호라성화
python3 -m venv venv
source venv/bin/activate

# 필수 라이브러리 설치
pip install torch transformers datasets peft bitsandbytes accelerate trl
```
 * transformers: 허깅페이스 허브에 등록된 사전 학습 모델에 즉시 접근 할 수 있습니다. Gemma, Llama, BERT등 거의 동일한 방식으로 모델과 토크나이저를 로드하고 사용 할 수 있는 인터페이스를 ㅈ공합니다.
 * peft: LoRA, QLoRA 등 효율적인 파인튜닝을 위한 라이브러리
 * bitsandbytes: 모델 양좌(Quantization)를 위한 라이브러리
 * accelerate: PyTorch 모델 학습을 도와주는 라이브러리
 * trl: 보상 모델링 및 피드백을 통한 학습을 위한 라이브러리

## 데이터셋 준비
파인튜닝의 효과를 검증하기 위한 **한국어 데이터셋**을 준비하기로 했습니다. **Beomi/KoAlpaca-v1.1a** 데이터셋을 선택했습니다. 이 데이터셋은 스탠포드 대학의 Alpaca 데이터셋을 한국어로 번역하고 생성한 것으로, 다양한 종류의 지시(instruction)과 응답으로 구성되어 있어 모델의 성능을 테스트하기에 적합합니다.

모델 훈련을 위해서는 데이터셋을 **JSONL** 형태의 파일로 변환해야 합니다. 먼저 pandas 라이브러리를 설치합니다.
```
pip install pandas
``` 

데이터셋을 JSONL 으로 변경하는 코드를 만들었습니다. 
```python
import json
import pandas as pd

# 데이터셋 URL
url = "https://raw.githubusercontent.com/Beomi/KoAlpaca/main/ko_alpaca_data.json"

print("KoAlpaca 데이터셋을 다운로드합니다...")
# 데이터셋 불러오기
df = pd.read_json(url)

print("train_dataset.jsonl 파일 형식으로 변환합니다...")
# JSONL 형식으로 변환하여 저장
with open("train_dataset.jsonl", "w", encoding="utf-8") as f:
    for index, row in df.iterrows():
        json_obj = {
            "instruction": row["instruction"],
            "input": row.get("input", ""),  # input 필드가 없는 경우를 대비
            "output": row["output"]
        }
        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print(f"완료! train_dataset.jsonl 파일이 생성되었으며, 총 {len(df)}개의 데이터가 포함되었습니다.")
```
코드를 실행하면 아래와 같은 JSONL 포맷의 훈련 데이터셋이 만들어집니다.
```
{"instruction": "가족 사진 촬영에 대한 좋은 아이디어 제안하기.", "input": "", "output": "가족 사진 촬영 아이디어를 공유하세요.\n\n- 가족 셀카 제작 경험에서, 가족 셀카에 효과적인 위치, 포즈 및 액세서리를 제안할 수 있습니다. 아침이나 일몰 등 시간, 배경 그리고 옷 착장에 대해 고려할 경우 여러 아이디어를 제공할 수 있습니다."}
{"instruction": "환불 문제에 대한 고객의 이메일에 대한 응답을 작성합니다.", "input": "회사]에게,\n\n최근에 스토어에서 제품을 구매했는데 품질에 실망했습니다. 환불을 요청하고 싶습니다.", "output": "환불 요청 이메일에 대한 답변 작성\n\n- 알림 해 주셔서 감사합니다. 환불 절차를 위해, [회사 이름] 홈페이지를 방문하시거나 [고객 지원 이메일]로 연락해주세요. 빠르게 처리되도록 최선을 다하겠습니다."}
{"instruction": "주어진 뉴스 기사를 한 문장으로 요약하세요.", "input": "최근 예산안 발표에서 총리는 중소기업 직원을 돕기 위해 새로운 고용 수당을 도입하겠다고 약속했습니다.", "output": "뉴스 기사 요약하기\n\n- 총리가 중소기업 직원 지원을 위 해 고용 수당 도입"}
```
instruction, input, output 은 언어 모델을 특정 작업에 맞게 파인튜닝할 때 사용되는 매우 일반적인 데이터 구조 입니다. 각각의 의미는 아래와 같습니다.
 * instruction(지시): 모델에게 무엇을 해야 할지 알려주는 명령 또는 질문입니다. 파인튜닝의 목표는 모델이 이 instruction을 잘 이해하고 따르게 하는데 있습니다.
 * input: instruction을 수행하는 데 필요한 추가적인 정보나 맥락입니다. 모든 instruction에 input이 필요한 건 아니며, 필요 없다면 비워둡니다. 
 * output: 주어진 instruction에 대해서 input이 주어졌을 때 모델이 생성해야 할 **가장 이상적인 정답** 입니다. 모델은 파인튜닝 과정에서 이 output 과 자신의 예측을 비교하여 학습합니다.

## gemma3-270m 모델 파인튜닝 
이제 준비된 데이터셋을 이용해서 gemma3 270m 모델을 파인튜닝해보겠습니다.
```python
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# 1. 모델 및 토크나이저 설정
model_id = "google/gemma-3-270m"  # 사용할 모델 ID
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

# 2. LoRA 설정
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 3. 데이터셋 로드 및 포맷팅
def formatting_prompts_func(example):
    # Process a single example
    text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}
"""
    return {"text": text}

dataset = load_dataset("json", data_files="train_dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func)

# 4. 트레이너 설정 및 학습
training_args = SFTConfig(
    output_dir="./gemma3-270m-finetuned-adapter",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    max_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

# 학습 시작
trainer.train()

# 학습된 어댑터 저장
trainer.save_model("./gemma3-270m-finetuned-adapter")
```

코드를 분석해보겠습니다.
```python
model_id = "google/gemma-3-270m"  # 사용할 모델 ID
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
```
 * 모델 선택: 파인튜닝의 기반이 될 사전 학습 모델로 **google/gemma-3-270m**을 선택했습니다.
 * QLoRA 설정: 모델을 4비트 양자화하여 로드합니다. 이렇게 하면 모델이 차지하는 메모리가 크게 줄어들어, 고사양 GPU 없이도 파인튜닝을 진행 할 수 있습니다.
 * 모델 및 토크나이저 로드: 양자화 구성을 적용하여 모델과 토크나이저를 불러옵니다. device_map-{"",0}은 사용가능한 첫번째 GPU를 사용하라는 의미입니다.

다음 LoRA 설정을 합니다.
```python
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```
 * LoRA(Low-Rank Adaptation): 모델의 모든 파라미터를 학습시키는 대신, **어댑터(adapter)** 라고 불리는 훨씬 작은 규모의 파라미터만 추가하여 학습하는 방법입니다. 이를 더 적은 컴퓨팅 리소스로 더 빠르게 학습할 수 있습니다.
 * r=8: LoRA 어댑터의 차원(rank)를 8차원으로 설정했습니다. 값이 클수록 더 복잡한 패턴을 학습 할 수 있지만, 그만큼 파라미터의 수도 늘어납니다.
 * target_modules: LoRA 어뎁터를 적용할 모델의 레이어를 지정합니다. 주로 어텐선(attention)관련 레이어를 지정합니다.
 * 모델 준비:  prepare_model_for_kbit_trainingrhk get_peft_model 함수를 차례로 호출하여 4비트 양자화된 모델에 LoRA 설정을 적용하여 학습 준비를 마칩니다.

데이터셋을 로드합니다.
```phton
def formatting_prompts_func(example):
    # Process a single example
    text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}
"""
    return {"text": text}

dataset = load_dataset("json", data_files="train_dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func)
```
 * 데이터셋 로드: **train_dataset.jsonl** 파일에 있는 학습 데이터를 로드합니다.
 * 프롬프트 포맷팅: 모델이 instruction과 input이 주어지면 response를 생성하는 특정 작업 패턴을 학습하도록, 각 데이터 샘플을 정해진 프롬프트 형식의 문자열로 변환합니다.

트레이너 설정 및 학습
```python
training_args = SFTConfig(
    output_dir="./gemma3-270m-finetuned-adapter",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    max_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

# 학습 시작
trainer.train()

# 학습된 어댑터 저장
trainer.save_model("./gemma3-270m-finetuned-adapter")
```
 * 학습 설정(SFTConfig): 파인튜닝에 필요한 하이퍼파라미터를 설정합니다.
   * output_dir: 학습된 모델(LoRA 어댑터)가 저장될 경로입니다.
   * num_train_epochs=3: 전체 데이터셋을 3번 반복하여 학습합니다.
   * learning_rate, per_device_train_batch_size: 학습에 영향을 미치는 주요 값들을 지정
 * 트레이너 생성(SFTTrainer): SFTTrainer은 Supervised Fine-Tuning를 쉽게 할 수 있도록 도와주는 도구입니다. 준비된 모델, 데이터셋, 학습 설정을 하나로 묶어줍니다.
 * 학습 시작 및 저장: trainer.train()을 호출하여 실제 파인튜닝을 시작하고, 학습이 완료되면 trainer.save_model()을 통해 학습된 LoRA 어댑터 가중치를 지정된 경로에 저장합니다.

## 훈련
아래와 같이 훈련을 합니다. 
```bash
python finetune.py 
```
훈련 결과는 아래와 같습니다.
```json
{
  "train_runtime": 7558.359, 
  "train_samples_per_second": 19.695, 
  "train_steps_per_second": 1.231, 
  "train_loss": 3.488577838408263, 
  "num_tokens": 11568384.0, 
  "mean_token_accuracy": 0.36196496798878625, 
  "epoch": 3.0
}
```
결과를 분석해보겠습니다.
  * train_runtime: 49620 개의 데이터를 훈련하는데 7558초의 시간이 걸렸습니다. 
  * 학습에 오류 없이 3에포크까지 성공적으로 완료되었습니다.
  * 학습효과: mean_token_accuracy 가 약 36.2%로 나왔는데, 이는 매우 긍정적인 신호입니다. 이는 모델이 단순히 무작위로 예측하는 것이 아니라, 학습 데이터의 패턴과 스타일을 성공적으로 학습하여 상당한 수준의 예측 능력을 갖추게 되었음을 의미합니다. 언어 모델의 핵심 능력은 다음 토큰을 예측하는 것인데, 이 정도 수치는 학습이 잘 되었다는 것을 보여줍니다.
  * 손실 값: train_loss가 3.488로 수렴했다는 것은 모델이 사용할 만한 수준의 도달했음을 나타냅니다.

## ollama 모델 변환
이제 Ollama를 이용해서 모델을 서비스해보도록 하겠습니다. merge_model.py 파일을 만들어서 Ollama에서 쉽게 사용 하도록 원본 모델과 학습된 어뎁터를 하나의 모델로 병합하겠습니다. 
```
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 기본 모델 및 토크나이저 로드
base_model_id = "google/gemma-3-270m"
adapter_path = "./gemma3-270m-finetuned-adapter"
merged_model_path = "./gemma3-270m-finetuned-merged"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 2. PEFT 모델 로드 및 병합
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

# 3. 병합된 모델과 토크나이저 저장
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"모델 병합 완료! 저장 경로: {merged_model_path}")
```

Ollama가 모델을 인식 할 수 있도록 `Modelfile`을 작성합니다.
```
# Modelfile
FROM ./gemma3-270m-finetuned-merged

TEMPLATE """
### Instruction:
{{ .Prompt }}

### Response:
"""

PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"
```

모델을 생성합니다.
```
ollama create my-gemma -f ./Modelfile
```

ollama list 명령으로 모델이 성공적으로 등록됐는지 확인해 보겠습니다.
```
ollama list
NAME                         ID              SIZE      MODIFIED       
my-gemma:latest              0e2a19ef3707    552 MB    28 seconds ago    
gemma3:270m                  e7d36fb2c3b3    291 MB    26 hours ago      
qwen3:14b                    bdbd181c33f2    9.3 GB    4 days ago        
```
