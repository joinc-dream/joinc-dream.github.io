---
layout: single 
title: "Gemini 사용에 대한 코드 정리(VertexAI vs GenAI)"
date: 2025-08-17 21:00:00 +0900
categories: LLM 생성형-AI Quick Lab
card_description: "GCP에서 크게 두가지 방법으로 Gemini 언어 모델을 사용할 수 있습니다. 기존 Vertex AI API를 사용하는 방식과 GEMINI API를 사용하는 최신 방식을 빠르게 실행해보세요."
author: sangwoonam
---

# 라이브러리 Import
- 공통적으로 사용하는 라이브러리를 불러옵니다.


```python
import os
from dotenv import load_dotenv
```

## 기존 VertexAI SDK 사용
- google.oauth2의 service_account를 활용 
- GCP의 Credential 파일을 통해 별도 로그인 없이 VertexAI 사용


```python
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import (
    GenerativeModel, 
    GenerationConfig,
    Tool,
    grounding,
)
```

- VERTEXAI_PROJECT_ID: Vertex AI 사용 권한이 있는 GCP 프로젝트의 ID
- VERTEXAI_CREDENTIALS_PATH: Vertex AI에 접근 권한이 있는 서비스 계정의 “JSON 키 파일” 경로
- **별도 Client를 받아 올 필요없이, init 한 번으로 호출 끝**


```python
load_dotenv()

project_id = os.getenv("VERTEXAI_PROJECT_ID")
credential_path = os.getenv("VERTEXAI_CREDENTIALS_PATH")
credentials = service_account.Credentials.from_service_account_file(credential_path)

vertexai.init(
    project=project_id, 
    credentials=credentials
)
```

### Vertex AI 답변 생성 함수


```python
def vertexai_generate(
    prompt,
    model_name="gemini-1.5-flash",
    generation_config=GenerationConfig(),
    tools=None
):
    model = GenerativeModel(
        model_name=model_name,
        tools=tools,
        generation_config=generation_config,
    )

    response = model.generate_content(prompt)
    return response.text
```

- 기본 값(Default)으로 답변


```python
prompt = "2025년 한국 대통령 누구야?"

result = vertexai_generate(prompt)
print("result: ", result, flush=True)
```

    result:  아직 2025년 한국 대통령이 누가 될지는 알 수 없습니다.  2022년 대선 이후로는  대한민국 대통령 선거가 5년마다 치러지기 때문에 다음 대선은 2027년에 있을 예정입니다.  따라서 2025년에는 현직 대통령이 계속해서 직무를 수행할 것입니다.
    


- 최신 모델(gemini 2.5 버전) 사용


```python
model_name = "gemini-2.5-flash"
prompt = "2025년 한국 대통령 누구야?"

result = vertexai_generate(prompt, model_name)
print("result: ", result, flush=True)
```

    result:  2025년 한국 대통령은 현재와 동일하게 **윤석열** 대통령입니다.
    
    윤석열 대통령은 2022년 5월 10일에 취임했으며, 대한민국 대통령의 임기는 5년 단임이므로 2027년 5월까지 재임하게 됩니다.


- Config 변경


```python
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.95,
    top_k=20,
    candidate_count=1,
    seed=5,
)
model_name = "gemini-2.5-flash"
prompt = "2025년 한국 대통령 누구야?"

result = vertexai_generate(prompt, model_name, generation_config)
print("result: ", result, flush=True)
```

    result:  현재 대한민국 대통령은 **윤석열 대통령**입니다.
    
    대한민국 대통령의 임기는 5년 단임이며, 윤석열 대통령은 2022년 5월에 취임했습니다.
    
    따라서 **2025년에도 윤석열 대통령이 재임 중일 것입니다.** 다음 대통령 선거는 2027년에 치러질 예정입니다.


- Tools 사용(Google 웹 검색)
    - Vertex AI에서 2.0 이상 모델은 사용 가능하지만, **웹 검색은 1.5 버전까지만 지원**


```python
google_search_retrieval = grounding.GoogleSearchRetrieval()
tools = [Tool.from_google_search_retrieval(google_search_retrieval)]
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.95,
    top_k=20,
    candidate_count=1,
    seed=5,
)
model_name = "gemini-2.5-flash"
prompt = "2025년 한국 대통령 누구야?"

result = vertexai_generate(prompt, model_name, generation_config, tools)
print("result: ", result, flush=True)
```


    ---------------------------------------------------------------------------

    _InactiveRpcError                         Traceback (most recent call last)

    File ~/miniconda3/lib/python3.12/site-packages/google/api_core/grpc_helpers.py:76, in _wrap_unary_errors.<locals>.error_remapped_callable(*args, **kwargs)
         75 try:
    ---> 76     return callable_(*args, **kwargs)
         77 except grpc.RpcError as exc:


    File ~/miniconda3/lib/python3.12/site-packages/grpc/_interceptor.py:277, in _UnaryUnaryMultiCallable.__call__(self, request, timeout, metadata, credentials, wait_for_ready, compression)
        268 def __call__(
        269     self,
        270     request: Any,
       (...)
        275     compression: Optional[grpc.Compression] = None,
        276 ) -> Any:
    --> 277     response, ignored_call = self._with_call(
        278         request,
        279         timeout=timeout,
        280         metadata=metadata,
        281         credentials=credentials,
        282         wait_for_ready=wait_for_ready,
        283         compression=compression,
        284     )
        285     return response


    File ~/miniconda3/lib/python3.12/site-packages/grpc/_interceptor.py:332, in _UnaryUnaryMultiCallable._with_call(self, request, timeout, metadata, credentials, wait_for_ready, compression)
        329 call = self._interceptor.intercept_unary_unary(
        330     continuation, client_call_details, request
        331 )
    --> 332 return call.result(), call


    File ~/miniconda3/lib/python3.12/site-packages/grpc/_channel.py:440, in _InactiveRpcError.result(self, timeout)
        439 """See grpc.Future.result."""
    --> 440 raise self


    File ~/miniconda3/lib/python3.12/site-packages/grpc/_interceptor.py:315, in _UnaryUnaryMultiCallable._with_call.<locals>.continuation(new_details, request)
        314 try:
    --> 315     response, call = self._thunk(new_method).with_call(
        316         request,
        317         timeout=new_timeout,
        318         metadata=new_metadata,
        319         credentials=new_credentials,
        320         wait_for_ready=new_wait_for_ready,
        321         compression=new_compression,
        322     )
        323     return _UnaryOutcome(response, call)


    File ~/miniconda3/lib/python3.12/site-packages/grpc/_channel.py:1198, in _UnaryUnaryMultiCallable.with_call(self, request, timeout, metadata, credentials, wait_for_ready, compression)
       1192 (
       1193     state,
       1194     call,
       1195 ) = self._blocking(
       1196     request, timeout, metadata, credentials, wait_for_ready, compression
       1197 )
    -> 1198 return _end_unary_response_blocking(state, call, True, None)


    File ~/miniconda3/lib/python3.12/site-packages/grpc/_channel.py:1006, in _end_unary_response_blocking(state, call, with_call, deadline)
       1005 else:
    -> 1006     raise _InactiveRpcError(state)


    _InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
    	status = StatusCode.INVALID_ARGUMENT
    	details = "Unable to submit request because google_search_retrieval is not supported; please use google_search field instead. Learn more: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini"
    	debug_error_string = "UNKNOWN:Error received from peer ipv4:142.250.76.10:443 {grpc_message:"Unable to submit request because google_search_retrieval is not supported; please use google_search field instead. Learn more: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini", grpc_status:3, created_time:"2025-08-17T15:56:13.024487793+09:00"}"
    >

    
    The above exception was the direct cause of the following exception:


    InvalidArgument                           Traceback (most recent call last)

    Cell In[29], line 15
         12 model_name = "gemini-2.5-flash"
         13 prompt = "2025년 한국 대통령 누구야?"
    ---> 15 result = generate_response(prompt, model_name, generation_config, tools)
         16 print("result: ", result, flush=True)


    Cell In[26], line 13, in generate_response(prompt, model_name, generation_config, tools)
          1 def generate_response(
          2     prompt,
          3     model_name="gemini-1.5-flash",
          4     generation_config=GenerationConfig(),
          5     tools=None
          6 ):
          7     model = GenerativeModel(
          8         model_name=model_name,
          9         tools=tools,
         10         generation_config=generation_config,
         11     )
    ---> 13     response = model.generate_content(prompt)
         14     return response.text


    File ~/miniconda3/lib/python3.12/site-packages/vertexai/generative_models/_generative_models.py:695, in _GenerativeModel.generate_content(self, contents, generation_config, safety_settings, tools, tool_config, labels, stream)
        686     return self._generate_content_streaming(
        687         contents=contents,
        688         generation_config=generation_config,
       (...)
        692         labels=labels,
        693     )
        694 else:
    --> 695     return self._generate_content(
        696         contents=contents,
        697         generation_config=generation_config,
        698         safety_settings=safety_settings,
        699         tools=tools,
        700         tool_config=tool_config,
        701         labels=labels,
        702     )


    File ~/miniconda3/lib/python3.12/site-packages/vertexai/generative_models/_generative_models.py:820, in _GenerativeModel._generate_content(self, contents, generation_config, safety_settings, tools, tool_config, labels)
        793 """Generates content.
        794 
        795 Args:
       (...)
        810     A single GenerationResponse object
        811 """
        812 request = self._prepare_request(
        813     contents=contents,
        814     generation_config=generation_config,
       (...)
        818     labels=labels,
        819 )
    --> 820 gapic_response = self._prediction_client.generate_content(request=request)
        821 return self._parse_response(gapic_response)


    File ~/miniconda3/lib/python3.12/site-packages/google/cloud/aiplatform_v1/services/prediction_service/client.py:2275, in PredictionServiceClient.generate_content(self, request, model, contents, retry, timeout, metadata)
       2272 self._validate_universe_domain()
       2274 # Send the request.
    -> 2275 response = rpc(
       2276     request,
       2277     retry=retry,
       2278     timeout=timeout,
       2279     metadata=metadata,
       2280 )
       2282 # Done; return the response.
       2283 return response


    File ~/miniconda3/lib/python3.12/site-packages/google/api_core/gapic_v1/method.py:131, in _GapicCallable.__call__(self, timeout, retry, compression, *args, **kwargs)
        128 if self._compression is not None:
        129     kwargs["compression"] = compression
    --> 131 return wrapped_func(*args, **kwargs)


    File ~/miniconda3/lib/python3.12/site-packages/google/api_core/grpc_helpers.py:78, in _wrap_unary_errors.<locals>.error_remapped_callable(*args, **kwargs)
         76     return callable_(*args, **kwargs)
         77 except grpc.RpcError as exc:
    ---> 78     raise exceptions.from_grpc_error(exc) from exc


    InvalidArgument: 400 Unable to submit request because google_search_retrieval is not supported; please use google_search field instead. Learn more: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini


- 1.5 버전으로 모델 변경 후 재실행


```python
google_search_retrieval = grounding.GoogleSearchRetrieval()
tools = [Tool.from_google_search_retrieval(google_search_retrieval)]
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.95,
    top_k=20,
    candidate_count=1,
    seed=5,
)
model_name = "gemini-1.5-flash"
prompt = "2025년 한국 대통령 누구야?"

result = vertexai_generate(prompt, model_name, generation_config, tools)
print("result: ", result, flush=True)
```

    result:  2025년 대한민국 대통령은 이재명입니다.  2025년 6월 3일에 치러진 제21대 대통령 선거에서 당선되었습니다.  윤석열 전 대통령의 탄핵으로 인한 조기 대선이었습니다.
    


## 신규 GenAI SDK 사용
- GenAI SDK는 GEMINI API를 사용하지만, 기존 VertexAI API를 필요에 따라 끌어 올 수 있다.
- google.oauth2의 service_account 필요 없음
- **Gemini API Key를 통해 Client 호출**


```python
from google import genai
from google.genai.types import (
    Content,
    Part,
    GoogleSearch,
    GenerateContentConfig,
    Tool
)
```

1) VertexAI API 사용 시
    - VERTEXAI_USE: Vertex AI 사용 유무
    - VERTEXAI_PROJECT_ID: Vertex AI 사용 권한이 있는 GCP 프로젝트의 ID
    - VERTEXAI_LOCATION: Vertex AI에 접근 권한이 있는 GCP 프로젝트의 LOCATION


```python
load_dotenv()

# 반드시 해줘야 함
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("VERTEXAI_CREDENTIALS_PATH")

client = genai.Client(
    vertexai=True,
    project="gai-llm-poc",
    location="us-central1",
)
```

2) 순수 GEMINI API 사용 시
    - GENAI_API_KEY: GEMINI API 활성화 시 발급된 API Key


```python
load_dotenv()

api_key = os.getenv("GENAI_API_KEY")

client = genai.Client(
    vertexai=False,
    api_key=api_key
)
```

### Gen AI 답변 생성 함수


```python
def genai_generate(
    client,
    prompt,
    model_name="gemini-1.5-flash",
    generation_config=GenerateContentConfig(),
):
    contents = []
    contents.append(
        Content(
            role="user",
            parts=[
                Part.from_text(text=prompt)
            ]
        ),
    )
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generation_config,
    )
    return response.text
```

- 기본 값(Default)으로 답변


```python
prompt = "2025년 한국 대통령 누구야?"

result = genai_generate(client, prompt)
print("result: ", result, flush=True)
```

    result:  2025년 한국 대통령은 아직 결정되지 않았습니다.  2022년 대통령 선거에서 윤석열 후보가 당선되었고, 그의 임기는 2022년 5월 10일부터 2027년 5월 9일까지 입니다. 따라서 2025년에도 윤석열 대통령이 한국의 대통령일 것입니다.
    


- 최신 모델(gemini 2.5 버전) 사용


```python
model_name = "gemini-2.5-flash"
prompt = "2025년 한국 대통령 누구야?"

result = genai_generate(client, prompt, model_name)
print("result: ", result, flush=True)
```

    result:  2025년 한국 대통령은 **윤석열 대통령**입니다.
    
    윤석열 대통령은 2022년 5월에 취임했으며, 대한민국 대통령의 임기는 5년 단임이기 때문에 2027년 5월까지가 그의 임기입니다. 따라서 2025년에도 윤석열 대통령이 재임 중입니다.


- Config 변경


```python
generation_config = GenerateContentConfig(
    temperature=0.1,
    top_p=0.95,
    top_k=20,
    candidate_count=1,
    seed=5,
)
model_name = "gemini-2.5-flash"
prompt = "2025년 한국 대통령 누구야?"

result = genai_generate(client, prompt, model_name, generation_config)
print("result: ", result, flush=True)
```

    result:  2025년에도 **윤석열** 대통령입니다.
    
    대한민국 대통령의 임기는 5년 단임제이며, 윤석열 대통령은 2022년 5월 10일에 취임했으므로, 임기는 2027년 5월 9일까지입니다. 따라서 2025년에는 윤석열 대통령이 재임 중입니다.


- Tools 사용(Google 웹 검색)
    - **2.0 이상 모델 또한 웹 검색 사용 가능**
    - Tools가 GenerateContentConfig 안에 포함되는 구조


```python
google_search = GoogleSearch()
tools = [Tool(google_search=google_search)]
generation_config = GenerateContentConfig(
    temperature=0.1,
    top_p=0.95,
    top_k=20,
    candidate_count=1,
    seed=5,
    tools=tools  # 안에 포함됨
)
model_name = "gemini-2.5-flash"
prompt = "2025년 한국 대통령 누구야?"

result = genai_generate(client, prompt, model_name, generation_config)
print("result: ", result, flush=True)
```

    result:  2025년 대한민국의 대통령은 이재명입니다. 그는 2025년 6월 3일에 치러진 제21대 대통령 선거에서 당선되었으며, 임기는 2025년 6월 4일부터 2030년 6월 3일까지입니다. 이 선거는 윤석열 전 대통령의 탄핵으로 인해 조기에 실시되었습니다.

