import threading
import time
from threading import Event
from typing import Union

from vllm import LLM, SamplingParams
from vllm.outputs import PoolingRequestOutput, RequestOutput


class MyLLM(LLM):
    def keep_running(
        self,
        *,
        stop_event: Event,
    ):
        self.output_dict = {}
        while True:
            outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
            if stop_event.is_set():
                break
            if not self.llm_engine.has_unfinished_requests():
                time.sleep(0.0000001)
                continue
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
            if len(outputs) > 0:
                for output in outputs:
                    self.output_dict[output.request_id] = output

    def add_requests(self, prompt: str, sampling_params: SamplingParams):
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
            request_id,
            prompt,
            sampling_params,
        )
        return request_id

    def get_output(self, request_id: str):
        while True:
            if request_id in self.output_dict:
                return self.output_dict[request_id]
            time.sleep(0.0000001)


if __name__ == "__main__":
    llm = MyLLM(model="meta-llama/Llama-3.2-1B", enforce_eager=True)
    stop_event = Event()
    threading.Thread(
        target=llm.keep_running,
        kwargs={"stop_event": stop_event},
    ).start()
    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?",
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
    )
    request_ids = []
    for prompt in prompts:
        request_ids.append(llm.add_requests(prompt, sampling_params))
    for request_id in request_ids:
        output = llm.get_output(request_id)
        print(output)
    stop_event.set()
