from __future__ import annotations

import time

import pytest
from vllm import LLM, SamplingParams


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0,
                          max_tokens=10,
                          ignore_eos=True)


@pytest.fixture
# TODO(amacaskill): Replace with GKE owned GCS bucket, and a smaller model.
def gcs_model_name():
    return "gs://vertex-model-garden-public-us/llama3/llama3-8b-hf"


@pytest.fixture
def hf_model_name():
    return "meta-llama/Meta-Llama-3-8B"


@pytest.fixture
def prompt():
    return "Hello, my name is"


def test_correctness(
    sampling_config: SamplingParams,
    gcs_model_name: str,
    hf_model_name: str,
    prompt: str,
):
    '''
    Compare the outputs of a model loaded from GCS via runai_model_streamer
    and a model loaded from Hugging Face. The outputs should be the same.
    '''
    # Test with GCS model using runai_model_streamer
    gcs_llm = LLM(model=gcs_model_name,
                  model_impl_type="runai_model_streamer",
                  max_model_len=128,
                  max_num_seqs=16,
                  max_num_batched_tokens=256)
    gcs_outputs = gcs_llm.generate([prompt], sampling_config)
    gcs_output_text = gcs_outputs[0].outputs[0].text
    del gcs_llm
    time.sleep(10)  # Wait for TPUs to be released

    # Test with Hugging Face model
    hf_llm = LLM(model=hf_model_name,
                 max_model_len=128,
                 max_num_seqs=16,
                 max_num_batched_tokens=256)
    hf_outputs = hf_llm.generate([prompt], sampling_config)
    hf_output_text = hf_outputs[0].outputs[0].text
    del hf_llm
    time.sleep(10) # Wait for TPUs to be released

    assert gcs_output_text == hf_output_text, (
        f"Outputs do not match! "
        f"GCS output: {gcs_output_text}, HF output: {hf_output_text}"
    )


def test_performance(
    gcs_model_name: str,
    hf_model_name: str,
):
    '''
    Compare the model load time of a model loaded from GCS via
    runai_model_streamer and a model loaded from Hugging Face.
    '''
    # Time loading from GCS
    start_time = time.time()
    gcs_llm = LLM(model=gcs_model_name,
                  model_impl_type="runai_model_streamer",
                  max_model_len=128,
                  max_num_seqs=16,
                  max_num_batched_tokens=256)
    gcs_load_time = time.time() - start_time
    print(f"GCS model load time: {gcs_load_time:.2f} seconds")
    del gcs_llm
    time.sleep(10)

    # Time loading from Hugging Face
    start_time = time.time()
    hf_llm = LLM(model=hf_model_name,
                 max_model_len=128,
                 max_num_seqs=16,
                 max_num_batched_tokens=256)
    hf_load_time = time.time() - start_time
    print(f"Hugging Face model load time: {hf_load_time:.2f} seconds")
    del hf_llm
    time.sleep(10)

    print(f"GCS load time: {gcs_load_time:.2f}s, HF load time: {hf_load_time:.2f}s")
