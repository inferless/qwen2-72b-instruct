from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class InferlessPythonModel:

    def initialize(self):
        model_id = "Qwen/Qwen2-72B-Instruct-AWQ"  # Specify the model repository ID
        # Initialize the LLM object with the downloaded model directory
        self.llm = LLM(model=model_id, enforce_eager=True, quantization="AWQ")

        # Load the tokenizer associated with the pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def infer(self, inputs):
        prompts = inputs["prompt"]  # Extract the prompt from the input
        temperature = inputs["temperature"]
        top_p = inputs["top_p"]
        repetition_penalty = inputs["repetition_penalty"]
        top_k = inputs["top_k"]
        max_tokens = inputs["max_tokens"]

        # Define sampling parameters for model generation
        # You can set max_tokens to 1024 for complete answer to your question
        sampling_params = SamplingParams(temperature=temperature,top_p=top_p,repetition_penalty=repetition_penalty,
                                         top_k=top_k,max_tokens=max_tokens)
        # Apply the chat template and convert to a list of strings (without tokenization)
        input_text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompts}], tokenize=False)

        # Generate text using the LLM with the specified sampling parameters
        result = self.llm.generate(input_text, sampling_params)

        # Extract the generated text from the result object
        result_output = [output.outputs[0].text for output in result]

        # Return a dictionary containing the generated text
        return {"generated_result": result_output[0]}

    def finalize(self):
        self.llm = None
