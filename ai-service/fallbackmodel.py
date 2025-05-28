from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from prompt import prompt_by_id

class FallbackModel:
    def __init__(self, model_name: str = "gpt2-medium", id=None, temperature=None):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Use eos_token_id for padding to avoid warning
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Placeholder prompt and temperature setup
        self.prompt = prompt_by_id(id) if id is not None else "Please provide an ID for your prompt."
        self.temperature = temperature if temperature is not None else 0.7

    def params(self, temperature: float, prompt: str):
        """
        Set generation parameters.
        """
        self.temperature = temperature
        self.prompt = prompt

    def call_fallback(self, **kwargs) -> str:
        """
        Generate a blog snippet using the fallback model.

        Returns:
            str: Generated blog text.
        """
        try:
            response = self.generator(
                self.prompt,
                max_new_tokens=256,  
                num_return_sequences=1,
                do_sample=True,
                temperature=self.temperature,
                top_k=50,
                top_p=0.95,
                truncation=True,  # Prevent token overflow
                **kwargs
            )
            return response[0]["generated_text"]
        except Exception as e:
            print(f"[FallbackModel Error] {e}")
            return "Error: Failed to generate blog using fallback model."


