import time
import google.api_core.exceptions
import google.generativeai as genai

class GeminiModel:
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.max_tokens = 64

    def generate(self, prompt, max_new_tokens=None):
        max_tokens_to_use = max_new_tokens if max_new_tokens is not None else self.max_tokens

        for attempt in range(10):  # Retry up to 10 times
            try:
                # Call Gemini API
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tokens_to_use
                    }
                )
                return response.text.strip()

            except (google.api_core.exceptions.Cancelled,
                    google.api_core.exceptions.InternalServerError) as e:
                print(f"[Retry {attempt + 1}/10] Temporary error ({type(e).__name__}): {e}. Retrying...")
                time.sleep(500)

            except Exception as e:
                print(f"[Error] {e}")
                break  # break for any other error not intended to retry

        raise RuntimeError("Failed to generate content after 3 retries.")
