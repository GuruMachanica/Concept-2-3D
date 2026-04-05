import json
import os

from gemini_client import gemini_generate_text


class IntentAnalyzer:
    def __init__(self):
        # Configure Gemini API if key is available
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    def parse(self, query: str) -> dict:
        """
        Takes a natural language query and breaks it down into intent using LLM.
        """
        if self.api_key:
            prompt = f"""
            Analyze the following query to extract structural intent for a 3D model generator.
            Return ONLY a valid JSON object with these keys:
            - primary_keywords (list of strings)
            - structural_components (list of strings representing 3D shapes or distinct parts)
            - context (string, e.g., 'Medical', 'Educational', 'Gaming')

            Query: "{query}"
            """
            try:
                text = gemini_generate_text(prompt=prompt, model=self.model_name, api_key=self.api_key) or ""
                text = text.strip()
                # Remove markdown formatting if present
                if text.startswith("```json"):
                    text = text[7:-3]
                elif text.startswith("```"):
                    text = text[3:-3]
                if not text:
                    return self._naive_parse(query)
                return json.loads(text.strip())
            except Exception as e:
                print(f"Error accessing Gemini: {e}")
                # Fallback to naive parsing
                return self._naive_parse(query)
        else:
            print("Gemini API key not found. Using naive intent parsing.")
            return self._naive_parse(query)

    def _naive_parse(self, query: str) -> dict:
        """
        A fallback rule-based parser when no LLM is configured.
        """
        words = query.lower().split()
        return {
            "primary_keywords": words,
            "structural_components": words,
            "context": "General",
        }
