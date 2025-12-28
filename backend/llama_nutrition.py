"""
LLaMA Nutrition Insights Generator
Uses Hugging Face LLaMA model to generate nutrition information
"""

import os
import json
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LLaMaNutritionAnalyzer:
    """Generate nutrition insights using LLaMA via Hugging Face."""

    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", use_api=True):
        """
        Initialize LLaMA nutrition analyzer.

        Args:
            model_name: Hugging Face model name
            use_api: Use Hugging Face Inference API (recommended) vs local model
        """
        self.model_name = model_name
        self.use_api = use_api
        self.hf_token = os.getenv('HUGGINGFACE_API_KEY')

        if self.use_api:
            # Use Hugging Face Inference API (faster, no local resources needed)
            from huggingface_hub import InferenceClient
            if not self.hf_token:
                raise ValueError("HUGGINGFACE_API_KEY environment variable required for API access")
            self.client = InferenceClient(token=self.hf_token)
        else:
            # Load model locally (requires significant RAM/VRAM)
            print("Loading LLaMA model locally (this may take several minutes)...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=self.hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_auth_token=self.hf_token,
                torch_dtype=torch.float16,
                device_map='auto'
            )
            print("Model loaded successfully!")

    def generate_nutrition_insights(self, food_items: List[str], temperature=0.7, max_tokens=500):
        """
        Generate nutrition insights for detected food items.

        Args:
            food_items: List of detected food names
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with nutrition information
        """

        if not food_items:
            return {
                'success': False,
                'error': 'No food items provided'
            }

        # Build prompt
        prompt = self._build_nutrition_prompt(food_items)

        try:
            # Generate response
            if self.use_api:
                response = self._generate_with_api(prompt, temperature, max_tokens)
            else:
                response = self._generate_with_local_model(prompt, temperature, max_tokens)

            # Parse response
            nutrition_data = self._parse_nutrition_response(response, food_items)

            return {
                'success': True,
                'data': nutrition_data
            }

        except Exception as e:
            print(f"Error generating nutrition insights: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _build_nutrition_prompt(self, food_items: List[str]) -> str:
        """Build prompt for LLaMA to generate nutrition information."""

        foods_list = ", ".join(food_items)

        prompt = f"""You are a nutrition expert. Analyze the following food items and provide detailed nutritional information.

Food items detected: {foods_list}

For EACH food item, provide:
1. Estimated calories per serving
2. Protein (grams)
3. Carbohydrates (grams)
4. Fats (grams)
5. Key vitamins and minerals
6. Health benefits or concerns

Also provide:
- Total estimated calories for all items
- Overall nutritional balance assessment
- Health score (0-100)
- Recommendations for improvement

Respond in JSON format:
{{
  "foods": [
    {{
      "name": "food_name",
      "calories": 0,
      "protein": 0,
      "carbs": 0,
      "fat": 0,
      "vitamins": ["vitamin1", "vitamin2"],
      "minerals": ["mineral1", "mineral2"],
      "benefits": "health benefits"
    }}
  ],
  "total_calories": 0,
  "total_protein": 0,
  "total_carbs": 0,
  "total_fat": 0,
  "health_score": 0,
  "balance_assessment": "assessment text",
  "recommendations": ["recommendation1", "recommendation2"]
}}

Respond ONLY with valid JSON, no additional text."""

        return prompt

    def _generate_with_api(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate response using Hugging Face Inference API."""

        response = self.client.text_generation(
            prompt,
            model=self.model_name,
            temperature=temperature,
            max_new_tokens=max_tokens,
            return_full_text=False
        )

        return response

    def _generate_with_local_model(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate response using locally loaded model."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from response
        response = response[len(prompt):].strip()

        return response

    def _parse_nutrition_response(self, response: str, food_items: List[str]) -> Dict:
        """Parse LLaMA response and extract nutrition data."""

        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                # No JSON found, use fallback
                return self._generate_fallback_nutrition(food_items)

            json_str = response[start_idx:end_idx]
            nutrition_data = json.loads(json_str)

            # Validate required fields
            if 'foods' not in nutrition_data:
                return self._generate_fallback_nutrition(food_items)

            return nutrition_data

        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing nutrition response: {e}")
            print(f"Raw response: {response}")
            return self._generate_fallback_nutrition(food_items)

    def _generate_fallback_nutrition(self, food_items: List[str]) -> Dict:
        """Generate basic fallback nutrition data when LLaMA fails."""

        # Basic nutrition estimates (very rough)
        basic_estimates = {
            'fruits': {'calories': 60, 'protein': 1, 'carbs': 15, 'fat': 0.2},
            'vegetables': {'calories': 25, 'protein': 2, 'carbs': 5, 'fat': 0.1},
            'meat': {'calories': 200, 'protein': 25, 'carbs': 0, 'fat': 10},
            'grain': {'calories': 150, 'protein': 5, 'carbs': 30, 'fat': 2},
            'dairy': {'calories': 100, 'protein': 8, 'carbs': 12, 'fat': 5},
            'default': {'calories': 100, 'protein': 5, 'carbs': 15, 'fat': 3}
        }

        foods = []
        total_cal = total_pro = total_carb = total_fat = 0

        for item in food_items:
            # Use default estimates
            estimate = basic_estimates.get('default', basic_estimates['default'])

            foods.append({
                'name': item,
                'calories': estimate['calories'],
                'protein': estimate['protein'],
                'carbs': estimate['carbs'],
                'fat': estimate['fat'],
                'vitamins': ['Various'],
                'minerals': ['Various'],
                'benefits': 'Nutritional information not available'
            })

            total_cal += estimate['calories']
            total_pro += estimate['protein']
            total_carb += estimate['carbs']
            total_fat += estimate['fat']

        return {
            'foods': foods,
            'total_calories': round(total_cal, 1),
            'total_protein': round(total_pro, 1),
            'total_carbs': round(total_carb, 1),
            'total_fat': round(total_fat, 1),
            'health_score': 50,
            'balance_assessment': 'Basic nutrition estimate. For accurate information, consult a nutritionist.',
            'recommendations': [
                'Verify nutrition information with a professional',
                'Maintain balanced diet with variety',
                'Consider portion sizes'
            ]
        }


# Convenience function
def analyze_food_nutrition(food_items: List[str], use_api=True):
    """
    Convenience function to analyze food nutrition with LLaMA.

    Args:
        food_items: List of detected food names
        use_api: Use Hugging Face API (True) or local model (False)

    Returns:
        Nutrition data dictionary
    """
    try:
        analyzer = LLaMaNutritionAnalyzer(use_api=use_api)
        return analyzer.generate_nutrition_insights(food_items)
    except Exception as e:
        print(f"LLaMA nutrition analysis failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }
