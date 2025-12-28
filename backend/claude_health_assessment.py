"""
Claude AI Health Assessment
Uses Anthropic's Claude API to provide doctor-perspective health scoring
"""

import os
import json
from typing import List, Dict
import anthropic


def get_health_score_from_claude(food_items: List[str], nutrition_data: Dict, user_profile: Dict = None) -> Dict:
    """
    Get health score and assessment from Claude AI from a doctor's perspective.

    Args:
        food_items: List of detected food names
        nutrition_data: Dictionary with nutrition information
        user_profile: Optional dictionary with user's health information (weight, height, age, etc.)

    Returns:
        Dictionary with health_score (0-100) and balance_assessment
    """

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("⚠ ANTHROPIC_API_KEY not set, using fallback health score")
        return _fallback_health_score(nutrition_data)

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Build the prompt for Claude
        prompt = _build_health_assessment_prompt(food_items, nutrition_data, user_profile)

        # Call Claude API
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0.3,  # Lower temperature for more consistent medical advice
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Parse Claude's response
        response_text = message.content[0].text
        result = _parse_claude_response(response_text)

        return result

    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return _fallback_health_score(nutrition_data)


def _build_health_assessment_prompt(food_items: List[str], nutrition_data: Dict, user_profile: Dict = None) -> str:
    """Build the prompt for Claude to assess health score."""

    foods_list = ", ".join(food_items)
    total_calories = nutrition_data.get('total_calories', 0)
    total_protein = nutrition_data.get('total_protein', 0)
    total_carbs = nutrition_data.get('total_carbs', 0)
    total_fat = nutrition_data.get('total_fat', 0)

    # Build user profile section
    user_context = ""
    if user_profile:
        weight_kg = user_profile.get('weight_kg')
        height_cm = user_profile.get('height_cm')
        age = user_profile.get('age')
        gender = user_profile.get('gender')
        activity_level = user_profile.get('activity_level')
        health_goals = user_profile.get('health_goals')

        # Calculate BMI if available
        bmi = None
        bmi_category = ""
        if weight_kg and height_cm:
            height_m = height_cm / 100
            bmi = round(weight_kg / (height_m * height_m), 1)

            if bmi < 18.5:
                bmi_category = "Underweight"
            elif bmi < 25:
                bmi_category = "Normal weight"
            elif bmi < 30:
                bmi_category = "Overweight"
            else:
                bmi_category = "Obese"

        user_context = f"""

**User Profile:**
- Age: {age or 'Not specified'}
- Gender: {gender or 'Not specified'}
- Weight: {weight_kg}kg, Height: {height_cm}cm
- BMI: {bmi} ({bmi_category}) if available else 'Not calculated'
- Activity Level: {activity_level or 'Not specified'}
- Health Goals: {health_goals or 'Not specified'}

**IMPORTANT:** Please personalize the health score and recommendations based on the user's BMI, age, and health goals. Consider:
- For overweight/obese users: Be more critical of high-calorie, high-fat meals
- For underweight users: Be more lenient with calorie-dense foods
- Adjust portion size recommendations based on BMI and activity level
- Tailor advice to their specific health goals
"""

    prompt = f"""You are a medical doctor and nutritionist. Analyze this meal from a health perspective and provide a personalized health score out of 100 (100 being perfectly healthy).

**Detected Foods:** {foods_list}

**Nutrition Summary:**
- Total Calories: {total_calories} kcal
- Protein: {total_protein}g
- Carbohydrates: {total_carbs}g
- Fat: {total_fat}g{user_context}

**Instructions:**
1. Evaluate this meal from a doctor's perspective considering:
   - Nutritional balance (protein, carbs, fats ratio)
   - Overall healthiness of food choices
   - Portion sizes (based on typical servings)
   - Presence of processed vs whole foods
   - Micronutrient diversity
   - Potential health concerns (high sugar, sodium, saturated fats, etc.)

2. Provide a health score from 0-100 where:
   - 85-100: Excellent - Very healthy, nutrient-dense, well-balanced meal (vegetables, lean proteins, whole grains)
   - 70-84: Good - Healthy home-cooked meal with good nutritional value
   - 55-69: Fair - Acceptable meal, typical everyday food, could be improved
   - 35-54: Poor - Heavily processed or fast food, nutritional concerns
   - 0-34: Very Poor - Deep-fried, high sugar, or extremely unhealthy choices

**Important Scoring Guidelines:**
- Be realistic and generous - most home-cooked meals should score 60-80
- Reserve scores below 50 for clearly unhealthy fast food or junk food
- A typical balanced meal with vegetables and protein should score 70+
- Only extremely unhealthy meals (deep-fried, high sugar desserts, etc.) should score below 40

3. Provide a brief medical assessment (2-3 sentences) explaining the score.

**Response Format (JSON only):**
{{
  "health_score": <number 0-100>,
  "balance_assessment": "<your 2-3 sentence medical assessment>",
  "recommendations": ["<recommendation 1>", "<recommendation 2>", "<recommendation 3>"]
}}

Respond ONLY with valid JSON, no additional text."""

    return prompt


def _parse_claude_response(response: str) -> Dict:
    """Parse Claude's JSON response."""

    try:
        # Try to extract JSON from response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")

        json_str = response[start_idx:end_idx]
        result = json.loads(json_str)

        # Validate required fields
        if 'health_score' not in result:
            raise ValueError("Missing health_score in response")

        # Ensure health score is within 0-100
        result['health_score'] = max(0, min(100, int(result['health_score'])))

        # Ensure we have defaults for optional fields
        result.setdefault('balance_assessment', 'Health assessment unavailable')
        result.setdefault('recommendations', [])

        return result

    except Exception as e:
        print(f"Error parsing Claude response: {e}")
        print(f"Raw response: {response}")
        raise


def _fallback_health_score(nutrition_data: Dict) -> Dict:
    """
    Generate a basic health score when Claude API is unavailable.
    Uses simple heuristics based on nutrition values.
    """

    total_calories = nutrition_data.get('total_calories', 0)
    total_protein = nutrition_data.get('total_protein', 0)
    total_carbs = nutrition_data.get('total_carbs', 0)
    total_fat = nutrition_data.get('total_fat', 0)

    # Simple scoring heuristic
    score = 50  # Start at neutral

    # Protein check (higher is generally better)
    if total_protein > 20:
        score += 10
    elif total_protein > 10:
        score += 5

    # Calorie check (moderate is better)
    if 300 <= total_calories <= 600:
        score += 10
    elif total_calories > 800:
        score -= 10

    # Balance check (protein to carb ratio)
    if total_carbs > 0:
        protein_carb_ratio = total_protein / total_carbs
        if 0.2 <= protein_carb_ratio <= 0.5:
            score += 10

    # Fat check (moderate is better)
    if total_fat > 30:
        score -= 5
    elif 10 <= total_fat <= 20:
        score += 5

    # Clamp score to 0-100
    score = max(0, min(100, score))

    return {
        'health_score': score,
        'balance_assessment': 'Basic health estimate. For accurate assessment, configure ANTHROPIC_API_KEY.',
        'recommendations': [
            'Consider portion sizes',
            'Balance macronutrients',
            'Include variety in your diet'
        ]
    }
