// Modern Food Recognition Interface JavaScript

// Sidebar toggle functionality
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('expanded');
}

// Update slider value displays
function updateSliderValue(type) {
    if (type === 'confidence') {
        const slider = document.getElementById('confidence-range');
        const label = document.getElementById('confidence-value');
        label.textContent = slider.value + '%';
    } else if (type === 'threshold') {
        const slider = document.getElementById('threshold-range');
        const label = document.getElementById('threshold-value');
        label.textContent = slider.value + '%';
    }
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    // Preview the image
    const reader = new FileReader();
    reader.onload = function(e) {
        const placeholder = document.getElementById('placeholder');
        const previewImage = document.getElementById('preview-image');

        placeholder.classList.add('hidden');
        previewImage.src = e.target.result;
        previewImage.classList.remove('hidden');

        // Enable analyze button
        document.getElementById('analyze-btn').disabled = false;
        updateStatusMessage('Image loaded. Click "Analyze Food" to detect.', 'success');
    };

    reader.readAsDataURL(file);
}

// Update status message
function updateStatusMessage(message, type = 'info') {
    const statusElement = document.getElementById('status-message');
    statusElement.textContent = message;

    // Reset classes
    statusElement.className = 'status-message';

    // Add type-specific class
    if (type === 'success') {
        statusElement.style.color = '#1DB954';
    } else if (type === 'error') {
        statusElement.style.color = '#ff6b6b';
    } else {
        statusElement.style.color = '#b3b3b3';
    }
}

// Handle form submission
document.getElementById('detection-form')?.addEventListener('submit', function(e) {
    e.preventDefault();

    updateStatusMessage('Analyzing food...', 'info');
    document.getElementById('analyze-btn').disabled = true;

    const formData = new FormData(this);

    fetch(this.action, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update preview image with detection result
            if (data.detection_image) {
                document.getElementById('preview-image').src = data.detection_image;
            }

            // Display nutritional data
            if (data.nutrition_data) {
                displayNutritionData(data.nutrition_data);
            }

            // Get health analysis from Claude
            if (data.nutrition_data) {
                analyzeHealthWithClaude(data.nutrition_data);
            }

            updateStatusMessage('Analysis complete!', 'success');
        } else {
            updateStatusMessage('Error: ' + (data.error || 'Analysis failed'), 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        updateStatusMessage('Error: ' + error.message, 'error');
    })
    .finally(() => {
        document.getElementById('analyze-btn').disabled = false;
    });
});

// Display nutrition data in column 3
function displayNutritionData(nutritionData) {
    const nutritionContent = document.getElementById('nutrition-content');
    nutritionContent.innerHTML = '';

    if (!nutritionData || !nutritionData.foods || nutritionData.foods.length === 0) {
        nutritionContent.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-utensils"></i>
                <p>No nutritional data available</p>
            </div>
        `;
        return;
    }

    // Display individual food items
    nutritionData.foods.forEach(food => {
        const foodItem = document.createElement('div');
        foodItem.className = 'nutrition-item';
        foodItem.innerHTML = `
            <div class="nutrition-item-name">${food.name}</div>
            <div class="nutrition-details">
                Calories: ${food.calories || 0} kcal<br>
                Protein: ${food.protein || 0}g<br>
                Carbs: ${food.carbs || 0}g<br>
                Fat: ${food.fat || 0}g
            </div>
        `;
        nutritionContent.appendChild(foodItem);
    });

    // Display total nutrition
    if (nutritionData.total_calories !== undefined) {
        const totalItem = document.createElement('div');
        totalItem.className = 'nutrition-total';
        totalItem.innerHTML = `
            <div class="nutrition-total-title">Total Nutrition</div>
            <div class="nutrition-total-details">
                Calories: ${nutritionData.total_calories || 0} kcal<br>
                Protein: ${nutritionData.total_protein || 0}g<br>
                Carbs: ${nutritionData.total_carbs || 0}g<br>
                Fat: ${nutritionData.total_fat || 0}g
            </div>
        `;
        nutritionContent.appendChild(totalItem);
    }
}

// Analyze health with Claude API
async function analyzeHealthWithClaude(nutritionData) {
    try {
        updateHealthScore(null, 'Analyzing healthiness with AI...');

        const response = await fetch('/api/analyze-health', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ nutrition_data: nutritionData })
        });

        if (!response.ok) {
            throw new Error('Failed to analyze health');
        }

        const healthAnalysis = await response.json();

        if (healthAnalysis.success) {
            displayHealthAnalysis(healthAnalysis.data);
        } else {
            updateHealthScore(null, 'Unable to analyze health score');
        }
    } catch (error) {
        console.error('Error analyzing health:', error);
        updateHealthScore(null, 'Error analyzing health score');
    }
}

// Display health analysis results
function displayHealthAnalysis(analysis) {
    const healthScore = analysis.health_score || 50;
    const explanation = analysis.explanation || '';
    const recommendations = analysis.recommendations || [];

    // Update health bar
    updateHealthBar(healthScore);

    // Update explanation
    const explanationElement = document.getElementById('health-explanation');
    explanationElement.innerHTML = `
        <strong>Score: ${healthScore}/100</strong><br>
        ${explanation}
    `;

    // Display recommendations if available
    if (recommendations.length > 0) {
        const recsHTML = `
            <div class="health-recommendations">
                <h4>AI Recommendations:</h4>
                <ul>
                    ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        `;
        explanationElement.innerHTML += recsHTML;
    }
}

// Update health bar visual
function updateHealthBar(score) {
    const healthBarFill = document.getElementById('health-bar-fill');
    const healthScoreLabel = document.getElementById('health-score-label');

    // Animate the health bar
    setTimeout(() => {
        healthBarFill.style.width = score + '%';
    }, 100);

    // Update label
    healthScoreLabel.textContent = score + '/100';

    // Update color based on score
    let color;
    if (score < 40) {
        color = 'var(--health-low)';
    } else if (score < 70) {
        color = 'var(--health-medium)';
    } else {
        color = 'var(--health-high)';
    }

    healthScoreLabel.style.color = color;
}

// Update health score (legacy function for compatibility)
function updateHealthScore(score, message) {
    const healthScoreLabel = document.getElementById('health-score-label');
    const healthExplanation = document.getElementById('health-explanation');

    if (score !== null) {
        healthScoreLabel.textContent = score + '/100';
        updateHealthBar(score);
    } else {
        healthScoreLabel.textContent = '---';
    }

    if (message) {
        healthExplanation.textContent = message;
    }
}

// Menu item click handlers
document.querySelectorAll('.menu-item').forEach(item => {
    item.addEventListener('click', function() {
        // Remove active class from all items
        document.querySelectorAll('.menu-item').forEach(i => i.classList.remove('active'));

        // Add active class to clicked item
        this.classList.add('active');

        // Handle navigation based on menu item
        // For now, only Food Detection is implemented
    });
});

// Initialize interface
document.addEventListener('DOMContentLoaded', function() {
    console.log('Modern Food Recognition Interface loaded');

    // Initialize slider values
    updateSliderValue('confidence');
    updateSliderValue('threshold');
});
