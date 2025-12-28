// Food Nutrition CV Dashboard JavaScript

// State
let currentImage = null;
let detectionHistory = [];
let stats = {
    totalScans: 0,
    foodsDetected: 0,
    avgHealth: 0
};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('Food Nutrition CV Dashboard loaded');

    // Setup event listeners
    setupEventListeners();

    // Load history from localStorage
    loadHistory();

    // Update stats
    updateStats();
});

function setupEventListeners() {
    // File input change
    const fileInput = document.getElementById('file-input');
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    const imageContainer = document.getElementById('image-container');
    imageContainer.addEventListener('dragover', handleDragOver);
    imageContainer.addEventListener('drop', handleDrop);
    imageContainer.addEventListener('click', () => fileInput.click());

    // Form submit
    const form = document.getElementById('detection-form');
    form.addEventListener('submit', handleFormSubmit);

    // Confidence slider
    const confidenceSlider = document.getElementById('confidence');
    confidenceSlider.addEventListener('input', function() {
        document.getElementById('confidence-val').textContent = this.value + '%';
    });
}

// File Selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        showStatus('Please select an image file', 'error');
        return;
    }

    currentImage = file;
    previewImage(file);
    document.getElementById('analyze-btn').disabled = false;
    showStatus('Image loaded. Click "Analyze Food" to detect.', 'success');
}

// Drag and Drop
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
}

function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();

    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        const fileInput = document.getElementById('file-input');
        fileInput.files = event.dataTransfer.files;
        handleFileSelect({ target: fileInput });
    }
}

// Preview Image
function previewImage(file) {
    const reader = new FileReader();

    reader.onload = function(e) {
        const placeholder = document.getElementById('upload-placeholder');
        const previewImage = document.getElementById('preview-image');

        placeholder.style.display = 'none';
        previewImage.src = e.target.result;
        previewImage.classList.remove('hidden');
    };

    reader.readAsDataURL(file);
}

// Form Submit - Analyze Food
async function handleFormSubmit(event) {
    event.preventDefault();

    if (!currentImage) {
        showStatus('Please upload an image first', 'error');
        return;
    }

    showStatus('Analyzing food...', 'info');
    document.getElementById('analyze-btn').disabled = true;

    const formData = new FormData();
    formData.append('file', currentImage);
    formData.append('confidence', document.getElementById('confidence').value);
    formData.append('model', document.getElementById('model-select').value);

    try {
        const response = await fetch('/api/detect-food', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Detection failed');
        }

        const data = await response.json();

        if (data.success) {
            // Update preview with detection result
            if (data.detection_image) {
                document.getElementById('preview-image').src = data.detection_image;
            }

            // Display nutrition results
            displayNutritionResults(data.nutrition_data);

            // Add to history
            addToHistory(data);

            // Update stats
            stats.totalScans++;
            stats.foodsDetected += data.nutrition_data.foods.length;
            updateStats();

            showStatus('Analysis complete!', 'success');
        } else {
            throw new Error(data.error || 'Analysis failed');
        }

    } catch (error) {
        console.error('Error:', error);
        showStatus('Error: ' + error.message, 'error');
    } finally {
        document.getElementById('analyze-btn').disabled = false;
    }
}

// Display Nutrition Results
function displayNutritionResults(nutritionData) {
    // Hide empty state
    document.getElementById('nutrition-empty').style.display = 'none';

    // Show nutrition content
    const nutritionContent = document.getElementById('nutrition-content');
    nutritionContent.classList.remove('hidden');

    // Display detected foods
    const detectedFoodsDiv = document.getElementById('detected-foods');
    detectedFoodsDiv.innerHTML = '<h4 style="font-size: 14px; margin-bottom: 10px; color: var(--primary-light);">Detected Foods</h4>';

    if (nutritionData.foods && nutritionData.foods.length > 0) {
        nutritionData.foods.forEach(food => {
            const foodItem = document.createElement('div');
            foodItem.className = 'food-item';
            foodItem.innerHTML = `
                <div>
                    <div class="food-item-name">${food.name}</div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                        P: ${food.protein}g | C: ${food.carbs}g | F: ${food.fat}g
                    </div>
                </div>
                <div class="food-item-calories">${food.calories} cal</div>
            `;
            detectedFoodsDiv.appendChild(foodItem);
        });
    }

    // Update nutrition summary
    document.getElementById('total-calories').textContent = nutritionData.total_calories || 0;
    document.getElementById('total-protein').textContent = (nutritionData.total_protein || 0) + 'g';
    document.getElementById('total-carbs').textContent = (nutritionData.total_carbs || 0) + 'g';
    document.getElementById('total-fat').textContent = (nutritionData.total_fat || 0) + 'g';

    // Update health score
    const healthScore = nutritionData.health_score || 50;
    updateHealthScore(healthScore, nutritionData.balance_assessment);

    // Update health badge
    const healthBadge = document.getElementById('health-badge');
    healthBadge.innerHTML = `<span class="badge">${healthScore}/100</span>`;

    // Display recommendations
    if (nutritionData.recommendations && nutritionData.recommendations.length > 0) {
        const recDiv = document.getElementById('recommendations');
        recDiv.innerHTML = '<h4>Recommendations</h4><ul>';
        nutritionData.recommendations.forEach(rec => {
            recDiv.innerHTML += `<li>${rec}</li>`;
        });
        recDiv.innerHTML += '</ul>';
    }

    // Update average health score
    if (stats.totalScans > 0) {
        const totalHealth = (stats.avgHealth * (stats.totalScans - 1) + healthScore) / stats.totalScans;
        stats.avgHealth = Math.round(totalHealth);
        updateStats();
    }
}

// Update Health Score Bar
function updateHealthScore(score, explanation) {
    const healthScoreFill = document.getElementById('health-score-fill');
    const healthExplanation = document.getElementById('health-explanation');

    setTimeout(() => {
        healthScoreFill.style.width = score + '%';
    }, 100);

    healthExplanation.textContent = explanation || `Health score: ${score}/100`;
}

// Add to History
function addToHistory(detectionData) {
    const historyItem = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        image: document.getElementById('preview-image').src,
        foodCount: detectionData.nutrition_data.foods.length,
        healthScore: detectionData.nutrition_data.health_score || 50
    };

    detectionHistory.unshift(historyItem);

    // Keep only last 10
    if (detectionHistory.length > 10) {
        detectionHistory = detectionHistory.slice(0, 10);
    }

    // Save to localStorage
    saveHistory();

    // Update UI
    renderHistory();
}

// Render History
function renderHistory() {
    const historyList = document.getElementById('history-list');

    if (detectionHistory.length === 0) {
        historyList.innerHTML = `
            <div class="history-empty">
                <i class="fas fa-inbox"></i>
                <p>No scans yet</p>
            </div>
        `;
        return;
    }

    historyList.innerHTML = '';

    detectionHistory.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';

        const date = new Date(item.timestamp);
        const timeStr = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

        historyItem.innerHTML = `
            <img src="${item.image}" alt="Scan">
            <div class="history-item-info">
                <div style="font-weight: 600; font-size: 12px; color: var(--text-primary); margin-bottom: 2px;">
                    ${item.foodCount} items
                </div>
                <div>${dateStr} at ${timeStr}</div>
                <div style="color: var(--success);">Score: ${item.healthScore}</div>
            </div>
        `;

        historyItem.addEventListener('click', () => loadHistoryItem(item));

        historyList.appendChild(historyItem);
    });
}

// Load History Item
function loadHistoryItem(item) {
    // TODO: Implement loading previous analysis
    console.log('Loading history item:', item);
}

// Save History to localStorage
function saveHistory() {
    try {
        localStorage.setItem('detectionHistory', JSON.stringify(detectionHistory));
        localStorage.setItem('stats', JSON.stringify(stats));
    } catch (e) {
        console.error('Error saving history:', e);
    }
}

// Load History from localStorage
function loadHistory() {
    try {
        const savedHistory = localStorage.getItem('detectionHistory');
        const savedStats = localStorage.getItem('stats');

        if (savedHistory) {
            detectionHistory = JSON.parse(savedHistory);
            renderHistory();
        }

        if (savedStats) {
            stats = JSON.parse(savedStats);
            updateStats();
        }
    } catch (e) {
        console.error('Error loading history:', e);
    }
}

// Update Stats Display
function updateStats() {
    document.getElementById('total-scans').textContent = stats.totalScans;
    document.getElementById('foods-detected').textContent = stats.foodsDetected;
    document.getElementById('avg-health').textContent = stats.avgHealth > 0 ? stats.avgHealth : '--';
}

// Show Status Message
function showStatus(message, type = 'info') {
    const statusElement = document.getElementById('status-message');
    statusElement.textContent = message;

    statusElement.style.color = {
        'success': 'var(--success)',
        'error': 'var(--danger)',
        'info': 'var(--text-secondary)'
    }[type] || 'var(--text-secondary)';
}
