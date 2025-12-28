// Food Nutrition CV Dashboard JavaScript

// State
let currentImage = null;
let currentView = 'dashboard';
let nutritionChart = null;
let currentTimeframe = 'all';
let allNutritionData = [];

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('Food Nutrition CV Dashboard loaded');
    setupEventListeners();
    loadHistory();
    loadStats();
});

function setupEventListeners() {
    const headerToggle = document.getElementById('header-toggle');
    const topHeader = document.getElementById('top-header');
    if (headerToggle && topHeader) {
        headerToggle.addEventListener('click', () => {
            topHeader.classList.toggle('collapsed');
        });
    }

    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const view = item.getAttribute('data-view');
            if (view) switchView(view);
        });
    });

    const fileInput = document.getElementById('file-input');
    if (fileInput) fileInput.addEventListener('change', handleFileSelect);

    const imageContainer = document.getElementById('image-container');
    if (imageContainer) {
        imageContainer.addEventListener('dragover', handleDragOver);
        imageContainer.addEventListener('drop', handleDrop);
        // Click handler removed - button handles file selection to avoid double-trigger
    }

    const form = document.getElementById('detection-form');
    if (form) form.addEventListener('submit', handleFormSubmit);

    const confidenceSlider = document.getElementById('confidence');
    if (confidenceSlider) {
        confidenceSlider.addEventListener('input', function() {
            document.getElementById('confidence-val').textContent = this.value + '%';
        });
    }

    // Timeframe filter buttons
    const timeframeBtns = document.querySelectorAll('.timeframe-btn');
    timeframeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            timeframeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTimeframe = btn.getAttribute('data-timeframe');
            updateChartWithTimeframe(currentTimeframe);
        });
    });
}

function switchView(viewName) {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('data-view') === viewName) item.classList.add('active');
    });
    document.querySelectorAll('.view-container').forEach(view => view.classList.remove('active'));
    const targetView = document.getElementById(viewName + '-view');
    if (targetView) targetView.classList.add('active');
    currentView = viewName;
    if (viewName === 'analytics') {
        loadStats();
        loadAnalyticsData();
        initNutritionChart();
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        showStatus('Please select an image file', 'error');
        return;
    }
    currentImage = file;
    previewImage(file);
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) analyzeBtn.disabled = false;
    showStatus('Image loaded. Click Analyze to detect.', 'success');
}

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

function previewImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const placeholder = document.getElementById('upload-placeholder');
        const previewImage = document.getElementById('preview-image');
        if (placeholder && previewImage) {
            placeholder.style.display = 'none';
            previewImage.src = e.target.result;
            previewImage.classList.remove('hidden');
        }
    };
    reader.readAsDataURL(file);
}

async function handleFormSubmit(event) {
    event.preventDefault();
    if (!currentImage) {
        showStatus('Please upload an image first', 'error');
        return;
    }
    showStatus('Analyzing food...', 'info');
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', currentImage);
    formData.append('confidence', document.getElementById('confidence')?.value || 50);
    formData.append('model', document.getElementById('model-select')?.value || 'best');
    formData.append('segmentation', document.getElementById('segmentation-toggle')?.checked || false);

    try {
        const response = await fetch('/api/detect-food', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Detection failed');
        const data = await response.json();
        if (data.success) {
            // Display detection image
            if (data.detection_image) {
                const previewImage = document.getElementById('preview-image');
                if (previewImage) previewImage.src = data.detection_image;
            }

            // Display segmentation overlay if available
            if (data.segmentation_overlay) {
                console.log('Segmentation overlay available:', data.segmentation_overlay);
                // You can add UI to display segmentation overlay here
                // For now, we'll log it and could show in a modal or separate view
            }

            displayNutritionResults(data.nutrition_data);
            loadHistory();
            loadStats();

            let statusMsg = 'Analysis complete!';
            if (data.segmentation_overlay) {
                statusMsg += ' (with segmentation)';
            }
            showStatus(statusMsg, 'success');
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showStatus('Error: ' + error.message, 'error');
    } finally {
        if (analyzeBtn) analyzeBtn.disabled = false;
    }
}

function displayNutritionResults(nutritionData) {
    const nutritionEmpty = document.getElementById('nutrition-empty');
    if (nutritionEmpty) nutritionEmpty.style.display = 'none';
    const nutritionContent = document.getElementById('nutrition-content');
    if (nutritionContent) nutritionContent.classList.remove('hidden');
    const detectedFoodsDiv = document.getElementById('detected-foods');
    if (detectedFoodsDiv && nutritionData.foods) {
        detectedFoodsDiv.innerHTML = '<h4 style="font-size: 14px; margin-bottom: 10px; color: var(--primary-light);">Detected Foods</h4>';
        nutritionData.foods.forEach(food => {
            const foodItem = document.createElement('div');
            foodItem.className = 'food-item';
            foodItem.innerHTML = '<div><div class="food-item-name">' + food.name + '</div><div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">P: ' + food.protein + 'g | C: ' + food.carbs + 'g | F: ' + food.fat + 'g</div></div><div class="food-item-calories">' + food.calories + ' cal</div>';
            detectedFoodsDiv.appendChild(foodItem);
        });
    }
    const updateEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    updateEl('total-calories', nutritionData.total_calories || 0);
    updateEl('total-protein', (nutritionData.total_protein || 0) + 'g');
    updateEl('total-carbs', (nutritionData.total_carbs || 0) + 'g');
    updateEl('total-fat', (nutritionData.total_fat || 0) + 'g');
    const healthScore = nutritionData.health_score || 50;
    updateHealthScore(healthScore, nutritionData.balance_assessment);
    const healthBadge = document.getElementById('health-badge');
    if (healthBadge) healthBadge.innerHTML = '<span class="badge">' + healthScore + '/100</span>';
    if (nutritionData.recommendations) {
        const recDiv = document.getElementById('recommendations');
        if (recDiv) {
            recDiv.innerHTML = '<h4>Recommendations</h4><ul>';
            nutritionData.recommendations.forEach(rec => { recDiv.innerHTML += '<li>' + rec + '</li>'; });
            recDiv.innerHTML += '</ul>';
        }
    }
}

function updateHealthScore(score, explanation) {
    const fill = document.getElementById('health-score-fill');
    const exp = document.getElementById('health-explanation');
    if (fill) setTimeout(() => fill.style.width = score + '%', 100);
    if (exp) exp.textContent = explanation || 'Health score: ' + score + '/100';
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history?limit=3');
        if (!response.ok) return;
        const data = await response.json();
        if (data.success) renderHistory(data.history);
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function renderHistory(history) {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;
    if (!history || history.length === 0) {
        historyList.innerHTML = '<div class="history-empty"><i class="fas fa-inbox"></i><p>No scans yet</p></div>';
        return;
    }
    historyList.innerHTML = '';
    history.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        const date = new Date(item.timestamp);
        const timeStr = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        const foodCount = item.detected_foods?.length || 0;
        historyItem.innerHTML = '<img src="' + (item.detection_image_path || item.image_path) + '" alt="Scan"><div class="history-item-info"><div style="font-weight: 600; font-size: 12px; color: var(--text-primary); margin-bottom: 2px;">' + foodCount + ' items</div><div>' + dateStr + ' at ' + timeStr + '</div><div style="color: var(--success);">Score: ' + (item.health_score || 0) + '</div></div>';
        historyItem.addEventListener('click', () => loadHistoryItem(item.id));
        historyList.appendChild(historyItem);
    });
}

async function loadHistoryItem(id) {
    try {
        const response = await fetch('/api/history/' + id);
        if (!response.ok) return;
        const data = await response.json();
        if (data.success) console.log('Loading history item:', data.detection);
    } catch (error) {
        console.error('Error loading history item:', error);
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) return;
        const data = await response.json();
        if (data.success) updateStats(data.stats);
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function updateStats(stats) {
    const updateEl = (id, val) => {
        document.querySelectorAll('#' + id).forEach(el => el.textContent = val);
    };
    updateEl('total-scans', stats.total_scans || 0);
    updateEl('foods-detected', stats.total_foods_detected || 0);
    updateEl('avg-health', stats.avg_health_score > 0 ? stats.avg_health_score : '--');
    updateEl('total-calories-tracked', stats.total_calories_tracked || 0);

    // Update analytics nutrition summary
    updateEl('analytics-total-calories', stats.total_calories_tracked || 0);
}

async function loadAnalyticsData() {
    try {
        // Load recent detections for the table and aggregate nutrition data
        const response = await fetch('/api/history?limit=20');
        if (!response.ok) return;
        const data = await response.json();

        if (data.success && data.history) {
            populateDetectionsTable(data.history);
            calculateAggregateNutrition(data.history);
        }
    } catch (error) {
        console.error('Error loading analytics data:', error);
    }
}

function calculateAggregateNutrition(history) {
    let totalProtein = 0;
    let totalCarbs = 0;
    let totalFat = 0;
    let micronutrientSet = new Set();

    history.forEach(item => {
        // Parse nutrition data if available
        if (item.nutrition_data) {
            const nutrition = typeof item.nutrition_data === 'string'
                ? JSON.parse(item.nutrition_data)
                : item.nutrition_data;

            totalProtein += nutrition.total_protein || 0;
            totalCarbs += nutrition.total_carbs || 0;
            totalFat += nutrition.total_fat || 0;

            // Collect micronutrients
            if (nutrition.foods) {
                nutrition.foods.forEach(food => {
                    if (food.vitamins) {
                        food.vitamins.forEach(v => micronutrientSet.add(v));
                    }
                    if (food.minerals) {
                        food.minerals.forEach(m => micronutrientSet.add(m));
                    }
                });
            }
        }
    });

    // Update analytics nutrition display
    const updateEl = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    updateEl('analytics-total-protein', Math.round(totalProtein));
    updateEl('analytics-total-carbs', Math.round(totalCarbs));
    updateEl('analytics-total-fat', Math.round(totalFat));

    // Update micronutrients if we have any
    if (micronutrientSet.size > 0) {
        const microTagsDiv = document.getElementById('micronutrients-tags');
        if (microTagsDiv) {
            microTagsDiv.innerHTML = '';
            Array.from(micronutrientSet).slice(0, 10).forEach(nutrient => {
                const tag = document.createElement('span');
                tag.className = 'micro-tag';
                tag.textContent = nutrient;
                microTagsDiv.appendChild(tag);
            });
        }
    }
}

function populateDetectionsTable(history) {
    const tableBody = document.getElementById('detections-table-body');
    if (!tableBody) return;

    if (!history || history.length === 0) {
        tableBody.innerHTML = '<tr class="empty-row"><td colspan="7"><div class="table-empty"><i class="fas fa-inbox"></i><p>No detection data available</p></div></td></tr>';
        return;
    }

    tableBody.innerHTML = '';

    history.forEach(item => {
        const row = document.createElement('tr');

        // Format date/time
        const date = new Date(item.timestamp);
        const dateStr = date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
        });
        const timeStr = date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });

        // Get food items
        const detectedFoods = item.detected_foods || [];
        const foodNames = detectedFoods.map(f => f.name).join(', ');

        // Get nutrition data
        let calories = item.total_calories || 0;
        let protein = item.total_protein || 0;
        let carbs = item.total_carbs || 0;
        let fat = item.total_fat || 0;
        let healthScore = item.health_score || 0;

        // Determine health score badge class
        let badgeClass = 'poor';
        if (healthScore >= 80) badgeClass = 'excellent';
        else if (healthScore >= 60) badgeClass = 'good';
        else if (healthScore >= 40) badgeClass = 'fair';

        row.innerHTML = `
            <td>${dateStr}<br><span style="font-size: 11px; color: var(--text-muted);">${timeStr}</span></td>
            <td><div style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${foodNames}">${foodNames || 'Unknown'}</div></td>
            <td>${calories}</td>
            <td>${Math.round(protein)}</td>
            <td>${Math.round(carbs)}</td>
            <td>${Math.round(fat)}</td>
            <td><span class="health-score-badge ${badgeClass}">${healthScore}/100</span></td>
        `;

        tableBody.appendChild(row);
    });
}

function showStatus(message, type = 'info') {
    const statusEl = document.getElementById('status-message');
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.style.color = { 'success': 'var(--success)', 'error': 'var(--danger)', 'info': 'var(--text-secondary)' }[type] || 'var(--text-secondary)';
}

// Nutrition Chart Functions
function initNutritionChart() {
    const ctx = document.getElementById('nutrition-chart');
    if (!ctx) return;

    // Destroy existing chart if it exists
    if (nutritionChart) {
        nutritionChart.destroy();
    }

    // Create the chart with black background and styled lines
    nutritionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Calories',
                    data: [],
                    borderColor: '#FF6B6B',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: '#FF6B6B',
                    pointBorderColor: '#000',
                    pointBorderWidth: 2,
                    fill: false
                },
                {
                    label: 'Protein (g)',
                    data: [],
                    borderColor: '#4ECDC4',
                    backgroundColor: 'rgba(78, 205, 196, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: '#4ECDC4',
                    pointBorderColor: '#000',
                    pointBorderWidth: 2,
                    fill: false
                },
                {
                    label: 'Carbs (g)',
                    data: [],
                    borderColor: '#FFE66D',
                    backgroundColor: 'rgba(255, 230, 109, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: '#FFE66D',
                    pointBorderColor: '#000',
                    pointBorderWidth: 2,
                    fill: false
                },
                {
                    label: 'Fat (g)',
                    data: [],
                    borderColor: '#FF9F1C',
                    backgroundColor: 'rgba(255, 159, 28, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: '#FF9F1C',
                    pointBorderColor: '#000',
                    pointBorderWidth: 2,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#444',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += Math.round(context.parsed.y);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#fff',
                        font: {
                            size: 11
                        }
                    },
                    border: {
                        display: false
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#fff',
                        font: {
                            size: 11
                        },
                        maxRotation: 45,
                        minRotation: 0
                    },
                    border: {
                        display: false
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });

    // Load initial data
    loadNutritionChartData();
}

async function loadNutritionChartData() {
    try {
        const response = await fetch('/api/history?limit=100');
        if (!response.ok) return;
        const data = await response.json();

        if (data.success && data.history) {
            allNutritionData = data.history;
            updateChartWithTimeframe(currentTimeframe);
        }
    } catch (error) {
        console.error('Error loading chart data:', error);
    }
}

function updateChartWithTimeframe(timeframe) {
    if (!nutritionChart || !allNutritionData.length) return;

    // Filter data based on timeframe
    const now = new Date();
    let filteredData = allNutritionData;

    if (timeframe === 'day') {
        const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        filteredData = allNutritionData.filter(item => new Date(item.timestamp) >= oneDayAgo);
    } else if (timeframe === 'week') {
        const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        filteredData = allNutritionData.filter(item => new Date(item.timestamp) >= oneWeekAgo);
    } else if (timeframe === 'month') {
        const oneMonthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        filteredData = allNutritionData.filter(item => new Date(item.timestamp) >= oneMonthAgo);
    }

    // Reverse to show oldest first (left to right)
    filteredData = filteredData.reverse();

    // Prepare chart data
    const labels = [];
    const caloriesData = [];
    const proteinData = [];
    const carbsData = [];
    const fatData = [];

    filteredData.forEach(item => {
        const date = new Date(item.timestamp);
        let label;

        if (timeframe === 'day') {
            label = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        } else if (timeframe === 'week') {
            label = date.toLocaleDateString('en-US', { weekday: 'short', hour: '2-digit' });
        } else {
            label = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        }

        labels.push(label);
        caloriesData.push(item.total_calories || 0);
        proteinData.push(item.total_protein || 0);
        carbsData.push(item.total_carbs || 0);
        fatData.push(item.total_fat || 0);
    });

    // Update chart
    nutritionChart.data.labels = labels;
    nutritionChart.data.datasets[0].data = caloriesData;
    nutritionChart.data.datasets[1].data = proteinData;
    nutritionChart.data.datasets[2].data = carbsData;
    nutritionChart.data.datasets[3].data = fatData;
    nutritionChart.update();
}
