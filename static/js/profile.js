// ============================================================================
// Profile Modal Functions
// ============================================================================

let currentUser = null;

// Load user profile on page load
async function loadUserProfile() {
    try {
        const response = await fetch('/api/user/profile');
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                currentUser = data.user;
                updateProfileUI(currentUser);
            }
        }
    } catch (error) {
        console.error('Error loading profile:', error);
    }
}

function updateProfileUI(user) {
    // Update profile picture
    const profilePic = document.getElementById('profile-picture');
    if (user.profile_picture) {
        profilePic.src = user.profile_picture;
    } else if (user.name) {
        const encodedName = encodeURIComponent(user.name);
        profilePic.src = 'https://ui-avatars.com/api/?name=' + encodedName + '&background=7C5CFF&color=fff';
    }

    // Populate profile form
    if (user.name) document.getElementById('profile-name').value = user.name || '';
    if (user.email) document.getElementById('profile-email').value = user.email || '';

    // Set weight in kg field (default view)
    if (user.weight_kg) {
        document.getElementById('profile-weight-kg').value = user.weight_kg || '';
    }

    // Set height in cm field (default view)
    if (user.height_cm) {
        document.getElementById('profile-height-cm').value = user.height_cm || '';
    }

    if (user.age) document.getElementById('profile-age').value = user.age || '';
    if (user.gender) document.getElementById('profile-gender').value = user.gender || '';
    if (user.activity_level) document.getElementById('profile-activity').value = user.activity_level || '';
    if (user.health_goals) document.getElementById('profile-goals').value = user.health_goals || '';
    if (user.dietary_restrictions) document.getElementById('profile-restrictions').value = user.dietary_restrictions || '';

    // Calculate and display BMI
    if (user.weight_kg && user.height_cm) {
        updateBMI();
    }
}

function getWeightInKg() {
    // Check which weight input is active
    const kgContainer = document.getElementById('weight-kg-container');
    const isKgActive = kgContainer.style.display !== 'none';

    if (isKgActive) {
        return parseFloat(document.getElementById('profile-weight-kg').value) || null;
    } else {
        // lbs mode - convert to kg
        const lbs = parseFloat(document.getElementById('profile-weight-lbs').value) || null;
        if (!lbs) return null;
        return lbs * 0.453592; // Convert lbs to kg
    }
}

function getHeightInCm() {
    // Check which height input is active
    const cmContainer = document.getElementById('height-cm-container');
    const isCmActive = cmContainer.style.display !== 'none';

    if (isCmActive) {
        return parseFloat(document.getElementById('profile-height-cm').value) || null;
    } else {
        // ft/in mode
        const feet = parseFloat(document.getElementById('profile-height-ft').value) || 0;
        const inches = parseFloat(document.getElementById('profile-height-in').value) || 0;

        if (feet === 0 && inches === 0) return null;

        // Convert feet and inches to cm: (feet * 12 + inches) * 2.54
        const totalInches = (feet * 12) + inches;
        return totalInches * 2.54;
    }
}

function updateBMI() {
    const weightKg = getWeightInKg();
    const heightCm = getHeightInCm();

    if (!weightKg || !heightCm) {
        document.getElementById('bmi-display').style.display = 'none';
        return;
    }

    const heightM = heightCm / 100;
    const bmi = (weightKg / (heightM * heightM)).toFixed(1);

    document.getElementById('bmi-value').textContent = bmi;
    document.getElementById('bmi-display').style.display = 'block';
}

// Profile modal handlers
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('user-profile-btn')?.addEventListener('click', () => {
        document.getElementById('profile-modal').style.display = 'flex';
        loadUserProfile();
    });

    document.getElementById('close-profile-modal')?.addEventListener('click', () => {
        document.getElementById('profile-modal').style.display = 'none';
    });

    document.getElementById('cancel-profile-btn')?.addEventListener('click', () => {
        document.getElementById('profile-modal').style.display = 'none';
    });

    // Close modal when clicking outside
    document.getElementById('profile-modal')?.addEventListener('click', (e) => {
        if (e.target.id === 'profile-modal') {
            document.getElementById('profile-modal').style.display = 'none';
        }
    });

    // Real-time BMI calculation - Weight inputs
    document.getElementById('profile-weight-kg')?.addEventListener('input', updateBMI);
    document.getElementById('profile-weight-lbs')?.addEventListener('input', updateBMI);

    // Real-time BMI calculation - Height inputs
    document.getElementById('profile-height-cm')?.addEventListener('input', updateBMI);
    document.getElementById('profile-height-ft')?.addEventListener('input', updateBMI);
    document.getElementById('profile-height-in')?.addEventListener('input', updateBMI);

    // Weight unit toggle handler
    document.querySelectorAll('.weight-unit-toggle .unit-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const unit = btn.getAttribute('data-unit');
            const kgContainer = document.getElementById('weight-kg-container');
            const lbsContainer = document.getElementById('weight-lbs-container');

            // Update active state
            document.querySelectorAll('.weight-unit-toggle .unit-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            if (unit === 'lbs') {
                // Switching to lbs
                const kg = parseFloat(document.getElementById('profile-weight-kg').value);

                if (kg && !isNaN(kg)) {
                    // Convert kg to lbs: kg * 2.20462
                    const lbs = (kg * 2.20462).toFixed(1);
                    document.getElementById('profile-weight-lbs').value = lbs;
                }

                kgContainer.style.display = 'none';
                lbsContainer.style.display = 'flex';
            } else {
                // Switching to kg
                const lbs = parseFloat(document.getElementById('profile-weight-lbs').value);

                if (lbs && !isNaN(lbs)) {
                    // Convert lbs to kg: lbs * 0.453592
                    const kg = (lbs * 0.453592).toFixed(1);
                    document.getElementById('profile-weight-kg').value = kg;
                }

                lbsContainer.style.display = 'none';
                kgContainer.style.display = 'flex';
            }

            // Recalculate BMI
            updateBMI();
        });
    });

    // Height unit toggle handler
    document.querySelectorAll('.height-unit-toggle .unit-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const unit = btn.getAttribute('data-unit');
            const cmContainer = document.getElementById('height-cm-container');
            const ftContainer = document.getElementById('height-ft-container');

            // Update active state
            document.querySelectorAll('.height-unit-toggle .unit-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            if (unit === 'ft') {
                // Switching to feet/inches
                const cm = parseFloat(document.getElementById('profile-height-cm').value);

                if (cm && !isNaN(cm)) {
                    // Convert cm to feet and inches
                    const totalInches = cm / 2.54;
                    const feet = Math.floor(totalInches / 12);
                    const inches = Math.round(totalInches % 12);

                    document.getElementById('profile-height-ft').value = feet;
                    document.getElementById('profile-height-in').value = inches;
                }

                cmContainer.style.display = 'none';
                ftContainer.style.display = 'flex';
            } else {
                // Switching to cm
                const feet = parseFloat(document.getElementById('profile-height-ft').value) || 0;
                const inches = parseFloat(document.getElementById('profile-height-in').value) || 0;

                if (feet > 0 || inches > 0) {
                    // Convert feet and inches to cm
                    const totalInches = (feet * 12) + inches;
                    const cm = (totalInches * 2.54).toFixed(1);

                    document.getElementById('profile-height-cm').value = cm;
                }

                ftContainer.style.display = 'none';
                cmContainer.style.display = 'flex';
            }

            // Recalculate BMI
            updateBMI();
        });
    });

    // Profile form submission
    document.getElementById('profile-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Get weight in kg and height in cm regardless of input mode
        const weightKg = getWeightInKg();
        const heightCm = getHeightInCm();

        const formData = {
            name: document.getElementById('profile-name').value,
            weight_kg: weightKg,
            height_cm: heightCm,
            age: parseInt(document.getElementById('profile-age').value) || null,
            gender: document.getElementById('profile-gender').value || null,
            activity_level: document.getElementById('profile-activity').value || null,
            health_goals: document.getElementById('profile-goals').value || null,
            dietary_restrictions: document.getElementById('profile-restrictions').value || null
        };

        try {
            const response = await fetch('/api/user/profile', {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            if (data.success) {
                currentUser = data.user;
                updateProfileUI(currentUser);
                document.getElementById('profile-modal').style.display = 'none';
                alert('Profile updated successfully!');
            } else {
                alert('Failed to update profile: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error updating profile:', error);
            alert('Network error. Please try again.');
        }
    });

    // Logout handler
    document.getElementById('logout-btn')?.addEventListener('click', async (e) => {
        e.preventDefault();

        try {
            const response = await fetch('/api/auth/logout', {method: 'POST'});
            if (response.ok) {
                window.location.href = '/login';
            }
        } catch (error) {
            console.error('Logout error:', error);
            alert('Failed to logout. Please try again.');
        }
    });

    // Load profile on page load
    loadUserProfile();
});
