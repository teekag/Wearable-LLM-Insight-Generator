/**
 * Main JavaScript for Wearable Insight Generator
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize UI components
    initializeUIComponents();
    
    // Set up event listeners
    setupEventListeners();
});

/**
 * Initialize UI components
 */
function initializeUIComponents() {
    // Toggle user menu
    const userMenuBtn = document.getElementById('userMenuBtn');
    const userMenu = document.getElementById('userMenu');
    
    if (userMenuBtn && userMenu) {
        userMenuBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            userMenu.classList.toggle('hidden');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!userMenuBtn.contains(e.target) && !userMenu.contains(e.target)) {
                userMenu.classList.add('hidden');
            }
        });
    }
    
    // Sync button functionality
    const syncBtn = document.getElementById('syncBtn');
    const syncModal = document.getElementById('syncModal');
    const cancelSyncBtn = document.getElementById('cancelSyncBtn');
    
    if (syncBtn && syncModal && cancelSyncBtn) {
        syncBtn.addEventListener('click', function() {
            syncModal.classList.remove('hidden');
            simulateSyncProgress();
        });
        
        cancelSyncBtn.addEventListener('click', function() {
            syncModal.classList.add('hidden');
        });
    }
}

/**
 * Set up event listeners for interactive elements
 */
function setupEventListeners() {
    // Chat functionality on dashboard
    const chatInput = document.getElementById('chatInput');
    const sendChatBtn = document.getElementById('sendChatBtn');
    
    if (chatInput && sendChatBtn) {
        sendChatBtn.addEventListener('click', function() {
            sendChatMessage();
        });
        
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });
    }
}

/**
 * Send chat message and handle response
 */
function sendChatMessage() {
    const chatInput = document.getElementById('chatInput');
    const chatContainer = document.getElementById('chatContainer');
    
    if (!chatInput || !chatContainer) return;
    
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Add user message to chat
    const userMessageHTML = `
        <div class="flex items-start justify-end">
            <div class="bg-gray-200 rounded-lg p-3 max-w-md">
                <p class="text-sm text-gray-800">${escapeHTML(message)}</p>
            </div>
            <div class="bg-gray-300 rounded-full p-2 ml-2 flex-shrink-0">
                <i class="fas fa-user text-gray-600"></i>
            </div>
        </div>
    `;
    
    chatContainer.querySelector('.flex-col').insertAdjacentHTML('beforeend', userMessageHTML);
    chatInput.value = '';
    
    // Simulate response (in a real app, this would be an API call)
    setTimeout(() => {
        simulateChatResponse();
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 1000);
    
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Simulate chat response from AI
 */
function simulateChatResponse() {
    const chatContainer = document.getElementById('chatContainer');
    if (!chatContainer) return;
    
    const responses = [
        "Based on your data, I recommend focusing on sleep quality tonight. Try to get at least 8 hours.",
        "Your recovery is improving! Keep up the balanced activity and good sleep habits.",
        "I notice your HRV tends to drop after high-intensity workouts. Consider adding an extra recovery day.",
        "Your sleep patterns show disruption around 2-3 AM. Limiting screen time before bed might help."
    ];
    
    const randomResponse = responses[Math.floor(Math.random() * responses.length)];
    
    const botMessageHTML = `
        <div class="flex items-start">
            <div class="bg-indigo-100 rounded-full p-2 mr-2 flex-shrink-0">
                <i class="fas fa-robot text-indigo-600"></i>
            </div>
            <div class="bg-indigo-100 rounded-lg p-3 max-w-md">
                <p class="text-sm text-gray-800">${randomResponse}</p>
            </div>
        </div>
    `;
    
    chatContainer.querySelector('.flex-col').insertAdjacentHTML('beforeend', botMessageHTML);
}

/**
 * Simulate device sync progress
 */
function simulateSyncProgress() {
    const progressBar = document.getElementById('syncProgress');
    const statusText = document.getElementById('syncStatus');
    
    if (!progressBar || !statusText) return;
    
    let width = 0;
    
    const statuses = [
        "Found Apple Watch nearby – syncing last 7 days of recovery data...",
        "Downloading heart rate data...",
        "Processing sleep metrics...",
        "Analyzing activity patterns...",
        "Generating insights...",
        "Finalizing sync..."
    ];
    
    let currentStatus = 0;
    statusText.textContent = statuses[currentStatus];
    
    const interval = setInterval(() => {
        if (width >= 100) {
            clearInterval(interval);
            document.getElementById('syncModal').classList.add('hidden');
            
            // Update the last synced time
            const deviceStatus = document.getElementById('deviceStatus');
            if (deviceStatus) {
                deviceStatus.querySelector('span:last-child').textContent = "Apple Watch Connected | Last synced just now";
            }
            
            // Show a success notification
            showNotification("Sync completed successfully!");
        } else {
            width += 2;
            progressBar.style.width = width + "%";
            
            // Update status text at certain points
            if (width === 20) {
                currentStatus = 1;
                statusText.textContent = statuses[currentStatus];
            } else if (width === 40) {
                currentStatus = 2;
                statusText.textContent = statuses[currentStatus];
            } else if (width === 60) {
                currentStatus = 3;
                statusText.textContent = statuses[currentStatus];
            } else if (width === 80) {
                currentStatus = 4;
                statusText.textContent = statuses[currentStatus];
            } else if (width === 90) {
                currentStatus = 5;
                statusText.textContent = statuses[currentStatus];
            }
        }
    }, 50);
}

/**
 * Show notification
 * @param {string} message - Notification message
 */
function showNotification(message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg transform transition-all duration-500 translate-x-full opacity-0 z-50';
    notification.textContent = message;
    document.body.appendChild(notification);
    
    // Show notification
    setTimeout(() => {
        notification.classList.remove('translate-x-full', 'opacity-0');
    }, 100);
    
    // Hide notification after 3 seconds
    setTimeout(() => {
        notification.classList.add('translate-x-full', 'opacity-0');
        setTimeout(() => {
            notification.remove();
        }, 500);
    }, 3000);
}

/**
 * Helper function to escape HTML
 * @param {string} str - String to escape
 * @returns {string} - Escaped string
 */
function escapeHTML(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

/**
 * Create placeholder images for visualizations
 * This would be replaced with actual chart rendering in a production app
 */
function createPlaceholderImages() {
    // This function would be used to create placeholder images for visualizations
    // In a real app, this would be replaced with actual chart rendering
}

/**
 * Handle device connection
 * @param {string} deviceType - Type of device to connect
 */
function connectDevice(deviceType) {
    showNotification(`Connecting to ${deviceType}...`);
    
    // Simulate connection process
    setTimeout(() => {
        showNotification(`${deviceType} connected successfully!`);
        
        // Update UI to show connected state
        const deviceStatus = document.getElementById('deviceStatus');
        if (deviceStatus) {
            deviceStatus.querySelector('span:last-child').textContent = `${deviceType} Connected | Last synced just now`;
        }
    }, 2000);
}

/**
 * Format date for display
 * @param {Date} date - Date to format
 * @returns {string} - Formatted date string
 */
function formatDate(date) {
    return new Date(date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
}

/**
 * Format time for display
 * @param {Date} date - Date to format
 * @returns {string} - Formatted time string
 */
function formatTime(date) {
    return new Date(date).toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
}
