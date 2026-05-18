const API_URL = '/api/status';
let currentPollInterval = null;

// Format seconds into HH:MM:SS
function formatDuration(seconds) {
    if (seconds <= 0) return "Ready";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    
    if (h > 0) {
        return `${h}h ${m}m ${s}s`;
    }
    return `${m}m ${s}s`;
}

// Format timestamp
function formatTime(timestamp) {
    const d = new Date(timestamp * 1000);
    return d.toLocaleTimeString();
}

async function fetchStatus() {
    try {
        const response = await fetch(API_URL);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        updateDashboard(data);
        updateSystemStatus(true);
    } catch (error) {
        console.error("Failed to fetch status:", error);
        updateSystemStatus(false);
    }
}

function updateSystemStatus(isOnline) {
    const dot = document.querySelector('.pulse-dot');
    const text = document.getElementById('server-status');
    
    if (isOnline) {
        dot.className = 'pulse-dot active';
        text.textContent = 'System Online';
    } else {
        dot.className = 'pulse-dot error';
        text.textContent = 'Connection Lost';
    }
}

function updateDashboard(data) {
    // Update Metrics
    document.getElementById('active-providers').textContent = data.providers_count || 0;
    document.getElementById('total-models').textContent = data.models_count || 0;
    
    const uptimeSecs = Math.floor(Date.now()/1000) - data.start_time;
    document.getElementById('uptime').textContent = formatDuration(uptimeSecs);

    // Update Claude Accounts
    const claudeData = data.claude || {};
    const accounts = claudeData.accounts || [];
    const cooldowns = claudeData.cooldowns || {};
    const currentIndex = claudeData.current_index;

    document.getElementById('claude-total').textContent = `${accounts.length} Loaded`;

    const container = document.getElementById('accounts-container');
    
    if (accounts.length === 0) {
        container.innerHTML = `<div class="loading-state">No Claude accounts configured.</div>`;
        return;
    }

    // Build the grid
    let html = '';
    const now = Math.floor(Date.now() / 1000);

    accounts.forEach((acc, index) => {
        const email = acc.email;
        const isCurrent = (index === currentIndex);
        
        let statusClass = 'status-active';
        let statusText = 'Available';
        let cooldownHtml = '';
        
        // Check cooldown
        if (cooldowns[email]) {
            const cooldownEnd = cooldowns[email];
            const remaining = Math.max(0, cooldownEnd - now);
            
            if (remaining > 0) {
                statusClass = 'status-cooldown';
                statusText = 'Rate Limited';
                cooldownHtml = `
                    <div class="detail-row">
                        <span class="detail-label">Cooldown ends in:</span>
                        <span class="detail-value cooldown-timer">${formatDuration(remaining)}</span>
                    </div>
                `;
            }
        }

        html += `
            <div class="account-card glass-panel ${statusClass}">
                ${isCurrent && statusClass === 'status-active' ? '<div class="current-indicator">Next Up</div>' : ''}
                <div class="account-header">
                    <span class="account-email">${email}</span>
                    <span class="account-status-badge">${statusText}</span>
                </div>
                <div class="account-details">
                    <div class="detail-row">
                        <span class="detail-label">Status:</span>
                        <span class="detail-value">${acc.status}</span>
                    </div>
                    ${cooldownHtml}
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// Initialize
function init() {
    fetchStatus();
    // Poll every 5 seconds
    currentPollInterval = setInterval(fetchStatus, 5000);
}

// Start app
document.addEventListener('DOMContentLoaded', init);
