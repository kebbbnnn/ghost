const API_STATUS_URL = '/api/status';
const API_ACCOUNTS_URL = '/api/accounts';

let currentPollInterval = null;
let eventSource = null;
let activeThreadId = null;
let attachedImages = []; // Array of base64 data URIs
let attachedVideo = null; // { file: File, thumbnail: string (base64), duration: string, name: string }

// ---------------------------------------------------------------------------
// 🛠️ Utilities
// ---------------------------------------------------------------------------
function formatDuration(seconds) {
    if (seconds <= 0) return "Ready";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    return `${m}m ${s}s`;
}

function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

const AVAILABLE_TOOLS = {
    calculator: {
        type: "function",
        function: {
            name: "calculator",
            description: "Perform mathematical evaluations (e.g. arithmetic, trigonometry, math expressions).",
            parameters: {
                type: "object",
                properties: {
                    expression: {
                        type: "string",
                        description: "The mathematical expression to evaluate, e.g. '125 * 37' or 'Math.sqrt(144)'"
                    }
                },
                required: ["expression"]
            }
        }
    },
    get_weather: {
        type: "function",
        function: {
            name: "get_weather",
            description: "Retrieve current weather report and temperature for a given city or coordinates.",
            parameters: {
                type: "object",
                properties: {
                    location: {
                        type: "string",
                        description: "City or region name, e.g. 'Tokyo' or 'Paris'"
                    },
                    unit: {
                        type: "string",
                        enum: ["celsius", "fahrenheit"],
                        description: "Temperature unit"
                    }
                },
                required: ["location"]
            }
        }
    },
    get_current_time: {
        type: "function",
        function: {
            name: "get_current_time",
            description: "Get current date, time, and timezone information.",
            parameters: {
                type: "object",
                properties: {
                    timezone: {
                        type: "string",
                        description: "Timezone name or 'local'"
                    }
                }
            }
        }
    },
    web_search: {
        type: "function",
        function: {
            name: "web_search",
            description: "Simulate searching the web for real-time information and reference facts.",
            parameters: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "Search keywords or question"
                    }
                },
                required: ["query"]
            }
        }
    }
};

function getActiveTools() {
    const tools = [];
    if (document.getElementById('tool-opt-calc')?.checked) tools.push(AVAILABLE_TOOLS.calculator);
    if (document.getElementById('tool-opt-weather')?.checked) tools.push(AVAILABLE_TOOLS.get_weather);
    if (document.getElementById('tool-opt-time')?.checked) tools.push(AVAILABLE_TOOLS.get_current_time);
    if (document.getElementById('tool-opt-search')?.checked) tools.push(AVAILABLE_TOOLS.web_search);
    return tools;
}

function parseMarkdown(text) {
    if (!text) return "";
    let escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    
    // Code blocks
    escaped = escaped.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang || 'txt'}">${code.trim()}</code></pre>`;
    });
    
    // Inline code
    escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Newlines
    escaped = escaped.replace(/\n/g, "<br>");
    
    return escaped;
}

// Write to console panel
function logToConsole(message, type = 'system') {
    const consoleOutput = document.getElementById('console-output');
    if (!consoleOutput) return;

    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    const timestamp = new Date().toLocaleTimeString();
    line.textContent = `[${timestamp}] ${message}`;
    
    consoleOutput.appendChild(line);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// ---------------------------------------------------------------------------
// 📊 System Monitor & Telemetry Subscriber
// ---------------------------------------------------------------------------
async function fetchStatus() {
    try {
        const response = await fetch(API_STATUS_URL);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();
        updateSystemStatus(true);
        updateDashboard(data);
    } catch (error) {
        console.error("Failed to fetch system status:", error);
        updateSystemStatus(false);
    }
}

function updateSystemStatus(isOnline) {
    const dot = document.querySelector('.pulse-dot');
    const text = document.getElementById('server-status');
    if (!dot || !text) return;

    if (isOnline) {
        dot.className = 'pulse-dot active';
        text.textContent = 'System Online';
    } else {
        dot.className = 'pulse-dot error';
        text.textContent = 'Connection Lost';
        document.getElementById('active-providers').textContent = '-';
        document.getElementById('total-models').textContent = '-';
        document.getElementById('uptime').textContent = '-';
    }
}

function updateDashboard(data) {
    document.getElementById('active-providers').textContent = data.providers_count || 0;
    document.getElementById('total-models').textContent = data.models_count || 0;
    
    const uptimeSecs = Math.floor(Date.now() / 1000) - data.start_time;
    document.getElementById('uptime').textContent = formatDuration(uptimeSecs);
}



// Telemetry SSE listener
function startTelemetryStream() {
    if (eventSource) {
        eventSource.close();
    }
    
    eventSource = new EventSource('/api/events');

    eventSource.addEventListener('request_start', (e) => {
        const data = JSON.parse(e.data);
        logToConsole(`[Request Start] ID: ${data.id.slice(-6)} | Model: ${data.model} | Stream: ${data.stream}`, 'request');
        triggerRoutingAnimation(data.model, true);
    });

    eventSource.addEventListener('request_token', (e) => {
        const data = JSON.parse(e.data);
        // Quick visual pulse
        const proxyNode = document.getElementById('node-proxy');
        if (proxyNode) {
            proxyNode.classList.add('active');
            setTimeout(() => proxyNode.classList.remove('active'), 150);
        }
    });

    eventSource.addEventListener('request_end', (e) => {
        const data = JSON.parse(e.data);
        logToConsole(`[Request Complete] ID: ${data.id.slice(-6)}`, 'response');
        triggerRoutingAnimation(null, false);
    });

    eventSource.addEventListener('request_error', (e) => {
        const data = JSON.parse(e.data);
        logToConsole(`[Request Error] ID: ${data.id.slice(-6)} | Error: ${data.error}`, 'error');
        triggerRoutingAnimation(null, false);
    });

    eventSource.onerror = (err) => {
        console.error("Telemetry Stream encountered error. Reconnecting...");
        eventSource.close();
        setTimeout(startTelemetryStream, 5000);
    };
}

function triggerRoutingAnimation(model, active) {
    const clientLine = document.getElementById('line-client-proxy');
    const proxyNode = document.getElementById('node-proxy');
    
    const lines = {
        chatgpt: document.getElementById('line-proxy-chatgpt'),
        gemini: document.getElementById('line-proxy-gemini'),
        perplexity: document.getElementById('line-proxy-perplexity')
    };
    
    const nodes = {
        chatgpt: document.getElementById('node-chatgpt'),
        gemini: document.getElementById('node-gemini'),
        perplexity: document.getElementById('node-perplexity')
    };

    // Reset all
    clientLine.classList.remove('active');
    proxyNode.classList.remove('active');
    Object.values(lines).forEach(l => l && l.classList.remove('active'));
    Object.values(nodes).forEach(n => n && n.classList.remove('active'));

    if (active && model) {
        clientLine.classList.add('active');
        proxyNode.classList.add('active');
        
        const lowerModel = model.toLowerCase();
        if (lowerModel.startsWith('spectre')) {
            if (lines['chatgpt']) lines['chatgpt'].classList.add('active');
            if (nodes['chatgpt']) nodes['chatgpt'].classList.add('active');
            if (lines['gemini']) lines['gemini'].classList.add('active');
            if (nodes['gemini']) nodes['gemini'].classList.add('active');
        } else {
            let provider = 'chatgpt';
            if (lowerModel.startsWith('gemini')) {
                provider = 'gemini';
            } else if (lowerModel.startsWith('perplexity')) {
                provider = 'perplexity';
            }
            if (lines[provider]) lines[provider].classList.add('active');
            if (nodes[provider]) nodes[provider].classList.add('active');
        }
    }
}

// ---------------------------------------------------------------------------
// 💬 Chat Playground Logic (Local Caching)
// ---------------------------------------------------------------------------
function getThreads() {
    const data = localStorage.getItem('ghosttunnel_threads');
    return data ? JSON.parse(data) : [];
}

async function saveThreads(threads) {
    try {
        localStorage.setItem('ghosttunnel_threads', JSON.stringify(threads));
    } catch (e) {
        console.error("Failed to save threads to LocalStorage:", e);
        if (e.name === 'QuotaExceededError') {
            alert("LocalStorage quota exceeded. Please delete some old conversations to free up space.");
        }
    }
    try {
        await fetch('/api/threads', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(threads)
        });
    } catch (e) {
        console.error("Failed to save threads to backend persistence:", e);
    }
}

function createThread(model = 'auto') {
    const threads = getThreads();
    const newThread = {
        id: 'thread_' + Date.now(),
        model: model,
        title: `Chat with ${model}`,
        messages: [],
        conversationId: null,
        parentMessageId: null,
        created: Date.now()
    };
    threads.unshift(newThread);
    saveThreads(threads);
    selectThread(newThread.id);
    renderThreadsSidebar();
}

function deleteThread(id, event) {
    if (event) event.stopPropagation();
    const threads = getThreads();
    const filtered = threads.filter(t => t.id !== id);
    saveThreads(filtered);
    
    if (activeThreadId === id) {
        activeThreadId = filtered.length > 0 ? filtered[0].id : null;
    }
    
    renderThreadsSidebar();
    renderChatMessages();
}

function selectThread(id) {
    activeThreadId = id;
    renderThreadsSidebar();
    renderChatMessages();
}

function renderThreadsSidebar() {
    const list = document.getElementById('threads-list');
    if (!list) return;

    const threads = getThreads();
    if (threads.length === 0) {
        list.innerHTML = `<div style="text-align:center; padding: 2rem 0; color: var(--text-muted); font-size: 0.8rem;">No conversations.</div>`;
        return;
    }

    list.innerHTML = threads.map(t => `
        <button class="thread-item ${t.id === activeThreadId ? 'active' : ''}" onclick="selectThread('${t.id}')">
            <span class="thread-label">${t.title}</span>
            <span class="thread-delete" onclick="deleteThread('${t.id}', event)">🗑️</span>
        </button>
    `).join('');
}

function renderChatMessages() {
    const area = document.getElementById('chat-messages-area');
    const welcome = document.getElementById('empty-chat-welcome');
    if (!area || !welcome) return;

    const threads = getThreads();
    const current = threads.find(t => t.id === activeThreadId);

    if (!current || current.messages.length === 0) {
        area.innerHTML = '';
        welcome.style.display = 'flex';
        // Reset inputs
        attachedImages = [];
        attachedVideo = null;
        document.getElementById('attachment-previews-bar').style.display = 'none';
        document.getElementById('attachment-previews-bar').innerHTML = '';
        return;
    }

    welcome.style.display = 'none';

    // Populate active model dropdown to match thread selection
    const select = document.getElementById('playground-model');
    if (select && select.value !== current.model) {
        select.value = current.model;
    }

    let html = '';
    current.messages.forEach((msg, idx) => {
        if (msg.role === 'tool') {
            html += `
                <div class="message-bubble tool-result">
                    <div class="message-avatar">🔧</div>
                    <div class="message-content">
                        <div class="tool-result-header">
                            <span>Tool Output for <code>${escapeHtml(msg.tool_call_id || msg.name || 'tool')}</code></span>
                        </div>
                        <pre class="tool-result-body"><code>${escapeHtml(typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content, null, 2))}</code></pre>
                    </div>
                </div>
            `;
            return;
        }

        const isUser = msg.role === 'user';
        const name = isUser ? 'You' : 'AI';
        
        let imageHtml = '';
        let textContent = '';

        if (Array.isArray(msg.content)) {
            // Structured prompt parts
            msg.content.forEach(part => {
                if (part.type === 'text') {
                    textContent += part.text;
                } else if (part.type === 'image_url') {
                    imageHtml += `<img src="${part.image_url.url}" class="message-img-preview">`;
                }
            });
        } else {
            textContent = msg.content || '';
        }

        let toolCallsHtml = '';
        if (msg.tool_calls && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
            const subsequentToolOutputs = current.messages.slice(idx + 1).filter(m => m.role === 'tool');
            const executedCallIds = new Set(subsequentToolOutputs.map(m => m.tool_call_id));
            const pendingCalls = msg.tool_calls.filter(tc => !executedCallIds.has(tc.id));
            const isLatestAssistantWithPending = (idx === current.messages.length - 1 - subsequentToolOutputs.length) && pendingCalls.length > 0;

            toolCallsHtml = `<div class="tool-calls-container">`;
            if (isLatestAssistantWithPending && pendingCalls.length > 1) {
                toolCallsHtml += `
                    <div style="margin-bottom: 0.6rem;">
                        <button class="btn-execute-tool btn-execute-all" 
                                onclick="window.executeAllMockTools()">
                            ⚡ Execute All (${pendingCalls.length}) Pending Tools & Reply
                        </button>
                    </div>
                `;
            }
            msg.tool_calls.forEach(tc => {
                const callId = tc.id || `call_${Date.now()}`;
                const isExecuted = executedCallIds.has(callId);
                const funcName = tc.function?.name || 'unknown_tool';
                const argsStr = tc.function?.arguments || '{}';
                let formattedArgs = argsStr;
                try {
                    formattedArgs = JSON.stringify(JSON.parse(argsStr), null, 2);
                } catch (e) {}

                toolCallsHtml += `
                    <div class="tool-call-card ${isExecuted ? 'executed' : ''}">
                        <div class="tool-call-header">
                            <span class="tool-tag">🛠️ Tool Call</span>
                            <span class="tool-call-name">${escapeHtml(funcName)}</span>
                            <span class="tool-call-id">${escapeHtml(callId)}</span>
                            ${isExecuted ? '<span class="tool-tag" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3); margin-left: auto;">✓ Executed</span>' : ''}
                        </div>
                        <div class="tool-call-args">
                            <pre><code>${escapeHtml(formattedArgs)}</code></pre>
                        </div>
                        ${(!isExecuted && isLatestAssistantWithPending) ? `
                            <button class="btn-execute-tool" 
                                    data-call-id="${escapeHtml(callId)}" 
                                    data-tool-name="${escapeHtml(funcName)}" 
                                    data-args="${encodeURIComponent(argsStr)}"
                                    onclick="window.handleExecuteButtonClick(this)">
                                ⚡ ${pendingCalls.length > 1 ? 'Execute This Tool' : 'Execute & Reply to Agent'}
                            </button>
                        ` : ''}
                    </div>
                `;
            });
            toolCallsHtml += `</div>`;
        }

        html += `
            <div class="message-bubble ${isUser ? 'user' : 'assistant'}">
                <div class="message-avatar">${name[0]}</div>
                <div class="message-content">
                    ${imageHtml ? `<div class="message-images">${imageHtml}</div>` : ''}
                    ${textContent ? `<div>${parseMarkdown(textContent)}</div>` : ''}
                    ${toolCallsHtml}
                </div>
            </div>
        `;
    });

    area.innerHTML = html;
    area.scrollTop = area.scrollHeight;
}

// ---------------------------------------------------------------------------
// 🎬 Video Attachment Handling
// ---------------------------------------------------------------------------
function attachVideoFile(file) {
    // Clear any existing image attachments (mutual exclusion)
    attachedImages = [];

    // Create a temporary video element to extract thumbnail and duration
    const videoEl = document.createElement('video');
    videoEl.preload = 'metadata';
    videoEl.muted = true;

    const objectUrl = URL.createObjectURL(file);
    videoEl.src = objectUrl;

    videoEl.onloadedmetadata = () => {
        const duration = videoEl.duration;
        const durationStr = formatDuration(Math.floor(duration));

        // Seek to 1 second (or start) to get a representative thumbnail
        videoEl.currentTime = Math.min(1, duration / 2);
    };

    videoEl.onseeked = () => {
        // Draw the current frame to a canvas for thumbnail
        const canvas = document.createElement('canvas');
        const maxThumb = 320;
        let w = videoEl.videoWidth;
        let h = videoEl.videoHeight;

        if (w > maxThumb || h > maxThumb) {
            if (w > h) {
                h = Math.round((h * maxThumb) / w);
                w = maxThumb;
            } else {
                w = Math.round((w * maxThumb) / h);
                h = maxThumb;
            }
        }

        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoEl, 0, 0, w, h);

        const thumbnail = canvas.toDataURL('image/jpeg', 0.7);
        const durationStr = formatDuration(Math.floor(videoEl.duration));

        attachedVideo = {
            file: file,
            thumbnail: thumbnail,
            duration: durationStr,
            name: file.name
        };

        URL.revokeObjectURL(objectUrl);
        renderAttachmentPreviews();
    };

    videoEl.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        console.error('Failed to load video for thumbnail extraction');
        // Still attach without thumbnail
        attachedVideo = {
            file: file,
            thumbnail: null,
            duration: '?',
            name: file.name
        };
        renderAttachmentPreviews();
    };
}

function removeVideoAttachment() {
    attachedVideo = null;
    renderAttachmentPreviews();
}

// Image Loading and drag/drop
function addImageAttachment(base64Uri) {
    if (attachedImages.length >= 3) {
        alert("Maximum of 3 image attachments allowed per message.");
        return;
    }
    attachedImages.push(base64Uri);
    renderAttachmentPreviews();
}

function removeImageAttachment(index) {
    attachedImages.splice(index, 1);
    renderAttachmentPreviews();
}

function renderAttachmentPreviews() {
    const bar = document.getElementById('attachment-previews-bar');
    if (!bar) return;

    if (attachedImages.length === 0 && !attachedVideo) {
        bar.style.display = 'none';
        bar.innerHTML = '';
        return;
    }

    bar.style.display = 'flex';
    let html = '';

    // Video preview (if attached)
    if (attachedVideo) {
        const thumbStyle = attachedVideo.thumbnail
            ? `background-image: url('${attachedVideo.thumbnail}')`
            : `background: var(--bg-tertiary)`;
        html += `
            <div class="preview-thumbnail video-preview" style="${thumbStyle}">
                <div class="video-badge">🎬 ${attachedVideo.duration}</div>
                <div class="video-name">${attachedVideo.name}</div>
                <button class="remove-btn" onclick="removeVideoAttachment()">×</button>
            </div>
        `;
    }

    // Image previews
    html += attachedImages.map((uri, idx) => `
        <div class="preview-thumbnail" style="background-image: url('${uri}')">
            <button class="remove-btn" onclick="removeImageAttachment(${idx})">×</button>
        </div>
    `).join('');

    bar.innerHTML = html;
}

function handleFileSelect(files) {
    for (let file of files) {
        if (file.type.startsWith('video/')) {
            // Video attachment (mutual exclusion with images)
            attachVideoFile(file);
            return; // Only one video at a time
        }
        if (!file.type.startsWith('image/')) continue;

        // Image attachment — clear any video first (mutual exclusion)
        if (attachedVideo) {
            attachedVideo = null;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const maxDim = 1024;
                let width = img.width;
                let height = img.height;
                
                if (width > maxDim || height > maxDim) {
                    if (width > height) {
                        height = Math.round((height * maxDim) / width);
                        width = maxDim;
                    } else {
                        width = Math.round((width * maxDim) / height);
                        height = maxDim;
                    }
                }
                
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);
                
                try {
                    const compressedBase64 = canvas.toDataURL('image/jpeg', 0.85);
                    addImageAttachment(compressedBase64);
                } catch (err) {
                    console.error("Failed to compress image:", err);
                    addImageAttachment(e.target.result);
                }
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
}

// Send local API completion request
async function sendMessage(isFollowup = false) {
    const textarea = document.getElementById('chat-input');
    const text = textarea ? textarea.value.trim() : '';

    let threads = getThreads();
    let current = threads.find(t => t.id === activeThreadId);
    if (!current) {
        const selectModel = document.getElementById('playground-model').value;
        const newThread = {
            id: 'thread_' + Date.now(),
            model: selectModel,
            title: text.length > 25 ? text.slice(0, 22) + '...' : text || 'New Chat',
            messages: [],
            conversationId: null,
            parentMessageId: null,
            created: Date.now()
        };
        threads.unshift(newThread);
        activeThreadId = newThread.id;
        current = newThread;
        saveThreads(threads);
        renderThreadsSidebar();
    }

    const selectModel = document.getElementById('playground-model').value;
    current.model = selectModel;

    if (!isFollowup) {
        window._autoExecTurnCount = 0;
        if (!text && attachedImages.length === 0 && !attachedVideo) return;

        // --- VIDEO PATH ---
        if (attachedVideo) {
            const displayContent = `\uD83C\uDFAC [Video: ${attachedVideo.name}]\n\n${text}`;
            const userMsg = { role: 'user', content: displayContent };
            current.messages.push(userMsg);

            if (current.messages.length === 1 && text) {
                current.title = text.length > 25 ? text.slice(0, 22) + '...' : text;
            }

            const videoFile = attachedVideo.file;
            textarea.value = '';
            attachedVideo = null;
            attachedImages = [];
            renderAttachmentPreviews();
            saveThreads(threads);
            renderThreadsSidebar();
            renderChatMessages();

            const area = document.getElementById('chat-messages-area');
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble assistant';
            bubble.innerHTML = `
                <div class="message-avatar">A</div>
                <div class="message-content">
                    <div class="video-processing-indicator">
                        <div class="processing-spinner"></div>
                        <span>Processing video frames...</span>
                    </div>
                </div>
            `;
            area.appendChild(bubble);
            area.scrollTop = area.scrollHeight;

            const bubbleContent = bubble.querySelector('.message-content');
            let fullResponse = '';

            try {
                const formData = new FormData();
                formData.append('video', videoFile);
                formData.append('prompt', text);
                formData.append('model', selectModel);
                if (current.conversationId) formData.append('conversation_id', current.conversationId);
                if (current.parentMessageId) formData.append('parent_message_id', current.parentMessageId);
                const priorMessages = current.messages.slice(0, -1);
                formData.append('history', JSON.stringify(priorMessages));

                const response = await fetch('/api/video/chat', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errText = await response.text();
                    throw new Error(`HTTP ${response.status}: ${errText}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');

                    for (let line of lines) {
                        line = line.trim();
                        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.conversation_id) current.conversationId = data.conversation_id;
                                if (data.parent_message_id) current.parentMessageId = data.parent_message_id;
                                const contentChunk = data.choices?.[0]?.delta?.content;
                                if (contentChunk) {
                                    fullResponse += contentChunk;
                                    bubbleContent.innerHTML = parseMarkdown(fullResponse);
                                    area.scrollTop = area.scrollHeight;
                                }
                            } catch (err) {
                                // ignore
                            }
                        }
                    }
                }

                if (fullResponse) {
                    current.messages.push({ role: 'assistant', content: fullResponse });
                    saveThreads(threads);
                    renderChatMessages();
                } else {
                    bubble.remove();
                }
            } catch (e) {
                console.error(e);
                bubbleContent.innerHTML = `<span style="color: var(--danger-color); font-weight: 500;">Video Processing Failed: ${e.message}</span>`;
                logToConsole(`Video chat failure: ${e.message}`, 'error');
            }
            return;
        }

        // --- NORMAL TEXT / MULTIMODAL PATH ---
        let content = text;
        if (attachedImages.length > 0) {
            content = [];
            attachedImages.forEach(uri => {
                content.push({ type: 'image_url', image_url: { url: uri } });
            });
            content.push({ type: 'text', text: text });
        }

        const userMsg = { role: 'user', content: content };
        current.messages.push(userMsg);

        if (current.messages.length === 1 && text) {
            current.title = text.length > 25 ? text.slice(0, 22) + '...' : text;
        }

        textarea.value = '';
        attachedImages = [];
        renderAttachmentPreviews();
        saveThreads(threads);
        renderThreadsSidebar();
        renderChatMessages();
    }

    // --- DISPATCH CHAT COMPLETIONS (OPENAI OR ANTHROPIC) ---
    const isAnthropic = (getActiveProtocol() === 'anthropic');
    const endpoint = isAnthropic ? '/v1/messages' : '/v1/chat/completions';

    const payload = {
        model: selectModel,
        stream: true
    };
    if (current.conversationId) payload.conversation_id = current.conversationId;
    if (current.parentMessageId) payload.parent_message_id = current.parentMessageId;

    // Attach active parameters & tools
    const tempEl = document.getElementById('param-temperature');
    if (tempEl && tempEl.value !== "") {
        payload.temperature = parseFloat(tempEl.value);
    }
    const maxTokensEl = document.getElementById('param-max-tokens');
    if (maxTokensEl && maxTokensEl.value) {
        payload.max_tokens = parseInt(maxTokensEl.value, 10);
    }
    const stopEl = document.getElementById('param-stop');
    if (stopEl && stopEl.value.trim()) {
        payload.stop = stopEl.value.split(',').map(s => s.trim()).filter(Boolean);
    }

    const activeTools = getActiveTools();
    if (activeTools.length > 0) {
        payload.tools = activeTools;
        const toolChoiceEl = document.getElementById('param-tool-choice');
        if (toolChoiceEl) {
            payload.tool_choice = toolChoiceEl.value;
        }
    }

    // Build protocol-specific request body
    let requestBody;
    if (isAnthropic) {
        const { systemPrompt, anthropicMsgs } = formatMessagesForAnthropic(current.messages);
        requestBody = {
            model: selectModel,
            messages: anthropicMsgs,
            stream: true,
            max_tokens: payload.max_tokens || 2048,
        };
        if (current.conversationId) requestBody.conversation_id = current.conversationId;
        if (current.parentMessageId) requestBody.parent_message_id = current.parentMessageId;
        if (systemPrompt) requestBody.system = systemPrompt;
        if (payload.temperature !== undefined) requestBody.temperature = payload.temperature;
        if (payload.stop && payload.stop.length > 0) requestBody.stop_sequences = payload.stop;

        if (payload.tools && payload.tools.length > 0) {
            requestBody.tools = payload.tools.map(t => ({
                name: t.function.name,
                description: t.function.description,
                input_schema: t.function.parameters
            }));
            if (payload.tool_choice === 'required') {
                requestBody.tool_choice = { type: 'any' };
            } else if (payload.tool_choice === 'none') {
                requestBody.tool_choice = { type: 'none' };
            } else {
                requestBody.tool_choice = { type: 'auto' };
            }
        }
        logToConsole(`[Anthropic] POST /v1/messages | Model: ${selectModel} | MaxTokens: ${requestBody.max_tokens} | Temp: ${requestBody.temperature ?? 'default'}`, 'system');
    } else {
        payload.messages = current.messages;
        requestBody = payload;
        logToConsole(`[OpenAI] POST /v1/chat/completions | Model: ${selectModel} | Stream: true | Temp: ${payload.temperature ?? 'default'}`, 'system');
    }

    // Append streaming skeleton bubble
    const area = document.getElementById('chat-messages-area');
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble assistant';
    bubble.innerHTML = `
        <div class="message-avatar">A</div>
        <div class="message-content">
            <span class="pulse-dot active" style="display:inline-block; vertical-align:middle; width:6px; height:6px;"></span>
        </div>
    `;
    area.appendChild(bubble);
    area.scrollTop = area.scrollHeight;

    const bubbleContent = bubble.querySelector('.message-content');
    let fullResponse = '';
    let activeToolCalls = [];
    const blockIndexToToolCall = {};

    function updateLiveBubble() {
        let liveHtml = '';
        if (fullResponse) {
            liveHtml += parseMarkdown(fullResponse);
        }
        if (activeToolCalls.length > 0) {
            liveHtml += `<div class="tool-calls-container">`;
            activeToolCalls.forEach(tc => {
                let argsDisplay = tc.function?.arguments || 'Generating arguments...';
                try {
                    argsDisplay = JSON.stringify(JSON.parse(tc.function.arguments), null, 2);
                } catch(e) {}
                liveHtml += `
                    <div class="tool-call-card">
                        <div class="tool-call-header">
                            <span class="tool-tag">🛠️ Tool Call</span>
                            <span class="tool-call-name">${escapeHtml(tc.function?.name || '...')}</span>
                            <span class="tool-call-id">${escapeHtml(tc.id)}</span>
                        </div>
                        <div class="tool-call-args">
                            <pre><code>${escapeHtml(argsDisplay)}</code></pre>
                        </div>
                    </div>
                `;
            });
            liveHtml += `</div>`;
        }
        if (!liveHtml) {
            bubbleContent.innerHTML = `<span class="pulse-dot active" style="display:inline-block; vertical-align:middle; width:6px; height:6px;"></span>`;
        } else {
            bubbleContent.innerHTML = liveHtml;
        }
        area.scrollTop = area.scrollHeight;
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let streamError = null;
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (let line of lines) {
                line = line.trim();
                if (!line) continue;
                if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.conversation_id) current.conversationId = data.conversation_id;
                        if (data.parent_message_id) current.parentMessageId = data.parent_message_id;
                        
                        if (data.error) {
                            streamError = data.error.message || JSON.stringify(data.error);
                            logToConsole(`Upstream error: ${streamError}`, 'error');
                        }

                        // Anthropic SSE Events
                        if (isAnthropic || data.type) {
                            if (data.type === 'content_block_start') {
                                const cb = data.content_block;
                                if (cb && cb.type === 'tool_use') {
                                    const tcObj = {
                                        id: cb.id,
                                        type: 'function',
                                        function: { name: cb.name, arguments: '' }
                                    };
                                    if (data.index !== undefined) {
                                        blockIndexToToolCall[data.index] = tcObj;
                                    }
                                    activeToolCalls.push(tcObj);
                                    updateLiveBubble();
                                }
                            } else if (data.type === 'content_block_delta') {
                                const d = data.delta;
                                if (d && d.type === 'text_delta') {
                                    fullResponse += d.text;
                                    updateLiveBubble();
                                } else if (d && d.type === 'input_json_delta') {
                                    const target = (data.index !== undefined && blockIndexToToolCall[data.index])
                                        ? blockIndexToToolCall[data.index]
                                        : activeToolCalls[activeToolCalls.length - 1];
                                    if (target) {
                                        target.function.arguments += d.partial_json;
                                        updateLiveBubble();
                                    }
                                }
                            } else if (data.type === 'message_delta') {
                                const d = data.delta;
                                if (d?.stop_reason) {
                                    const stopInfo = d.stop_sequence ? ` ("${d.stop_sequence}")` : '';
                                    logToConsole(`[Anthropic] Stream stop: ${d.stop_reason}${stopInfo} | Output tokens: ${data.usage?.output_tokens || 'N/A'}`, 'system');
                                }
                            }
                        }

                        // OpenAI SSE Events
                        if (!isAnthropic && data.choices) {
                            if (data.conversation_id) current.conversationId = data.conversation_id;
                            if (data.parent_message_id) current.parentMessageId = data.parent_message_id;
                            
                            const delta = data.choices[0]?.delta;
                            if (delta) {
                                if (delta.content) {
                                    fullResponse += delta.content;
                                    updateLiveBubble();
                                }
                                if (delta.tool_calls && Array.isArray(delta.tool_calls)) {
                                    delta.tool_calls.forEach(tc => {
                                        const idx = tc.index ?? activeToolCalls.length;
                                        if (!activeToolCalls[idx]) {
                                            activeToolCalls[idx] = {
                                                id: tc.id || `call_${Date.now()}`,
                                                type: 'function',
                                                function: { name: tc.function?.name || '', arguments: '' }
                                            };
                                        }
                                        if (tc.function?.name) activeToolCalls[idx].function.name = tc.function.name;
                                        if (tc.function?.arguments) activeToolCalls[idx].function.arguments += tc.function.arguments;
                                    });
                                    updateLiveBubble();
                                }
                            }
                        }
                    } catch (err) {
                        // ignore malformed lines
                    }
                }
            }
        }

        // Save assistant message to thread or show error
        if (streamError) {
            bubbleContent.innerHTML = `<div style="padding: 10px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; color: var(--danger-color); font-size: 13px; line-height: 1.5;">
                <div style="font-weight: 600; margin-bottom: 4px;">⚠️ Provider Error</div>
                <div>${escapeHtml(streamError)}</div>
                <div style="margin-top: 8px; font-size: 11px; opacity: 0.85;">Tip: If using ChatGPT and getting a 403 device/activity flag, switch to Gemini or wait a few minutes for OpenAI's rate limiter to reset.</div>
            </div>`;
            current.messages.push({
                role: 'assistant',
                content: `⚠️ Provider Error: ${streamError}`
            });
            saveThreads(threads);
            logToConsole(`Stream error received: ${streamError}`, 'error');
        } else if (fullResponse || activeToolCalls.length > 0) {
            const assistantMsg = {
                role: 'assistant',
                content: fullResponse || null
            };
            if (activeToolCalls.length > 0) {
                assistantMsg.tool_calls = activeToolCalls;
            }
            current.messages.push(assistantMsg);
            saveThreads(threads);
            renderChatMessages();

            // Autonomous mock tool execution
            if (activeToolCalls.length > 0 && isAutoExecuteEnabled()) {
                if (window._autoExecTurnCount === undefined) window._autoExecTurnCount = 0;
                window._autoExecTurnCount++;
                if (window._autoExecTurnCount <= 5) {
                    logToConsole(`[Auto-Execute] Running ${activeToolCalls.length} tool call(s) autonomously (turn ${window._autoExecTurnCount}/5)...`, 'system');
                    setTimeout(() => {
                        window.executeAllMockTools();
                    }, 400);
                } else {
                    logToConsole('[Auto-Execute] Safety cap reached: 5 consecutive autonomous tool turns. Pausing for user review.', 'warning');
                    window._autoExecTurnCount = 0;
                }
            } else if (activeToolCalls.length === 0) {
                window._autoExecTurnCount = 0;
            }
        } else {
            bubbleContent.innerHTML = `<div style="padding: 10px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; color: var(--danger-color); font-size: 13px; line-height: 1.5;">
                <div style="font-weight: 600; margin-bottom: 4px;">⚠️ Empty Response</div>
                <div>The upstream model closed the connection without returning any content or tool calls.</div>
                <div style="margin-top: 8px; font-size: 11px; opacity: 0.85;">Tip: The provider may be temporarily throttling this IP. Switch to Gemini or wait a few minutes.</div>
            </div>`;
            logToConsole('Empty response received from provider.', 'warning');
        }
    } catch (e) {
        console.error(e);
        bubbleContent.innerHTML = `<div style="padding: 10px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; color: var(--danger-color); font-size: 13px; line-height: 1.5;">
            <div style="font-weight: 600; margin-bottom: 4px;">⚠️ Connection Failed</div>
            <div>${escapeHtml(e.message)}</div>
        </div>`;
        logToConsole(`Completions loop failure: ${e.message}`, 'error');
    }
}

// ---------------------------------------------------------------------------
// 🛠️ Interactive Tool Execution Bridge
// ---------------------------------------------------------------------------
window.handleExecuteButtonClick = function(btn) {
    const callId = btn.getAttribute('data-call-id');
    const toolName = btn.getAttribute('data-tool-name');
    const argsEncoded = btn.getAttribute('data-args');
    let argsStr = '{}';
    try {
        argsStr = decodeURIComponent(argsEncoded);
    } catch (e) {
        argsStr = argsEncoded;
    }
    window.executeMockTool(callId, toolName, argsStr);
};

function computeMockResult(toolName, argsRaw) {
    let args = {};
    if (typeof argsRaw === 'string') {
        try { args = JSON.parse(argsRaw); } catch (e) { args = { raw: argsRaw }; }
    } else if (typeof argsRaw === 'object' && argsRaw !== null) {
        args = argsRaw;
    }

    if (toolName === 'calculator') {
        const expr = String(args.expression || '0');
        try {
            const sanitized = expr.replace(/[^0-9+\-*/().%\s]/g, '');
            const calcVal = Function(`'use strict'; return (${sanitized})`)();
            return JSON.stringify({ expression: expr, result: calcVal });
        } catch (e) {
            return JSON.stringify({ error: `Evaluation error: ${e.message}` });
        }
    } else if (toolName === 'get_weather') {
        const loc = args.location || 'Tokyo';
        const unit = args.unit || 'celsius';
        return JSON.stringify({
            location: loc,
            temperature: unit === 'fahrenheit' ? "72°F" : "22°C",
            condition: "Partly Cloudy",
            humidity: "58%",
            forecast: "Clear skies with light breeze"
        });
    } else if (toolName === 'get_current_time') {
        return JSON.stringify({
            iso: new Date().toISOString(),
            local_time: new Date().toLocaleString(),
            timezone: args.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone
        });
    } else if (toolName === 'web_search') {
        const query = args.query || '';
        return JSON.stringify({
            query: query,
            results: [
                { title: `Overview: ${query}`, snippet: `Top verified facts and findings regarding '${query}' via GhostTunnel tool emulation.` },
                { title: `Documentation reference`, snippet: `Parameters and tool-calling verified operational in multi-turn mode.` }
            ]
        });
    } else {
        return JSON.stringify({ status: "success", executed: true, tool: toolName, arguments: args });
    }
}

window.executeAllMockTools = function() {
    const threads = getThreads();
    const current = threads.find(t => t.id === activeThreadId);
    if (!current || current.messages.length === 0) return;

    let lastAssistantIdx = -1;
    for (let i = current.messages.length - 1; i >= 0; i--) {
        if (current.messages[i].role === 'assistant' && Array.isArray(current.messages[i].tool_calls) && current.messages[i].tool_calls.length > 0) {
            lastAssistantIdx = i;
            break;
        }
    }
    if (lastAssistantIdx === -1) return;

    const lastAssistant = current.messages[lastAssistantIdx];
    const answeredIds = new Set(
        current.messages.slice(lastAssistantIdx + 1).filter(m => m.role === 'tool').map(m => m.tool_call_id)
    );

    let executedCount = 0;
    lastAssistant.tool_calls.forEach(tc => {
        const callId = tc.id || `call_${Date.now()}`;
        if (!answeredIds.has(callId)) {
            const toolName = tc.function?.name || 'unknown_tool';
            const argsStr = tc.function?.arguments || '{}';
            const result = computeMockResult(toolName, argsStr);

            current.messages.push({
                role: 'tool',
                tool_call_id: callId,
                name: toolName,
                content: result
            });
            executedCount++;
        }
    });

    if (executedCount > 0) {
        saveThreads(threads);
        renderChatMessages();
        sendMessage(true);
    }
};

window.executeMockTool = function(callId, toolName, argsRaw) {
    const threads = getThreads();
    const current = threads.find(t => t.id === activeThreadId);
    if (!current) return;

    const result = computeMockResult(toolName, argsRaw);

    // Append tool response message
    current.messages.push({
        role: 'tool',
        tool_call_id: callId,
        name: toolName,
        content: result
    });

    saveThreads(threads);
    renderChatMessages();

    // Check if there are any remaining tool calls from the last assistant message
    let lastAssistantIdx = -1;
    for (let i = current.messages.length - 1; i >= 0; i--) {
        if (current.messages[i].role === 'assistant' && Array.isArray(current.messages[i].tool_calls) && current.messages[i].tool_calls.length > 0) {
            lastAssistantIdx = i;
            break;
        }
    }

    if (lastAssistantIdx !== -1) {
        const lastAssistant = current.messages[lastAssistantIdx];
        const answeredIds = new Set(
            current.messages.slice(lastAssistantIdx + 1).filter(m => m.role === 'tool').map(m => m.tool_call_id)
        );
        const hasPending = lastAssistant.tool_calls.some(tc => !answeredIds.has(tc.id));
        if (hasPending) {
            // Still waiting for user to execute remaining tool calls
            return;
        }
    }

    // Trigger next turn so assistant responds to tool result!
    sendMessage(true);
};

function isAutoExecuteEnabled() {
    return localStorage.getItem('ghosttunnel_auto_exec_tools') === 'true';
}

// ---------------------------------------------------------------------------
// 🌐 API Protocol Switcher (OpenAI vs Anthropic)
// ---------------------------------------------------------------------------
function getActiveProtocol() {
    return localStorage.getItem('ghosttunnel_active_protocol') || 'openai';
}

function formatMessagesForAnthropic(rawMessages) {
    let systemPrompt = "";
    const anthropicMsgs = [];
    let pendingToolResults = [];

    function flushToolResults() {
        if (pendingToolResults.length > 0) {
            anthropicMsgs.push({
                role: 'user',
                content: pendingToolResults
            });
            pendingToolResults = [];
        }
    }

    for (const msg of rawMessages) {
        if (msg.role === 'system') {
            if (typeof msg.content === 'string') {
                systemPrompt += (systemPrompt ? "\n\n" : "") + msg.content;
            }
            continue;
        }

        if (msg.role === 'tool') {
            pendingToolResults.push({
                type: 'tool_result',
                tool_use_id: msg.tool_call_id || 'call_0',
                content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
            });
            continue;
        }

        flushToolResults();

        if (msg.role === 'assistant') {
            const contentBlocks = [];
            if (msg.content) {
                contentBlocks.push({ type: 'text', text: msg.content });
            }
            if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
                for (const tc of msg.tool_calls) {
                    let parsedArgs = {};
                    try {
                        parsedArgs = typeof tc.function?.arguments === 'string'
                            ? JSON.parse(tc.function.arguments)
                            : (tc.function?.arguments || {});
                    } catch (e) {
                        parsedArgs = { raw: tc.function?.arguments };
                    }
                    contentBlocks.push({
                        type: 'tool_use',
                        id: tc.id || `toolu_${Date.now()}`,
                        name: tc.function?.name || 'tool',
                        input: parsedArgs
                    });
                }
            }
            anthropicMsgs.push({
                role: 'assistant',
                content: contentBlocks.length === 1 && contentBlocks[0].type === 'text'
                    ? contentBlocks[0].text
                    : (contentBlocks.length > 0 ? contentBlocks : '')
            });
        } else if (msg.role === 'user') {
            anthropicMsgs.push({
                role: 'user',
                content: msg.content
            });
        }
    }

    flushToolResults();
    return { systemPrompt, anthropicMsgs };
}

function setupProtocolSwitcher() {
    const openaiBtn = document.getElementById('proto-openai-btn');
    const anthropicBtn = document.getElementById('proto-anthropic-btn');
    const badge = document.getElementById('param-active-protocol-badge');

    function updateProtocolUI(proto) {
        localStorage.setItem('ghosttunnel_active_protocol', proto);

        if (openaiBtn && anthropicBtn) {
            openaiBtn.classList.toggle('active', proto === 'openai');
            anthropicBtn.classList.toggle('active', proto === 'anthropic');
        }

        if (badge) {
            if (proto === 'anthropic') {
                badge.className = 'protocol-pill-badge anthropic';
                badge.textContent = '🟠 Anthropic Mode (/v1/messages)';
            } else {
                badge.className = 'protocol-pill-badge';
                badge.textContent = '🟢 OpenAI Mode (/v1/chat/completions)';
            }
        }
    }

    if (openaiBtn) {
        openaiBtn.addEventListener('click', () => {
            updateProtocolUI('openai');
            logToConsole('[Protocol Switch] Switched to OpenAI Chat Completions (POST /v1/chat/completions)', 'system');
        });
    }

    if (anthropicBtn) {
        anthropicBtn.addEventListener('click', () => {
            updateProtocolUI('anthropic');
            logToConsole('[Protocol Switch] Switched to Anthropic Messages API (POST /v1/messages)', 'system');
        });
    }

    updateProtocolUI(getActiveProtocol());
}

function setupParametersDrawer() {
    const toggleBtn = document.getElementById('params-toggle-btn');
    const panel = document.getElementById('chat-params-panel');
    const badge = document.getElementById('params-badge');
    const tempSlider = document.getElementById('param-temperature');
    const tempIndicator = document.getElementById('temp-indicator');
    const maxTokensInput = document.getElementById('param-max-tokens');
    const stopInput = document.getElementById('param-stop');
    const toolCheckboxes = document.querySelectorAll('.tool-chip input');
    const autoExecParamCb = document.getElementById('param-auto-execute-tools');
    const autoExecSettingsCb = document.getElementById('settings-auto-execute-tools');

    function updateBadge() {
        const activeTools = getActiveTools();
        const hasTools = activeTools.length > 0;
        const temp = tempSlider ? parseFloat(tempSlider.value) : 0.7;
        const isNonDefaultTemp = temp !== 0.7;
        const hasMaxTokens = Boolean(maxTokensInput?.value);
        const hasStop = Boolean(stopInput?.value?.trim());
        const autoExec = isAutoExecuteEnabled();

        const activeCount = (hasTools ? activeTools.length : 0) + (isNonDefaultTemp ? 1 : 0) + (hasMaxTokens ? 1 : 0) + (hasStop ? 1 : 0) + (autoExec ? 1 : 0);
        if (activeCount > 0) {
            badge.textContent = `${activeCount} custom`;
            badge.style.background = 'rgba(0, 242, 254, 0.2)';
            badge.style.color = 'var(--accent-cyan)';
        } else {
            badge.textContent = 'Default';
            badge.style.background = 'rgba(255, 255, 255, 0.08)';
            badge.style.color = 'var(--text-muted)';
        }
    }

    // Initialize auto-execute checkbox states from localStorage
    const initialAutoExec = isAutoExecuteEnabled();
    if (autoExecParamCb) autoExecParamCb.checked = initialAutoExec;
    if (autoExecSettingsCb) autoExecSettingsCb.checked = initialAutoExec;

    function handleAutoExecChange(e) {
        const checked = Boolean(e.target.checked);
        localStorage.setItem('ghosttunnel_auto_exec_tools', checked ? 'true' : 'false');
        if (autoExecParamCb) autoExecParamCb.checked = checked;
        if (autoExecSettingsCb) autoExecSettingsCb.checked = checked;
        updateBadge();
        logToConsole(`[Settings] Autonomous tool execution ${checked ? 'enabled' : 'disabled'}.`, 'system');
    }

    if (autoExecParamCb) autoExecParamCb.addEventListener('change', handleAutoExecChange);
    if (autoExecSettingsCb) autoExecSettingsCb.addEventListener('change', handleAutoExecChange);

    if (toggleBtn && panel) {
        toggleBtn.addEventListener('click', () => {
            const isHidden = panel.style.display === 'none';
            panel.style.display = isHidden ? 'flex' : 'none';
            toggleBtn.classList.toggle('active', isHidden);
        });
    }

    if (tempSlider && tempIndicator) {
        tempSlider.addEventListener('input', () => {
            const val = parseFloat(tempSlider.value);
            let label = "Balanced";
            if (val <= 0.3) label = "Deterministic / Precise";
            else if (val >= 0.8) label = "Creative / Explorative";
            tempIndicator.textContent = `${val.toFixed(2)} (${label})`;
            updateBadge();
        });
    }

    if (maxTokensInput) maxTokensInput.addEventListener('input', updateBadge);
    if (stopInput) stopInput.addEventListener('input', updateBadge);
    toolCheckboxes.forEach(cb => cb.addEventListener('change', updateBadge));
    updateBadge();
}

// ---------------------------------------------------------------------------
// 📂 Tab Switching & Bridge Initialization
// ---------------------------------------------------------------------------
function setupTabs() {
    const buttons = document.querySelectorAll('.nav-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const tabId = btn.getAttribute('data-tab');
            document.querySelectorAll('.tab-content').forEach(tc => {
                tc.classList.remove('active');
            });
            document.getElementById(tabId).classList.add('active');
            
            // Adjust chat scrollbar on activation
            if (tabId === 'chat-tab') {
                renderThreadsSidebar();
                renderChatMessages();
            }
        });
    });
}

async function init() {
    setupTabs();
    setupProtocolSwitcher();
    setupParametersDrawer();
    fetchStatus();
    startTelemetryStream();
    
    // Poll stats every 5s
    currentPollInterval = setInterval(fetchStatus, 5000);

    // Initialize Thread Caching from backend, fallback to LocalStorage
    let threads = [];
    try {
        const resp = await fetch('/api/threads');
        if (resp.ok) {
            threads = await resp.json();
            try {
                localStorage.setItem('ghosttunnel_threads', JSON.stringify(threads));
            } catch (se) {
                console.error("Failed to sync backend threads to LocalStorage:", se);
            }
        } else {
            threads = getThreads();
        }
    } catch (e) {
        console.error("Failed to fetch threads from backend:", e);
        threads = getThreads();
    }

    if (threads.length > 0) {
        activeThreadId = threads[0].id;
    }
    renderThreadsSidebar();
    renderChatMessages();

    // Event Bindings


    const clearConsoleBtn = document.getElementById('clear-console-btn');
    if (clearConsoleBtn) {
        clearConsoleBtn.addEventListener('click', () => {
            const consoleOutput = document.getElementById('console-output');
            if (consoleOutput) consoleOutput.innerHTML = '';
        });
    }

    const modelSelect = document.getElementById('playground-model');
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            const newModel = e.target.value;
            let threads = getThreads();
            let current = threads.find(t => t.id === activeThreadId);
            if (current) {
                current.model = newModel;
                if (current.title.startsWith("Chat with ")) {
                    current.title = `Chat with ${newModel}`;
                }
                saveThreads(threads);
                renderThreadsSidebar();
            }
        });
    }

    const newThreadBtn = document.getElementById('new-thread-btn');
    if (newThreadBtn) {
        newThreadBtn.addEventListener('click', () => {
            const currentModel = document.getElementById('playground-model').value;
            createThread(currentModel);
        });
    }

    const sendBtn = document.getElementById('send-chat-btn');
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }

    const chatTextarea = document.getElementById('chat-input');
    if (chatTextarea) {
        chatTextarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Auto resize height
        chatTextarea.addEventListener('input', () => {
            chatTextarea.style.height = 'auto';
            chatTextarea.style.height = (chatTextarea.scrollHeight) + 'px';
        });

        // Image Drag & Drop
        const chatWindow = document.querySelector('.chat-window-wrapper');
        chatWindow.addEventListener('dragover', (e) => {
            e.preventDefault();
            chatWindow.style.borderColor = 'var(--accent-cyan)';
        });

        chatWindow.addEventListener('dragleave', (e) => {
            e.preventDefault();
            chatWindow.style.borderColor = 'var(--border-color)';
        });

        chatWindow.addEventListener('drop', (e) => {
            e.preventDefault();
            chatWindow.style.borderColor = 'var(--border-color)';
            if (e.dataTransfer.files) {
                handleFileSelect(e.dataTransfer.files);
            }
        });
    }

    const fileBtn = document.getElementById('attach-file-btn');
    const fileInput = document.getElementById('hidden-file-input');
    if (fileBtn && fileInput) {
        fileBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files) {
                handleFileSelect(e.target.files);
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', init);
