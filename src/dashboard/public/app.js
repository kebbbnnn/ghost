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
        gemini: document.getElementById('line-proxy-gemini')
    };
    
    const nodes = {
        chatgpt: document.getElementById('node-chatgpt'),
        gemini: document.getElementById('node-gemini')
    };

    // Reset all
    clientLine.classList.remove('active');
    proxyNode.classList.remove('active');
    Object.values(lines).forEach(l => l.classList.remove('active'));
    Object.values(nodes).forEach(n => n.classList.remove('active'));

    if (active && model) {
        clientLine.classList.add('active');
        proxyNode.classList.add('active');
        
        let provider = 'chatgpt';
        const lowerModel = model.toLowerCase();
        if (lowerModel.startsWith('gemini')) {
            provider = 'gemini';
        }

        if (lines[provider]) lines[provider].classList.add('active');
        if (nodes[provider]) nodes[provider].classList.add('active');
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
    current.messages.forEach(msg => {
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
            textContent = msg.content;
        }

        html += `
            <div class="message-bubble ${isUser ? 'user' : 'assistant'}">
                <div class="message-avatar">${name[0]}</div>
                <div class="message-content">
                    ${imageHtml ? `<div class="message-images">${imageHtml}</div>` : ''}
                    <div>${parseMarkdown(textContent)}</div>
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
async function sendMessage() {
    const textarea = document.getElementById('chat-input');
    const text = textarea.value.trim();
    if (!text && attachedImages.length === 0 && !attachedVideo) return;

    let threads = getThreads();
    let current = threads.find(t => t.id === activeThreadId);
    if (!current) {
        // Automatically create a thread if none exists
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

    // --- VIDEO PATH ---
    if (attachedVideo) {
        // Store user message with a video marker for display
        const displayContent = `\uD83C\uDFAC [Video: ${attachedVideo.name}]\n\n${text}`;
        const userMsg = { role: 'user', content: displayContent };
        current.messages.push(userMsg);

        if (current.messages.length === 1 && text) {
            current.title = text.length > 25 ? text.slice(0, 22) + '...' : text;
        }

        // Clear UI state
        const videoFile = attachedVideo.file;
        textarea.value = '';
        attachedVideo = null;
        attachedImages = [];
        renderAttachmentPreviews();
        saveThreads(threads);
        renderThreadsSidebar();
        renderChatMessages();

        // Append streaming skeleton bubble with video processing indicator
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
            // Build FormData for video upload
            const formData = new FormData();
            formData.append('video', videoFile);
            formData.append('prompt', text);
            formData.append('model', selectModel);
            if (current.conversationId) formData.append('conversation_id', current.conversationId);
            if (current.parentMessageId) formData.append('parent_message_id', current.parentMessageId);
            // Send prior messages as history (exclude the video message we just added)
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

            // Stream SSE response (identical to normal chat)
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
                            // ignore malformed lines
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

    // --- NORMAL IMAGE/TEXT PATH ---
    // Build standard message payload content
    let content = text;
    if (attachedImages.length > 0) {
        content = [];
        attachedImages.forEach(uri => {
            content.push({ type: 'image_url', image_url: { url: uri } });
        });
        content.push({ type: 'text', text: text });
    }

    // Append user message
    const userMsg = { role: 'user', content: content };
    current.messages.push(userMsg);
    
    // If it's the first message, change the title to a summary of the text
    if (current.messages.length === 1 && text) {
        current.title = text.length > 25 ? text.slice(0, 22) + '...' : text;
    }

    // Clear UI state
    textarea.value = '';
    attachedImages = [];
    renderAttachmentPreviews();
    saveThreads(threads);
    renderThreadsSidebar();
    renderChatMessages();

    // Build request body
    const payload = {
        model: selectModel,
        messages: current.messages,
        stream: true
    };
    if (current.conversationId) payload.conversation_id = current.conversationId;
    if (current.parentMessageId) payload.parent_message_id = current.parentMessageId;

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

    try {
        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
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
                        
                        // Capture thread conversation tracking values
                        if (data.conversation_id) current.conversationId = data.conversation_id;
                        if (data.parent_message_id) current.parentMessageId = data.parent_message_id;
                        
                        const contentChunk = data.choices?.[0]?.delta?.content;
                        if (contentChunk) {
                            fullResponse += contentChunk;
                            bubbleContent.innerHTML = parseMarkdown(fullResponse);
                            area.scrollTop = area.scrollHeight;
                        }
                    } catch (err) {
                        // ignore malformed lines
                    }
                }
            }
        }

        // Save assistant message to thread
        if (fullResponse) {
            current.messages.push({ role: 'assistant', content: fullResponse });
            saveThreads(threads);
            renderChatMessages();
        } else {
            bubble.remove();
        }
    } catch (e) {
        console.error(e);
        bubbleContent.innerHTML = `<span style="color: var(--danger-color); font-weight: 500;">Connection Failed: ${e.message}</span>`;
        logToConsole(`Completions loop failure: ${e.message}`, 'error');
    }
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
