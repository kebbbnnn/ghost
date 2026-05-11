// Paste this into the Chrome Console on chatgpt.com to intercept the sentinel token
const originalFetch = window.fetch;
window.fetch = async function(...args) {
    if (args[0] && (args[0].includes('/backend-anon/conversation') || args[0].includes('/backend-anon/sentinel/req'))) {
        const headers = args[1]?.headers;
        if (headers && headers['openai-sentinel-chat-requirements-token']) {
            const token = headers['openai-sentinel-chat-requirements-token'];
            try {
                const decoded = atob(token);
                const parsed = JSON.parse(decoded);
                console.log("== REAL T TOKEN ==");
                console.log(parsed.t);
                console.log("Length:", parsed.t.length);
            } catch (e) {
                console.error("Failed to parse token", e);
            }
        }
    }
    return originalFetch.apply(this, args);
};
console.log("Interceptor installed. Now send a message or refresh.");
