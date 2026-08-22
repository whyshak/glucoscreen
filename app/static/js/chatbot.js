/**
 * chatbot.js — Dr. GlucoBot Floating RAG Chatbot Client Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    const chatWidget = document.getElementById("chat-widget");
    const chatTrigger = document.getElementById("chat-trigger");
    const chatContainer = document.getElementById("chat-container");
    const chatCloseBtn = document.getElementById("chat-close-btn");
    const chatResetBtn = document.getElementById("chat-reset-btn");
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");
    const chatSuggestions = document.getElementById("chat-suggestions");

    if (!chatTrigger || !chatContainer) return;

    let isChatOpen = false;

    // Toggle Chat Window
    chatTrigger.addEventListener("click", () => {
        toggleChat();
    });

    if (chatCloseBtn) {
        chatCloseBtn.addEventListener("click", () => {
            closeChat();
        });
    }

    if (chatResetBtn) {
        chatResetBtn.addEventListener("click", () => {
            resetChat();
        });
    }

    function toggleChat() {
        if (isChatOpen) {
            closeChat();
        } else {
            openChat();
        }
    }

    function openChat() {
        isChatOpen = true;
        chatContainer.classList.add("chat-container--active");
        chatTrigger.classList.add("chat-trigger--active");
        chatInput.focus();

        // Load initial suggested prompts if empty
        if (chatMessages.children.length <= 1) {
            fetchInitialSuggestions();
        }
    }

    function closeChat() {
        isChatOpen = false;
        chatContainer.classList.remove("chat-container--active");
        chatTrigger.classList.remove("chat-trigger--active");
    }

    function resetChat() {
        // Keep welcome message, remove rest
        while (chatMessages.children.length > 1) {
            chatMessages.removeChild(chatMessages.lastChild);
        }
        fetchInitialSuggestions();
    }

    // Fetch dynamic initial suggestion pills
    function fetchInitialSuggestions() {
        fetch("/api/chat/suggested")
            .then(res => res.json())
            .then(data => {
                if (data.suggested_questions) {
                    renderSuggestions(data.suggested_questions);
                }
            })
            .catch(() => {
                renderSuggestions([
                    "What are the main risk factors for diabetes?",
                    "How does GlucoScreen predict risk?",
                    "What foods lower blood sugar?"
                ]);
            });
    }

    // Handle Form Submit
    if (chatForm) {
        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const message = chatInput.value.trim();
            if (!message) return;

            sendMessage(message);
            chatInput.value = "";
        });
    }

    function sendMessage(messageText) {
        // Append User Message
        appendMessage("user", messageText);
        clearSuggestions();

        // Show Typing Indicator
        const typingId = showTypingIndicator();

        // Send to Backend API
        fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: messageText })
        })
        .then(res => {
            if (!res.ok) throw new Error("API request failed");
            return res.json();
        })
        .then(data => {
            removeTypingIndicator(typingId);
            appendMessage("bot", data.answer, data.sources);

            if (data.suggested_questions && data.suggested_questions.length > 0) {
                renderSuggestions(data.suggested_questions);
            }
        })
        .catch(err => {
            console.error(err);
            removeTypingIndicator(typingId);
            appendMessage("bot", "I am having trouble connecting right now. Please try asking again in a moment.");
        });
    }

    function appendMessage(sender, text, sources = []) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-msg chat-msg--${sender}`;

        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "chat-msg__bubble";

        // Simple Markdown Renderer (Bold, Lists, Horizontal Rules, Line breaks)
        let formattedText = formatMarkdown(text);
        bubbleDiv.innerHTML = formattedText;

        // Append Sources Badges if present
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement("div");
            sourcesDiv.className = "chat-msg__sources";
            sourcesDiv.innerHTML = `<span class="sources-label">Sources:</span> ` + 
                sources.map(s => `<span class="source-tag">${escapeHtml(s)}</span>`).join(" ");
            bubbleDiv.appendChild(sourcesDiv);
        }

        msgDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const id = "typing-" + Date.now();
        const typingDiv = document.createElement("div");
        typingDiv.className = "chat-msg chat-msg--bot";
        typingDiv.id = id;
        typingDiv.innerHTML = `
            <div class="chat-msg__bubble chat-msg__bubble--typing">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        scrollToBottom();
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function renderSuggestions(questions) {
        if (!chatSuggestions) return;
        chatSuggestions.innerHTML = "";
        
        const label = document.createElement("div");
        label.className = "suggestions-title";
        label.innerText = "Suggested Questions:";
        chatSuggestions.appendChild(label);

        const container = document.createElement("div");
        container.className = "suggestions-chips";

        questions.forEach(q => {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "suggestion-chip";
            chip.innerText = q;
            chip.addEventListener("click", () => {
                sendMessage(q);
            });
            container.appendChild(chip);
        });

        chatSuggestions.appendChild(container);
        scrollToBottom();
    }

    function clearSuggestions() {
        if (chatSuggestions) {
            chatSuggestions.innerHTML = "";
        }
    }

    function scrollToBottom() {
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 50);
    }

    function formatMarkdown(text) {
        let html = escapeHtml(text);
        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Italics
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Horizontal Rule
        html = html.replace(/---/g, '<hr class="chat-hr">');
        // Bullet Points
        html = html.replace(/^- (.*$)/gim, '• $1');
        // Linebreaks
        html = html.replace(/\n/g, '<br>');
        return html;
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }
});
