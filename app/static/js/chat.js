/**
 * chat.js — Dia Conversational Assistant Frontend Controller
 * 
 * Handles floating chat widget and dedicated full-page chat interface,
 * message rendering, markdown formatting, suggestion chips, and API integration.
 */

(function () {
  'use strict';

  const WELCOME_MESSAGE = {
    role: 'assistant',
    text: "Hello! 👋 I'm Dia, your diabetes assistant. I can answer questions about diabetes risk, symptoms, diet plans, exercise, blood sugar and more.\n\nWhat would you like to know today?",
    suggestions: [
      "Am I at risk of diabetes?",
      "What should I eat?",
      "Best exercises for diabetes",
      "What are the symptoms?",
    ],
  };

  const QUICK_PROMPTS = [
    "What are the symptoms of diabetes?",
    "What should I eat for breakfast?",
    "How much should I exercise?",
    "What is a normal blood sugar level?",
    "How can I prevent diabetes?",
  ];

  const STORAGE_KEY = 'dia_chat_history_v1';

  // ── State ────────────────────────────────────────────────────────────────
  let messages = loadChatHistory();
  let isTyping = false;

  function loadChatHistory() {
    try {
      const saved = sessionStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
    } catch (e) {
      console.warn('Could not read chat history from sessionStorage', e);
    }
    return [{ ...WELCOME_MESSAGE }];
  }

  function saveChatHistory() {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch (e) {
      console.warn('Could not save chat history to sessionStorage', e);
    }
  }

  function clearHistory() {
    messages = [{ ...WELCOME_MESSAGE }];
    saveChatHistory();
    renderAllViews();
    resetTextarea(document.getElementById('dia-chat-input'), 'dia-chat-input', 'dia-chat-send');
    resetTextarea(document.getElementById('dia-fullpage-input'), 'dia-fullpage-input', 'dia-fullpage-send');
  }

  // ── Markdown / Rich text formatter ───────────────────────────────────────
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function formatRichText(rawText) {
    if (!rawText) return '';
    
    // Split into lines
    const lines = rawText.split('\n');
    const formattedLines = lines.map((line) => {
      const trimmed = line.trim();
      let isBullet = false;
      let lineContent = line;

      if (trimmed.startsWith('•') || trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
        isBullet = true;
        lineContent = trimmed.replace(/^[•\-\*]\s*/, '');
      }

      // Escape base HTML
      let html = escapeHtml(lineContent);

      // Bold: **text**
      html = html.replace(/\*\*([^*]+)\*\*/g, '<strong class="dia-bold">$1</strong>');

      // Italic: _text_
      html = html.replace(/_([^_]+)_/g, '<em class="dia-italic">$1</em>');

      if (isBullet) {
        return `<div class="dia-bullet-item"><span class="dia-bullet-point">•</span><span>${html}</span></div>`;
      }
      return html === '' ? '<div class="dia-spacer"></div>' : `<p class="dia-text-line">${html}</p>`;
    });

    return formattedLines.join('');
  }

  // ── DOM Helpers ──────────────────────────────────────────────────────────
  function createAvatarSvg(sizeClass = 'dia-avatar--sm') {
    const span = document.createElement('span');
    span.className = `dia-avatar ${sizeClass}`;
    span.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8"/>
      </svg>
    `;
    return span;
  }

  function createTypingIndicator() {
    const wrapper = document.createElement('div');
    wrapper.className = 'dia-message-row dia-message-row--assistant dia-typing-row';
    wrapper.id = 'dia-typing-indicator';

    const avatar = createAvatarSvg('dia-avatar--sm');
    const bubble = document.createElement('div');
    bubble.className = 'dia-bubble dia-bubble--assistant dia-bubble--typing';
    bubble.innerHTML = `
      <span class="dia-typing-dot"></span>
      <span class="dia-typing-dot" style="animation-delay: 0.2s"></span>
      <span class="dia-typing-dot" style="animation-delay: 0.4s"></span>
    `;

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    return wrapper;
  }

  function createMessageElement(msg, onSuggestionClick) {
    const isUser = msg.role === 'user';
    const row = document.createElement('div');
    row.className = `dia-message-row ${isUser ? 'dia-message-row--user' : 'dia-message-row--assistant'}`;

    if (isUser) {
      const bubble = document.createElement('div');
      bubble.className = 'dia-bubble dia-bubble--user';
      bubble.textContent = msg.text;
      row.appendChild(bubble);
    } else {
      const avatar = createAvatarSvg('dia-avatar--sm');
      const contentWrapper = document.createElement('div');
      contentWrapper.className = 'dia-assistant-content';

      const bubble = document.createElement('div');
      bubble.className = 'dia-bubble dia-bubble--assistant';
      bubble.innerHTML = formatRichText(msg.text);
      contentWrapper.appendChild(bubble);

      if (msg.suggestions && msg.suggestions.length > 0) {
        const chipsContainer = document.createElement('div');
        chipsContainer.className = 'dia-suggestion-chips';

        msg.suggestions.forEach((suggestion) => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'dia-suggestion-chip';
          btn.textContent = suggestion;
          btn.addEventListener('click', () => {
            if (!isTyping && onSuggestionClick) {
              onSuggestionClick(suggestion);
            }
          });
          chipsContainer.appendChild(btn);
        });

        contentWrapper.appendChild(chipsContainer);
      }

      row.appendChild(avatar);
      row.appendChild(contentWrapper);
    }

    return row;
  }

  // ── Render Views ─────────────────────────────────────────────────────────
  function renderMessagesForContainer(container, onSuggestionClick) {
    if (!container) return;
    container.innerHTML = '';

    messages.forEach((msg) => {
      container.appendChild(createMessageElement(msg, onSuggestionClick));
    });

    if (isTyping) {
      container.appendChild(createTypingIndicator());
    }

    // Scroll smoothly to bottom
    setTimeout(() => {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth',
      });
    }, 10);
  }

  function renderQuickPromptsForContainer(container, onPromptClick) {
    if (!container) return;
    container.innerHTML = '';

    QUICK_PROMPTS.forEach((promptText) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'dia-quick-prompt-btn';
      btn.textContent = promptText;
      btn.disabled = isTyping;
      btn.addEventListener('click', () => {
        if (!isTyping && onPromptClick) {
          onPromptClick(promptText);
        }
      });
      container.appendChild(btn);
    });
  }

  // ── API Communication ────────────────────────────────────────────────────
  async function sendMessage(rawText) {
    const text = (rawText || '').trim();
    if (!text || isTyping) return;

    // 1. Add user message
    messages.push({ role: 'user', text });
    saveChatHistory();
    isTyping = true;
    renderAllViews();

    // 2. Fetch API response
    try {
      const payload = {
        message: text,
        history: messages.slice(-6).map((m) => ({ role: m.role, content: m.text })),
      };

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server returned ${response.status}`);
      }

      const data = await response.json();
      messages.push({
        role: 'assistant',
        text: data.text || "Sorry, I couldn't process your question. Please try asking again.",
        suggestions: data.suggestions || [],
      });
    } catch (err) {
      console.error('Dia chat error:', err);
      messages.push({
        role: 'assistant',
        text: "I encountered a momentary connection issue. Please check your internet or try asking again in a moment.",
        suggestions: ["Am I at risk of diabetes?", "What should I eat?", "What are the symptoms?"],
      });
    } finally {
      isTyping = false;
      saveChatHistory();
      renderAllViews();
    }
  }

  function renderAllViews() {
    // Widget elements
    const widgetScroll = document.getElementById('dia-messages-scroll');
    const widgetQuick = document.getElementById('dia-quick-prompts');
    renderMessagesForContainer(widgetScroll, sendMessage);
    renderQuickPromptsForContainer(widgetQuick, sendMessage);

    // Fullpage elements
    const fullpageScroll = document.getElementById('dia-fullpage-messages');
    const fullpageQuick = document.getElementById('dia-fullpage-quick-prompts');
    renderMessagesForContainer(fullpageScroll, sendMessage);
    renderQuickPromptsForContainer(fullpageQuick, sendMessage);

    // Update input send button states
    updateInputState('dia-chat-input', 'dia-chat-send');
    updateInputState('dia-fullpage-input', 'dia-fullpage-send');
  }

  function updateInputState(inputId, sendBtnId) {
    const input = document.getElementById(inputId);
    const btn = document.getElementById(sendBtnId);
    if (input && btn) {
      const hasText = input.value.trim().length > 0;
      btn.disabled = !hasText || isTyping;
    }
  }

  function autoResizeTextarea(textarea) {
    if (!textarea) return;
    const maxHeight = 120;
    // Temporarily reset height to auto to recalculate scrollHeight accurately
    textarea.style.height = 'auto';
    const scrollH = textarea.scrollHeight;
    if (scrollH > maxHeight) {
      textarea.style.height = `${maxHeight}px`;
      textarea.style.overflowY = 'auto';
    } else {
      const targetH = Math.max(scrollH, 38);
      textarea.style.height = `${targetH}px`;
      textarea.style.overflowY = 'hidden';
    }
  }

  function resetTextarea(textarea, inputId, sendBtnId) {
    if (!textarea) return;
    textarea.value = '';
    textarea.style.height = '';
    textarea.style.overflowY = 'hidden';
    autoResizeTextarea(textarea);
    if (inputId && sendBtnId) {
      updateInputState(inputId, sendBtnId);
    }
  }

  function setupInputHandlers(inputId, sendBtnId, formId) {
    const input = document.getElementById(inputId);
    const form = document.getElementById(formId);

    if (!input || !form) return;

    // Initial resize calculation
    autoResizeTextarea(input);

    // Auto-resize textarea on typing or paste/cut
    input.addEventListener('input', () => {
      autoResizeTextarea(input);
      updateInputState(inputId, sendBtnId);
    });

    // Enter to submit (Shift+Enter for newline)
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const text = input.value.trim();
        if (text && !isTyping) {
          resetTextarea(input, inputId, sendBtnId);
          sendMessage(text);
        }
      }
    });

    // Form submission
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (text && !isTyping) {
        resetTextarea(input, inputId, sendBtnId);
        sendMessage(text);
      }
    });
  }

  // ── Floating Widget Toggle Controller ────────────────────────────────────
  function updateViewportHeight() {
    const chatWindow = document.getElementById('dia-chat-window');
    if (!chatWindow) return;

    if (window.innerWidth <= 640 && chatWindow.classList.contains('dia-chat-window--open')) {
      if (window.visualViewport) {
        const vpHeight = window.visualViewport.height;
        const vpTop = window.visualViewport.offsetTop;
        document.documentElement.style.setProperty('--dia-viewport-height', `${vpHeight}px`);
        document.documentElement.style.setProperty('--dia-viewport-top', `${vpTop}px`);
      }
    } else {
      document.documentElement.style.removeProperty('--dia-viewport-height');
      document.documentElement.style.removeProperty('--dia-viewport-top');
    }
  }

  function setupWidgetController() {
    const toggleBtn = document.getElementById('dia-chat-toggle');
    const chatWindow = document.getElementById('dia-chat-window');
    const closeBtn = document.getElementById('dia-chat-close');
    const clearBtn = document.getElementById('dia-clear-chat');
    const chatInput = document.getElementById('dia-chat-input');

    if (!toggleBtn || !chatWindow) return;

    function openWidget() {
      chatWindow.removeAttribute('hidden');
      chatWindow.classList.add('dia-chat-window--open');
      toggleBtn.setAttribute('aria-expanded', 'true');
      toggleBtn.classList.add('dia-chat-toggle--active');
      renderAllViews();
      updateViewportHeight();
      setTimeout(() => {
        if (chatInput) {
          chatInput.focus();
          autoResizeTextarea(chatInput);
        }
      }, 100);
    }

    function closeWidget() {
      chatWindow.classList.remove('dia-chat-window--open');
      toggleBtn.setAttribute('aria-expanded', 'false');
      toggleBtn.classList.remove('dia-chat-toggle--active');
      document.documentElement.style.removeProperty('--dia-viewport-height');
      document.documentElement.style.removeProperty('--dia-viewport-top');
      setTimeout(() => {
        if (!chatWindow.classList.contains('dia-chat-window--open')) {
          chatWindow.setAttribute('hidden', '');
        }
      }, 250);
    }

    toggleBtn.addEventListener('click', () => {
      const isOpen = chatWindow.classList.contains('dia-chat-window--open');
      if (isOpen) {
        closeWidget();
      } else {
        openWidget();
      }
    });

    if (closeBtn) {
      closeBtn.addEventListener('click', closeWidget);
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        clearHistory();
      });
    }

    // Escape key closes floating window
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && chatWindow.classList.contains('dia-chat-window--open')) {
        closeWidget();
        toggleBtn.focus();
      }
    });
  }

  function setupFullpageController() {
    const clearBtn = document.getElementById('dia-fullpage-clear');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        clearHistory();
      });
    }
  }

  function setupViewportListeners() {
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', updateViewportHeight);
      window.visualViewport.addEventListener('scroll', updateViewportHeight);
    }
    window.addEventListener('resize', updateViewportHeight);
  }

  // ── Initialization on DOM ready ──────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    setupWidgetController();
    setupFullpageController();
    setupViewportListeners();
    setupInputHandlers('dia-chat-input', 'dia-chat-send', 'dia-chat-form');
    setupInputHandlers('dia-fullpage-input', 'dia-fullpage-send', 'dia-fullpage-form');
    renderAllViews();
  });
})();
