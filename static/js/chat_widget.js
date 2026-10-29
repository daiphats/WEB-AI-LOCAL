/**
 * ============================================================================
 * NEURO STATION - FLOATING CBT CLINICAL CHAT WIDGET
 * ============================================================================
 * Clean Vanilla ES6 Implementation - Zero Emojis - Isolated Namespace
 * Provides unified cross-tab clinical counseling for Tab 1, Tab 2, and Tab 3.
 * ============================================================================
 */

(function () {
  "use strict";

  const STORAGE_KEY = "cw_chat_history_v1";

  const DEFAULT_WELCOME_MSG = {
    sender: "bot",
    text: "[CLINICAL STATION READY: SESSION INITIALIZED]\n" +
      "Chào bạn. Tôi là Trợ lý Tham vấn Tâm lý Lâm sàng dựa trên Liệu pháp Nhận thức Hành vi (CBT).\n" +
      "Hệ thống cảm biến sinh trắc học đang theo dõi liên tục mức độ tiêu hao năng lượng tinh thần, trương lực cơ mặt và giọng nói của bạn để hỗ trợ điều hòa thần kinh theo thời gian thực.\n" +
      "Hôm nay bạn đang trải nghiệm những cảm xúc hoặc áp lực cụ thể nào?",
    time: formatCurrentTime()
  };

  const TAB_CHIPS = {
    "tab-monitor": [
      { label: "Trầm Cảm", query: "Trầm cảm là gì và cách vượt qua?" },
      { label: "Mất Ngủ", query: "Tôi bị mất ngủ cả tuần nay, làm sao để ngủ ngon?" },
      { label: "Overthinking", query: "Làm sao chữa chứng overthinking hay suy nghĩ quá mức?" },
      { label: "Hoảng Loạn", query: "Cơn hoảng loạn (Panic Attack) lúc tim đập thình thịch thì xử lý thế nào?" },
      { label: "Kiệt Sức", query: "Tôi đang bị kiệt sức nghề nghiệp (Burnout), tôi cần làm gì?" },
      { label: "Lo Âu", query: "Rối loạn lo âu lan tỏa (GAD) có chữa khỏi được không?" },
      { label: "+ Dạy Bác Sĩ", query: "Dạy bác sĩ: [Tên bệnh] : [Phác đồ điều trị của bạn]", teach: true }
    ],
    "tab-mimic": [
      { label: "Nụ Cười Duchenne", query: "Làm thế nào để cười nụ cười Duchenne chuẩn lâm sàng kích hoạt cơ gò má lớn?" },
      { label: "Thư Giãn Quai Hàm", query: "Làm sao để thả lỏng cơ cắn (Masseter) và giải tỏa căng cứng khớp thái dương hàm?" },
      { label: "Tập Trung Cao Độ", query: "Cách điều hòa trương lực cơ hạ mày để duy trì sự chú ý bền vững?" },
      { label: "Thở Chúm Môi", query: "Kỹ thuật thở chúm môi giải tỏa căng thẳng hoạt động theo cơ chế nào?" },
      { label: "+ Dạy Bác Sĩ", query: "Dạy bác sĩ: [Tên bài tập] : [Cơ chế vận động cơ mặt]", teach: true }
    ],
    "tab-forecast": [
      { label: "Bài Tập Chin Tuck", query: "Hướng dẫn thực hiện bài tập thu cằm Chin Tuck để chỉnh tư thế cổ rùa?" },
      { label: "Chỉ Số Tuổi Sinh Học", query: "Tuổi sinh học biểu cảm (Bio-Age) được tính toán dựa trên các cơ chế nào?" },
      { label: "Hội Chứng Slump", query: "Tư thế gập cổ về trước (Forward Head Slump) gây tổn thương thần kinh ra sao?" },
      { label: "rPPG Đo Nhịp Tim", query: "Công nghệ rPPG quang học qua camera đo nhịp tim từ biến thiên màu da như thế nào?" },
      { label: "+ Dạy Bác Sĩ", query: "Dạy bác sĩ: [Chủ đề sinh trắc] : [Phác đồ can thiệp]", teach: true }
    ]
  };

  const TAB_NAMES = {
    "tab-monitor": "Tab 1: Giám Sát",
    "tab-mimic": "Tab 2: Cơ Mặt",
    "tab-forecast": "Tab 3: Động Học"
  };

  function formatCurrentTime() {
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, "0");
    const mm = String(now.getMinutes()).padStart(2, "0");
    return `${hh}:${mm}`;
  }

  function escapeHTML(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  class FloatingChatWidget {
    constructor() {
      this.isOpen = false;
      this.isExpanded = false;
      this.unreadCount = 0;
      this.activeTabId = "tab-monitor";
      this.messages = [];
      this.isSending = false;
      this.lastNudgeTime = 0;
      this.nudgeTimer = null;
      this.nudgeEl = null;

      // DOM Elements
      this.launcher = null;
      this.panel = null;
      this.messagesContainer = null;
      this.chipsContainer = null;
      this.inputField = null;
      this.sendBtn = null;
      this.contextTabEl = null;
      this.miniTelemetryEl = null;
      this.telValEl = null;
      this.telArouEl = null;
      this.telExhaustEl = null;
      this.unreadBadge = null;
      this.expandBtn = null;
    }

    init() {
      // 1. Hide the static CBT chat card in Tab 1 (keeps DOM intact for main.js)
      this.hideLegacyChatCard();

      // 2. Load conversation history from sessionStorage
      this.loadHistory();

      // 3. Render Widget DOM
      this.renderWidgetDOM();

      // 4. Bind event listeners
      this.bindEvents();

      // 5. Setup Tab 1 Companion Card
      this.setupCompanionCard();

      // 6. Detect initial tab & start telemetry synchronization
      this.detectActiveTab();
      this.startTelemetrySync();

      // 7. Start Proactive Nudge Checker
      this.startProactiveNudgeChecker();

      // 8. Render current messages and chips
      this.renderMessages();
      this.renderChips();

      // 9. Expose global cbtWidget API
      this.exposeGlobalAPI();
    }

    hideLegacyChatCard() {
      const oldContainer = document.getElementById("chat-messages-container");
      if (oldContainer) {
        const oldCard = oldContainer.closest(".monitor-card");
        if (oldCard) {
          oldCard.style.display = "none";
        }
      }
    }

    loadHistory() {
      try {
        const stored = sessionStorage.getItem(STORAGE_KEY);
        if (stored) {
          const parsed = JSON.parse(stored);
          if (Array.isArray(parsed) && parsed.length > 0) {
            this.messages = parsed;
            return;
          }
        }
      } catch (e) {
        console.warn("[CHAT_WIDGET] Failed to parse sessionStorage history:", e);
      }
      this.messages = [Object.assign({}, DEFAULT_WELCOME_MSG)];
      this.saveHistory();
    }

    saveHistory() {
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(this.messages));
      } catch (e) {
        console.warn("[CHAT_WIDGET] Failed to save history to sessionStorage:", e);
      }
    }

    renderWidgetDOM() {
      // Create Launcher Button
      const launcher = document.createElement("button");
      launcher.type = "button";
      launcher.className = "cw-launcher";
      launcher.id = "cw-launcher-btn";
      launcher.setAttribute("aria-label", "Mở Trợ lý Tâm lý CBT");
      launcher.innerHTML = `
        <span class="cw-launcher-icon cw-icon-chat">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        </span>
        <span class="cw-launcher-icon cw-icon-close">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </span>
        <span class="cw-unread-badge" id="cw-unread-badge">0</span>
      `;

      // Create Panel
      const panel = document.createElement("div");
      panel.className = "cw-panel cw-closed";
      panel.id = "cw-chat-panel";
      panel.setAttribute("role", "dialog");
      panel.setAttribute("aria-label", "Hộp thoại Trợ lý Tâm lý CBT");
      panel.innerHTML = `
        <!-- Header -->
        <div class="cw-header">
          <div class="cw-header-left">
            <div class="cw-avatar">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
              </svg>
            </div>
            <div class="cw-header-title-box">
              <span class="cw-header-title">Trợ lý Tâm lý CBT</span>
              <span class="cw-header-subtitle">
                <span class="cw-status-dot"></span>
                OFFLINE REASONING
              </span>
            </div>
          </div>
          <div class="cw-header-actions">
            <button type="button" class="cw-icon-btn" id="cw-expand-btn" title="Phóng to kích thước (520x720)">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="15 3 21 3 21 9"></polyline>
                <polyline points="9 21 3 21 3 15"></polyline>
                <line x1="21" y1="3" x2="14" y2="10"></line>
                <line x1="3" y1="21" x2="10" y2="14"></line>
              </svg>
            </button>
            <button type="button" class="cw-icon-btn" id="cw-clear-btn" title="Xóa lịch sử hội thoại">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
            </button>
            <button type="button" class="cw-icon-btn" id="cw-close-btn" title="Thu nhỏ (Esc)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </button>
          </div>
        </div>

        <!-- Context Bar (Active Tab + Mini Telemetry Badges) -->
        <div class="cw-context-bar" id="cw-context-bar">
          <span class="cw-context-tab" id="cw-context-tab">TAB 1: GIÁM SÁT</span>
          <div class="cw-mini-telemetry" id="cw-mini-telemetry">
            <span class="cw-telemetry-badge cw-stat-green" id="cw-tel-val">VAL: +0.05</span>
            <span class="cw-telemetry-badge cw-stat-green" id="cw-tel-arou">AROU: +0.02</span>
            <span class="cw-telemetry-badge cw-stat-green" id="cw-tel-exhaust">KIỆT SỨC: 1/5</span>
          </div>
        </div>

        <!-- Messages History -->
        <div class="cw-messages" id="cw-messages"></div>

        <!-- Dynamic Suggestion Chips -->
        <div class="cw-chips" id="cw-chips"></div>

        <!-- Input Row -->
        <div class="cw-input-row">
          <input type="text" class="cw-input" id="cw-input-field"
            placeholder="Hỏi về bệnh lý, hoặc gõ 'Dạy bác sĩ: [Bệnh] : [Phác đồ]'..."
            autocomplete="off">
          <button type="button" class="cw-send-btn" id="cw-send-btn">
            [GỬI]
          </button>
        </div>
      `;

      document.body.appendChild(panel);
      document.body.appendChild(launcher);

      this.launcher = launcher;
      this.panel = panel;
      this.messagesContainer = panel.querySelector("#cw-messages");
      this.chipsContainer = panel.querySelector("#cw-chips");
      this.inputField = panel.querySelector("#cw-input-field");
      this.sendBtn = panel.querySelector("#cw-send-btn");
      this.contextTabEl = panel.querySelector("#cw-context-tab");
      this.miniTelemetryEl = panel.querySelector("#cw-mini-telemetry");
      this.telValEl = panel.querySelector("#cw-tel-val");
      this.telArouEl = panel.querySelector("#cw-tel-arou");
      this.telExhaustEl = panel.querySelector("#cw-tel-exhaust");
      this.unreadBadge = launcher.querySelector("#cw-unread-badge");
      this.expandBtn = panel.querySelector("#cw-expand-btn");
    }

    bindEvents() {
      // Toggle panel on launcher click
      this.launcher.addEventListener("click", () => this.togglePanel());

      // Expand / Compress toggle
      if (this.expandBtn) {
        this.expandBtn.addEventListener("click", () => this.toggleExpanded());
      }

      // Close panel on header close button
      const closeBtn = this.panel.querySelector("#cw-close-btn");
      if (closeBtn) {
        closeBtn.addEventListener("click", () => this.closePanel());
      }

      // Clear history on header trash button
      const clearBtn = this.panel.querySelector("#cw-clear-btn");
      if (clearBtn) {
        clearBtn.addEventListener("click", () => this.clearHistory());
      }

      // Close on Esc key
      window.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && this.isOpen) {
          this.closePanel();
        }
      });

      // Send on button click
      this.sendBtn.addEventListener("click", () => this.sendMessage());

      // Send on Enter key (without shift)
      this.inputField.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this.sendMessage();
        }
      });

      // Listen for tab switching on page
      const navTabs = document.querySelectorAll(".nav-tabs .tab-btn");
      navTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
          setTimeout(() => this.detectActiveTab(), 60);
        });
      });
    }

    toggleExpanded() {
      this.isExpanded = !this.isExpanded;
      if (this.isExpanded) {
        this.panel.classList.add("cw-expanded");
        if (this.expandBtn) {
          this.expandBtn.title = "Thu nhỏ kích thước (380x560)";
          this.expandBtn.innerHTML = `
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="4 14 10 14 10 20"></polyline>
              <polyline points="20 10 14 10 14 4"></polyline>
              <line x1="14" y1="10" x2="21" y2="3"></line>
              <line x1="3" y1="21" x2="10" y2="14"></line>
            </svg>
          `;
        }
      } else {
        this.panel.classList.remove("cw-expanded");
        if (this.expandBtn) {
          this.expandBtn.title = "Phóng to kích thước (520x720)";
          this.expandBtn.innerHTML = `
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="15 3 21 3 21 9"></polyline>
              <polyline points="9 21 3 21 3 15"></polyline>
              <line x1="21" y1="3" x2="14" y2="10"></line>
              <line x1="3" y1="21" x2="10" y2="14"></line>
            </svg>
          `;
        }
      }
    }

    exposeGlobalAPI() {
      const api = {
        open: () => this.openPanel(),
        close: () => this.closePanel(),
        toggle: () => this.togglePanel(),
        send: (text) => this.sendExternalMessage(text)
      };
      window.cbtWidget = api;
      window.cbtChatWidget = this;
    }

    sendExternalMessage(text) {
      if (!text) return;
      if (!this.isOpen) {
        this.openPanel();
      }
      this.inputField.value = text;
      this.sendMessage();
    }

    setupCompanionCard() {
      const openBtn = document.getElementById("cw-open-cbt-btn");
      if (openBtn) {
        openBtn.addEventListener("click", () => this.openPanel());
      }

      const topicBtns = document.querySelectorAll(".cw-topic-btn");
      topicBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
          const q = btn.getAttribute("data-query") || btn.textContent.trim();
          this.sendExternalMessage(q);
        });
      });

      this.updateCompanionCard();
    }

    updateCompanionCard() {
      const snapEmotion = document.getElementById("cw-snap-emotion");
      const snapVA = document.getElementById("cw-snap-va");
      const snapExhaust = document.getElementById("cw-snap-exhaustion");
      const snapPosture = document.getElementById("cw-snap-posture");
      const insightText = document.getElementById("cw-companion-insight-text");
      const statusBadge = document.getElementById("cw-companion-status");

      if (!snapEmotion && !insightText) return;

      let emotion = "NEUTRAL / CALM";
      let val = 0.05;
      let arou = 0.02;
      let exhaust = 1;
      let postureWarn = false;
      let inconf = false;

      if (window.clinicalStation && window.clinicalStation.currentBiometrics) {
        const b = window.clinicalStation.currentBiometrics;
        emotion = b.emotion || "NEUTRAL / CALM";
        val = Number(b.valence ?? 0.05);
        arou = Number(b.arousal ?? 0.02);
        exhaust = Number(b.mental_exhaustion_level ?? b.exhaustion_level ?? 1);
        postureWarn = Boolean(b.posture_warning);
        inconf = Boolean(b.incongruence_detected);
      }

      if (snapEmotion) {
        snapEmotion.textContent = String(emotion).toUpperCase();
      }

      if (snapVA) {
        const vStr = (val >= 0 ? "+" : "") + val.toFixed(2);
        const aStr = (arou >= 0 ? "+" : "") + arou.toFixed(2);
        snapVA.textContent = `${vStr} / ${aStr}`;
      }

      if (snapExhaust) {
        snapExhaust.textContent = `CẤP ${exhaust}/5`;
        snapExhaust.classList.remove("cw-val-normal", "cw-val-warning", "cw-val-alert");
        if (exhaust <= 2) {
          snapExhaust.classList.add("cw-val-normal");
        } else if (exhaust === 3) {
          snapExhaust.classList.add("cw-val-warning");
        } else {
          snapExhaust.classList.add("cw-val-alert");
        }
      }

      if (snapPosture) {
        snapPosture.classList.remove("cw-val-normal", "cw-val-alert");
        if (postureWarn) {
          snapPosture.textContent = "CẢNH BÁO (LỆCH TRỤC)";
          snapPosture.classList.add("cw-val-alert");
        } else {
          snapPosture.textContent = "BÌNH THƯỜNG";
          snapPosture.classList.add("cw-val-normal");
        }
      }

      if (insightText) {
        let suggestion = "";
        if (exhaust >= 3) {
          suggestion = `Phát hiện mức tiêu hao năng lượng tinh thần cao (Cấp ${exhaust}/5). Khuyến nghị tạm dừng tác vụ 5 phút, thực hiện kỹ thuật thở 4-7-8 để hạ nhịp tim.`;
        } else if (postureWarn) {
          suggestion = "Trục đầu - cổ đang lệch khỏi vị trí trung tính kéo dài. Khuyến nghị thực hiện bài tập thu cằm (Chin Tuck) 5 nhịp để giải phóng chèn ép đốt sống cổ.";
        } else if (inconf) {
          suggestion = "Phát hiện mâu thuẫn giữa vi biểu cảm cơ mặt và nội dung biểu đạt. Hãy thử chia sẻ chân thực cảm xúc hiện tại với bác sĩ CBT.";
        } else {
          suggestion = "Chỉ số sinh trắc học đang ở ngưỡng cân bằng tự nhiên. Bạn có thể duy trì nhịp làm việc hiện tại kết hợp bổ sung nước ấm.";
        }
        insightText.textContent = suggestion;
      }

      if (statusBadge) {
        statusBadge.textContent = "[SẴN SÀNG]";
      }
    }

    startProactiveNudgeChecker() {
      setInterval(() => {
        if (this.isOpen) return;
        const now = Date.now();
        if (now - this.lastNudgeTime < 120000) return; // 2 minutes cooldown

        let exhaust = 1;
        let postureWarn = false;
        if (window.clinicalStation && window.clinicalStation.currentBiometrics) {
          const b = window.clinicalStation.currentBiometrics;
          exhaust = Number(b.mental_exhaustion_level ?? b.exhaustion_level ?? 1);
          postureWarn = Boolean(b.posture_warning);
        }

        if (exhaust >= 3 || postureWarn) {
          this.lastNudgeTime = now;
          this.showProactiveNudge(postureWarn ? "posture" : "exhaustion", exhaust);
        }
      }, 3000);
    }

    showProactiveNudge(type, exhaust) {
      this.dismissNudge();

      this.unreadCount += 1;
      this.updateUnreadBadge();

      const bubble = document.createElement("div");
      bubble.className = "cw-nudge-bubble";
      bubble.id = "cw-nudge-bubble";

      const header = document.createElement("div");
      header.className = "cw-nudge-header";
      header.innerHTML = `
        <span>[NHẮC NHỞ LÂM SÀNG]</span>
        <button type="button" class="cw-nudge-close" aria-label="Đóng">&times;</button>
      `;

      const body = document.createElement("div");
      body.className = "cw-nudge-body";
      if (type === "posture") {
        body.textContent = "Cảnh báo trục đầu - cổ lệch vị trí trung tính kéo dài. Bấm để xem hướng dẫn bài tập Chin Tuck.";
      } else {
        body.textContent = `Phát hiện mức kiệt sức Cấp ${exhaust}/5. Bấm để trao đổi cùng trợ lý CBT và kích hoạt nhịp thở điều hòa.`;
      }

      bubble.appendChild(header);
      bubble.appendChild(body);

      bubble.addEventListener("click", () => {
        this.openPanel();
        this.dismissNudge();
      });

      const closeBtn = header.querySelector(".cw-nudge-close");
      if (closeBtn) {
        closeBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.dismissNudge();
        });
      }

      document.body.appendChild(bubble);
      this.nudgeEl = bubble;

      if (this.nudgeTimer) clearTimeout(this.nudgeTimer);
      this.nudgeTimer = setTimeout(() => this.dismissNudge(), 8000);
    }

    dismissNudge() {
      if (this.nudgeTimer) {
        clearTimeout(this.nudgeTimer);
        this.nudgeTimer = null;
      }
      if (this.nudgeEl) {
        const el = this.nudgeEl;
        this.nudgeEl = null;
        el.classList.add("cw-nudge-hiding");
        setTimeout(() => {
          if (el.parentNode) {
            el.parentNode.removeChild(el);
          }
        }, 260);
      }
    }

    togglePanel() {
      if (this.isOpen) {
        this.closePanel();
      } else {
        this.openPanel();
      }
    }

    openPanel() {
      this.isOpen = true;
      this.panel.classList.remove("cw-closed");
      this.panel.classList.add("cw-open");
      this.launcher.classList.add("cw-active");

      // Dismiss any open nudge bubble
      this.dismissNudge();

      // Reset unread badge
      this.unreadCount = 0;
      this.updateUnreadBadge();

      // Scroll messages to bottom and focus input
      this.scrollToBottom();
      setTimeout(() => {
        if (window.innerWidth > 520) {
          this.inputField.focus();
        }
      }, 150);
    }

    closePanel() {
      this.isOpen = false;
      this.panel.classList.remove("cw-open");
      this.panel.classList.add("cw-closed");
      this.launcher.classList.remove("cw-active");
    }

    updateUnreadBadge() {
      if (this.unreadCount > 0) {
        this.unreadBadge.textContent = this.unreadCount > 9 ? "9+" : String(this.unreadCount);
        this.unreadBadge.classList.add("cw-visible");
      } else {
        this.unreadBadge.textContent = "0";
        this.unreadBadge.classList.remove("cw-visible");
      }
    }

    detectActiveTab() {
      const activeTabBtn = document.querySelector(".nav-tabs .tab-btn.active");
      if (activeTabBtn) {
        const tabId = activeTabBtn.getAttribute("data-tab") || "tab-monitor";
        if (this.activeTabId !== tabId) {
          this.activeTabId = tabId;
          this.updateContextBar();
          this.renderChips();
        }
      }
    }

    startTelemetrySync() {
      const updateTelemetry = () => {
        this.updateContextBar();
        this.updateCompanionCard();
      };
      updateTelemetry();
      setInterval(updateTelemetry, 1500);
    }

    updateContextBar() {
      if (this.contextTabEl) {
        const name = TAB_NAMES[this.activeTabId] || "Tab: Lâm Sàng";
        this.contextTabEl.textContent = name;
      }

      let val = 0.05;
      let arou = 0.02;
      let exhaust = 1;

      if (window.clinicalStation && window.clinicalStation.currentBiometrics) {
        const b = window.clinicalStation.currentBiometrics;
        val = Number(b.valence ?? 0.05);
        arou = Number(b.arousal ?? 0.02);
        exhaust = Number(b.mental_exhaustion_level ?? b.exhaustion_level ?? 1);
      }

      if (this.telValEl) {
        const vStr = (val >= 0 ? "+" : "") + val.toFixed(2);
        this.telValEl.textContent = `VAL: ${vStr}`;
        this.telValEl.classList.remove("cw-stat-green", "cw-stat-amber", "cw-stat-red");
        if (val >= 0) {
          this.telValEl.classList.add("cw-stat-green");
        } else if (val >= -0.2) {
          this.telValEl.classList.add("cw-stat-amber");
        } else {
          this.telValEl.classList.add("cw-stat-red");
        }
      }

      if (this.telArouEl) {
        const aStr = (arou >= 0 ? "+" : "") + arou.toFixed(2);
        this.telArouEl.textContent = `AROU: ${aStr}`;
        this.telArouEl.classList.remove("cw-stat-green", "cw-stat-amber", "cw-stat-red");
        if (arou <= 0.3) {
          this.telArouEl.classList.add("cw-stat-green");
        } else if (arou <= 0.6) {
          this.telArouEl.classList.add("cw-stat-amber");
        } else {
          this.telArouEl.classList.add("cw-stat-red");
        }
      }

      if (this.telExhaustEl) {
        this.telExhaustEl.textContent = `KIỆT SỨC: ${exhaust}/5`;
        this.telExhaustEl.classList.remove("cw-stat-green", "cw-stat-amber", "cw-stat-red");
        if (exhaust <= 2) {
          this.telExhaustEl.classList.add("cw-stat-green");
        } else if (exhaust === 3) {
          this.telExhaustEl.classList.add("cw-stat-amber");
        } else {
          this.telExhaustEl.classList.add("cw-stat-red");
        }
      }
    }

    renderChips() {
      if (!this.chipsContainer) return;
      const chips = TAB_CHIPS[this.activeTabId] || TAB_CHIPS["tab-monitor"];

      this.chipsContainer.innerHTML = "";
      chips.forEach((c) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "cw-chip" + (c.teach ? " cw-chip-teach" : "");
        btn.textContent = c.label;
        btn.addEventListener("click", () => {
          this.inputField.value = c.query;
          this.inputField.focus();
          if (!c.teach) {
            this.sendMessage();
          }
        });
        this.chipsContainer.appendChild(btn);
      });
    }

    renderMessages() {
      if (!this.messagesContainer) return;
      this.messagesContainer.innerHTML = "";

      this.messages.forEach((msg) => {
        this.appendMessageElement(msg, false);
      });
      this.scrollToBottom();
    }

    appendMessageElement(msg, autoScroll = true, stream = false) {
      const isUser = msg.sender === "user";

      if (isUser) {
        const wrap = document.createElement("div");
        wrap.className = "cw-msg cw-msg-user";

        const bubble = document.createElement("div");
        bubble.className = "cw-bubble-user";
        bubble.textContent = msg.text;

        const timeEl = document.createElement("span");
        timeEl.className = "cw-msg-time";
        timeEl.textContent = msg.time || formatCurrentTime();

        wrap.appendChild(bubble);
        wrap.appendChild(timeEl);
        this.messagesContainer.appendChild(wrap);

        if (autoScroll) {
          this.scrollToBottom();
        }
        return;
      }

      // Bot Message with Avatar, Structured Sections, Typewriter, Copy, Chips
      const wrap = document.createElement("div");
      wrap.className = "cw-msg cw-msg-bot";

      const layout = document.createElement("div");
      layout.className = "cw-msg-bot-layout";

      // Small round bot avatar
      const avatar = document.createElement("div");
      avatar.className = "cw-bot-avatar";
      avatar.setAttribute("aria-hidden", "true");
      avatar.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"></path>
          <path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"></path>
          <circle cx="20" cy="10" r="2"></circle>
        </svg>
      `;

      const body = document.createElement("div");
      body.className = "cw-msg-bot-body";

      const bubble = document.createElement("div");
      bubble.className = "cw-bubble-bot";
      bubble.setAttribute("title", "Chạm hoặc bấm vào để hiện đầy đủ ngay lập tức");

      // Meta row (time + copy button)
      const metaRow = document.createElement("div");
      metaRow.className = "cw-msg-meta-row";

      const timeEl = document.createElement("span");
      timeEl.className = "cw-msg-time";
      timeEl.textContent = msg.time || formatCurrentTime();

      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "cw-copy-btn";
      copyBtn.setAttribute("aria-label", "Sao chép câu trả lời vào bộ nhớ tạm");
      copyBtn.innerHTML = `
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        <span>[SAO CHÉP]</span>
      `;

      copyBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        try {
          await navigator.clipboard.writeText(msg.text);
          const span = copyBtn.querySelector("span");
          if (span) span.textContent = "[ĐÃ CHÉP]";
          copyBtn.classList.add("cw-copied");
          setTimeout(() => {
            if (span) span.textContent = "[SAO CHÉP]";
            copyBtn.classList.remove("cw-copied");
          }, 2000);
        } catch (err) {
          console.warn("[CHAT_WIDGET] Clipboard copy failed:", err);
        }
      });

      metaRow.appendChild(timeEl);
      metaRow.appendChild(copyBtn);

      // Quick follow-up chips
      const quickReplyRow = document.createElement("div");
      quickReplyRow.className = "cw-quick-reply-row";

      const quickReplies = [
        { label: "Giải thích thêm", query: "Bác sĩ có thể giải thích chi tiết hơn về cơ chế này được không?" },
        { label: "Cho bài tập 2 phút", query: "Cho tôi một bài tập thực hành nhanh trong 2 phút để xử lý ngay." },
        { label: "Tôi vẫn thấy lo", query: "Tôi vẫn cảm thấy bất an và căng thẳng, tôi nên làm gì tiếp theo?" }
      ];

      quickReplies.forEach((qr) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "cw-quick-reply-chip";
        chip.textContent = qr.label;
        chip.addEventListener("click", () => {
          this.sendExternalMessage(qr.query);
        });
        quickReplyRow.appendChild(chip);
      });

      body.appendChild(bubble);
      body.appendChild(metaRow);
      body.appendChild(quickReplyRow);

      layout.appendChild(avatar);
      layout.appendChild(body);
      wrap.appendChild(layout);

      this.messagesContainer.appendChild(wrap);

      // Check prefers-reduced-motion
      const prefersReducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      if (!stream || prefersReducedMotion) {
        this.renderFormattedBotContent(bubble, msg.text);
        if (autoScroll) {
          this.scrollToBottom();
        }
        return;
      }

      // Typewriter Simulation: ~12ms per char
      const fullText = String(msg.text || "");
      let charIdx = 0;
      let isDone = false;
      metaRow.style.opacity = "0.3";
      quickReplyRow.style.opacity = "0.3";

      const finishImmediately = () => {
        if (isDone) return;
        isDone = true;
        if (typewriterTimer) clearInterval(typewriterTimer);
        this.renderFormattedBotContent(bubble, fullText);
        metaRow.style.opacity = "1";
        quickReplyRow.style.opacity = "1";
        this.scrollToBottom();
      };

      bubble.addEventListener("click", finishImmediately);

      const typewriterTimer = setInterval(() => {
        if (charIdx < fullText.length) {
          charIdx += 2;
          bubble.textContent = fullText.slice(0, Math.min(charIdx, fullText.length));
          this.scrollToBottom();
        } else {
          finishImmediately();
        }
      }, 24);
    }

    parseBotSections(text) {
      const rawLines = String(text || "").split("\n");
      const sections = [];
      let currentSection = null;

      rawLines.forEach((line) => {
        const trimmed = line.trim();
        const headerMatch = trimmed.match(/^\[([A-ZÀ-Ỹ0-9\s:_\-&()]+)\]$/i);
        if (headerMatch && trimmed.length <= 95) {
          if (currentSection) {
            sections.push(currentSection);
          }
          currentSection = {
            rawHeader: trimmed,
            title: headerMatch[1].trim(),
            lines: []
          };
        } else {
          if (currentSection) {
            currentSection.lines.push(line);
          } else {
            currentSection = {
              rawHeader: "",
              title: "",
              lines: [line]
            };
          }
        }
      });

      if (currentSection) {
        sections.push(currentSection);
      }

      return sections;
    }

    renderFormattedBotContent(bubbleEl, text) {
      bubbleEl.textContent = "";
      const sections = this.parseBotSections(text);

      const hasHeaders = sections.some((s) => s.title !== "");

      if (!hasHeaders) {
        // Simple plain text with line breaks
        const lines = String(text || "").split("\n");
        lines.forEach((line, idx) => {
          bubbleEl.appendChild(document.createTextNode(line));
          if (idx < lines.length - 1) {
            bubbleEl.appendChild(document.createElement("br"));
          }
        });
        return;
      }

      // Structured section blocks
      sections.forEach((sec) => {
        if (!sec.title && sec.lines.join("").trim() === "") {
          return;
        }

        if (!sec.title) {
          // Introductory text before headers
          const introBox = document.createElement("div");
          introBox.style.marginBottom = "6px";
          sec.lines.forEach((line, idx) => {
            introBox.appendChild(document.createTextNode(line));
            if (idx < sec.lines.length - 1) {
              introBox.appendChild(document.createElement("br"));
            }
          });
          bubbleEl.appendChild(introBox);
          return;
        }

        // Determine section category and tag
        const titleUpper = sec.title.toUpperCase();
        let secClass = "cw-section-block";
        let displayTag = `[${sec.title}]`;

        if (titleUpper.includes("CHẨN ĐOÁN") || titleUpper.includes("CƠ CHẾ") || titleUpper.includes("DIAGNOSIS")) {
          secClass += " cw-sec-diagnosis";
          displayTag = "[CHẨN ĐOÁN LÂM SÀNG & CƠ CHẾ]";
        } else if (titleUpper.includes("CÂU HỎI") || titleUpper.includes("SOCRATIC")) {
          secClass += " cw-sec-socratic";
          displayTag = "[CÂU HỎI TÁI CẤU TRÚC SOCRATIC]";
        } else if (titleUpper.includes("PHÁC ĐỒ") || titleUpper.includes("CAN THIỆP") || titleUpper.includes("ĐIỀU TRỊ") || titleUpper.includes("PROTOCOL")) {
          secClass += " cw-sec-protocol";
          displayTag = "[PHÁC ĐỒ CAN THIỆP CBT]";
        } else if (titleUpper.includes("MÂU THUẪN") || titleUpper.includes("INCONGRUENCE")) {
          secClass += " cw-sec-warning";
          displayTag = "[QUAN SÁT LÂM SÀNG: MÂU THUẪN]";
        } else if (titleUpper.includes("TƯ THẾ") || titleUpper.includes("POSTURE")) {
          secClass += " cw-sec-posture";
          displayTag = "[CHỈNH SỬA TƯ THẾ]";
        } else if (titleUpper.includes("SINH TRẮC") || titleUpper.includes("TÌNH TRẠNG")) {
          secClass += " cw-sec-diagnosis";
          displayTag = "[BỐI CẢNH SINH TRẮC HỌC]";
        } else if (titleUpper.includes("XÁC NHẬN") || titleUpper.includes("KIẾN THỨC MỚI")) {
          secClass += " cw-sec-protocol";
          displayTag = "[TRI THỨC MỚI ĐÃ LƯU TRỮ]";
        }

        const secBlock = document.createElement("div");
        secBlock.className = secClass;

        const tagEl = document.createElement("span");
        tagEl.className = "cw-section-tag";
        tagEl.textContent = displayTag;
        secBlock.appendChild(tagEl);

        const contentEl = document.createElement("div");
        contentEl.className = "cw-section-content";
        const contentLines = sec.lines;
        contentLines.forEach((line, idx) => {
          contentEl.appendChild(document.createTextNode(line));
          if (idx < contentLines.length - 1) {
            contentEl.appendChild(document.createElement("br"));
          }
        });
        secBlock.appendChild(contentEl);

        bubbleEl.appendChild(secBlock);
      });
    }

    showTypingIndicator() {
      const existing = document.getElementById("cw-typing-indicator");
      if (existing) return existing;

      const typing = document.createElement("div");
      typing.className = "cw-typing";
      typing.id = "cw-typing-indicator";
      typing.innerHTML = `
        <span class="cw-typing-dot"></span>
        <span class="cw-typing-dot"></span>
        <span class="cw-typing-dot"></span>
        <span class="cw-typing-text">Bác sĩ đang phân tích...</span>
      `;
      this.messagesContainer.appendChild(typing);
      this.scrollToBottom();
      return typing;
    }

    hideTypingIndicator() {
      const el = document.getElementById("cw-typing-indicator");
      if (el && el.parentNode) {
        el.parentNode.removeChild(el);
      }
    }

    scrollToBottom() {
      if (this.messagesContainer) {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
      }
    }

    clearHistory() {
      const confirmed = window.confirm("Xác nhận xóa toàn bộ lịch sử hội thoại hiện tại?");
      if (!confirmed) return;

      this.messages = [Object.assign({}, DEFAULT_WELCOME_MSG)];
      this.saveHistory();
      this.renderMessages();
    }

    async sendMessage() {
      if (this.isSending) return;
      const text = this.inputField.value.trim();
      if (!text) return;

      this.isSending = true;
      this.inputField.value = "";
      this.sendBtn.disabled = true;

      // 1. Add user message
      const userMsg = {
        sender: "user",
        text: text,
        time: formatCurrentTime()
      };
      this.messages.push(userMsg);
      this.saveHistory();
      this.appendMessageElement(userMsg, true);

      // 2. Show typing indicator
      this.showTypingIndicator();

      // 3. Assemble biometric context safely from clinicalStation
      let bioContext = {
        emotion: "NEUTRAL / CALM",
        valence: 0.05,
        arousal: 0.02,
        exhaustion_level: 1,
        incongruence_detected: false,
        posture_warning: false,
        active_tab: this.activeTabId
      };

      if (window.clinicalStation && window.clinicalStation.currentBiometrics) {
        const b = window.clinicalStation.currentBiometrics;
        bioContext.emotion = b.emotion || "NEUTRAL / CALM";
        bioContext.valence = Number(b.valence ?? 0.05);
        bioContext.arousal = Number(b.arousal ?? 0.02);
        bioContext.exhaustion_level = Number(b.mental_exhaustion_level ?? b.exhaustion_level ?? 1);
        bioContext.incongruence_detected = Boolean(b.incongruence_detected);
        bioContext.posture_warning = Boolean(b.posture_warning);
      }

      // 4. Send request to /api/cbt/chat
      try {
        const response = await fetch("/api/cbt/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: text,
            biometric_context: bioContext
          })
        });

        if (!response.ok) {
          throw new Error(`Máy chủ trả về mã HTTP ${response.status}`);
        }

        const data = await response.json();
        this.hideTypingIndicator();

        const botReply = data && data.response
          ? data.response
          : "Hệ thống đã tiếp nhận thông tin nhưng chưa có phản hồi phù hợp.";

        const botMsg = {
          sender: "bot",
          text: botReply,
          time: formatCurrentTime()
        };

        this.messages.push(botMsg);
        this.saveHistory();
        this.appendMessageElement(botMsg, true, true); // Typewriter streaming enabled!

        // If widget is closed, increase unread count
        if (!this.isOpen) {
          this.unreadCount += 1;
          this.updateUnreadBadge();
        }
      } catch (err) {
        console.error("[CHAT_WIDGET] Network / API error:", err);
        this.hideTypingIndicator();

        const errorMsg = {
          sender: "bot",
          text: "[CẢNH BÁO KẾT NỐI HỆ THỐNG]\n" +
            "Không thể kết nối đến dịch vụ bác sĩ CBT tại máy chủ cục bộ (127.0.0.1:8000).\n" +
            "Vui lòng kiểm tra lại tiến trình backend FastAPI và thử lại.",
          time: formatCurrentTime()
        };

        this.messages.push(errorMsg);
        this.saveHistory();
        this.appendMessageElement(errorMsg, true, false);
      } finally {
        this.isSending = false;
        this.sendBtn.disabled = false;
        this.inputField.focus();
      }
    }
  }

  // Initialize once DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      window.cbtChatWidget = new FloatingChatWidget();
      window.cbtChatWidget.init();
    });
  } else {
    window.cbtChatWidget = new FloatingChatWidget();
    window.cbtChatWidget.init();
  }
})();
