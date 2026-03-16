/**
 * Quickstart guide — animated terminal + scroll reveals.
 * Zero dependencies, vanilla JS.
 */

(function () {
  "use strict";

  // ── Terminal typing animation ────────────────────────────

  const INIT_LINES = [
    { text: "$ ", cls: "prompt", delay: 0 },
    { text: "nmem init --full", cls: "cmd", typing: true },
    { text: "\n", delay: 300 },
    { text: "✓ ", cls: "ok", delay: 200 },
    { text: "Config created: ~/.neuralmemory/config.toml", delay: 0 },
    { text: "\n", delay: 150 },
    { text: "✓ ", cls: "ok", delay: 200 },
    { text: "Brain ready: default", delay: 0 },
    { text: "\n", delay: 150 },
    { text: "✓ ", cls: "ok", delay: 200 },
    { text: "MCP configured: Claude Code", delay: 0 },
    { text: "\n", delay: 150 },
    { text: "✓ ", cls: "ok", delay: 200 },
    { text: "Hooks installed: PreCompact, Stop, PostToolUse", delay: 0 },
    { text: "\n", delay: 150 },
    { text: "✓ ", cls: "ok", delay: 200 },
    { text: "Embeddings: Sentence Transformers (multilingual)", delay: 0 },
    { text: "\n", delay: 150 },
    { text: "✓ ", cls: "ok", delay: 200 },
    { text: "Dedup: enabled", delay: 0 },
    { text: "\n", delay: 150 },
    { text: "✓ ", cls: "ok", delay: 200 },
    { text: "Skills: 3 installed", delay: 0 },
    { text: "\n\n", delay: 300 },
    { text: "  Neural Memory is ready!", cls: "accent", delay: 0 },
    { text: "\n", delay: 100 },
    { text: "  Restart your AI tool to activate memory.", cls: "dim", delay: 0 },
  ];

  const DOCTOR_LINES = [
    { text: "$ ", cls: "prompt", delay: 0 },
    { text: "nmem doctor", cls: "cmd", typing: true },
    { text: "\n\n", delay: 300 },
    { text: "  NeuralMemory Doctor\n", cls: "accent", delay: 100 },
    { text: "  ───────────────────\n\n", cls: "dim", delay: 150 },
    { text: "  [OK] ", cls: "ok", delay: 120 },
    { text: "Python version       3.12.1\n", delay: 0 },
    { text: "  [OK] ", cls: "ok", delay: 120 },
    { text: "Configuration        config.toml (brain: default)\n", delay: 0 },
    { text: "  [OK] ", cls: "ok", delay: 120 },
    { text: "Brain database       default (2,847 KB)\n", delay: 0 },
    { text: "  [OK] ", cls: "ok", delay: 120 },
    { text: "Dependencies         all core deps available\n", delay: 0 },
    { text: "  [OK] ", cls: "ok", delay: 120 },
    { text: "Embedding provider   sentence_transformer (multilingual)\n", delay: 0 },
    { text: "  [OK] ", cls: "ok", delay: 120 },
    { text: "Schema version       v26 (current)\n", delay: 0 },
    { text: "  [OK] ", cls: "ok", delay: 120 },
    { text: "MCP configuration    neural-memory registered\n", delay: 0 },
    { text: "  [OK] ", cls: "ok", delay: 120 },
    { text: "Hooks                3/3 installed\n", delay: 0 },
    { text: "  [OK] ", cls: "ok", delay: 120 },
    { text: "Dedup                enabled\n", delay: 0 },
    { text: "  [OK] ", cls: "ok", delay: 120 },
    { text: "Knowledge surface    default.nm (1.2 KB)\n", delay: 0 },
    { text: "  [OK] ", cls: "ok", delay: 120 },
    { text: "CLI tools            nmem + nmem-mcp on PATH\n", delay: 0 },
    { text: "\n", delay: 200 },
    { text: "  11/11 passed", cls: "ok", delay: 0 },
  ];

  function animateTerminal(container, lines, onDone) {
    var body = container.querySelector(".qs-terminal__body");
    if (!body) return;
    body.innerHTML = "";
    var lineIdx = 0;
    var charIdx = 0;

    function next() {
      if (lineIdx >= lines.length) {
        if (onDone) onDone();
        return;
      }

      var line = lines[lineIdx];
      var delay = line.delay || 0;

      if (line.typing && charIdx === 0) {
        // Start typing mode
        setTimeout(typeChar, delay);
        return;
      }

      setTimeout(function () {
        appendText(body, line.text, line.cls);
        lineIdx++;
        charIdx = 0;
        next();
      }, delay);
    }

    function typeChar() {
      var line = lines[lineIdx];
      if (charIdx < line.text.length) {
        appendText(body, line.text[charIdx], line.cls);
        charIdx++;
        setTimeout(typeChar, 30);
      } else {
        lineIdx++;
        charIdx = 0;
        next();
      }
    }

    next();
  }

  function appendText(parent, text, cls) {
    if (cls) {
      var span = document.createElement("span");
      span.className = cls;
      span.textContent = text;
      parent.appendChild(span);
    } else {
      parent.appendChild(document.createTextNode(text));
    }
    parent.scrollTop = parent.scrollHeight;
  }

  // ── Scroll-triggered reveals ─────────────────────────────

  function setupReveals() {
    var elements = document.querySelectorAll(".qs-reveal");
    if (!elements.length) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("qs-reveal--visible");
          }
        });
      },
      { threshold: 0.15 }
    );

    elements.forEach(function (el) {
      observer.observe(el);
    });
  }

  // ── Terminal auto-play on scroll ─────────────────────────

  function setupTerminals() {
    var terminals = document.querySelectorAll(".qs-terminal[data-lines]");
    terminals.forEach(function (terminal) {
      var linesKey = terminal.getAttribute("data-lines");
      var lines = linesKey === "doctor" ? DOCTOR_LINES : INIT_LINES;
      var played = false;

      var observer = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting && !played) {
              played = true;
              animateTerminal(terminal, lines);
            }
          });
        },
        { threshold: 0.3 }
      );

      observer.observe(terminal);

      // Replay button
      var replay = terminal.querySelector(".qs-terminal__replay");
      if (replay) {
        replay.addEventListener("click", function () {
          played = true;
          animateTerminal(terminal, lines);
        });
      }
    });
  }

  // ── Init ─────────────────────────────────────────────────

  function init() {
    setupTerminals();
    setupReveals();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Re-init on MkDocs instant navigation
  if (typeof document$ !== "undefined") {
    document$.subscribe(init);
  }
})();
