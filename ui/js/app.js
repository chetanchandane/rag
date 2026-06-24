/**
 * app.js — ClinicalRAG frontend
 *
 * State machine with 4 states:
 *   idle     → user can type and submit
 *   loading  → request in-flight, input frozen, skeleton visible
 *   answer   → response rendered, sources grid populated
 *   error    → error card shown with retry button
 *
 * To iterate on this file:
 *   - Rendering logic is in the "Render helpers" section
 *   - State transitions are in the "State machine" section
 *   - API call is in "Core query"
 *   - Event wiring is at the bottom
 */

'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const queryInput    = document.getElementById('queryInput');
const askBtn        = document.getElementById('askBtn');
const inputWrapper  = document.getElementById('inputWrapper');
const statusDot     = document.getElementById('statusDot');
const statusText    = document.getElementById('statusText');

const loadingState  = document.getElementById('loadingState');
const answerState   = document.getElementById('answerState');
const errorState    = document.getElementById('errorState');

const answerText    = document.getElementById('answerText');
const sourcesGrid   = document.getElementById('sourcesGrid');
const chunksUsed    = document.getElementById('chunksUsed');
const errorMessage  = document.getElementById('errorMessage');
const retryBtn      = document.getElementById('retryBtn');
const chips         = document.querySelectorAll('.chip');

// ── App state ─────────────────────────────────────────────────────────────────
let lastQuery = '';

// ── State machine ─────────────────────────────────────────────────────────────
function setState(state) {
  // Hide everything first
  loadingState.classList.add('hidden');
  answerState.classList.add('hidden');
  answerState.classList.remove('visible');
  errorState.classList.add('hidden');
  inputWrapper.classList.remove('dimmed');
  askBtn.disabled = false;

  switch (state) {
    case 'loading':
      loadingState.classList.remove('hidden');
      inputWrapper.classList.add('dimmed');
      askBtn.disabled = true;
      setStatus('Searching…', 'busy');
      break;

    case 'answer':
      answerState.classList.remove('hidden');
      // Double rAF ensures the element is painted before the transition fires
      requestAnimationFrame(() =>
        requestAnimationFrame(() => answerState.classList.add('visible'))
      );
      setStatus('System Ready', 'ready');
      break;

    case 'error':
      errorState.classList.remove('hidden');
      setStatus('System Ready', 'ready');
      break;

    default: // 'idle'
      setStatus('System Ready', 'ready');
  }
}

function setStatus(label, mode) {
  statusText.textContent = label;
  const color = mode === 'busy' ? 'var(--accent)' : 'var(--success)';
  statusDot.style.background  = color;
  statusDot.style.boxShadow   = `0 0 6px ${color}`;
}

// ── Render helpers ─────────────────────────────────────────────────────────────
function renderSources(sources) {
  sourcesGrid.innerHTML = '';

  if (!sources.length) {
    sourcesGrid.innerHTML = '<p style="font-size:0.8125rem;color:var(--text-dim)">No sources returned.</p>';
    return;
  }

  sources.forEach((s, i) => {
    const rank = i + 1;
    const pct  = Math.round((s.relevance_score ?? s.score ?? 0) * 100);
    const name = s.source.replace(/\.pdf$/i, '');

    const card = document.createElement('div');
    card.className = 'source-card';
    card.innerHTML = `
      <div class="source-card-header">
        <div class="source-name" title="${escapeHtml(s.source)}">${escapeHtml(name)}</div>
        <span class="rank-badge">#${rank}</span>
      </div>
      <div class="source-page">Page ${s.page}</div>
      <div class="relevance-track">
        <div class="relevance-fill" data-pct="${pct}"></div>
      </div>
      <div class="relevance-score">${pct}% relevance</div>
    `;
    sourcesGrid.appendChild(card);
  });

  // Trigger bar animations after elements are in the DOM
  requestAnimationFrame(() => {
    sourcesGrid.querySelectorAll('.relevance-fill').forEach(bar => {
      bar.style.width = bar.dataset.pct + '%';
    });
  });

  // Legend — rendered once after the grid
  const legend = document.createElement('p');
  legend.className = 'sources-legend';
  legend.innerHTML =
    '<span class="legend-dot legend-dot--score"></span>% = Cohere cross-encoder relevance score &nbsp;·&nbsp; ' +
    '<span class="legend-dot legend-dot--rank"></span>#rank = order after hybrid BM25 + semantic reranking';
  sourcesGrid.after(legend);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Core query ────────────────────────────────────────────────────────────────
async function runQuery(question) {
  const q = question.trim();
  if (!q) return;
  lastQuery = q;

  setState('loading');

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, top_k: 5 }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Server error ${res.status}`);
    }

    const data = await res.json();

    // Render markdown, then style inline [Source: ...] citations as badges
    let html = marked.parse(data.answer);
    html = html.replace(/\[Source: ([^\]]+)\]/g, '<cite class="source-cite">$1</cite>');
    answerText.innerHTML = html;
    chunksUsed.textContent = `${data.chunks_used} chunk${data.chunks_used !== 1 ? 's' : ''} used`;
    renderSources(data.sources || []);
    setState('answer');

    // Scroll to results smoothly
    answerState.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch (err) {
    errorMessage.textContent = err.message || 'An unexpected error occurred. Please try again.';
    setState('error');
  }
}

// ── Auto-resize textarea ───────────────────────────────────────────────────────
function resizeTextarea() {
  queryInput.style.height = 'auto';
  queryInput.style.height = queryInput.scrollHeight + 'px';
}

// ── Event wiring ──────────────────────────────────────────────────────────────

// Submit on button click
askBtn.addEventListener('click', () => runQuery(queryInput.value));

// Submit on Enter (Shift+Enter = newline)
queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    runQuery(queryInput.value);
  }
});

// Auto-resize as user types
queryInput.addEventListener('input', resizeTextarea);

// Chips — fill input and submit immediately
chips.forEach(chip => {
  chip.addEventListener('click', () => {
    const q = chip.dataset.q;
    queryInput.value = q;
    resizeTextarea();
    runQuery(q);
  });
});

// Retry button
retryBtn.addEventListener('click', () => runQuery(lastQuery));
