import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlignLeft,
  BookText,
  CircleAlert,
  CircleHelp,
  Download,
  FileText,
  Info,
  Lightbulb,
  NotebookPen,
  RefreshCw,
  Sparkles,
  Trash2,
  TriangleAlert,
  Upload,
  X,
} from "lucide-react";

/* -------------------------------------------------------------------------- */
/* Configuration                                                              */
/* -------------------------------------------------------------------------- */

const API_BASE = "http://127.0.0.1:8000";
const MAX_BYTES = 10 * 1024 * 1024; // must match the backend limit
const ACCEPT = ".pdf,.txt,application/pdf,text/plain";
const MODEL_FALLBACK = "qwen2.5:3b";
const HEALTH_POLL_MS = 20000;

const TABS = [
  { id: "summary", label: "Summary", Icon: AlignLeft },
  { id: "key_concepts", label: "Key concepts", Icon: Lightbulb },
  { id: "definitions", label: "Definitions", Icon: BookText },
  { id: "revision_questions", label: "Revision questions", Icon: CircleHelp },
];

const STATUS_COPY = {
  checking: { label: "Checking backend", tone: "neutral" },
  online: { label: "Backend connected", tone: "ok" },
  offline: { label: "Backend offline", tone: "bad" },
  ollama_down: { label: "Ollama not running", tone: "warn" },
  model_missing: { label: "Model not installed", tone: "warn" },
};

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

class ApiError extends Error {
  constructor(info) {
    super(info.message);
    this.info = info;
  }
}

function getExtension(name) {
  const index = name.lastIndexOf(".");
  return index === -1 ? "" : name.slice(index).toLowerCase();
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatElapsed(seconds) {
  const m = Math.floor(seconds / 60);
  const s = String(seconds % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function fileKindLabel(name) {
  return getExtension(name) === ".pdf" ? "PDF document" : "Text file";
}

function validateFile(file) {
  const ext = getExtension(file.name);
  if (ext !== ".pdf" && ext !== ".txt") {
    return {
      tone: "error",
      title: "That file type isn’t supported",
      message: `“${file.name}” isn’t a PDF or TXT file.`,
      hint: "Export the document as a PDF or save it as plain text, then try again.",
    };
  }
  if (file.size === 0) {
    return {
      tone: "error",
      title: "That file is empty",
      message: `“${file.name}” has no content.`,
      hint: "Choose a file that contains text.",
    };
  }
  if (file.size > MAX_BYTES) {
    return {
      tone: "error",
      title: "That file is too large",
      message: `“${file.name}” is ${formatBytes(file.size)}. The limit is ${formatBytes(MAX_BYTES)}.`,
      hint: "Split the document into smaller parts and analyse them one at a time.",
    };
  }
  return null;
}

async function readApiError(res) {
  let detail = "";
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    /* the error body was not JSON */
  }

  const byStatus = {
    413: {
      title: "That file is too large",
      fallback: "Choose a smaller file and try again.",
    },
    415: {
      title: "That file type isn’t supported",
      fallback: "Upload a PDF or TXT file.",
    },
    422: {
      title: "This document couldn’t be read",
      fallback:
        "The backend couldn’t use this upload. The file may be empty, password-protected or a scan.",
    },
    502: {
      title: "The model’s answer couldn’t be used",
      fallback: "Try again. If it keeps happening, try a shorter document.",
    },
    503: {
      title: "Ollama isn’t ready",
      fallback: "Start Ollama and make sure the model is installed.",
    },
    504: {
      title: "The model took too long",
      fallback: "Try a shorter document, or close other heavy programs and try again.",
    },
  };
  const entry = byStatus[res.status] || {
    title: "Something went wrong",
    fallback: `The backend returned an error (status ${res.status}).`,
  };
  return { tone: "error", title: entry.title, message: detail || entry.fallback };
}

const stringList = (value) =>
  Array.isArray(value) ? value.filter((v) => typeof v === "string" && v.trim()) : [];

/** Maps the backend SummarizeResponse onto a safe shape without inventing fields. */
function normalizeResult(data) {
  return {
    filename: typeof data?.filename === "string" ? data.filename : "document",
    model: typeof data?.model === "string" ? data.model : MODEL_FALLBACK,
    characters_analyzed: Number.isFinite(data?.characters_analyzed) ? data.characters_analyzed : 0,
    truncated: Boolean(data?.truncated),
    summary: typeof data?.summary === "string" ? data.summary.trim() : "",
    key_concepts: stringList(data?.key_concepts),
    definitions: Array.isArray(data?.definitions)
      ? data.definitions.filter(
          (d) => d && typeof d.term === "string" && typeof d.definition === "string"
        )
      : [],
    revision_questions: stringList(data?.revision_questions),
    warnings: stringList(data?.warnings),
  };
}

function buildMarkdown(r) {
  const lines = [];
  lines.push(`# Study notes: ${r.filename}`, "");
  lines.push(
    `_Generated locally by LazyNotes with ${r.model}. ${r.characters_analyzed.toLocaleString(
      "en-US"
    )} characters analysed${r.truncated ? " (the document was truncated)" : ""}._`,
    ""
  );
  if (r.warnings.length) {
    lines.push("> **Notes from the generator**");
    r.warnings.forEach((w) => lines.push(`> - ${w}`));
    lines.push("");
  }

  lines.push("## Summary", "", r.summary || "_No summary was generated._", "");

  lines.push("## Key concepts", "");
  if (r.key_concepts.length) r.key_concepts.forEach((c) => lines.push(`- ${c}`));
  else lines.push("_None found in the document._");
  lines.push("");

  lines.push("## Definitions", "");
  if (r.definitions.length) r.definitions.forEach((d) => lines.push(`- **${d.term}**: ${d.definition}`));
  else lines.push("_None found in the document._");
  lines.push("");

  lines.push("## Revision questions", "");
  if (r.revision_questions.length) r.revision_questions.forEach((q, i) => lines.push(`${i + 1}. ${q}`));
  else lines.push("_None generated._");
  lines.push("");

  return lines.join("\n");
}

function markdownFilename(original) {
  const base =
    original
      .replace(/\.[^.]+$/, "")
      .replace(/[\\/:*?"<>|]+/g, "_")
      .trim() || "notes";
  return `${base}-notes.md`;
}

function statusHint(status) {
  switch (status.phase) {
    case "offline":
      return {
        tone: "bad",
        body: (
          <>
            Can’t reach <code>{API_BASE}</code>. Start the backend with{" "}
            <code>uvicorn main:app --reload --port 8000</code> and open this page at{" "}
            <code>http://localhost:5173</code>.
          </>
        ),
      };
    case "ollama_down":
      return {
        tone: "warn",
        body: (
          <>
            The backend is running but can’t reach Ollama. Start Ollama, then select the status
            indicator to check again.
          </>
        ),
      };
    case "model_missing":
      return {
        tone: "warn",
        body: (
          <>
            Ollama is running but the model is missing. Run <code>ollama pull {status.model}</code>,
            then check again.
          </>
        ),
      };
    default:
      return null;
  }
}

/* -------------------------------------------------------------------------- */
/* Backend status hook                                                        */
/* -------------------------------------------------------------------------- */

function useBackendStatus() {
  const [status, setStatus] = useState({ phase: "checking", model: MODEL_FALLBACK });
  const aliveRef = useRef(true);

  const check = useCallback(async (quiet = false) => {
    if (!quiet) setStatus((s) => ({ ...s, phase: "checking" }));
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    try {
      const res = await fetch(`${API_BASE}/api/health`, { signal: controller.signal });
      if (!res.ok) throw new Error(`Health check returned ${res.status}`);
      const data = await res.json();
      let phase = "online";
      if (data.ollama !== "up") phase = "ollama_down";
      else if (!data.model_available) phase = "model_missing";
      if (aliveRef.current) setStatus({ phase, model: data.model || MODEL_FALLBACK });
    } catch {
      if (aliveRef.current) setStatus((s) => ({ ...s, phase: "offline" }));
    } finally {
      clearTimeout(timer);
    }
  }, []);

  useEffect(() => {
    aliveRef.current = true;
    check();
    const id = setInterval(() => check(true), HEALTH_POLL_MS);
    return () => {
      aliveRef.current = false;
      clearInterval(id);
    };
  }, [check]);

  return [status, check];
}

/* -------------------------------------------------------------------------- */
/* Small components                                                           */
/* -------------------------------------------------------------------------- */

function StatusPill({ status, onRefresh }) {
  const copy = STATUS_COPY[status.phase];
  return (
    <button
      type="button"
      className={`status-pill tone-${copy.tone}`}
      onClick={() => onRefresh(false)}
      aria-label={`${copy.label}. Select to check the connection again.`}
      title="Check the connection again"
    >
      <span className="status-dot" aria-hidden="true" />
      <span>{copy.label}</span>
      <RefreshCw size={14} aria-hidden="true" className={status.phase === "checking" ? "spin" : ""} />
    </button>
  );
}

function Alert({ info, onDismiss }) {
  const Icon = info.tone === "error" ? CircleAlert : info.tone === "warn" ? TriangleAlert : Info;
  return (
    <div className={`alert alert-${info.tone}`} role={info.tone === "error" ? "alert" : "status"}>
      <Icon size={20} aria-hidden="true" />
      <div className="alert-body">
        <p className="alert-title">{info.title}</p>
        {info.message && <p className="alert-msg">{info.message}</p>}
        {info.items && (
          <ul className="alert-list">
            {info.items.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        )}
        {info.hint && <p className="alert-hint">{info.hint}</p>}
      </div>
      {onDismiss && (
        <button type="button" className="icon-btn" onClick={onDismiss} aria-label="Dismiss message">
          <X size={18} aria-hidden="true" />
        </button>
      )}
    </div>
  );
}

function Dropzone({ dragging, hasFile, onBrowse, onDragState, onDropFiles }) {
  return (
    <div
      className={`dropzone${dragging ? " is-dragging" : ""}`}
      role="button"
      tabIndex={0}
      aria-label="Choose a PDF or TXT file, or drop one here"
      onClick={onBrowse}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onBrowse();
        }
      }}
      onDragEnter={(e) => {
        e.preventDefault();
        onDragState(true);
      }}
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        onDragState(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) onDragState(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        onDragState(false);
        onDropFiles(e.dataTransfer.files);
      }}
    >
      <Upload size={28} aria-hidden="true" />
      <p className="dropzone-title">
        {hasFile ? "Drop another file to replace it" : "Drop a PDF or TXT file here"}
      </p>
      <p className="dropzone-sub">or click to browse your files</p>
      <p className="dropzone-limit">One file, up to {formatBytes(MAX_BYTES)}</p>
    </div>
  );
}

function FileDetails({ file, disabled, onRemove }) {
  return (
    <div className="file-details">
      <span className="file-icon" aria-hidden="true">
        <FileText size={22} />
      </span>
      <div className="file-meta">
        <p className="file-name" title={file.name}>
          {file.name}
        </p>
        <p className="file-sub">
          {fileKindLabel(file.name)}, {formatBytes(file.size)}
        </p>
      </div>
      <button
        type="button"
        className="icon-btn"
        onClick={onRemove}
        disabled={disabled}
        aria-label={`Remove ${file.name}`}
        title="Remove file"
      >
        <Trash2 size={18} aria-hidden="true" />
      </button>
    </div>
  );
}

function IntroSheet() {
  return (
    <section className="sheet" aria-label="Notes">
      <div className="sheet-head">
        <div className="sheet-title">
          <NotebookPen size={20} aria-hidden="true" />
          <div>
            <h2>Notes</h2>
            <p className="sheet-sub">Nothing generated yet</p>
          </div>
        </div>
      </div>
      <div className="sheet-body">
        <p>Your study notes will appear on this page.</p>
        <ol className="steps">
          <li>Add a PDF or TXT file.</li>
          <li>Select Generate notes.</li>
          <li>Read the summary, then work through the revision questions.</li>
        </ol>
        <p className="empty-line">
          Notes only use what the document says. If it doesn’t cover something, that section stays
          empty.
        </p>
      </div>
    </section>
  );
}

function LoadingSheet({ filename, elapsed }) {
  return (
    <section className="sheet" aria-label="Generating notes" aria-busy="true">
      <div className="sheet-head">
        <div className="sheet-title">
          <FileText size={20} aria-hidden="true" />
          <div>
            <h2 title={filename}>{filename}</h2>
            <p className="sheet-sub">Working on it</p>
          </div>
        </div>
      </div>
      <div className="sheet-body" role="status" aria-live="polite">
        <p className="loading-line">
          <span className="spinner spinner-ink" aria-hidden="true" />
          Reading the document and writing your notes
        </p>
        <p>
          The model runs on this computer, so a long document can take a minute or two. Elapsed time:{" "}
          <strong>{formatElapsed(elapsed)}</strong>
        </p>
      </div>
    </section>
  );
}

function NotesSheet({ result, activeTab, onTabChange, reviewed, onToggleReviewed, onDownload }) {
  const tabRefs = useRef([]);

  const counts = {
    key_concepts: result.key_concepts.length,
    definitions: result.definitions.length,
    revision_questions: result.revision_questions.length,
  };

  function onTabKeyDown(e, index) {
    const last = TABS.length - 1;
    let next = null;
    if (e.key === "ArrowRight") next = index === last ? 0 : index + 1;
    else if (e.key === "ArrowLeft") next = index === 0 ? last : index - 1;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = last;
    if (next !== null) {
      e.preventDefault();
      onTabChange(TABS[next].id);
      tabRefs.current[next]?.focus();
    }
  }

  const doneCount = result.revision_questions.filter((_, i) => reviewed[i]).length;
  const summaryParagraphs = result.summary.split(/\n+/).filter((p) => p.trim());

  return (
    <section className="sheet" aria-label="Generated notes">
      <div className="sheet-head">
        <div className="sheet-title">
          <FileText size={20} aria-hidden="true" />
          <div>
            <h2 title={result.filename}>{result.filename}</h2>
            <p className="sheet-sub">
              Analysed {result.characters_analyzed.toLocaleString("en-US")} characters with{" "}
              {result.model}
            </p>
          </div>
        </div>
        <button type="button" className="btn btn-quiet" onClick={onDownload}>
          <Download size={18} aria-hidden="true" />
          Download Markdown
        </button>
      </div>

      {result.warnings.length > 0 && (
        <div className="sheet-notice">
          <Alert
            info={{
              tone: "warn",
              title: result.warnings.length === 1 ? "A note about these results" : "Notes about these results",
              items: result.warnings,
            }}
          />
        </div>
      )}

      <div className="tabs" role="tablist" aria-label="Note sections">
        {TABS.map(({ id, label, Icon }, index) => {
          const selected = activeTab === id;
          return (
            <button
              key={id}
              ref={(el) => (tabRefs.current[index] = el)}
              type="button"
              role="tab"
              id={`tab-${id}`}
              aria-selected={selected}
              aria-controls={`panel-${id}`}
              tabIndex={selected ? 0 : -1}
              className="tab"
              onClick={() => onTabChange(id)}
              onKeyDown={(e) => onTabKeyDown(e, index)}
            >
              <Icon size={17} aria-hidden="true" />
              {label}
              {id !== "summary" && <span className="tab-count">{counts[id]}</span>}
            </button>
          );
        })}
      </div>

      <div
        className="sheet-body"
        role="tabpanel"
        id={`panel-${activeTab}`}
        aria-labelledby={`tab-${activeTab}`}
        tabIndex={0}
      >
        {activeTab === "summary" &&
          (summaryParagraphs.length ? (
            summaryParagraphs.map((p, i) => <p key={i}>{p}</p>)
          ) : (
            <p className="empty-line">No summary came back for this document. Try generating again.</p>
          ))}

        {activeTab === "key_concepts" &&
          (result.key_concepts.length ? (
            <ul className="concept-list">
              {result.key_concepts.map((concept, i) => (
                <li key={i}>
                  <mark>{concept}</mark>
                </li>
              ))}
            </ul>
          ) : (
            <>
              <p className="empty-line">No key concepts were found.</p>
              <p className="empty-line">
                The document may be too short or too general. LazyNotes leaves a section empty rather
                than guessing.
              </p>
            </>
          ))}

        {activeTab === "definitions" &&
          (result.definitions.length ? (
            <dl className="def-list">
              {result.definitions.map((d, i) => (
                <div className="def" key={i}>
                  <dt>{d.term}</dt>
                  <dd>{d.definition}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <>
              <p className="empty-line">No definitions were found.</p>
              <p className="empty-line">
                A term appears here only when the document itself defines or explains it.
              </p>
            </>
          ))}

        {activeTab === "revision_questions" &&
          (result.revision_questions.length ? (
            <>
              <p className="q-progress" aria-live="polite">
                {doneCount} of {result.revision_questions.length} reviewed
              </p>
              <ul className="q-list">
                {result.revision_questions.map((q, i) => (
                  <li key={i} className={`q-item${reviewed[i] ? " is-done" : ""}`}>
                    <input
                      type="checkbox"
                      className="q-check"
                      id={`q-${i}`}
                      checked={Boolean(reviewed[i])}
                      onChange={() => onToggleReviewed(i)}
                    />
                    <label htmlFor={`q-${i}`}>{q}</label>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="empty-line">No revision questions were generated for this document.</p>
          ))}
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* App                                                                        */
/* -------------------------------------------------------------------------- */

export default function App() {
  const [status, checkStatus] = useBackendStatus();
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("summary");
  const [reviewed, setReviewed] = useState({});

  const inputRef = useRef(null);
  const abortRef = useRef(null);

  // Stop the browser from navigating away when a file is dropped outside the drop area.
  useEffect(() => {
    const prevent = (e) => e.preventDefault();
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", prevent);
    return () => {
      window.removeEventListener("dragover", prevent);
      window.removeEventListener("drop", prevent);
    };
  }, []);

  // Elapsed-time counter while generating.
  useEffect(() => {
    if (!loading) return undefined;
    setElapsed(0);
    const start = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, [loading]);

  // Abort any in-flight request on unmount.
  useEffect(() => () => abortRef.current?.abort(), []);

  const handleFiles = useCallback((fileList) => {
    const picked = fileList?.[0];
    if (!picked) return;
    const problem = validateFile(picked);
    if (problem) {
      setError(problem);
      return;
    }
    setFile(picked);
    setError(
      fileList.length > 1
        ? {
            tone: "info",
            title: "One file at a time",
            message: `Using “${picked.name}”. LazyNotes analyses a single document per run.`,
          }
        : null
    );
  }, []);

  function browse() {
    inputRef.current?.click();
  }

  function removeFile() {
    setFile(null);
    setError(null);
  }

  function cancel() {
    abortRef.current?.abort();
  }

  async function generate() {
    if (!file || loading) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError(null);

    try {
      const body = new FormData();
      body.append("file", file);

      const res = await fetch(`${API_BASE}/api/summarize`, {
        method: "POST",
        body,
        signal: controller.signal,
      });

      if (!res.ok) throw new ApiError(await readApiError(res));

      const data = await res.json();
      setResult(normalizeResult(data));
      setActiveTab("summary");
      setReviewed({});
    } catch (err) {
      if (err?.name === "AbortError") {
        setError({
          tone: "info",
          title: "Generation cancelled",
          message: "Your file is still selected. The model on your computer may take a moment to stop.",
        });
      } else if (err instanceof ApiError) {
        setError(err.info);
        checkStatus(true);
      } else if (err instanceof SyntaxError) {
        setError({
          tone: "error",
          title: "The backend sent an unexpected response",
          message: "The response wasn’t in the format LazyNotes expects.",
          hint: "Check the backend terminal for errors, then try again.",
        });
      } else {
        setError({
          tone: "error",
          title: "Can’t reach the backend",
          message: `The request to ${API_BASE} didn’t get a response.`,
          hint: "Check that uvicorn is running, and that this page is open at http://localhost:5173, because the backend only allows that address.",
        });
        checkStatus(true);
      }
    } finally {
      abortRef.current = null;
      setLoading(false);
    }
  }

  function downloadMarkdown() {
    if (!result) return;
    const blob = new Blob([buildMarkdown(result)], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = markdownFilename(result.filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const hint = statusHint(status);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <NotebookPen size={22} />
          </span>
          <div>
            <h1>LazyNotes</h1>
            <p>Study notes from your own documents, written on this computer.</p>
          </div>
        </div>
        <StatusPill status={status} onRefresh={checkStatus} />
      </header>

      <main className="workspace">
        <aside className="intake" aria-label="Upload">
          <h2 className="panel-title">Your document</h2>

          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            hidden
            onChange={(e) => {
              handleFiles(e.target.files);
              e.target.value = "";
            }}
          />

          <Dropzone
            dragging={dragging}
            hasFile={Boolean(file)}
            onBrowse={browse}
            onDragState={setDragging}
            onDropFiles={handleFiles}
          />

          {file && <FileDetails file={file} disabled={loading} onRemove={removeFile} />}

          <button
            type="button"
            className="btn btn-primary btn-block"
            onClick={generate}
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Generating notes
              </>
            ) : (
              <>
                <Sparkles size={18} aria-hidden="true" />
                Generate notes
              </>
            )}
          </button>

          {loading && (
            <button type="button" className="btn btn-quiet btn-block" onClick={cancel}>
              Cancel
            </button>
          )}

          {hint && <p className={`status-hint hint-${hint.tone}`}>{hint.body}</p>}

          <p className="privacy">
            Your file goes only to the backend on this computer. No cloud AI services are used.
          </p>
        </aside>

        <section className="notes-col" aria-label="Results">
          {error && <Alert info={error} onDismiss={() => setError(null)} />}

          {loading ? (
            <LoadingSheet filename={file?.name || "document"} elapsed={elapsed} />
          ) : result ? (
            <NotesSheet
              result={result}
              activeTab={activeTab}
              onTabChange={setActiveTab}
              reviewed={reviewed}
              onToggleReviewed={(i) => setReviewed((prev) => ({ ...prev, [i]: !prev[i] }))}
              onDownload={downloadMarkdown}
            />
          ) : (
            <IntroSheet />
          )}
        </section>
      </main>
    </div>
  );
}