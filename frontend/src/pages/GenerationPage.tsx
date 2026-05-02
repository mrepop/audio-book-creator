import { useCallback, useEffect, useRef, useState } from 'react';
import { AudioWaveform, Download, XCircle, Clock, CheckCircle2, AlertCircle, Loader2, Pause, Play, ChevronDown, ChevronRight, Volume2, Edit3, RefreshCw } from 'lucide-react';
import { clsx } from 'clsx';
import { books, generation, type GenerationJob, type Book, type ChapterSummary, type Segment } from '../lib/api';

// ---------------------------------------------------------------------------
// Status config
// ---------------------------------------------------------------------------
const STATUS_CONFIG: Record<string, { icon: typeof Clock; color: string; label: string }> = {
  pending:    { icon: Clock,         color: 'text-zinc-400',  label: 'Pending' },
  parsing:    { icon: Loader2,       color: 'text-blue-400',  label: 'Parsing' },
  analyzing:  { icon: Loader2,       color: 'text-blue-400',  label: 'Analyzing' },
  generating: { icon: Loader2,       color: 'text-accent',    label: 'Generating' },
  processing: { icon: Loader2,       color: 'text-accent',    label: 'Processing' },
  completed:  { icon: CheckCircle2,  color: 'text-success',   label: 'Completed' },
  failed:     { icon: AlertCircle,   color: 'text-danger',    label: 'Failed' },
  cancelled:  { icon: XCircle,       color: 'text-zinc-500',  label: 'Cancelled' },
  paused:     { icon: Pause,         color: 'text-amber-400', label: 'Paused' },
};
const RESUMABLE = ['paused', 'failed', 'cancelled'];
type GenMode = 'auto' | 'chapter' | 'sentence';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-surface-overlay px-3 py-2">
      <p className="text-[10px] uppercase tracking-wider text-zinc-600">{label}</p>
      <p className="text-sm font-semibold text-zinc-200">{value}</p>
    </div>
  );
}

function Tag({ label, color }: { label: string; color: string }) {
  return <span className={clsx("rounded px-1.5 py-0.5 text-[10px] font-medium", color)}>{label}</span>;
}

// ---------------------------------------------------------------------------
// Job Card
// ---------------------------------------------------------------------------
function JobCard({ job, onCancel, onPause, onResume }: {
  job: GenerationJob; onCancel: () => void; onPause: () => void; onResume: () => void;
}) {
  const cfg = STATUS_CONFIG[job.status] ?? STATUS_CONFIG.pending;
  const Icon = cfg.icon;
  const isActive = ['pending', 'parsing', 'analyzing', 'generating', 'processing'].includes(job.status);
  const isResumable = RESUMABLE.includes(job.status);

  return (
    <div className="rounded-xl border border-border-subtle bg-surface-raised p-5 transition hover:bg-surface-overlay">
      <div className="mb-4 flex items-start justify-between">
        <div className="flex items-center gap-2.5">
          <Icon size={16} className={clsx(cfg.color, isActive && 'animate-spin')} />
          <span className={clsx('text-sm font-semibold', cfg.color)}>{cfg.label}</span>
        </div>
        <span className="font-mono text-[10px] text-zinc-600">{job.job_id.slice(0, 8)}</span>
      </div>

      {isActive && (
        <div className="mb-4">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs text-zinc-300">{job.current_step ?? 'Starting...'}</span>
            <span className="text-sm font-semibold text-accent">{Math.round(job.progress * 10) / 10}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-surface-overlay">
            <div className="h-full rounded-full bg-gradient-to-r from-accent via-purple-400 to-accent transition-all duration-700 ease-out" style={{ width: `${Math.max(1, job.progress)}%` }} />
          </div>
          <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-500">
            <span>{job.completed_segments} of {job.total_segments} segments</span>
          </div>
        </div>
      )}

      <div className="mb-4 grid grid-cols-4 gap-3">
        <StatBox label="Segments" value={`${job.completed_segments}/${job.total_segments}`} />
        {job.current_chapter != null && <StatBox label="Chapter" value={String(job.current_chapter)} />}
        <StatBox label="Format" value={job.output_format.toUpperCase()} />
        <StatBox label="Mode" value={job.is_auto_mode ? 'Auto' : 'Manual'} />
      </div>

      {job.error_message && (
        <p className="mb-3 rounded-lg bg-danger/10 px-3 py-2 text-xs text-danger">{job.error_message}</p>
      )}

      <div className="flex gap-2">
        {isActive && (
          <>
            <button onClick={onPause} className="flex items-center gap-1.5 rounded-lg border border-border-default px-3 py-1.5 text-xs font-medium text-amber-400 transition hover:border-amber-400/30 hover:text-amber-300">
              <Pause size={12} /> Pause
            </button>
            <button onClick={onCancel} className="flex items-center gap-1.5 rounded-lg border border-border-default px-3 py-1.5 text-xs font-medium text-zinc-400 transition hover:border-danger/30 hover:text-danger">
              <XCircle size={12} /> Cancel
            </button>
          </>
        )}
        {isResumable && (
          <button onClick={onResume} className="flex items-center gap-1.5 rounded-lg bg-accent/15 px-3 py-1.5 text-xs font-medium text-accent hover:bg-accent/25">
            <Play size={12} /> Resume
          </button>
        )}
        {job.status === 'completed' && (
          <a href={`/api/generation/jobs/${job.job_id}/download`} className="flex items-center gap-1.5 rounded-lg bg-success/15 px-3 py-1.5 text-xs font-medium text-success hover:bg-success/25">
            <Download size={12} /> Download
          </a>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Segment Row (sentence-level editor)
// ---------------------------------------------------------------------------
function SegmentRow({ seg }: { seg: Segment }) {
  const [editing, setEditing] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const handlePreview = async () => {
    setPreviewing(true);
    try {
      const blob = await generation.previewSegment(seg.id);
      setAudioUrl(URL.createObjectURL(blob));
    } catch (e) { console.error('Preview failed:', e); }
    setPreviewing(false);
  };

  const typeColor = seg.segment_type === 'dialogue' ? 'text-blue-400'
    : seg.segment_type === 'internal_thought' ? 'text-purple-400' : 'text-zinc-400';
  const typeLabel = seg.segment_type === 'dialogue' ? 'DLG'
    : seg.segment_type === 'internal_thought' ? 'THT' : 'NAR';

  return (
    <div className="group border-b border-border-subtle py-3 px-4 hover:bg-surface-overlay/50">
      <div className="flex items-start gap-3">
        <div className="flex flex-col items-center gap-1 pt-0.5">
          <span className="font-mono text-[10px] text-zinc-600">#{seg.sequence_number}</span>
          <span className={clsx("text-[9px] uppercase font-semibold", typeColor)}>{typeLabel}</span>
        </div>

        <div className="flex-1 min-w-0">
          <p className="text-sm text-zinc-300 leading-relaxed">{seg.user_text_override || seg.text}</p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {seg.emotion && <Tag label={seg.emotion} color="bg-amber-500/10 text-amber-400" />}
            {seg.emphasis && seg.emphasis !== 'normal' && <Tag label={seg.emphasis} color="bg-purple-500/10 text-purple-400" />}
            {seg.pacing && seg.pacing !== 'normal' && <Tag label={seg.pacing} color="bg-blue-500/10 text-blue-400" />}
            {seg.is_generated && <Tag label={`${seg.audio_duration_seconds?.toFixed(1)}s`} color="bg-success/10 text-success" />}
          </div>

          {editing && (
            <div className="mt-3 rounded-lg bg-surface-overlay p-3 space-y-2">
              <div>
                <label className="text-[10px] uppercase text-zinc-500">Instruct (emotion/style for TTS)</label>
                <input type="text" placeholder="e.g. Speak with warmth and a cheerful tone"
                  className="mt-1 w-full rounded border border-border-default bg-surface-raised px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:border-accent focus:outline-none" />
              </div>
              <div className="flex gap-2">
                {['Emotion', 'Emphasis', 'Pacing'].map(label => (
                  <select key={label} className="rounded border border-border-default bg-surface-raised px-2 py-1 text-xs text-zinc-200">
                    <option>{label}: {label === 'Emotion' ? (seg.emotion ?? 'neutral') : label === 'Emphasis' ? (seg.emphasis ?? 'normal') : (seg.pacing ?? 'normal')}</option>
                    {label === 'Emotion' && ['neutral','happy','sad','angry','tender','excited','contemplative'].map(v => <option key={v} value={v}>{v}</option>)}
                    {label === 'Emphasis' && ['normal','whispered','shouted','soft','strong'].map(v => <option key={v} value={v}>{v}</option>)}
                    {label === 'Pacing' && ['normal','slow','fast','urgent'].map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                ))}
              </div>
            </div>
          )}

          {audioUrl && <div className="mt-2"><audio controls src={audioUrl} className="h-8 w-full" /></div>}
        </div>

        <div className="flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition">
          <button onClick={handlePreview} disabled={previewing} title="Preview audio"
            className="rounded p-1.5 text-zinc-500 hover:bg-accent/10 hover:text-accent disabled:animate-pulse">
            {previewing ? <Loader2 size={14} className="animate-spin" /> : <Volume2 size={14} />}
          </button>
          <button onClick={() => setEditing(!editing)} title="Edit parameters"
            className="rounded p-1.5 text-zinc-500 hover:bg-accent/10 hover:text-accent">
            <Edit3 size={14} />
          </button>
          <button onClick={() => generation.regenerateSegment(seg.id)} title="Regenerate"
            className="rounded p-1.5 text-zinc-500 hover:bg-accent/10 hover:text-accent">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chapter Accordion
// ---------------------------------------------------------------------------
function ChapterAccordion({ chapter }: { chapter: ChapterSummary }) {
  const [open, setOpen] = useState(false);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [loadingSegs, setLoadingSegs] = useState(false);

  const toggle = async () => {
    if (!open && segments.length === 0) {
      setLoadingSegs(true);
      try { setSegments(await generation.segments(chapter.id)); } catch { /* empty */ }
      setLoadingSegs(false);
    }
    setOpen(!open);
  };

  return (
    <div className="rounded-xl border border-border-subtle bg-surface-raised overflow-hidden">
      <button onClick={toggle} className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-surface-overlay/50 transition">
        {open ? <ChevronDown size={14} className="text-zinc-500" /> : <ChevronRight size={14} className="text-zinc-500" />}
        <span className="text-sm font-medium text-zinc-200">
          Ch. {chapter.number}{chapter.title ? `: ${chapter.title}` : ''}
        </span>
        <span className="text-[11px] text-zinc-500 ml-auto">
          {chapter.word_count} words
          {chapter.is_generated && <span className="ml-2 text-success">[GENERATED {chapter.audio_duration_seconds?.toFixed(0)}s]</span>}
        </span>
      </button>
      {open && (
        <div className="border-t border-border-subtle">
          {loadingSegs ? (
            <div className="flex items-center justify-center py-6"><Loader2 size={16} className="animate-spin text-zinc-500" /></div>
          ) : segments.length === 0 ? (
            <p className="px-4 py-4 text-xs text-zinc-500">No segments found. Run analysis first.</p>
          ) : (
            <div className="max-h-[500px] overflow-y-auto">
              {segments.map(seg => <SegmentRow key={seg.id} seg={seg} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------
export default function GenerationPage() {
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState<GenMode>('auto');
  const [bookList, setBookList] = useState<Book[]>([]);
  const [selectedBook, setSelectedBook] = useState<Book | null>(null);
  const [chapters, setChapters] = useState<ChapterSummary[]>([]);
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;

  const load = useCallback(async () => {
    try { setJobs(await generation.list()); } catch { /* empty */ }
    setLoading(false);
  }, []);

  useEffect(() => { books.list().then(d => setBookList(d.books)).catch(() => {}); }, []);
  useEffect(() => {
    if (selectedBook?.id) books.get(selectedBook.id).then(b => setChapters(b.chapters ?? [])).catch(() => {});
  }, [selectedBook]);

  useEffect(() => {
    load();
    const tick = () => jobsRef.current.some(j => ['pending','parsing','analyzing','generating','processing'].includes(j.status)) ? 5_000 : 30_000;
    let timer: ReturnType<typeof setTimeout>;
    const schedule = () => { timer = setTimeout(async () => { await load(); schedule(); }, tick()); };
    schedule();
    return () => clearTimeout(timer);
  }, [load]);

  const handleCancel = async (id: string) => { await generation.cancel(id); await load(); };
  const handlePause  = async (id: string) => { await generation.pause(id);  await load(); };
  const handleResume = async (id: string) => { await generation.resume(id); await load(); };

  return (
    <div className="flex h-full flex-col min-h-0">
      <div className="shrink-0 px-8 pt-8 pb-4">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">Generation Studio</h1>
        <p className="mt-1 text-sm text-zinc-500">Generate audiobooks with full control over every sentence</p>
        <div className="mt-4 flex gap-1 rounded-lg bg-surface-overlay p-1 w-fit">
          {(['auto', 'chapter', 'sentence'] as GenMode[]).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={clsx("px-4 py-1.5 rounded-md text-xs font-medium transition",
                mode === m ? "bg-accent text-white" : "text-zinc-400 hover:text-zinc-200")}>
              {m === 'auto' ? 'Auto' : m === 'chapter' ? 'Chapter' : 'Sentence'}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-8 pb-8">
        {mode === 'auto' && (
          <>
            {loading ? (
              <div className="flex h-40 items-center justify-center">
                <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent border-t-transparent" />
              </div>
            ) : jobs.length === 0 ? (
              <div className="flex h-60 flex-col items-center justify-center rounded-xl border border-dashed border-border-default">
                <AudioWaveform size={36} className="mb-3 text-zinc-700" />
                <p className="text-sm text-zinc-500">No generation jobs yet</p>
                <p className="mt-1 text-xs text-zinc-600">Upload a book and click "Generate Audiobook" to start</p>
              </div>
            ) : (
              <div className="space-y-4">
                {jobs.map(job => (
                  <JobCard key={job.job_id} job={job}
                    onCancel={() => handleCancel(job.job_id)}
                    onPause={() => handlePause(job.job_id)}
                    onResume={() => handleResume(job.job_id)} />
                ))}
              </div>
            )}
          </>
        )}

        {(mode === 'chapter' || mode === 'sentence') && (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <label className="text-xs text-zinc-500 uppercase tracking-wider">Book</label>
              <select value={selectedBook?.id ?? ''}
                onChange={e => setSelectedBook(bookList.find(b => b.id === Number(e.target.value)) ?? null)}
                className="rounded-lg border border-border-default bg-surface-raised px-3 py-2 text-sm text-zinc-200 focus:border-accent focus:outline-none">
                <option value="">Select a book...</option>
                {bookList.filter(b => b.is_analyzed).map(b => <option key={b.id} value={b.id}>{b.title}</option>)}
              </select>
              {selectedBook && <span className="text-xs text-zinc-500">{chapters.length} chapters</span>}
            </div>

            {selectedBook && chapters.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs text-zinc-500 mb-2">
                  {mode === 'chapter'
                    ? 'Click a chapter to expand and view segments.'
                    : 'Expand chapters to edit individual segments. Use the preview button to hear audio before generating.'}
                </p>
                {chapters.map(ch => <ChapterAccordion key={ch.id} chapter={ch} />)}
              </div>
            )}

            {selectedBook && chapters.length === 0 && (
              <div className="flex h-40 flex-col items-center justify-center rounded-xl border border-dashed border-border-default">
                <p className="text-sm text-zinc-500">No chapters found</p>
                <p className="mt-1 text-xs text-zinc-600">Parse and analyze the book first from the Library page</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
