import { useCallback, useEffect, useRef, useState } from 'react';
import { AudioWaveform, Download, XCircle, Clock, CheckCircle2, AlertCircle, Loader2, Pause, Play } from 'lucide-react';
import { clsx } from 'clsx';
import { generation, type GenerationJob } from '../lib/api';

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

function JobCard({ job, onCancel, onPause, onResume }: {
  job: GenerationJob;
  onCancel: () => void;
  onPause: () => void;
  onResume: () => void;
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

      {/* Progress bar */}
      {isActive && (
        <div className="mb-4">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs text-zinc-300">{job.current_step ?? 'Starting...'}</span>
            <span className="text-sm font-semibold text-accent">{Math.round(job.progress * 10) / 10}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-surface-overlay">
            <div
              className="h-full rounded-full bg-gradient-to-r from-accent via-purple-400 to-accent transition-all duration-700 ease-out"
              style={{ width: `${Math.max(1, job.progress)}%` }}
            />
          </div>
          <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-500">
            <span>{job.completed_segments} of {job.total_segments} segments processed</span>
            {job.total_segments > 0 && job.completed_segments > 0 && (
              <span className="font-mono">
                {Math.round((job.completed_segments / job.total_segments) * 100)}% complete
              </span>
            )}
          </div>
        </div>
      )}

      {/* Stats grid */}
      <div className="mb-4 grid grid-cols-4 gap-3">
        <div className="rounded-lg bg-surface-overlay px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-zinc-600">Segments</p>
          <p className="text-sm font-semibold text-zinc-200">{job.completed_segments}<span className="text-zinc-500">/{job.total_segments}</span></p>
        </div>
        {job.current_chapter != null && (
          <div className="rounded-lg bg-surface-overlay px-3 py-2">
            <p className="text-[10px] uppercase tracking-wider text-zinc-600">Chapter</p>
            <p className="text-sm font-semibold text-zinc-200">{job.current_chapter}</p>
          </div>
        )}
        <div className="rounded-lg bg-surface-overlay px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-zinc-600">Format</p>
          <p className="text-sm font-semibold text-zinc-200">{job.output_format.toUpperCase()}</p>
        </div>
        <div className="rounded-lg bg-surface-overlay px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-zinc-600">Mode</p>
          <p className="text-sm font-semibold text-zinc-200">{job.is_auto_mode ? 'Auto' : 'Manual'}</p>
        </div>
      </div>

      {job.error_message && (
        <p className="mb-3 rounded-lg bg-danger/10 px-3 py-2 text-xs text-danger">{job.error_message}</p>
      )}

      {/* Actions */}
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
          <button onClick={onResume} className="flex items-center gap-1.5 rounded-lg bg-accent/15 px-3 py-1.5 text-xs font-medium text-accent transition hover:bg-accent/25">
            <Play size={12} /> Resume
          </button>
        )}
        {job.status === 'completed' && (
          <a href={`/api/generation/jobs/${job.job_id}/download`} className="flex items-center gap-1.5 rounded-lg bg-success/15 px-3 py-1.5 text-xs font-medium text-success transition hover:bg-success/25">
            <Download size={12} /> Download
          </a>
        )}
      </div>
    </div>
  );
}

export default function GenerationPage() {
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [loading, setLoading] = useState(true);
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;

  const load = useCallback(async () => {
    try {
      const j = await generation.list();
      setJobs(j);
    } catch { /* empty */ }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
    // Check the ref (not state) so this effect only runs once on mount.
    // Active jobs: poll every 5s.  Idle: every 30s.
    const tick = () => {
      const hasActive = jobsRef.current.some(j =>
        ['pending', 'parsing', 'analyzing', 'generating', 'processing'].includes(j.status),
      );
      return hasActive ? 5_000 : 30_000;
    };
    let timer: ReturnType<typeof setTimeout>;
    const schedule = () => {
      timer = setTimeout(async () => {
        await load();
        schedule();
      }, tick());
    };
    schedule();
    return () => clearTimeout(timer);
  }, [load]);

  const handleCancel = async (jobId: string) => {
    await generation.cancel(jobId);
    await load();
  };

  const handlePause = async (jobId: string) => {
    await generation.pause(jobId);
    await load();
  };

  const handleResume = async (jobId: string) => {
    await generation.resume(jobId);
    await load();
  };

  return (
    <div className="min-h-screen p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">Generation</h1>
        <p className="mt-1 text-sm text-zinc-500">Monitor audiobook generation jobs</p>
      </div>

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
            <JobCard
              key={job.job_id}
              job={job}
              onCancel={() => handleCancel(job.job_id)}
              onPause={() => handlePause(job.job_id)}
              onResume={() => handleResume(job.job_id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
