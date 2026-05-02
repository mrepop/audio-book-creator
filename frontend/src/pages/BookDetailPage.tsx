import { useCallback, useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Users, List, Sparkles, Play, ChevronDown } from 'lucide-react';
import { clsx } from 'clsx';
import { books, characters, generation, type Book, type Character, type ChapterSummary } from '../lib/api';

type Tab = 'chapters' | 'characters';

function RoleBadge({ role }: { role: string | null }) {
  const colors: Record<string, string> = {
    protagonist: 'bg-accent/15 text-accent',
    supporting: 'bg-blue-500/15 text-blue-400',
    minor: 'bg-zinc-700/40 text-zinc-400',
  };
  return (
    <span className={clsx('rounded-full px-2 py-0.5 text-[10px] font-medium capitalize', colors[role ?? 'minor'] ?? colors.minor)}>
      {role ?? 'unknown'}
    </span>
  );
}

export default function BookDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [book, setBook] = useState<Book | null>(null);
  const [chars, setChars] = useState<Character[]>([]);
  const [tab, setTab] = useState<Tab>('chapters');
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    if (!id) return;
    try {
      const b = await books.get(Number(id));
      setBook(b);
      const c = await characters.listForBook(Number(id));
      setChars(c);
    } catch { /* empty */ }
    setLoading(false);
  }, [id]);

  useEffect(() => { load(); }, [load]);

  const handleAnalyze = async () => {
    if (!book) return;
    await books.analyze(book.id);
    setTimeout(load, 2000);
  };

  const handleGenerate = async () => {
    if (!book) return;
    await generation.create({ book_id: book.id, is_auto_mode: true });
    navigate('/generate');
  };

  if (loading) return (
    <div className="flex h-screen items-center justify-center">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent border-t-transparent" />
    </div>
  );

  if (!book) return (
    <div className="flex h-screen flex-col items-center justify-center text-zinc-500">
      <p>Book not found</p>
      <button onClick={() => navigate('/')} className="mt-3 text-sm text-accent hover:text-accent-hover">Back to Library</button>
    </div>
  );

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Top bar */}
      <div className="sticky top-0 z-30 flex items-center justify-between border-b border-border-subtle bg-surface/80 px-8 py-4 backdrop-blur-md">
        <div className="flex items-center gap-4">
          <button onClick={() => navigate('/')} className="rounded-lg p-2 text-zinc-500 transition hover:bg-white/5 hover:text-zinc-300">
            <ArrowLeft size={16} />
          </button>
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">{book.title}</h1>
            {book.author && <p className="text-xs text-zinc-500">{book.author}</p>}
          </div>
        </div>
        <div className="flex items-center gap-3">
          {book.is_parsed && !book.is_analyzed && (
            <button onClick={handleAnalyze} className="flex items-center gap-2 rounded-lg border border-border-default bg-surface-raised px-4 py-2 text-sm font-medium text-zinc-300 transition hover:bg-surface-overlay">
              <Sparkles size={14} />
              Analyze Characters
            </button>
          )}
          {book.is_parsed && (
            <button onClick={handleGenerate} className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white shadow-lg shadow-accent/20 transition hover:bg-accent-hover">
              <Play size={14} />
              Generate Audiobook
            </button>
          )}
        </div>
      </div>

      {/* Stats bar */}
      <div className="flex gap-6 border-b border-border-subtle px-8 py-4">
        {[
          { label: 'Chapters', value: book.total_chapters },
          { label: 'Words', value: book.total_words.toLocaleString() },
          { label: 'Characters', value: book.total_characters },
          { label: 'Format', value: book.format.toUpperCase() },
        ].map(s => (
          <div key={s.label}>
            <p className="text-xs text-zinc-500">{s.label}</p>
            <p className="text-sm font-semibold text-zinc-200">{s.value}</p>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-border-subtle px-8 pt-2">
        {([['chapters', List, 'Chapters'], ['characters', Users, 'Characters']] as const).map(([key, Icon, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={clsx(
              'flex items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium transition',
              tab === key ? 'border-accent text-accent' : 'border-transparent text-zinc-500 hover:text-zinc-300',
            )}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      {/* Content -- scrolls independently */}
      <div className="flex-1 overflow-y-auto p-8">
        {tab === 'chapters' && (
          <div className="space-y-2">
            {book.chapters?.length ? book.chapters.map(ch => (
              <ChapterRow key={ch.id} chapter={ch} />
            )) : (
              <p className="py-12 text-center text-sm text-zinc-500">
                {book.is_parsed ? 'No chapters found' : 'Book is being parsed...'}
              </p>
            )}
          </div>
        )}

        {tab === 'characters' && (
          <div className="space-y-2">
            {chars.length ? chars.map(c => (
              <div key={c.id} className="flex items-center justify-between rounded-lg border border-border-subtle bg-surface-raised p-4 transition hover:bg-surface-overlay">
                <div className="flex items-center gap-4">
                  <div className="flex h-9 w-9 items-center justify-center rounded-full bg-accent/10 text-xs font-bold text-accent">
                    {c.name.charAt(0)}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-zinc-200">{c.name}</p>
                    <p className="text-xs text-zinc-500">
                      {c.dialogue_count} lines
                      {c.inferred_gender && <> &middot; {c.inferred_gender}</>}
                    </p>
                  </div>
                </div>
                <RoleBadge role={c.role} />
              </div>
            )) : (
              <p className="py-12 text-center text-sm text-zinc-500">
                {book.is_analyzed ? 'No characters detected' : 'Run analysis to detect characters'}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ChapterRow({ chapter }: { chapter: ChapterSummary }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-border-subtle bg-surface-raised transition hover:bg-surface-overlay">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center justify-between p-4 text-left">
        <div className="flex items-center gap-3">
          <span className="flex h-7 w-7 items-center justify-center rounded-md bg-surface-overlay text-xs font-semibold text-zinc-400">
            {chapter.number}
          </span>
          <div>
            <p className="text-sm font-medium text-zinc-200">{chapter.title ?? `Chapter ${chapter.number}`}</p>
            <p className="text-xs text-zinc-500">{chapter.word_count.toLocaleString()} words</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {chapter.is_generated && (
            <span className="rounded-full bg-success/15 px-2 py-0.5 text-[10px] font-medium text-success">Audio ready</span>
          )}
          <ChevronDown size={14} className={clsx('text-zinc-500 transition', open && 'rotate-180')} />
        </div>
      </button>
      {open && (
        <div className="border-t border-border-subtle px-4 py-3 text-xs text-zinc-500">
          Segment editor will be available after analysis and generation.
        </div>
      )}
    </div>
  );
}
