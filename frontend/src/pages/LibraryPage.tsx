import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, BookOpen, Trash2, FileText, ChevronRight, Search } from 'lucide-react';
import { clsx } from 'clsx';
import { books, type Book } from '../lib/api';

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1048576).toFixed(1)} MB`;
}

function StatusPill({ parsed, analyzed }: { parsed: boolean; analyzed: boolean }) {
  if (analyzed) return <span className="rounded-full bg-success/15 px-2.5 py-0.5 text-[11px] font-medium text-success">Ready</span>;
  if (parsed) return <span className="rounded-full bg-warning/15 px-2.5 py-0.5 text-[11px] font-medium text-warning">Parsed</span>;
  return <span className="rounded-full bg-zinc-700/40 px-2.5 py-0.5 text-[11px] font-medium text-zinc-400">Uploaded</span>;
}

export default function LibraryPage() {
  const navigate = useNavigate();
  const [bookList, setBookList] = useState<Book[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  const load = useCallback(async () => {
    try {
      const { books: b } = await books.list();
      setBookList(b);
    } catch { /* empty */ }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleUpload = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        await books.upload(file);
      }
      await load();
    } catch (e) {
      console.error(e);
    }
    setUploading(false);
  };

  const handleDelete = async (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    if (!confirm('Delete this book and all associated data?')) return;
    await books.delete(id);
    await load();
  };

  const filtered = bookList.filter(b =>
    b.title.toLowerCase().includes(search.toLowerCase()) ||
    b.original_filename.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="h-full overflow-y-auto p-8">
      {/* Header */}
      <div className="mb-8 flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">Library</h1>
          <p className="mt-1 text-sm text-zinc-500">Upload and manage your ebooks</p>
        </div>
        <label className={clsx(
          'flex cursor-pointer items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all',
          'bg-accent text-white hover:bg-accent-hover shadow-lg shadow-accent/20',
          uploading && 'pointer-events-none opacity-60',
        )}>
          <Upload size={15} />
          {uploading ? 'Uploading...' : 'Upload Book'}
          <input type="file" className="hidden" accept=".epub,.pdf,.txt" multiple onChange={e => handleUpload(e.target.files)} />
        </label>
      </div>

      {/* Search */}
      <div className="relative mb-6">
        <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
        <input
          type="text"
          placeholder="Search books..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="w-full rounded-lg border border-border-default bg-surface-raised py-2.5 pl-9 pr-4 text-sm text-zinc-200 placeholder-zinc-600 outline-none transition-colors focus:border-accent/50 focus:bg-surface-overlay"
        />
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); handleUpload(e.dataTransfer.files); }}
        className={clsx(
          'mb-8 flex flex-col items-center justify-center rounded-xl border-2 border-dashed py-12 transition-all duration-200',
          dragging ? 'border-accent bg-accent/5' : 'border-border-default bg-surface-raised/50',
        )}
      >
        <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-surface-overlay">
          <FileText size={20} className="text-zinc-500" />
        </div>
        <p className="text-sm text-zinc-400">Drop ebook files here</p>
        <p className="mt-1 text-xs text-zinc-600">EPUB, PDF, or TXT</p>
      </div>

      {/* Book Grid */}
      {loading ? (
        <div className="flex h-40 items-center justify-center">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent border-t-transparent" />
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex h-40 flex-col items-center justify-center text-zinc-500">
          <BookOpen size={32} className="mb-3 text-zinc-600" />
          <p className="text-sm">{bookList.length ? 'No matches' : 'No books yet'}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map(book => (
            <button
              key={book.id}
              onClick={() => navigate(`/book/${book.id}`)}
              className="group relative flex flex-col rounded-xl border border-border-subtle bg-surface-raised p-5 text-left transition-all duration-200 hover:border-border-default hover:bg-surface-overlay"
            >
              <div className="mb-3 flex items-start justify-between">
                <StatusPill parsed={book.is_parsed} analyzed={book.is_analyzed} />
                <button
                  onClick={e => handleDelete(e, book.id)}
                  className="rounded-md p-1.5 text-zinc-600 opacity-0 transition-all hover:bg-danger/10 hover:text-danger group-hover:opacity-100"
                >
                  <Trash2 size={14} />
                </button>
              </div>

              <h3 className="mb-1 text-sm font-semibold text-zinc-100 line-clamp-2">{book.title}</h3>
              {book.author && <p className="mb-3 text-xs text-zinc-500">{book.author}</p>}

              <div className="mt-auto flex items-center gap-4 border-t border-border-subtle pt-3 text-[11px] text-zinc-500">
                <span>{book.total_chapters} chapters</span>
                <span>{book.total_words.toLocaleString()} words</span>
                <span>{formatBytes(book.file_size_bytes)}</span>
                <span className="uppercase">{book.format}</span>
              </div>

              <ChevronRight size={14} className="absolute right-4 top-1/2 -translate-y-1/2 text-zinc-600 opacity-0 transition-all group-hover:opacity-100" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
