import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { Library, BookOpen, Mic2, AudioWaveform, Activity } from 'lucide-react';
import { clsx } from 'clsx';
import LibraryPage from './pages/LibraryPage';
import BookDetailPage from './pages/BookDetailPage';
import VoiceStudioPage from './pages/VoiceStudioPage';
import GenerationPage from './pages/GenerationPage';

const NAV = [
  { to: '/', icon: Library, label: 'Library' },
  { to: '/voices', icon: Mic2, label: 'Voice Studio' },
  { to: '/generate', icon: AudioWaveform, label: 'Generate' },
];

function Sidebar() {
  const location = useLocation();
  return (
    <aside className="fixed inset-y-0 left-0 z-40 flex w-[220px] flex-col border-r border-border-subtle bg-surface">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-5 py-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/20">
          <BookOpen size={16} className="text-accent" />
        </div>
        <div className="leading-tight">
          <span className="text-sm font-semibold text-zinc-100">AudioBook</span>
          <span className="block text-[10px] font-medium uppercase tracking-widest text-zinc-500">Creator</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="mt-2 flex flex-1 flex-col gap-0.5 px-3">
        {NAV.map(({ to, icon: Icon, label }) => {
          const active = to === '/' ? location.pathname === '/' : location.pathname.startsWith(to);
          return (
            <NavLink
              key={to}
              to={to}
              className={clsx(
                'group flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] font-medium transition-all duration-150',
                active
                  ? 'bg-accent/10 text-accent'
                  : 'text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200',
              )}
            >
              <Icon size={16} className={clsx('shrink-0', active ? 'text-accent' : 'text-zinc-500 group-hover:text-zinc-300')} />
              {label}
            </NavLink>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-border-subtle px-3 py-3">
        <div className="flex items-center gap-2 px-3 py-1.5">
          <Activity size={12} className="text-success" />
          <span className="text-[11px] text-zinc-500">System Online</span>
        </div>
      </div>
    </aside>
  );
}

function Layout() {
  return (
    <div className="flex h-screen overflow-hidden bg-surface font-sans">
      <Sidebar />
      <main className="ml-[220px] flex min-h-0 flex-1 flex-col overflow-hidden">
        <Routes>
          <Route path="/" element={<LibraryPage />} />
          <Route path="/book/:id" element={<BookDetailPage />} />
          <Route path="/voices" element={<VoiceStudioPage />} />
          <Route path="/generate" element={<GenerationPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  );
}
