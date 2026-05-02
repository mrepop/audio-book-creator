import { useCallback, useEffect, useState } from 'react';
import { Plus, Trash2, Mic2, Sliders, Save } from 'lucide-react';
import { clsx } from 'clsx';
import { voices, type VoiceProfile } from '../lib/api';

function Slider({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-xs text-zinc-400">{label}</span>
        <span className="text-xs font-mono text-zinc-500">{value.toFixed(2)}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full accent-accent"
      />
    </div>
  );
}

export default function VoiceStudioPage() {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
  const [selected, setSelected] = useState<VoiceProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [draft, setDraft] = useState<Partial<VoiceProfile>>({});

  const load = useCallback(async () => {
    try {
      const p = await voices.list();
      setProfiles(p);
    } catch { /* empty */ }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const select = (p: VoiceProfile) => {
    setSelected(p);
    setDraft({ ...p });
  };

  const handleCreate = async () => {
    const name = prompt('Voice profile name:');
    if (!name) return;
    const created = await voices.create({ name, temperature: 0.7, speaking_rate: 1.0, pitch_shift: 0, top_k: 50, top_p: 1.0 });
    await load();
    select(created);
  };

  const handleSave = async () => {
    if (!selected) return;
    setSaving(true);
    await voices.update(selected.id, draft);
    await load();
    setSaving(false);
  };

  const handleDelete = async (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    if (!confirm('Delete this voice profile?')) return;
    await voices.delete(id);
    if (selected?.id === id) setSelected(null);
    await load();
  };

  return (
    <div className="flex h-full overflow-hidden">
      {/* Profile list */}
      <div className="flex w-[280px] shrink-0 flex-col border-r border-border-subtle">
        <div className="flex items-center justify-between border-b border-border-subtle px-5 py-4">
          <h2 className="text-sm font-semibold text-zinc-200">Voice Profiles</h2>
          <button onClick={handleCreate} className="rounded-md bg-accent/15 p-1.5 text-accent transition hover:bg-accent/25">
            <Plus size={14} />
          </button>
        </div>
        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto p-3">
          {loading ? (
            <div className="flex h-20 items-center justify-center">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-accent border-t-transparent" />
            </div>
          ) : profiles.length === 0 ? (
            <p className="py-8 text-center text-xs text-zinc-500">No voice profiles yet</p>
          ) : profiles.map(p => (
            <button
              key={p.id}
              onClick={() => select(p)}
              className={clsx(
                'group flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-left transition',
                selected?.id === p.id ? 'bg-accent/10' : 'hover:bg-white/[0.03]',
              )}
            >
              <div className="flex items-center gap-2.5">
                <div className={clsx(
                  'flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold',
                  selected?.id === p.id ? 'bg-accent/20 text-accent' : 'bg-surface-overlay text-zinc-500',
                )}>
                  <Mic2 size={12} />
                </div>
                <div>
                  <p className={clsx('text-xs font-medium', selected?.id === p.id ? 'text-accent' : 'text-zinc-300')}>{p.name}</p>
                  <p className="text-[10px] text-zinc-600">{p.gender ?? 'neutral'} &middot; {p.age_range ?? 'adult'}</p>
                </div>
              </div>
              <button onClick={e => handleDelete(e, p.id)} className="rounded-md p-1 text-zinc-600 opacity-0 transition hover:text-danger group-hover:opacity-100">
                <Trash2 size={12} />
              </button>
            </button>
          ))}
        </div>
      </div>

      {/* Editor */}
      <div className="min-h-0 flex-1 overflow-y-auto p-8">
        {selected ? (
          <div className="mx-auto max-w-lg">
            <div className="mb-6 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Sliders size={18} className="text-accent" />
                <h2 className="text-lg font-semibold text-zinc-100">{draft.name ?? selected.name}</h2>
              </div>
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white shadow-lg shadow-accent/20 transition hover:bg-accent-hover disabled:opacity-50"
              >
                <Save size={14} />
                {saving ? 'Saving...' : 'Save'}
              </button>
            </div>

            {/* Name + description */}
            <div className="mb-6 space-y-3">
              <div>
                <label className="mb-1 block text-xs text-zinc-400">Name</label>
                <input
                  value={draft.name ?? ''} onChange={e => setDraft(d => ({ ...d, name: e.target.value }))}
                  className="w-full rounded-lg border border-border-default bg-surface-raised px-3 py-2 text-sm text-zinc-200 outline-none focus:border-accent/50"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-1 block text-xs text-zinc-400">Gender</label>
                  <select value={draft.gender ?? ''} onChange={e => setDraft(d => ({ ...d, gender: e.target.value || null }))} className="w-full rounded-lg border border-border-default bg-surface-raised px-3 py-2 text-sm text-zinc-200 outline-none">
                    <option value="">Neutral</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-xs text-zinc-400">Age Range</label>
                  <select value={draft.age_range ?? ''} onChange={e => setDraft(d => ({ ...d, age_range: e.target.value || null }))} className="w-full rounded-lg border border-border-default bg-surface-raised px-3 py-2 text-sm text-zinc-200 outline-none">
                    <option value="">Adult</option>
                    <option value="child">Child</option>
                    <option value="young">Young</option>
                    <option value="elderly">Elderly</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Voice parameters */}
            <div className="space-y-5 rounded-xl border border-border-subtle bg-surface-raised p-5">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Voice Parameters</h3>
              <Slider label="Temperature" value={draft.temperature ?? 0.7} min={0.1} max={1.0} step={0.05} onChange={v => setDraft(d => ({ ...d, temperature: v }))} />
              <Slider label="Speaking Rate" value={draft.speaking_rate ?? 1.0} min={0.5} max={1.5} step={0.05} onChange={v => setDraft(d => ({ ...d, speaking_rate: v }))} />
              <Slider label="Pitch Shift" value={draft.pitch_shift ?? 0} min={-12} max={12} step={0.5} onChange={v => setDraft(d => ({ ...d, pitch_shift: v }))} />
              <div>
                <label className="mb-1 block text-xs text-zinc-400">Seed (for consistency)</label>
                <input
                  type="number" value={draft.seed ?? ''} placeholder="Random"
                  onChange={e => setDraft(d => ({ ...d, seed: e.target.value ? Number(e.target.value) : null }))}
                  className="w-full rounded-lg border border-border-default bg-surface-overlay px-3 py-2 text-sm text-zinc-200 outline-none focus:border-accent/50"
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="flex h-full flex-col items-center justify-center text-zinc-500">
            <Mic2 size={40} className="mb-4 text-zinc-700" />
            <p className="text-sm">Select a voice profile to edit</p>
            <p className="mt-1 text-xs text-zinc-600">or create a new one</p>
          </div>
        )}
      </div>
    </div>
  );
}
