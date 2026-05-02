const BASE = '';

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts?.headers },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// ---- Types ----
export interface Book {
  id: number;
  title: string;
  author: string | null;
  format: string;
  original_filename: string;
  file_size_bytes: number;
  total_chapters: number;
  total_words: number;
  total_characters: number;
  is_parsed: boolean;
  is_analyzed: boolean;
  language: string | null;
  created_at: string;
  chapters?: ChapterSummary[];
  characters?: Character[];
}
export interface ChapterSummary {
  id: number;
  number: number;
  title: string | null;
  word_count: number;
  is_generated: boolean;
  audio_duration_seconds: number | null;
}
export interface Character {
  id: number;
  name: string;
  description: string | null;
  role: string | null;
  inferred_gender: string | null;
  inferred_age: string | null;
  dialogue_count: number;
  voice_profile_id: number | null;
  first_appearance_chapter: number | null;
}
export interface VoiceProfile {
  id: number;
  name: string;
  description: string | null;
  gender: string | null;
  age_range: string | null;
  voice_quality: string | null;
  temperature: number;
  top_k: number;
  top_p: number;
  speaking_rate: number;
  pitch_shift: number;
  seed: number | null;
  is_default: boolean;
  created_at: string;
}
export interface GenerationJob {
  id: number;
  job_id: string;
  book_id: number;
  status: string;
  progress: number;
  current_step: string | null;
  current_chapter: number | null;
  total_segments: number;
  completed_segments: number;
  is_auto_mode: boolean;
  output_format: string;
  error_message: string | null;
  created_at: string;
}
export interface Segment {
  id: number;
  sequence_number: number;
  text: string;
  segment_type: string;
  character_id: number | null;
  emotion: string | null;
  emphasis: string | null;
  pacing: string | null;
  is_generated: boolean;
  audio_duration_seconds: number | null;
  user_text_override: string | null;
  user_emphasis_override: string | null;
}
export interface HealthInfo {
  status: string;
  platform: Record<string, unknown>;
  tts_engine: string;
}

// ---- Books ----
export const books = {
  list: () => request<{ books: Book[]; total: number }>('/api/books/'),
  get: (id: number) => request<Book>(`/api/books/${id}`),
  upload: async (file: File) => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/books/upload', { method: 'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<Book>;
  },
  delete: (id: number) => request(`/api/books/${id}`, { method: 'DELETE' }),
  parse: (id: number) => request(`/api/books/${id}/parse`, { method: 'POST' }),
  analyze: (id: number) => request(`/api/books/${id}/analyze`, { method: 'POST' }),
  dramatisPersonae: (id: number) => request<{ book_id: number; book_title: string; characters: Character[]; total_characters: number }>(`/api/books/${id}/dramatis-personae`),
};

// ---- Characters ----
export const characters = {
  listForBook: (bookId: number) => request<Character[]>(`/api/characters/book/${bookId}`),
  update: (id: number, data: Partial<Character>) => request<Character>(`/api/characters/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
};

// ---- Voices ----
export const voices = {
  list: () => request<VoiceProfile[]>('/api/voices/'),
  get: (id: number) => request<VoiceProfile>(`/api/voices/${id}`),
  create: (data: Partial<VoiceProfile>) => request<VoiceProfile>('/api/voices/', { method: 'POST', body: JSON.stringify(data) }),
  update: (id: number, data: Partial<VoiceProfile>) => request<VoiceProfile>(`/api/voices/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  delete: (id: number) => request(`/api/voices/${id}`, { method: 'DELETE' }),
};

// ---- Generation ----
export const generation = {
  create: (data: { book_id: number; is_auto_mode?: boolean; output_format?: string }) => request<GenerationJob>('/api/generation/jobs', { method: 'POST', body: JSON.stringify(data) }),
  list: (bookId?: number) => request<GenerationJob[]>(`/api/generation/jobs${bookId ? `?book_id=${bookId}` : ''}`),
  get: (jobId: string) => request<GenerationJob>(`/api/generation/jobs/${jobId}`),
  cancel: (jobId: string) => request(`/api/generation/jobs/${jobId}/cancel`, { method: 'POST' }),
  pause: (jobId: string) => request(`/api/generation/jobs/${jobId}/pause`, { method: 'POST' }),
  resume: (jobId: string) => request<GenerationJob>(`/api/generation/jobs/${jobId}/resume`, { method: 'POST' }),
  segments: (chapterId: number) => request<Segment[]>(`/api/generation/chapters/${chapterId}/segments`),
};

// ---- Health ----
export const health = {
  check: () => request<HealthInfo>('/health'),
};
