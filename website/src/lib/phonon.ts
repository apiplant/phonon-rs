/** Loads phonon-rs's wasm module (in a Web Worker) and the Phonon-2 model, fetched straight from the
 * Hugging Face Hub or read from a local folder, entirely client-side — no backend involved.
 *
 * The archive is cached with the browser Cache API (origin-scoped, so it survives reloads) keyed by its
 * Hugging Face URL, since `resolve/main/...` content is immutable per commit.
 */

import { createSignal } from "solid-js";
import type { WorkerRequest, WorkerResponse } from "./phonon.worker";
import {
  fsAccessSupported,
  forgetRememberedDirectory,
  loadRememberedDirectory,
  pickDirectory,
  queryReadPermission,
  rememberDirectory,
  requestReadPermission,
} from "./localFs";

export { fsAccessSupported };

export const MODEL = {
  label: "Phonon-2",
  hfRepo: "FermionResearch/Phonon-2",
  file: "phonon-2.bps.tar.zst",
  approxSizeMb: 164,
  unpackDir: "model_phonon2_c4c_int6",
};

export const MODEL_URL = `https://huggingface.co/${MODEL.hfRepo}/resolve/main/${MODEL.file}`;

export interface Word {
  word: string;
  start: number;
  end: number;
  confidence: number;
}

export interface Transcript {
  text: string;
  words: Word[];
  srt: string;
  confidence: number;
}

export interface DownloadProgress {
  loaded: number;
  total: number | null;
}

export type Source = "hf" | "local";

/* ------------------------------------------------------------------ */
/* The worker                                                          */
/* ------------------------------------------------------------------ */

let worker: Worker | null = null;
let nextId = 1;
const listeners = new Set<(r: WorkerResponse) => void>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./phonon.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => listeners.forEach((fn) => fn(e.data));
    worker.onerror = (e) => {
      listeners.forEach((fn) => fn({ type: "error", message: e.message || "the worker crashed (out of memory?)" }));
    };
  }
  return worker;
}

function send(msg: WorkerRequest, transfer: Transferable[] = []) {
  getWorker().postMessage(msg, transfer);
}

/** Resolves with the first response `pick` accepts; rejects on a worker error. */
function waitFor<T>(pick: (r: WorkerResponse) => T | undefined, onOther?: (r: WorkerResponse) => void): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const fn = (r: WorkerResponse) => {
      if (r.type === "error") {
        listeners.delete(fn);
        reject(new Error(r.message));
        return;
      }
      const got = pick(r);
      if (got !== undefined) {
        listeners.delete(fn);
        resolve(got);
      } else onOther?.(r);
    };
    listeners.add(fn);
  });
}

/* ------------------------------------------------------------------ */
/* Model state                                                         */
/* ------------------------------------------------------------------ */

const [loaded, setLoaded] = createSignal<{ source: Source; ms: number } | null>(null);
export const loadedModel = loaded;

const CACHE_NAME = "phonon-rs-model-v1";

async function fetchArchive(onProgress: (p: DownloadProgress) => void): Promise<ArrayBuffer> {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(MODEL_URL);
  if (cached) {
    const buf = await cached.arrayBuffer();
    onProgress({ loaded: buf.byteLength, total: buf.byteLength });
    return buf;
  }

  const resp = await fetch(MODEL_URL);
  if (!resp.ok || !resp.body) throw new Error(`fetching ${MODEL_URL}: HTTP ${resp.status}`);
  const totalHeader = resp.headers.get("content-length");
  const total = totalHeader ? Number(totalHeader) : null;

  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress({ loaded: received, total });
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }

  // Best-effort: cache writes can fail (private browsing, quota) — the model still loads.
  try {
    await cache.put(MODEL_URL, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
  } catch {
    /* ignore */
  }
  return bytes.buffer;
}

export async function clearModelCache(): Promise<void> {
  await caches.delete(CACHE_NAME);
}

export async function isCached(): Promise<boolean> {
  if (typeof caches === "undefined") return false;
  try {
    return (await (await caches.open(CACHE_NAME)).match(MODEL_URL)) !== undefined;
  } catch {
    return false;
  }
}

/** Downloads (or reuses the cached) archive and loads it into the worker. */
export async function loadFromHuggingFace(onProgress: (p: DownloadProgress) => void, onPhase: (phase: string) => void) {
  onPhase("Downloading");
  const archive = await fetchArchive(onProgress);
  onPhase("Unpacking and expanding the weights");
  const done = waitFor((r) => (r.type === "loaded" ? r.ms : undefined));
  send({ type: "load-archive", archive }, [archive]);
  setLoaded({ source: "hf", ms: await done });
}

/* ------------------------------------------------------------------ */
/* Local folder                                                        */
/* ------------------------------------------------------------------ */

type ModelFiles =
  | { kind: "archive"; file: File; where: string }
  | { kind: "container"; container: File; config: File; where: string };

async function maybeFile(dir: FileSystemDirectoryHandle, name: string): Promise<File | null> {
  try {
    return await (await dir.getFileHandle(name)).getFile();
  } catch {
    return null;
  }
}

async function maybeDir(dir: FileSystemDirectoryHandle, name: string): Promise<FileSystemDirectoryHandle | null> {
  try {
    return await dir.getDirectoryHandle(name);
  } catch {
    return null;
  }
}

/** Finds Phonon-2 in a directory: the release archive, an unpacked `model.fermion` + `config.json`, or
 * either of those one level down (a Hugging Face checkout, or the unpacked directory inside it). */
export async function findModelFiles(dir: FileSystemDirectoryHandle, depth = 0): Promise<ModelFiles | null> {
  const archive = await maybeFile(dir, MODEL.file);
  if (archive) return { kind: "archive", file: archive, where: `${dir.name}/${MODEL.file}` };
  const container = await maybeFile(dir, "model.fermion");
  const config = await maybeFile(dir, "config.json");
  if (container && config) return { kind: "container", container, config, where: `${dir.name}/model.fermion` };
  if (depth > 0) return null;
  for (const name of [MODEL.unpackDir, "Phonon-2"]) {
    const sub = await maybeDir(dir, name);
    if (sub) {
      const found = await findModelFiles(sub, depth + 1);
      if (found) return { ...found, where: `${dir.name}/${found.where}` };
    }
  }
  return null;
}

export interface LocalFolder {
  handle: FileSystemDirectoryHandle;
  name: string;
  files: ModelFiles | null;
}

const [localFolder, setLocalFolder] = createSignal<LocalFolder | null>(null);
export { localFolder };

/** A remembered folder whose read permission has to be re-granted by a click. */
const [pendingFolder, setPendingFolder] = createSignal<FileSystemDirectoryHandle | null>(null);
export { pendingFolder };

async function adopt(handle: FileSystemDirectoryHandle): Promise<LocalFolder> {
  const folder = { handle, name: handle.name, files: await findModelFiles(handle) };
  setLocalFolder(folder);
  setPendingFolder(null);
  return folder;
}

export async function chooseLocalFolder(): Promise<LocalFolder | null> {
  const handle = await pickDirectory();
  if (!handle) return null;
  await rememberDirectory(handle);
  return adopt(handle);
}

let resumeAttempted = false;
/** Reopens the folder remembered from a previous visit without a picker. If the browser still grants read
 * access it becomes active at once; otherwise it waits for a click on `grantPendingFolder`. */
export function resumeLocalFolder(): void {
  if (resumeAttempted || localFolder()) return;
  resumeAttempted = true;
  void (async () => {
    const handle = await loadRememberedDirectory();
    if (!handle) return;
    if ((await queryReadPermission(handle)) === "granted") await adopt(handle);
    else setPendingFolder(handle);
  })();
}

export async function grantPendingFolder(): Promise<LocalFolder | null> {
  const handle = pendingFolder();
  if (!handle) return null;
  if ((await requestReadPermission(handle)) !== "granted") return null;
  return adopt(handle);
}

export async function forgetLocalFolder(): Promise<void> {
  await forgetRememberedDirectory();
  setLocalFolder(null);
  setPendingFolder(null);
}

export async function loadFromLocal(folder: LocalFolder, onPhase: (phase: string) => void) {
  const files = folder.files;
  if (!files) throw new Error(`Phonon-2 was not found in ${folder.name}/`);
  onPhase("Reading the files");
  if (files.kind === "archive") {
    const archive = await files.file.arrayBuffer();
    onPhase("Unpacking and expanding the weights");
    const done = waitFor((r) => (r.type === "loaded" ? r.ms : undefined));
    send({ type: "load-archive", archive }, [archive]);
    setLoaded({ source: "local", ms: await done });
  } else {
    const container = await files.container.arrayBuffer();
    const config = await files.config.text();
    onPhase("Expanding the weights");
    const done = waitFor((r) => (r.type === "loaded" ? r.ms : undefined));
    send({ type: "load-model", container, config }, [container]);
    setLoaded({ source: "local", ms: await done });
  }
}

export function unloadModel(): void {
  send({ type: "unload" });
  setLoaded(null);
}

/* ------------------------------------------------------------------ */
/* Audio                                                               */
/* ------------------------------------------------------------------ */

export interface Audio {
  samples: Float32Array;
  sampleRate: number;
  duration: number;
}

/** Decodes any audio (or video) container the browser understands to mono samples at 16 kHz. Asking the
 * AudioContext for 16 kHz makes the browser resample while it decodes. */
export async function decodeAudio(data: ArrayBuffer): Promise<Audio> {
  const ctx = new AudioContext({ sampleRate: 16000 });
  try {
    const buffer = await ctx.decodeAudioData(data);
    const mono = new Float32Array(buffer.length);
    const channels = buffer.numberOfChannels;
    for (let c = 0; c < channels; c++) {
      const ch = buffer.getChannelData(c);
      for (let i = 0; i < mono.length; i++) mono[i] += ch[i] / channels;
    }
    return { samples: mono, sampleRate: buffer.sampleRate, duration: buffer.duration };
  } finally {
    void ctx.close();
  }
}

export async function transcribe(
  audio: Audio,
  onProgress: (done: number, total: number) => void,
  chunkSecs = 30,
): Promise<{ transcript: Transcript; ms: number }> {
  const id = nextId++;
  // Copy: the buffer is transferred to the worker, and the caller keeps its audio for replays.
  const samples = audio.samples.slice();
  const result = waitFor(
    (r) => (r.type === "result" && r.id === id ? { json: r.json, ms: r.ms } : undefined),
    (r) => {
      if (r.type === "progress" && r.id === id) onProgress(r.done, r.total);
    },
  );
  send({ type: "transcribe", id, samples, sampleRate: audio.sampleRate, chunkSecs }, [samples.buffer]);
  const { json, ms } = await result;
  return { transcript: JSON.parse(json) as Transcript, ms };
}
