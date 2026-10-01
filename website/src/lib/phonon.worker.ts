/** Runs the Phonon-2 wasm module off the main thread: loading expands ~2.4 GB of f32 weights and a
 * transcription takes seconds to minutes, either of which would freeze the page. */

import init, { WasmPhonon } from "../wasm-pkg/phonon.js";

export type WorkerRequest =
  | { type: "load-archive"; archive: ArrayBuffer }
  | { type: "load-model"; container: ArrayBuffer; config: string }
  | { type: "transcribe"; id: number; samples: Float32Array; sampleRate: number; chunkSecs: number }
  | { type: "unload" };

export type WorkerResponse =
  | { type: "loaded"; ms: number }
  | { type: "progress"; id: number; done: number; total: number }
  | { type: "result"; id: number; json: string; ms: number }
  | { type: "error"; id?: number; message: string };

let model: WasmPhonon | null = null;
let ready: Promise<unknown> | null = null;

function post(message: WorkerResponse) {
  (self as unknown as Worker).postMessage(message);
}

function describe(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  try {
    if (msg.type === "unload") {
      model?.free();
      model = null;
      return;
    }
    ready ??= init();
    await ready;

    if (msg.type === "load-archive" || msg.type === "load-model") {
      model?.free();
      model = null;
      const t = performance.now();
      model =
        msg.type === "load-archive"
          ? WasmPhonon.loadArchive(new Uint8Array(msg.archive))
          : WasmPhonon.loadModel(new Uint8Array(msg.container), msg.config);
      post({ type: "loaded", ms: performance.now() - t });
    } else if (msg.type === "transcribe") {
      if (!model) throw new Error("no model loaded");
      const t = performance.now();
      const json = model.transcribe(msg.samples, msg.sampleRate, msg.chunkSecs, (done: number, total: number) =>
        post({ type: "progress", id: msg.id, done, total }),
      );
      post({ type: "result", id: msg.id, json, ms: performance.now() - t });
    }
  } catch (e) {
    post({ type: "error", id: msg.type === "transcribe" ? msg.id : undefined, message: describe(e) });
  }
};
