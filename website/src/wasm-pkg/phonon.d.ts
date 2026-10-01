/* tslint:disable */
/* eslint-disable */

export class WasmPhonon {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Loads the model from the bytes of `phonon-2.bps.tar.zst`. Float32 on CPU is the only combination that
     * makes sense in a browser tab.
     */
    static loadArchive(archive: Uint8Array): WasmPhonon;
    /**
     * Loads the model from an unpacked `model.fermion` (as bytes) and `config.json` (as text).
     */
    static loadModel(container: Uint8Array, config_json: string): WasmPhonon;
    /**
     * Transcribes mono audio at `sample_rate` Hz (resampled to 16 kHz first when it differs). Audio longer than
     * `chunk_secs` is cut at quiet points. `on_progress(done, total)` runs after every chunk.
     */
    transcribe(samples: Float32Array, sample_rate: number, chunk_secs: number, on_progress: Function): string;
}

/**
 * Readable panic messages (shape mismatches inside candle, ...) in the browser console.
 */
export function init_panic_hook(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmphonon_free: (a: number, b: number) => void;
    readonly init_panic_hook: () => void;
    readonly wasmphonon_loadArchive: (a: number, b: number) => [number, number, number];
    readonly wasmphonon_loadModel: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly wasmphonon_transcribe: (a: number, b: number, c: number, d: number, e: number, f: any) => [number, number, number, number];
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
