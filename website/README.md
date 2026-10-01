# phonon-rs website

The website for [phonon-rs](https://github.com/apiplant/phonon-rs). Solid 2 RC + Tailwind v4 + Vite, static build, deployed to Cloudflare.

```bash
pnpm install
pnpm dev       # local dev server
pnpm build     # -> dist/
pnpm check     # types only
pnpm build:wasm  # regenerate src/wasm-pkg/ (needs Rust + wasm-pack)
```

## The wasm package is committed

The `/demo` page runs phonon-rs itself in the browser (in a Web Worker), so the site depends on `src/wasm-pkg/`,
the `wasm-pack` output for the crate one directory up. It is **committed** so a deploy needs only Node. After changing
anything under `../src/`, run `pnpm build:wasm` and commit the result.

The version shown on the site is read from `../Cargo.toml` at build time.
