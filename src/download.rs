//! Downloads the Phonon-2 release archive from Hugging Face into a cache directory.
//!
//! Cache location: `$XDG_CACHE_HOME/phonon-rs/Phonon-2/phonon-2.bps.tar.zst`, falling back to
//! `~/.cache/phonon-rs/...` when `$XDG_CACHE_HOME` is unset. The file is fetched to a `.part`
//! sibling and renamed into place once complete, so a killed download never leaves an archive
//! that looks done. [`crate::fermion::load_files`] reads the archive as it is.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// The Hugging Face repository the model comes from.
pub const HF_REPO: &str = "FermionResearch/Phonon-2";
/// The release archive inside it (about 164 MB).
pub const ARCHIVE: &str = "phonon-2.bps.tar.zst";

/// `$XDG_CACHE_HOME/phonon-rs`, defaulting to `~/.cache/phonon-rs`.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("phonon-rs"))
}

/// Where [`download`] puts (or would put) the archive.
pub fn archive_path() -> Option<PathBuf> {
    Some(cache_dir()?.join(HF_REPO.rsplit('/').next().unwrap_or(HF_REPO)).join(ARCHIVE))
}

fn fetch_to_file(url: &str, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let part = dest.with_extension("zst.part");

    let resp = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let len: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());

    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let (mut written, mut last_report) = (0u64, 0u64);
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("reading body for {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).with_context(|| format!("writing {}", part.display()))?;
        written += n as u64;
        if written - last_report >= 8 << 20 {
            last_report = written;
            match len {
                Some(len) => eprint!("\r  {ARCHIVE} {:.0}% ({} / {} MiB)  ", written as f64 * 100.0 / len as f64, written >> 20, len >> 20),
                None => eprint!("\r  {ARCHIVE} {} MiB  ", written >> 20),
            }
            std::io::stderr().flush().ok();
        }
    }
    drop(file);

    if let Some(len) = len {
        if written != len {
            let _ = std::fs::remove_file(&part);
            bail!("short read for {url}: got {written} bytes, expected {len}");
        }
    }
    std::fs::rename(&part, dest).with_context(|| format!("renaming {} to {}", part.display(), dest.display()))?;
    eprintln!("\r  {ARCHIVE} done ({} MiB)          ", written >> 20);
    Ok(())
}

/// Downloads the release archive into the cache directory unless it is already there, and returns
/// its path. Needs network access on first use; any failure (offline, 404, disk error, ...) is
/// returned as an error and nothing partial is left looking complete.
pub fn download() -> Result<PathBuf> {
    let dest = archive_path().context("no cache directory available (set $HOME or $XDG_CACHE_HOME)")?;
    if dest.is_file() {
        return Ok(dest);
    }
    eprintln!("Downloading {HF_REPO} (about 164 MB) from https://huggingface.co/{HF_REPO} into {}", dest.display());
    fetch_to_file(&format!("https://huggingface.co/{HF_REPO}/resolve/main/{ARCHIVE}"), &dest)?;
    Ok(dest)
}
