//! Push-button dictation for Linux (Wayland or X11): a hotkey starts and stops recording from the microphone,
//! and the transcript is typed into the focused window through a virtual keyboard.
//!
//! Keys are read straight from /dev/input (evdev) and text is typed through /dev/uinput, so it works under any
//! compositor. Both need the user in the `input` group (or equivalent udev rules).

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use evdev::uinput::VirtualDevice;
use evdev::{AttributeSet, EventSummary, KeyCode};
use phonon::engine::{DeviceArg, Engine, Precision};
use phonon::{audio, mic, text};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const VIRTUAL_KBD: &str = "phonon-dictate virtual keyboard";

/// Dictation with Phonon-2: press a hotkey, speak, press it again; the text is typed where your cursor is.
#[derive(Parser)]
#[command(name = "phonon-dictate", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    #[command(flatten)]
    run: RunArgs,
}

#[derive(Subcommand)]
enum Cmd {
    /// Press the key or key combination to use as the hotkey; it is saved to the config file
    Setup {
        /// Push-to-talk: record while the keys are held instead of toggling on each press
        #[arg(long)]
        hold: bool,
    },
    /// Start or stop recording in the running daemon (bind this in your desktop's shortcut settings instead of
    /// using the evdev hotkey)
    Toggle,
    /// Start recording in the running daemon
    Start,
    /// Stop recording in the running daemon and type the transcript
    Stop,
    /// Print where the config file lives and what it holds
    Config,
}

#[derive(Args)]
struct RunArgs {
    /// Hotkey, overriding the saved one: e.g. "F9", "ctrl+alt+d", "super+space", "KEY_PAUSE"
    #[arg(long)]
    key: Option<String>,

    /// Push-to-talk: record only while the hotkey is held
    #[arg(long)]
    hold: bool,

    /// No hotkey at all: only the control socket (`phonon-dictate toggle`) starts and stops recording
    #[arg(long, conflicts_with = "key")]
    no_hotkey: bool,

    /// Microphone: any part of a name from `phonon --list-mics` (default: the system default input)
    #[arg(long)]
    mic_device: Option<String>,

    /// How the transcript is delivered
    #[arg(long, value_enum, default_value_t = Output::Type)]
    output: Output,

    /// Shortcut pressed to paste (for --output paste, and when typing falls back to paste for characters like
    /// "è" that a US keymap can't type). Terminals usually want "ctrl+shift+v"
    #[arg(long, default_value = "ctrl+v")]
    paste_keys: String,

    /// Don't add a space after each dictation
    #[arg(long)]
    no_space: bool,

    /// No desktop notifications
    #[arg(long)]
    quiet: bool,

    /// Stop recording automatically after this many seconds
    #[arg(long, default_value_t = 300.0)]
    max_secs: f32,

    /// Model: a dir with model.fermion + config.json, a model.fermion file, or phonon-2.bps.tar.zst
    #[arg(short, long, env = "PHONON_MODEL")]
    model: Option<PathBuf>,

    #[arg(short, long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Encoder precision (default: f32 on CPU, f16 on CUDA)
    #[arg(long, value_enum)]
    dtype: Option<Precision>,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Output {
    /// Type through a virtual keyboard (US layout); falls back to paste for characters it cannot type
    Type,
    /// Put the text on the clipboard (wl-copy) and press Ctrl+V
    Paste,
    /// Only put the text on the clipboard
    Clipboard,
    /// Print to stdout
    Stdout,
}

// ---------------------------------------------------------------------------------------------------------------
// config

#[derive(Serialize, Deserialize, Default)]
struct Config {
    /// evdev key names, e.g. ["KEY_LEFTCTRL", "KEY_SPACE"]
    keys: Vec<String>,
    #[serde(default)]
    hold: bool,
}

fn config_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".config"));
    base.join("phonon").join("dictate.json")
}

fn load_config() -> Result<Option<Config>> {
    let p = config_path();
    match std::fs::read(&p) {
        Ok(b) => Ok(Some(serde_json::from_slice(&b).with_context(|| format!("parsing {}", p.display()))?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", p.display())),
    }
}

fn save_config(c: &Config) -> Result<PathBuf> {
    let p = config_path();
    std::fs::create_dir_all(p.parent().unwrap())?;
    std::fs::write(&p, serde_json::to_string_pretty(c)? + "\n")?;
    Ok(p)
}

fn socket_path() -> PathBuf {
    let dir = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    dir.join("phonon-dictate.sock")
}

// ---------------------------------------------------------------------------------------------------------------
// keys

/// Left and right Ctrl / Shift / Meta count as the same key, so a combo saved with one side matches either.
fn canonical(k: KeyCode) -> KeyCode {
    match k {
        KeyCode::KEY_RIGHTCTRL => KeyCode::KEY_LEFTCTRL,
        KeyCode::KEY_RIGHTSHIFT => KeyCode::KEY_LEFTSHIFT,
        KeyCode::KEY_RIGHTMETA => KeyCode::KEY_LEFTMETA,
        k => k,
    }
}

fn is_modifier(k: KeyCode) -> bool {
    matches!(
        canonical(k),
        KeyCode::KEY_LEFTCTRL | KeyCode::KEY_LEFTSHIFT | KeyCode::KEY_LEFTMETA | KeyCode::KEY_LEFTALT | KeyCode::KEY_RIGHTALT
    )
}

/// "ctrl+alt+d", "super+space", "F9", "KEY_PAUSE", "BTN_SIDE" -> canonical key codes
fn parse_combo(s: &str) -> Result<BTreeSet<KeyCode>> {
    let mut out = BTreeSet::new();
    for tok in s.split('+').map(str::trim).filter(|t| !t.is_empty()) {
        let name = match tok.to_lowercase().as_str() {
            "ctrl" | "control" => "KEY_LEFTCTRL".to_string(),
            "shift" => "KEY_LEFTSHIFT".to_string(),
            "alt" => "KEY_LEFTALT".to_string(),
            "altgr" => "KEY_RIGHTALT".to_string(),
            "super" | "meta" | "win" | "cmd" | "logo" => "KEY_LEFTMETA".to_string(),
            "esc" => "KEY_ESC".to_string(),
            "del" => "KEY_DELETE".to_string(),
            _ if tok.to_uppercase().starts_with("KEY_") || tok.to_uppercase().starts_with("BTN_") => tok.to_uppercase(),
            t => format!("KEY_{}", t.to_uppercase()),
        };
        let k = KeyCode::from_str(&name).map_err(|_| anyhow::anyhow!("unknown key {tok:?} ({name})"))?;
        out.insert(canonical(k));
    }
    if out.is_empty() {
        bail!("empty key combination");
    }
    Ok(out)
}

fn combo_name(keys: &BTreeSet<KeyCode>) -> String {
    // modifiers first
    let mut v: Vec<&KeyCode> = keys.iter().collect();
    v.sort_by_key(|k| !is_modifier(**k));
    v.iter()
        .map(|k| {
            let s = format!("{k:?}");
            let s = s.strip_prefix("KEY_").unwrap_or(&s).to_string();
            match s.as_str() {
                "LEFTCTRL" => "Ctrl".into(),
                "LEFTSHIFT" => "Shift".into(),
                "LEFTALT" => "Alt".into(),
                "RIGHTALT" => "AltGr".into(),
                "LEFTMETA" => "Super".into(),
                _ => s,
            }
        })
        .collect::<Vec<_>>()
        .join("+")
}

/// Key down/up events (canonicalised, repeats dropped) from every keyboard-like device, including ones plugged
/// in later.
fn spawn_key_listener() -> Result<Receiver<(KeyCode, bool)>> {
    let (tx, rx) = channel();
    let open: Arc<Mutex<HashSet<PathBuf>>> = Arc::default();
    let found = scan_devices(&tx, &open);
    if found == 0 {
        bail!(
            "cannot read any input device in /dev/input. Add yourself to the `input` group \
             (sudo usermod -aG input $USER, then log in again) or run with --no-hotkey"
        );
    }
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(2));
            scan_devices(&tx, &open);
        }
    });
    Ok(rx)
}

fn scan_devices(tx: &Sender<(KeyCode, bool)>, open: &Arc<Mutex<HashSet<PathBuf>>>) -> usize {
    for (path, mut dev) in evdev::enumerate() {
        if open.lock().unwrap().contains(&path) || dev.name() == Some(VIRTUAL_KBD) {
            continue;
        }
        // anything with keys or buttons: keyboards, macro pads, mouse side buttons
        let has_keys = dev.supported_keys().is_some_and(|k| k.iter().next().is_some());
        if !has_keys {
            continue;
        }
        log::debug!("listening on {} ({})", path.display(), dev.name().unwrap_or("?"));
        open.lock().unwrap().insert(path.clone());
        let (tx, open) = (tx.clone(), open.clone());
        std::thread::spawn(move || {
            'outer: loop {
                let events = match dev.fetch_events() {
                    Ok(e) => e,
                    Err(_) => break, // unplugged
                };
                for ev in events {
                    if let EventSummary::Key(_, code, v) = ev.destructure() {
                        if v == 2 {
                            continue; // autorepeat
                        }
                        if tx.send((canonical(code), v == 1)).is_err() {
                            break 'outer;
                        }
                    }
                }
            }
            open.lock().unwrap().remove(&path);
        });
    }
    open.lock().unwrap().len()
}

// ---------------------------------------------------------------------------------------------------------------
// setup

fn setup(hold: bool) -> Result<()> {
    let rx = spawn_key_listener()?;
    eprintln!("Press the key or key combination to use for dictation, then release it.");
    eprintln!("Tip: pick something that does nothing else (F9-F12, Pause, Super+<letter> you don't use, a mouse side button):");
    eprintln!("     the keys also reach the focused window.");
    // the Enter that started this command is still being released: drain it
    std::thread::sleep(Duration::from_millis(300));
    while rx.try_recv().is_ok() {}

    // capture twice so a stray key (or someone else typing) can't end up as the hotkey
    let chord = loop {
        let first = read_chord(&rx)?;
        eprintln!("Got {}. Press it again to confirm.", combo_name(&first));
        let second = read_chord(&rx)?;
        if second == first {
            break first;
        }
        eprintln!("That was {}, not {}. Start over: press the combination.", combo_name(&second), combo_name(&first));
    };
    // keys that produce text (and auto-repeat) in the focused window when not combined with Ctrl/Alt/Super
    let types = |k: &KeyCode| (32u8..127).filter_map(|c| us_key(c as char)).any(|(t, _)| t == *k)
        || matches!(*k, KeyCode::KEY_ENTER | KeyCode::KEY_TAB | KeyCode::KEY_BACKSPACE | KeyCode::KEY_DELETE);
    let has_mod = chord.iter().any(|k| is_modifier(*k) && !matches!(*k, KeyCode::KEY_LEFTSHIFT));
    if !has_mod && chord.iter().any(types) {
        eprintln!(
            "Warning: {} includes a key that types into the focused window (and auto-repeats while held). \
             Consider adding Ctrl/Alt/Super, or use F13-F24 / Pause.",
            combo_name(&chord)
        );
    }
    if chord.iter().all(|k| is_modifier(*k)) && chord.len() == 1 {
        eprintln!("Note: a lone modifier as the hotkey will also fire whenever you use it in other shortcuts.");
    }
    let cfg = Config { keys: chord.iter().map(|k| format!("{k:?}")).collect(), hold };
    let path = save_config(&cfg)?;
    eprintln!(
        "Saved {} ({}) to {}",
        combo_name(&chord),
        if hold { "hold to talk" } else { "press to start, press again to stop" },
        path.display()
    );
    eprintln!("Now run `phonon-dictate` (or restart it if it is already running).");
    Ok(())
}

/// Every key held together from the first press until all are released.
fn read_chord(rx: &Receiver<(KeyCode, bool)>) -> Result<BTreeSet<KeyCode>> {
    let mut down: HashSet<KeyCode> = HashSet::new();
    let mut chord: BTreeSet<KeyCode> = BTreeSet::new();
    loop {
        let (k, pressed) = rx.recv()?;
        if pressed {
            down.insert(k);
            chord.insert(k);
        } else if down.remove(&k) && down.is_empty() && !chord.is_empty() {
            return Ok(chord);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// typing

/// US-layout key for a character: (key, needs shift)
fn us_key(c: char) -> Option<(KeyCode, bool)> {
    use KeyCode as K;
    let letters = [
        K::KEY_A, K::KEY_B, K::KEY_C, K::KEY_D, K::KEY_E, K::KEY_F, K::KEY_G, K::KEY_H, K::KEY_I, K::KEY_J, K::KEY_K,
        K::KEY_L, K::KEY_M, K::KEY_N, K::KEY_O, K::KEY_P, K::KEY_Q, K::KEY_R, K::KEY_S, K::KEY_T, K::KEY_U, K::KEY_V,
        K::KEY_W, K::KEY_X, K::KEY_Y, K::KEY_Z,
    ];
    let digits = [K::KEY_0, K::KEY_1, K::KEY_2, K::KEY_3, K::KEY_4, K::KEY_5, K::KEY_6, K::KEY_7, K::KEY_8, K::KEY_9];
    Some(match c {
        'a'..='z' => (letters[c as usize - 'a' as usize], false),
        'A'..='Z' => (letters[c as usize - 'A' as usize], true),
        '0'..='9' => (digits[c as usize - '0' as usize], false),
        ' ' => (K::KEY_SPACE, false),
        '\n' => (K::KEY_ENTER, false),
        '\t' => (K::KEY_TAB, false),
        '!' => (K::KEY_1, true),
        '@' => (K::KEY_2, true),
        '#' => (K::KEY_3, true),
        '$' => (K::KEY_4, true),
        '%' => (K::KEY_5, true),
        '^' => (K::KEY_6, true),
        '&' => (K::KEY_7, true),
        '*' => (K::KEY_8, true),
        '(' => (K::KEY_9, true),
        ')' => (K::KEY_0, true),
        '-' => (K::KEY_MINUS, false),
        '_' => (K::KEY_MINUS, true),
        '=' => (K::KEY_EQUAL, false),
        '+' => (K::KEY_EQUAL, true),
        '[' => (K::KEY_LEFTBRACE, false),
        '{' => (K::KEY_LEFTBRACE, true),
        ']' => (K::KEY_RIGHTBRACE, false),
        '}' => (K::KEY_RIGHTBRACE, true),
        '\\' => (K::KEY_BACKSLASH, false),
        '|' => (K::KEY_BACKSLASH, true),
        ';' => (K::KEY_SEMICOLON, false),
        ':' => (K::KEY_SEMICOLON, true),
        '\'' => (K::KEY_APOSTROPHE, false),
        '"' => (K::KEY_APOSTROPHE, true),
        '`' => (K::KEY_GRAVE, false),
        '~' => (K::KEY_GRAVE, true),
        ',' => (K::KEY_COMMA, false),
        '<' => (K::KEY_COMMA, true),
        '.' => (K::KEY_DOT, false),
        '>' => (K::KEY_DOT, true),
        '/' => (K::KEY_SLASH, false),
        '?' => (K::KEY_SLASH, true),
        _ => return None,
    })
}

/// Typographic characters folded to what a US keyboard can type.
fn plain(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{2032}' => "'".to_string(),
            '\u{201C}' | '\u{201D}' | '\u{201F}' | '\u{2033}' => "\"".to_string(),
            '\u{2013}' | '\u{2014}' | '\u{2212}' => "-".to_string(),
            '\u{2026}' => "...".to_string(),
            '\u{00A0}' | '\u{202F}' => " ".to_string(),
            c => c.to_string(),
        })
        .collect()
}

struct Keyboard {
    dev: VirtualDevice,
}

impl Keyboard {
    fn new() -> Result<Self> {
        let mut keys = AttributeSet::<KeyCode>::new();
        for c in (32u8..127).map(char::from).chain(['\n', '\t']) {
            if let Some((k, _)) = us_key(c) {
                keys.insert(k);
            }
        }
        for k in [KeyCode::KEY_LEFTSHIFT, KeyCode::KEY_LEFTCTRL, KeyCode::KEY_LEFTALT, KeyCode::KEY_LEFTMETA, KeyCode::KEY_V, KeyCode::KEY_INSERT] {
            keys.insert(k);
        }
        let dev = VirtualDevice::builder()
            .context("cannot open /dev/uinput (needs the `input` group or a udev rule); try --output paste/stdout")?
            .name(VIRTUAL_KBD)
            .with_keys(&keys)?
            .build()?;
        Ok(Self { dev })
    }

    fn tap(&mut self, k: KeyCode, mods: &[KeyCode]) -> Result<()> {
        let ev = |k: KeyCode, v: i32| *evdev::KeyEvent::new(k, v);
        for &m in mods {
            self.dev.emit(&[ev(m, 1)])?;
        }
        self.dev.emit(&[ev(k, 1)])?;
        self.dev.emit(&[ev(k, 0)])?;
        for &m in mods.iter().rev() {
            self.dev.emit(&[ev(m, 0)])?;
        }
        // compositors drop keys that arrive faster than they process them
        std::thread::sleep(Duration::from_millis(2));
        Ok(())
    }

    fn type_text(&mut self, text: &str) -> Result<()> {
        for c in text.chars() {
            let (k, shift) = us_key(c).with_context(|| format!("cannot type {c:?}"))?;
            if shift {
                self.tap(k, &[KeyCode::KEY_LEFTSHIFT])?;
            } else {
                self.tap(k, &[])?;
            }
        }
        Ok(())
    }
}

fn wl_copy(text: &str) -> Result<()> {
    let mut child = std::process::Command::new("wl-copy")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .context("wl-copy not found (install wl-clipboard)")?;
    child.stdin.take().unwrap().write_all(text.as_bytes())?;
    child.wait()?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// notifications

struct Notifier {
    enabled: bool,
    id: Option<String>,
}

impl Notifier {
    fn show(&mut self, summary: &str, body: &str, ms: u32) {
        if !self.enabled {
            return;
        }
        let mut cmd = std::process::Command::new("notify-send");
        cmd.args(["-a", "Phonon dictation", "-i", "audio-input-microphone", "-t", &ms.to_string(), "-p"]);
        if let Some(id) = &self.id {
            cmd.args(["-r", id]);
        }
        cmd.arg(summary);
        if !body.is_empty() {
            cmd.arg(body);
        }
        if let Ok(out) = cmd.output() {
            let id = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !id.is_empty() {
                self.id = Some(id);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// daemon

enum Event {
    Key(KeyCode, bool),
    Command(String, UnixStream),
}

fn send_command(cmd: &str) -> Result<()> {
    let path = socket_path();
    let mut s = UnixStream::connect(&path)
        .with_context(|| format!("no phonon-dictate daemon listening on {}", path.display()))?;
    writeln!(s, "{cmd}")?;
    let mut reply = String::new();
    BufReader::new(s).read_line(&mut reply)?;
    eprint!("{reply}");
    Ok(())
}

fn spawn_socket(tx: Sender<Event>) -> Result<()> {
    let path = socket_path();
    if UnixStream::connect(&path).is_ok() {
        bail!("phonon-dictate is already running ({})", path.display());
    }
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).with_context(|| format!("binding {}", path.display()))?;
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut line = String::new();
            if BufReader::new(&stream).read_line(&mut line).is_ok() {
                let _ = tx.send(Event::Command(line.trim().to_string(), stream));
            }
        }
    });
    Ok(())
}

struct Recording {
    capture: mic::Capture,
    started: Instant,
}

struct Daemon {
    engine: Engine,
    args: RunArgs,
    keyboard: Option<Keyboard>,
    notifier: Notifier,
    down: HashSet<KeyCode>,
    rec: Option<Recording>,
}

impl Daemon {
    fn start(&mut self) -> Result<()> {
        if self.rec.is_some() {
            return Ok(());
        }
        let capture = mic::Capture::open(self.args.mic_device.as_deref())?;
        eprintln!("recording from {}", capture.name);
        self.notifier.show("Listening…", "", 0);
        self.rec = Some(Recording { capture, started: Instant::now() });
        Ok(())
    }

    fn stop(&mut self, rx: &Receiver<Event>) -> Result<()> {
        let Some(rec) = self.rec.take() else { return Ok(()) };
        let rate = rec.capture.rate;
        // keep listening briefly: people press stop while finishing the last word
        let samples = rec.capture.finish(Duration::from_millis(250));
        let secs = samples.len() as f32 / rate as f32;
        if secs < 0.3 {
            self.notifier.show("Too short", "", 1000);
            return Ok(());
        }
        let voiced = mic::voiced_secs(&samples, rate);
        if voiced < mic::MIN_VOICED_SECS {
            eprintln!("{secs:.1}s recorded, {voiced:.2}s of it speech: skipped");
            self.notifier.show("No speech heard", "", 1500);
            return Ok(());
        }
        self.notifier.show("Transcribing…", "", 0);
        let t = Instant::now();
        let audio16 = audio::resample(samples, rate)?;
        let words = self.engine.transcribe(&audio16, 30.0)?;
        if !mic::plausible(&words, 0.9) {
            eprintln!("dropped {:?} (confidence {:.2})", text::join(&words), text::confidence(&words));
            self.notifier.show("No speech heard", "", 1500);
            return Ok(());
        }
        let mut text = plain(&text::join(&words));
        eprintln!("{secs:.1}s of audio transcribed in {:.2?}: {text}", t.elapsed());
        if text.is_empty() {
            self.notifier.show("No speech heard", "", 1500);
            return Ok(());
        }
        if !self.args.no_space {
            text.push(' ');
        }
        self.wait_for_release(rx);
        self.deliver(&text)?;
        self.notifier.show("Dictated", text.trim(), 2500);
        Ok(())
    }

    /// Typing while the hotkey's modifiers are still held would turn letters into shortcuts.
    fn wait_for_release(&mut self, rx: &Receiver<Event>) {
        let deadline = Instant::now() + Duration::from_secs(3);
        while self.down.iter().any(|k| is_modifier(*k)) && Instant::now() < deadline {
            if let Ok(Event::Key(k, pressed)) = rx.recv_timeout(Duration::from_millis(50)) {
                if pressed {
                    self.down.insert(k);
                } else {
                    self.down.remove(&k);
                }
            }
        }
        std::thread::sleep(Duration::from_millis(30));
    }

    fn deliver(&mut self, text: &str) -> Result<()> {
        let typeable = text.chars().all(|c| us_key(c).is_some());
        match (self.args.output, self.keyboard.as_mut()) {
            (Output::Stdout, _) => {
                println!("{}", text.trim_end());
                Ok(())
            }
            (Output::Type, Some(kb)) if typeable => kb.type_text(text),
            (Output::Type | Output::Paste, Some(kb)) => {
                wl_copy(text)?;
                std::thread::sleep(Duration::from_millis(100));
                let keys = parse_combo(&self.args.paste_keys)?;
                let (mods, main): (Vec<KeyCode>, Vec<KeyCode>) = keys.iter().partition(|k| is_modifier(**k));
                let key = *main.first().context("--paste-keys needs a non-modifier key")?;
                kb.tap(key, &mods)
            }
            (Output::Clipboard, _) | (Output::Type | Output::Paste, None) => wl_copy(text),
        }
    }
}

fn run(args: RunArgs) -> Result<()> {
    // hotkey: --key, else the saved one, else none (socket only)
    let cfg = load_config()?;
    let hold = args.hold || cfg.as_ref().is_some_and(|c| c.hold);
    let combo: Option<BTreeSet<KeyCode>> = if args.no_hotkey {
        None
    } else if let Some(k) = &args.key {
        Some(parse_combo(k)?)
    } else if let Some(c) = cfg.as_ref().filter(|c| !c.keys.is_empty()) {
        Some(parse_combo(&c.keys.join("+"))?)
    } else {
        bail!("no hotkey configured: run `phonon-dictate setup`, pass --key, or use --no-hotkey with `phonon-dictate toggle`");
    };

    let (tx, rx) = channel::<Event>();
    spawn_socket(tx.clone())?;

    phonon::engine::init_threads(None);
    let t = Instant::now();
    let engine = Engine::load(args.model.clone(), args.device, args.dtype)?;
    engine.transcribe(&vec![0.0; audio::SAMPLE_RATE], 30.0)?; // CUDA / cuBLAS warm-up
    eprintln!(
        "model ready on {} ({:?}) in {:.1?}",
        if engine.cuda { "cuda" } else { "cpu" },
        engine.precision,
        t.elapsed()
    );

    let keyboard = match args.output {
        Output::Type | Output::Paste => match Keyboard::new() {
            Ok(k) => Some(k),
            Err(e) => {
                eprintln!("warning: {e:#}; falling back to the clipboard");
                None
            }
        },
        _ => None,
    };

    if let Some(c) = &combo {
        let keys = spawn_key_listener()?;
        let tx = tx.clone();
        std::thread::spawn(move || {
            for (k, p) in keys {
                if tx.send(Event::Key(k, p)).is_err() {
                    break;
                }
            }
        });
        eprintln!(
            "{}: {}  (or `phonon-dictate toggle`)",
            combo_name(c),
            if hold { "hold to talk" } else { "press to start, press again to stop" }
        );
    } else {
        eprintln!("no hotkey; use `phonon-dictate toggle|start|stop`");
    }

    let notifier = Notifier { enabled: !args.quiet, id: None };
    let max = Duration::from_secs_f32(args.max_secs);
    let mut d = Daemon { engine, args, keyboard, notifier, down: HashSet::new(), rec: None };
    let mut armed = false; // combo fully down, waiting for its release

    loop {
        let ev = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(ev) => Some(ev),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
            Err(_) => break,
        };
        let result = match ev {
            Some(Event::Key(k, pressed)) => {
                // a key can arrive from more than one device node (multi-interface keyboards, remappers):
                // only real transitions count
                let changed = if pressed { d.down.insert(k) } else { d.down.remove(&k) };
                log::debug!("{k:?} {} (changed: {changed}) down: {:?}", if pressed { "down" } else { "up" }, d.down);
                let Some(c) = &combo else { continue };
                if !changed || !c.contains(&k) {
                    Ok(())
                } else if pressed && !armed && d.down.len() == c.len() && c.iter().all(|x| d.down.contains(x)) {
                    armed = true;
                    if hold { d.start() } else if d.rec.is_some() { d.stop(&rx) } else { d.start() }
                } else if !pressed && armed && !c.iter().any(|x| d.down.contains(x)) {
                    // the whole combo is up (not just its first key: Fn keys often report press+release at once)
                    armed = false;
                    if hold { d.stop(&rx) } else { Ok(()) }
                } else {
                    Ok(())
                }
            }
            Some(Event::Command(cmd, mut stream)) => {
                let r = match cmd.as_str() {
                    "toggle" if d.rec.is_some() => d.stop(&rx),
                    "toggle" | "start" => d.start(),
                    "stop" => d.stop(&rx),
                    other => Err(anyhow::anyhow!("unknown command {other:?}")),
                };
                let reply = match &r {
                    Ok(()) => format!("ok: {}\n", if d.rec.is_some() { "recording" } else { "idle" }),
                    Err(e) => format!("error: {e:#}\n"),
                };
                let _ = stream.write_all(reply.as_bytes());
                r
            }
            None => {
                if d.rec.as_ref().is_some_and(|r| r.started.elapsed() > max) {
                    d.stop(&rx)
                } else {
                    Ok(())
                }
            }
        };
        if let Err(e) = result {
            eprintln!("error: {e:#}");
            d.notifier.show("Dictation error", &format!("{e:#}"), 4000);
        }
    }
    Ok(())
}

fn show_config() -> Result<()> {
    let p = config_path();
    match load_config()? {
        Some(c) => {
            let combo = parse_combo(&c.keys.join("+"))?;
            println!("{}: {} ({})", p.display(), combo_name(&combo), if c.hold { "hold" } else { "toggle" });
        }
        None => println!("{}: not set up (run `phonon-dictate setup`)", p.display()),
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,pulseaudio=off")).init();
    let cli = Cli::parse();
    match cli.cmd {
        Some(Cmd::Setup { hold }) => setup(hold),
        Some(Cmd::Toggle) => send_command("toggle"),
        Some(Cmd::Start) => send_command("start"),
        Some(Cmd::Stop) => send_command("stop"),
        Some(Cmd::Config) => show_config(),
        None => run(cli.run),
    }
}
