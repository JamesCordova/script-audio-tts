"""Lee un guion de entrevista ("NOMBRE: texto") con una voz distinta por personaje.

Motores:  edge (voces neuronales online, gratis)  |  piper (offline, local)

Uso:
  uv run main.py entrevista.txt --engine piper
  uv run main.py entrevista.txt --engine edge --no-play
  uv run main.py entrevista.txt --engine both --no-play    # compara tiempos
  uv run main.py --list-voices es-                          # voces de edge
"""
import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PAUSE_RE = re.compile(r"^\[pausa\s+([\d.]+)\]$", re.I)
LINE_RE = re.compile(r"^([^:]{1,40}):\s*(.+)$")
MP3_BYTES_PER_SEC = 6000  # edge-tts: 48 kbps
MODELS_DIR = Path("models")
CACHE_DIR = Path(".cache")


def parse(path: Path):
    items = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if m := PAUSE_RE.match(line):
            items.append(("pause", float(m[1])))
        elif m := LINE_RE.match(line):
            items.append(("say", m[1].strip(), m[2].strip()))
        else:
            print(f"aviso: línea {n} ignorada: {line!r}", file=sys.stderr)
    return items


def cache_get(engine, cfg, text, ext):
    key = hashlib.sha1(json.dumps([engine, cfg, text], sort_keys=True).encode()).hexdigest()
    return CACHE_DIR / engine / f"{key}.{ext}"


# ---------------------------------------------------------------- edge
async def edge_synth(text, cfg) -> bytes:
    import edge_tts

    comm = edge_tts.Communicate(
        text, cfg["voice"], rate=cfg.get("rate", "+0%"), pitch=cfg.get("pitch", "+0Hz")
    )
    audio = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return bytes(audio)


# --------------------------------------------------------------- piper
_piper_voices = {}


def piper_voice(name):
    if name not in _piper_voices:
        from piper import PiperVoice

        _piper_voices[name] = PiperVoice.load(MODELS_DIR / f"{name}.onnx")
    return _piper_voices[name]


def piper_synth(text, cfg):
    """Devuelve (sample_rate, pcm_int16_bytes)."""
    from piper import SynthesisConfig

    voice = piper_voice(cfg["model"])
    syn = SynthesisConfig(
        speaker_id=cfg.get("speaker"), length_scale=cfg.get("length_scale", 1.0)
    )
    pcm = bytearray()
    for chunk in voice.synthesize(text, syn_config=syn):
        pcm += chunk.audio_int16_bytes
    return voice.config.sample_rate, bytes(pcm)


# --------------------------------------------------------------- build
async def build_edge(items, voices, out, gap, conc):
    sem = asyncio.Semaphore(conc)

    async def job(it):
        if it[0] != "say":
            return None
        cfg = voices[it[1]]["edge"]
        f = cache_get("edge", cfg, it[2], "mp3")
        if f.exists():
            return f.read_bytes()
        async with sem:
            for attempt in range(1, 4):  # timeout + reintentos: edge a veces se cuelga
                try:
                    data = await asyncio.wait_for(edge_synth(it[2], cfg), timeout=30)
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    print(f"  reintento {attempt} ({type(e).__name__}): {it[2][:40]}", file=sys.stderr)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
        return data

    parts = await asyncio.gather(*(job(i) for i in items))
    silence = lambda s: b"\x00" * int(MP3_BYTES_PER_SEC * s)
    with out.open("wb") as f:
        for it, audio in zip(items, parts):
            if it[0] == "pause":
                f.write(silence(it[1]))
            else:
                f.write(audio + silence(gap))


async def build_piper(items, voices, out, gap, conc):
    loop = asyncio.get_running_loop()

    def job(it):
        cfg = voices[it[1]]["piper"]
        f = cache_get("piper", cfg, it[2], "pcm")
        meta = f.with_suffix(".sr")
        if f.exists() and meta.exists():
            return int(meta.read_text()), f.read_bytes()
        sr, pcm = piper_synth(it[2], cfg)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(pcm)
        meta.write_text(str(sr))
        return sr, pcm

    # Precarga de modelos en el hilo principal (evita carreras al cargar)
    for it in items:
        if it[0] == "say":
            piper_voice(voices[it[1]]["piper"]["model"])

    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [loop.run_in_executor(ex, job, i) if i[0] == "say" else None for i in items]
        results = [await f if f else None for f in futs]

    sr = next((r[0] for r in results if r), 22050)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for it, r in zip(items, results):
            if it[0] == "pause":
                w.writeframes(b"\x00\x00" * int(sr * it[1]))
                continue
            rsr, pcm = r
            if rsr != sr:  # remuestreo simple si las voces tienen distinta frecuencia
                import numpy as np

                a = np.frombuffer(pcm, dtype=np.int16)
                n = int(len(a) * sr / rsr)
                pcm = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.int16).tobytes()
            w.writeframes(pcm + b"\x00\x00" * int(sr * gap))


def run_engine(engine, items, voices, name, gap, conc):
    out = Path("salida") / f"{name}.{engine}.{'mp3' if engine == 'edge' else 'wav'}"
    out.parent.mkdir(exist_ok=True)
    t = time.perf_counter()
    fn = build_edge if engine == "edge" else build_piper
    asyncio.run(fn(items, voices, out, gap, conc))
    dt = time.perf_counter() - t
    print(f"[{engine}] {out}  ({dt:.1f}s)")
    return out, dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("script", nargs="?", type=Path)
    p.add_argument("--engine", choices=["edge", "piper", "both"], default="piper")
    p.add_argument("--voices", type=Path, default=Path("voices.json"))
    p.add_argument("--gap", type=float, default=0.4, help="segundos entre líneas")
    p.add_argument("--workers", type=int, help="concurrencia (edge: 6, piper: 4 por defecto)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--list-voices", metavar="PREFIJO")
    a = p.parse_args()

    if a.list_voices is not None:
        import edge_tts

        for v in asyncio.run(edge_tts.list_voices()):
            if v["ShortName"].lower().startswith(a.list_voices.lower()):
                print(f'{v["ShortName"]:28} {v["Gender"]}')
        return
    if not a.script:
        p.error("indica el guion")

    if a.no_cache:  # cache en carpeta temporal nueva -> siempre regenera
        import tempfile

        global CACHE_DIR
        CACHE_DIR = Path(tempfile.mkdtemp())

    voices = json.loads(a.voices.read_text(encoding="utf-8"))
    items = parse(a.script)
    missing = {i[1] for i in items if i[0] == "say" and i[1] not in voices}
    if missing:
        sys.exit(f"Faltan voces en {a.voices.name} para: {', '.join(sorted(missing))}")

    words = sum(len(i[2].split()) for i in items if i[0] == "say")
    print(f"{len(items)} bloques, {words} palabras")
    engines = ["edge", "piper"] if a.engine == "both" else [a.engine]
    last = None
    for e in engines:
        conc = a.workers or (6 if e == "edge" else 4)
        last, _ = run_engine(e, items, voices, a.script.stem, a.gap, conc)
    if not a.no_play and sys.platform == "win32" and last:
        os.startfile(last)


if __name__ == "__main__":
    main()
