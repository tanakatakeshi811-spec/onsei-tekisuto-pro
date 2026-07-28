"""
音声入力ツール(ディクテーションアプリ)

ホットキー(デフォルトF8)を1回押すと録音開始。話している間はリアルタイムに
文字起こし結果が専用ウィンドウに表示され続ける。もう1回押すと録音停止し、
より高精度なモデルで文字起こしし直したうえで、ローカルAI(Ollama)に文章の
整形(誤字脱字修正・句読点付与・数字表記の変換)をさせてから、
今フォーカスしているアプリに自動で貼り付け + 専用ウィンドウにも表示する。

すべてこのPC内で完結する(音声データや文字起こし結果を外部サーバーに送らない。
文章整形に使うAIもこのPC内で動いているOllamaのみ)。
"""

import collections
import contextlib
import ctypes
import json
import multiprocessing
import os
import re
import sys

# PyInstallerでexe化した子プロセス(faster-whisperの内部依存が使うことがある)が
# エントリポイントを丸ごと再実行してしまい、プロセスが際限なく増殖するバグへの対策。
# mainより前、できるだけ早い段階で呼ぶ必要がある(Mac実機での再現・原因特定は
# しゅんりさんの協力者による報告)。
multiprocessing.freeze_support()

# Windowsで開発者モードが有効になっていない環境でも、faster-whisperが
# モデルを初めてダウンロードするときにシンボリックリンク作成でエラーに
# ならないようにする(WinError 1314対策)。importより前に設定すること。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import pyperclip
import requests
import sounddevice as sd
import speech_recognition as sr
from faster_whisper import WhisperModel
from pynput import keyboard as pynkb

# WindowsとMacの両方で動かすため、OS固有の処理はこの2つのフラグで分岐する。
# (Windows専用だったwinsound/ctypes.windll/keyboardライブラリを、
#  sounddeviceでの音生成・pynput・OSごとのAPI呼び分けに置き換えている)
IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

if IS_MAC:
    # macOS 26以降、キーボードレイアウト(入力ソース)を調べるHIToolboxのAPI
    # (TSMGetInputSourceProperty等)は、メインスレッド以外から呼ぶと
    # dispatch_assert_queueにより即座にクラッシュするようになった。
    # pynputはキー入力の翻訳(pynput._util.darwin.keycode_context)でこのAPIを
    # 内部のリスナースレッドから直接呼んでしまうため、起動直後に確実にクラッシュ
    # する不具合が実機で確認された(Mac実機での再現・原因特定・修正方針の提示は
    # しゅんりさんの協力者による報告に基づく)。
    # 該当箇所を差し替え、実際のHIToolbox呼び出しだけをメインスレッド
    # (Tkinterが使っているCocoaのメインループ)にディスパッチし、
    # 呼び出し元のスレッドは完了まで待つ(同期呼び出しの形を保ったまま安全にする)。
    try:
        import pynput._util.darwin as _pynput_darwin
        from PyObjCTools import AppHelper

        _orig_keycode_context = _pynput_darwin.keycode_context

        def _run_on_main_sync(func, timeout=5):
            result = {}
            done = threading.Event()

            def _wrapper():
                try:
                    result["value"] = func()
                except Exception as e:
                    result["error"] = e
                finally:
                    done.set()

            AppHelper.callAfter(_wrapper)
            if not done.wait(timeout=timeout):
                raise RuntimeError("keycode_context: メインスレッドへの処理委譲がタイムアウトしました")
            if "error" in result:
                raise result["error"]
            return result.get("value")

        @contextlib.contextmanager
        def _main_thread_keycode_context():
            cm = _orig_keycode_context()
            ctx = _run_on_main_sync(cm.__enter__)
            try:
                yield ctx
            finally:
                _run_on_main_sync(lambda: cm.__exit__(None, None, None))

        _pynput_darwin.keycode_context = _main_thread_keycode_context
    except Exception as e:
        # ここが失敗しても致命的ではない(パッチが当たらないだけ、元の挙動に戻る)
        pass

# PyInstallerでexe化すると、__file__はexeの場所ではなく実行時に展開される
# 一時フォルダ(_MEIxxxxxx、終了時に消える)を指してしまう。exe化されている
# 場合はsys.executable(exe自身の場所)を使い、config.jsonが次回起動時にも
# 残るようにする。
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# 何かが起きても画面が固まったまま何も表示されない、という事態を防ぐため、
# バックグラウンドスレッドで起きた予期しない例外はログファイルに残す。
# --windowedビルドはコンソールが無く、エラーが完全に見えなくなるための対策。
LOG_PATH = os.path.join(BASE_DIR, "error_log.txt")


def _log_exception(context, exc):
    try:
        import traceback
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} [{context}] ---\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass

SAMPLE_RATE = 16000
DEFAULT_CONFIG = {
    "hotkey": "f8",
    # 翻訳モードのホットキー。押すと日本語で話した内容を外国語に翻訳する
    "translate_hotkey": "f7",
    # 翻訳先の言語(下の LANGUAGES のキー)
    "target_language": "英語",
    # 翻訳にローカルAI(Ollama)を使うか。
    # OFFのときはWhisper内蔵の翻訳機能を使う(英語にしか訳せない)。
    # 公開配布版なのでOllama無しの環境の方が多い想定 → 既定値はOFFにして、
    # Ollamaが入っている人だけ画面のチェックボックスからONにしてもらう。
    "translate_with_ai": False,
    # リアルタイム表示用(速さ優先)。話している間、ここで随時文字起こしする
    "realtime_model_size": "small",
    # 確定時(F8を2回目に押した時)用。録音停止後に1回だけ流す。
    # 実測でsmallはリアルタイムより速く(RTF約0.72)、かつリアルタイム表示
    # (beam_size=1の速さ優先)より精度が高いビームサーチが効くため、
    # 「そこそこ速いのに精度もそこそこ良い」の落としどころとして採用。
    # mediumはさらに精度が上がるがRTF約1.83と重く、確定までの待ち時間が
    # 2倍以上になるため既定値からは外した(必要ならここをmediumに戻せる)。
    "final_model_size": "small",
    # 確定テキストをローカルAI(Ollama)に整形させるか。
    # アプリのウィンドウ上のチェックボックスから切り替えられる。
    # OFFでもルールベースの整形(句読点・数字変換)は常に効く。
    "llm_cleanup_enabled": False,
    "llm_model": "gemma3:4b",
    "llm_base_url": "http://127.0.0.1:11434",
    # 確定時の文字起こしにGoogle音声認識(要ネット接続、音声がGoogleに送信される)を
    # 使うか。既定はOFF(faster-whisperのみで端末内完結)。ONにすると精度が上がる
    # 場面がある一方、外部送信を伴うためユーザーの明示的なON操作を必要とする。
    "use_google_stt": False,
}

# 翻訳できる言語。tts_culture はOS標準の読み上げ音声(Windows: System.Speech、
# Mac: sayコマンド)を選ぶときの目印
LANGUAGES = {
    "英語": {"code": "en", "name_en": "English", "tts_culture": "en"},
    "韓国語": {"code": "ko", "name_en": "Korean", "tts_culture": "ko"},
    "中国語": {"code": "zh", "name_en": "Chinese", "tts_culture": "zh"},
}

# ---------- 見た目(ネオングリーンのダークテーマ) ----------
BG = "#141715"          # 一番外側の背景
PANEL = "#1e211f"       # メニューバー・ツールバーなどの帯
PANEL_DARK = "#0d0f0e"  # 波形やテキスト欄の中身
NEON = "#3dff6e"        # 主役の蛍光グリーン
NEON_DIM = "#2bb84f"    # 少し落とした緑(グリッド線など)
NEON_FAINT = "#1d3a26"  # ほぼ背景に近い緑
TEXT_MAIN = "#d6f5de"
TEXT_DIM = "#7a8a7e"
REC_RED = "#e8433f"
BORDER = "#2f3a33"

# 波形の下に出すスペクトラム(音の高さごとの強さ)の帯の本数
SPECTRUM_BANDS = 28
WAVE_W, WAVE_H = 500, 260      # 波形エリアの大きさ
METER_W = 96                   # 右側のレベルメーターの幅


def _spec_color(t):
    """0〜1の強さを、暗い緑→蛍光グリーン→黄緑のグラデーションに変換する。"""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        k = t / 0.5
        rgb = (int(13 + (61 - 13) * k), int(43 + (255 - 43) * k),
               int(22 + (110 - 22) * k))
    else:
        k = (t - 0.5) / 0.5
        rgb = (int(61 + (200 - 61) * k), 255, int(110 - (110 - 74) * k))
    return "#%02x%02x%02x" % rgb


def load_config():
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(data)
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _beep(freq, ms):
    """周波数(Hz)と長さ(ミリ秒)を指定して短い正弦波の音を鳴らす。
    winsoundはWindows専用なので、録音にも使っているsounddeviceで
    音を生成して鳴らすことで、Windows/Macどちらでも同じ音が出るようにしている。"""
    duration = ms / 1000.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sd.play(wave, SAMPLE_RATE)
    sd.wait()


def play_start_beep():
    """録音開始音「ピコン」(2音、上がる感じ)"""
    def _run():
        try:
            _beep(1046, 180)
            _beep(1568, 220)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def play_stop_beep():
    """録音終了音「ピロロン」(3音、下がる感じ)"""
    def _run():
        try:
            _beep(1568, 150)
            _beep(1318, 150)
            _beep(1046, 260)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def play_cancel_beep():
    """強制停止(ESC)音。低い一音だけの短いブザーで、開始・終了音と区別する。"""
    def _run():
        try:
            _beep(300, 180)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


class AudioRecorder:
    """マイクからの録音を担当。録音中の音量も波形表示用に貯めていく。"""

    def __init__(self):
        self._stream = None
        self._chunks = []
        self._lock = threading.Lock()
        self.level_queue = queue.Queue()
        self.spectrum_queue = queue.Queue()

    def _callback(self, indata, frames, time_info, status):
        data = indata[:, 0].copy()
        with self._lock:
            self._chunks.append(data)
        try:
            self.level_queue.put_nowait(float(np.abs(data).mean()))
        except queue.Full:
            pass
        # スペクトラム(音を高さごとの強さに分解したもの)を作って渡す。
        # 画面下側のカラフルな帯の描画に使う。
        try:
            n = min(len(data), 1024)
            if n >= 64:
                spec = np.abs(np.fft.rfft(data[:n] * np.hanning(n)))
                # 低い音ほど太く見えるように、対数っぽく24本の帯にまとめる
                edges = np.geomspace(1, len(spec) - 1, SPECTRUM_BANDS + 1).astype(int)
                bands = np.array([
                    spec[edges[i]:max(edges[i + 1], edges[i] + 1)].mean()
                    for i in range(SPECTRUM_BANDS)
                ])
                self.spectrum_queue.put_nowait(bands)
        except Exception:
            pass

    def start(self):
        with self._lock:
            self._chunks = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def snapshot(self):
        """録音を止めずに、今までに録れた音声をコピーして返す(リアルタイム表示用)。"""
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks)

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._chunks)
            self._chunks = []
        return audio


LLM_CLEANUP_PROMPT = """あなたは日本語の音声認識結果を整形する校正者です。以下のルールに厳密に従い、入力文を修正してください。
- 誤字脱字を自然な日本語に直す
- 句点(。)と読点(、)を適切な位置に補う
- ひらがな・カタカナで読み上げられた数量表現(例:ひゃくえん)は算用数字表記(例:100円)に直す
- 元の意味や言い回し、文体(丁寧語/タメ口など)はできるだけ変えない
- 説明や前置き、括弧書きの注釈は一切つけず、修正後の文章だけをそのまま出力する

入力: {text}
出力:"""


# ---------- ルールベースの整形(ローカルAIを使わない軽い整形) ----------
# 「AIで文章を整える」がOFFの時でも、ここだけは常に効く。
# ローカルAIと違って一瞬で終わるので、確定の待ち時間が増えない。

KANJI_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
LARGE_UNITS = {"万": 10 ** 4, "億": 10 ** 8, "兆": 10 ** 12}

# ひらがな読みを漢数字に寄せるための対応表(長いものから順に照合する)
KANA_TOKENS = [
    ("じゅう", "十"), ("じゅっ", "十"), ("じっ", "十"),
    ("ひゃく", "百"), ("びゃく", "百"), ("ぴゃく", "百"),
    ("せん", "千"), ("ぜん", "千"),
    ("まん", "万"), ("おく", "億"),
    ("いち", "一"), ("いっ", "一"),
    ("さん", "三"), ("よん", "四"), ("なな", "七"),
    ("はち", "八"), ("はっ", "八"), ("きゅう", "九"),
    ("ろく", "六"), ("ろっ", "六"),
    ("ぜろ", "〇"), ("れい", "〇"),
    ("しち", "七"),
    ("に", "二"), ("し", "四"), ("ご", "五"), ("く", "九"),
]

# 数のあとに来たら「これは数量表現だ」と判断してよい助数詞(漢字)
COUNTERS = [
    "円", "個", "人", "本", "回", "時間", "分間", "秒", "歳", "才", "番",
    "度", "枚", "冊", "台", "匹", "階", "名", "件", "軒", "杯", "足",
    "頭", "羽", "章", "話", "巻", "パーセント", "キロ", "メートル",
    "グラム", "リットル", "センチ", "ミリ", "年", "週間", "か月", "ヶ月",
    "つ",
]
COUNTERS = sorted(set(COUNTERS), key=len, reverse=True)

# ひらがな読みの助数詞 → 変換後に使う漢字表記
KANA_COUNTERS = {
    "えん": "円", "こ": "個", "にん": "人", "ほん": "本", "ぼん": "本",
    "ぽん": "本", "かい": "回", "じかん": "時間", "ふん": "分",
    "ぷん": "分", "びょう": "秒", "さい": "歳", "ばん": "番",
    "ど": "度", "まい": "枚", "さつ": "冊", "だい": "台", "ひき": "匹",
    "ぴき": "匹", "びき": "匹", "めい": "名", "けん": "件", "はい": "杯",
    "パーセント": "パーセント", "ねん": "年", "つ": "つ",
}


def _kanji_seq_to_int(s):
    """漢数字の並び(例: 千二百三十四)を整数に変換する。"""
    result = 0
    block = 0
    digit = 0
    for ch in s:
        if ch in KANJI_DIGITS:
            digit = digit * 10 + KANJI_DIGITS[ch]
        elif ch in SMALL_UNITS:
            block += (digit if digit else 1) * SMALL_UNITS[ch]
            digit = 0
        elif ch in LARGE_UNITS:
            block += digit
            result += (block if block else 1) * LARGE_UNITS[ch]
            block = 0
            digit = 0
        else:
            return None
    return result + block + digit


_KANJI_NUM_CLASS = "".join(KANJI_DIGITS) + "".join(SMALL_UNITS) + "".join(LARGE_UNITS)
_COUNTER_ALT = "|".join(re.escape(c) for c in COUNTERS)
_KANJI_NUM_RE = re.compile(
    r"(?<![0-9])([%s]+)(?=(?:%s))" % (_KANJI_NUM_CLASS, _COUNTER_ALT)
)

_KANA_NUM_ALT = "|".join(re.escape(k) for k, _ in KANA_TOKENS)
_KANA_COUNTER_ALT = "|".join(
    re.escape(k) for k in sorted(KANA_COUNTERS, key=len, reverse=True)
)
_KANA_NUM_RE = re.compile(r"((?:%s)+)(%s)" % (_KANA_NUM_ALT, _KANA_COUNTER_ALT))


def _convert_kanji_numbers(text):
    """漢数字+助数詞(例: 百円)を算用数字+助数詞(例: 100円)に直す。
    助数詞が続く時だけ変換するので「一緒」「一生懸命」などは壊さない。"""
    def _sub(m):
        value = _kanji_seq_to_int(m.group(1))
        return m.group(1) if value is None else str(value)
    return _KANJI_NUM_RE.sub(_sub, text)


def _kana_to_kanji_num(s):
    out = []
    i = 0
    while i < len(s):
        for kana, kanji in KANA_TOKENS:
            if s.startswith(kana, i):
                out.append(kanji)
                i += len(kana)
                break
        else:
            return None
    return "".join(out)


def _convert_kana_numbers(text):
    """ひらがな読みの数量表現(例: ひゃくえん)を算用数字+漢字助数詞(100円)に直す。"""
    def _sub(m):
        kanji = _kana_to_kanji_num(m.group(1))
        if kanji is None:
            return m.group(0)
        # 「にほん(日本)」→「2本」のような誤変換を防ぐため、
        # 十・百・千・万・億のような位を表す語を含む時だけ変換する
        if not any(u in kanji for u in "十百千万億"):
            return m.group(0)
        value = _kanji_seq_to_int(kanji)
        if value is None:
            return m.group(0)
        return "%d%s" % (value, KANA_COUNTERS[m.group(2)])
    return _KANA_NUM_RE.sub(_sub, text)


SENTENCE_ENDINGS = ["ました", "でした", "ません", "でしょう", "ください",
                    "します", "です", "ます", "だった", "である"]
# これらが後ろに続く時は文の途中なので。を打たない
NO_BREAK_AFTER = ["から", "ので", "けど", "が", "け", "の", "か", "し",
                  "ね", "よ", "と", "な", "わ", "ぞ", "ぜ"]

_PUNCT_RE = re.compile(
    r"(%s)(?!%s)(?=[^。、！？!?\s])"
    % ("|".join(SENTENCE_ENDINGS), "|".join(NO_BREAK_AFTER))
)


def _insert_punctuation(text):
    """「です」「ます」などの文末表現のあとに文が続いていたら。を補う。
    「ですが」「ますので」のように接続が続く場合は打たない。"""
    return _PUNCT_RE.sub(r"\1。", text)


def rule_based_cleanup(text):
    """ローカルAIを使わない軽い整形。
    余計なスペース除去 → 数字表記の変換 → 句点の補いを行う。"""
    text = text.strip()
    if not text:
        return text
    # 日本語の文字どうしの間に入った余計なスペースだけ消す
    # (英単語の間のスペースは残す)
    text = re.sub(r"(?<=[^\x00-\x7F])[ 　]+(?=[^\x00-\x7F])", "", text)
    text = re.sub(r"[ 　]{2,}", " ", text)
    text = _convert_kana_numbers(text)
    text = _convert_kanji_numbers(text)
    text = _insert_punctuation(text)
    if text and text[-1] not in "。、！？!?":
        text += "。"
    return text


def cleanup_with_llm(text, cfg):
    """確定した文字起こし結果を、このPC内で動いているOllama(ローカルAI)に
    渡して、誤字脱字修正・句読点付与・数字表記変換をさせる。
    失敗時(Ollama未起動など)はルールベース整形にフォールバックする。"""
    if not text:
        return text
    base_url = cfg.get("llm_base_url", "http://127.0.0.1:11434")
    model = cfg.get("llm_model", "gemma3:4b")
    prompt = LLM_CLEANUP_PROMPT.format(text=text)
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=20,
        )
        resp.raise_for_status()
        cleaned = resp.json().get("response", "").strip()
        if cleaned:
            return cleaned
    except Exception:
        pass
    return rule_based_cleanup(text)


# ---------- ハングル → カタカナ読み(AIを使わない・一瞬で終わる) ----------
# ハングルは1文字が「子音+母音(+子音)」に必ず分解できる表音文字なので、
# 計算だけで正確にカタカナの読みを作れる。ローカルAIに読みを作らせると
# 崩れやすかったため(実測)、韓国語だけはこちらを使う。

_CHO_ROWS = ["k", "kk", "n", "t", "tt", "r", "m", "p", "pp", "s", "ss",
             "", "ch", "ch", "ch", "k", "t", "p", "h"]

_KANA_ROWS = {
    "": ("ア", "イ", "ウ", "エ", "オ"),
    "k": ("カ", "キ", "ク", "ケ", "コ"),
    "kk": ("カ", "キ", "ク", "ケ", "コ"),
    "n": ("ナ", "ニ", "ヌ", "ネ", "ノ"),
    "t": ("タ", "ティ", "トゥ", "テ", "ト"),
    "tt": ("タ", "ティ", "トゥ", "テ", "ト"),
    "r": ("ラ", "リ", "ル", "レ", "ロ"),
    "m": ("マ", "ミ", "ム", "メ", "モ"),
    "p": ("パ", "ピ", "プ", "ペ", "ポ"),
    "pp": ("パ", "ピ", "プ", "ペ", "ポ"),
    "s": ("サ", "シ", "ス", "セ", "ソ"),
    "ss": ("サ", "シ", "ス", "セ", "ソ"),
    "ch": ("チャ", "チ", "チュ", "チェ", "チョ"),
    "h": ("ハ", "ヒ", "フ", "ヘ", "ホ"),
}

# 母音21種 → (カタカナ5段のどれか, 前に付く音の種類)
_JUNG = [
    (0, None), (3, None), (0, "y"), (3, "y"), (4, None), (3, None),
    (4, "y"), (3, "y"), (4, None), (0, "w"), (3, "w"), (3, "w"),
    (4, "y"), (2, None), (4, "w"), (3, "w"), (1, "w"), (2, "y"),
    (2, None), (1, None), (1, None),
]

_Y_ALONE = {0: "ヤ", 1: "イ", 2: "ユ", 3: "イェ", 4: "ヨ"}
_W_ALONE = {0: "ワ", 1: "ウィ", 2: "ウ", 3: "ウェ", 4: "ウォ"}
_Y_SMALL = {0: "ャ", 1: "", 2: "ュ", 3: "ェ", 4: "ョ"}
_W_SMALL = {0: "ァ", 1: "ィ", 2: "", 3: "ェ", 4: "ォ"}

# 終わりの子音28種 → カタカナ
_JONG = ["", "ク", "ク", "ク", "ン", "ン", "ン", "ッ", "ル", "ク", "ム",
         "ル", "ル", "ル", "プ", "ル", "ム", "プ", "プ", "ッ", "ッ",
         "ン", "ッ", "ッ", "ク", "ッ", "プ", "ッ"]


def hangul_to_katakana(text):
    """ハングルの文章をカタカナの読みに変換する。"""
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if not (0 <= code < 11172):
            out.append(ch)
            continue
        cho, jung, jong = code // 588, (code // 28) % 21, code % 28
        row = _CHO_ROWS[cho]
        base, kind = _JUNG[jung]
        if kind == "y":
            kana = _Y_ALONE[base] if row == "" else _KANA_ROWS[row][1] + _Y_SMALL[base]
        elif kind == "w":
            kana = _W_ALONE[base] if row == "" else _KANA_ROWS[row][2] + _W_SMALL[base]
        else:
            kana = _KANA_ROWS[row][base]
        out.append(kana + _JONG[jong])
    return "".join(out)


# ---------- 英語 → カタカナ読み(AIを使わない) ----------
# 小さいローカルAIは英単語のカタカナ読みが苦手だった(実測で、英文ではなく
# 元の日本語を読み返してしまう)ため、英語もこちらで作る。
# 英語は綴りと発音がずれる言語なので完全ではないが、声に出せる程度を目指す。

# よく出る単語は辞書で直接指定する(規則では正しく読めないものが多いため)
_EN_WORDS = {
    "a": "ア", "the": "ザ", "is": "イズ", "are": "アー", "am": "アム",
    "i": "アイ", "you": "ユー", "he": "ヒー", "she": "シー", "we": "ウィー",
    "they": "ゼイ", "it": "イット", "this": "ディス", "that": "ザット",
    "to": "トゥ", "of": "オブ", "for": "フォー", "and": "アンド",
    "or": "オア", "but": "バット", "in": "イン", "on": "オン", "at": "アット",
    "with": "ウィズ", "from": "フロム", "by": "バイ", "as": "アズ",
    "will": "ウィル", "would": "ウッド", "can": "キャン", "could": "クッド",
    "have": "ハブ", "has": "ハズ", "had": "ハッド", "do": "ドゥー",
    "does": "ダズ", "did": "ディド", "not": "ノット", "no": "ノー",
    "yes": "イエス", "please": "プリーズ", "thank": "サンク",
    "thanks": "サンクス", "you're": "ユア", "i'm": "アイム",
    "meeting": "ミーティング", "tomorrow": "トゥモロー", "today": "トゥデイ",
    "time": "タイム", "start": "スタート", "starts": "スターツ",
    "my": "マイ", "your": "ユア", "me": "ミー", "be": "ビー",
    "was": "ワズ", "were": "ワー", "there": "ゼア", "here": "ヒア",
    "what": "ワット", "where": "ウェア", "when": "ウェン", "who": "フー",
    "how": "ハウ", "why": "ワイ", "very": "ベリー", "good": "グッド",
    "morning": "モーニング", "night": "ナイト", "hello": "ハロー",
    "goodbye": "グッバイ", "sorry": "ソーリー", "excuse": "エクスキューズ",
    "one": "ワン", "two": "トゥー", "three": "スリー", "four": "フォー",
    "five": "ファイブ", "pm": "ピーエム", "am_": "エーエム",
}

# 複数文字のかたまり(長いものから順に照合する)
_EN_CHUNKS = [
    ("tion", "ション"), ("sion", "ジョン"), ("ture", "チャー"),
    ("ough", "オー"), ("augh", "アー"), ("ight", "アイト"),
    ("ing", "イング"), ("all", "オール"), ("ck", "ック"), ("ch", "チ"),
    ("sh", "シュ"), ("th", "ス"), ("ph", "フ"), ("wh", "ワ"),
    ("ng", "ング"), ("qu", "ク"), ("ee", "イー"), ("ea", "イー"),
    ("oo", "ウー"), ("ai", "エイ"), ("ay", "エイ"), ("oa", "オウ"),
    ("oi", "オイ"), ("oy", "オイ"), ("ou", "アウ"), ("ow", "アウ"),
    ("au", "オー"), ("aw", "オー"), ("ar", "アー"), ("er", "アー"),
    ("ir", "アー"), ("ur", "アー"), ("or", "オー"),
]

_EN_CV = {
    "k": "カキクケコ", "c": "カキクケコ", "g": "ガギグゲゴ",
    "s": "サシスセソ", "z": "ザジズゼゾ", "t": "タチトゥテト",
    "d": "ダヂドゥデド", "n": "ナニヌネノ", "h": "ハヒフヘホ",
    "b": "バビブベボ", "p": "パピプペポ", "m": "マミムメモ",
    "y": "ヤイユイェヨ", "r": "ラリルレロ", "l": "ラリルレロ",
    "w": "ワウィウウェウォ", "f": "ファフィフフェフォ",
    "v": "バビブベボ", "j": "ジャジジュジェジョ",
}
_EN_VOWEL_IDX = {"a": 0, "i": 1, "u": 2, "e": 3, "o": 4}
_EN_VOWEL_KANA = "アイウエオ"
# 語末が「母音+子音+e」のときの、母音の読み方(make→メイク など)
_EN_MAGIC_E = {"a": "エイ", "i": "アイ", "o": "オウ", "u": "ユー", "e": "イー"}
_EN_CONSONANT_ALONE = {
    "k": "ク", "c": "ク", "g": "グ", "s": "ス", "z": "ズ", "t": "ト",
    "d": "ド", "n": "ン", "h": "フ", "b": "ブ", "p": "プ", "m": "ム",
    "r": "ー", "l": "ル", "v": "ブ", "f": "フ", "j": "ジ", "x": "クス",
    "w": "ウ", "y": "イ", "q": "ク",
}


def _english_word_to_katakana(word):
    if word in _EN_WORDS:
        return _EN_WORDS[word]

    # 語尾の e が読まれないパターン(make, time, note など)
    magic = ""
    if (len(word) >= 3 and word.endswith("e")
            and word[-2] not in "aeiou" and word[-3] in "aeiou"):
        magic = _EN_MAGIC_E[word[-3]]
        word = word[:-3] + "\0"      # 母音+子音+e の部分を後で差し替える

    out = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch == "\0":
            out.append(magic)
            i += 1
            continue
        for pat, kana in _EN_CHUNKS:
            if word.startswith(pat, i):
                out.append(kana)
                i += len(pat)
                break
        else:
            if ch in _EN_VOWEL_IDX:
                out.append(_EN_VOWEL_KANA[_EN_VOWEL_IDX[ch]])
                i += 1
            elif ch in _EN_CV and i + 1 < len(word) and word[i + 1] in _EN_VOWEL_IDX:
                out.append(_EN_CV[ch][_EN_VOWEL_IDX[word[i + 1]]])
                i += 2
            elif ch in _EN_CONSONANT_ALONE:
                out.append(_EN_CONSONANT_ALONE[ch])
                i += 1
            else:
                i += 1
    # 語尾に子音+e が残っていた場合の後始末
    if magic and "\0" not in word:
        out.append(magic)
    return "".join(out)


def english_to_katakana(text):
    """英文をカタカナの読みに変換する(おおよその近似)。"""
    out = []
    for token in re.findall(r"[A-Za-z']+|[0-9]+|[^A-Za-z'0-9]+", text):
        low = token.lower()
        if re.fullmatch(r"[a-z']+", low):
            out.append(_english_word_to_katakana(low))
        else:
            out.append(token)
    return "".join(out)


# ---------- 翻訳 ----------

TRANSLATE_PROMPT = """あなたはプロの翻訳者です。次の日本語を{lang}に翻訳してください。

出力は必ず次の2行だけにしてください。説明や前置きは一切書かないでください。
訳: <翻訳した文>
読み: <その翻訳文を日本人が声に出して読めるようにしたカタカナ表記>

日本語: {text}"""


def _parse_translation(raw):
    """LLMの返事から「訳」と「読み」を取り出す。"""
    translated, reading = "", ""
    for line in raw.splitlines():
        line = line.strip()
        for prefix in ("訳:", "訳:"):
            if line.startswith(prefix):
                translated = line[len(prefix):].strip()
        for prefix in ("読み:", "読み:"):
            if line.startswith(prefix):
                reading = line[len(prefix):].strip()
    if not translated:
        # 決まった形で返ってこなかった場合は、1行目を訳として扱う
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if lines:
            translated = lines[0]
    return translated, reading


def translate_with_ai(text, cfg):
    """ローカルAI(Ollama)で翻訳する。訳と読み方(カタカナ)を返す。"""
    lang_key = cfg.get("target_language", "英語")
    lang = LANGUAGES.get(lang_key, LANGUAGES["英語"])
    prompt = TRANSLATE_PROMPT.format(lang=lang_key, text=text)
    try:
        resp = requests.post(
            f"{cfg.get('llm_base_url', 'http://127.0.0.1:11434')}/api/generate",
            json={
                "model": cfg.get("llm_model", "gemma3:4b"),
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=60,
        )
        resp.raise_for_status()
        translated, reading = _parse_translation(resp.json().get("response", "").strip())
        # 韓国語はAIに読みを作らせると崩れやすいので、必ず計算で作り直す
        if lang["code"] == "ko" and translated:
            reading = hangul_to_katakana(translated)
        return translated, reading
    except Exception:
        return "", ""


def whisper_transcribe(model, audio):
    """faster-whisperでの文字起こし(端末内で完結)。失敗時は空文字を返す。"""
    try:
        if audio.size > SAMPLE_RATE * 0.2:
            segments, _ = model.transcribe(audio, language="ja", vad_filter=True)
            return "".join(seg.text for seg in segments).strip()
    except Exception as e:
        _log_exception("whisper_transcribe", e)
    return ""


def transcribe_with_google(audio):
    """Google音声認識(要ネット接続、録音した音声がGoogleのサーバーに送信される)。
    faster-whisperより認識精度が良い場面がある代わりに、外部送信を伴う代替エンジン。
    失敗(ネット無し・無音・タイムアウト等)した場合はNoneを返す
    (呼び出し側でfaster-whisperにフォールバックする)。"""
    try:
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        audio_data = sr.AudioData(pcm16, SAMPLE_RATE, 2)
        recognizer = sr.Recognizer()
        return recognizer.recognize_google(audio_data, language="ja-JP")
    except sr.UnknownValueError:
        return None
    except Exception as e:
        _log_exception("transcribe_with_google", e)
        return None


def translate_with_whisper(audio, model):
    """Whisperに元から付いている翻訳機能を使う。
    ローカルAIより軽いが、英語にしか訳せないのが弱点。"""
    try:
        segments, _ = model.transcribe(
            audio, task="translate", vad_filter=True,
        )
        return "".join(seg.text for seg in segments).strip(), ""
    except Exception as e:
        _log_exception("translate_with_whisper", e)
        return "", ""


# ---------- 読み上げ(Windows内蔵の音声を使う。追加インストール不要) ----------

def list_tts_voices():
    """このPCに入っている読み上げ音声の一覧を [(名前, 言語コード), ...] で返す。
    WindowsはPowerShell(System.Speech)、Macは標準の`say`コマンドで調べる。"""
    if IS_MAC:
        try:
            out = subprocess.run(
                ["say", "-v", "?"], capture_output=True, text=True, timeout=15,
            ).stdout
            voices = []
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    voices.append((parts[0], parts[1]))
            return voices
        except Exception:
            return []
    if not IS_WINDOWS:
        return []
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.GetInstalledVoices() | ForEach-Object {"
        " $i=$_.VoiceInfo; Write-Output \"$($i.Name)|$($i.Culture)\" }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=15,
        ).stdout
        voices = []
        for line in out.splitlines():
            if "|" in line:
                name, culture = line.strip().split("|", 1)
                voices.append((name.strip(), culture.strip()))
        return voices
    except Exception:
        return []


def speak_text(text, culture_prefix=None, on_start=None):
    """文章をスピーカーから読み上げる。
    culture_prefix("en"や"ja"など)に合う音声があればそれを使う。
    合う音声が無ければOS標準の音声で読み上げる。
    on_startが渡された場合、読み上げ中のPopenを引数に呼び出す(強制停止用に
    呼び出し元がプロセスを保持できるようにするため)。終了時にはNoneで呼び直す。
    戻り値は (成功したか, 使えなかった言語名 or None)。"""
    if not text.strip():
        return False, None

    voices = list_tts_voices()
    chosen = None
    if culture_prefix:
        for name, culture in voices:
            if culture.lower().startswith(culture_prefix.lower()):
                chosen = name
                break

    if IS_MAC:
        # Macは標準の`say`コマンドにテキストを標準入力で渡して読み上げる。
        # runではなくPopenにしているのは、途中でterminate()して
        # 強制的に読み上げを止められるようにするため(force_stop/ESCから使う)。
        try:
            args = ["say"] + (["-v", chosen] if chosen else [])
            proc = subprocess.Popen(args, stdin=subprocess.PIPE)
            if on_start:
                on_start(proc)
            try:
                proc.communicate(input=text.encode("utf-8"), timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
            return True, (None if chosen else culture_prefix)
        except Exception:
            return False, culture_prefix
        finally:
            if on_start:
                on_start(None)

    if not IS_WINDOWS:
        return False, culture_prefix

    # PowerShellに文章を直接渡すと引用符で壊れるので、一時ファイル経由で渡す
    tmp = os.path.join(BASE_DIR, "_tts_tmp.txt")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        select = f"$s.SelectVoice('{chosen}');" if chosen else ""
        ps = (
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            f"{select}"
            f"$t = Get-Content -Raw -Encoding UTF8 '{tmp}';"
            "$s.Speak($t);"
        )
        # runではなくPopenにしているのは、途中でterminate()して強制的に
        # 読み上げを止められるようにするため(force_stop/ESCから使う)。
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", ps],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if on_start:
            on_start(proc)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True, (None if chosen else culture_prefix)
    except Exception:
        return False, culture_prefix
    finally:
        if on_start:
            on_start(None)
        try:
            os.remove(tmp)
        except Exception:
            pass


def get_foreground_id():
    """今フォーカスしているアプリの識別子を返す。
    Windows: ウィンドウハンドル。Mac: フォーカス中アプリのプロセスID。
    このアプリ自身かどうかを比べる用途にしか使わないので、OSごとに
    中身の意味が違っていても「等しいかどうか」さえ比較できれば十分。"""
    if IS_WINDOWS:
        return ctypes.windll.user32.GetForegroundWindow()
    if IS_MAC:
        try:
            from AppKit import NSWorkspace
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            return app.processIdentifier() if app else None
        except Exception:
            return None
    return None


# キーの送信(貼り付けキー)に使う。ホットキー検出と合わせてpynputに統一。
_kb_controller = pynkb.Controller()

# 修飾キー単体は「ホットキー」として使わせない(押しっぱなしと区別できないため)
_MODIFIER_KEYS = {
    pynkb.Key.ctrl, pynkb.Key.ctrl_l, pynkb.Key.ctrl_r,
    pynkb.Key.alt, pynkb.Key.alt_l, pynkb.Key.alt_r, pynkb.Key.alt_gr,
    pynkb.Key.shift, pynkb.Key.shift_l, pynkb.Key.shift_r,
    pynkb.Key.cmd, pynkb.Key.cmd_l, pynkb.Key.cmd_r,
}


def _pynput_key_name(key):
    """pynputのキーオブジェクトを、config.jsonに保存する短い文字列名に変換する。
    修飾キー単体はNoneを返す(ホットキーとしては扱わない)。
    文字キーは小文字の1文字、それ以外(F8やescなど)は"f8"/"esc"のような名前。"""
    if key in _MODIFIER_KEYS:
        return None
    if isinstance(key, pynkb.KeyCode):
        return key.char.lower() if key.char else None
    return key.name


def _capture_single_key(timeout=15):
    """次に押された(修飾キー単体を除く)1つのキーを待って、その名前を返す。
    複数キーの組み合わせ(Ctrl+Shiftなど)は今回のシンプルさ優先の方針では
    サポートせず、最初に押された「実キー」だけを見る。"""
    result = {"key": None}

    def _on_press(key):
        name = _pynput_key_name(key)
        if name is None:
            return
        result["key"] = name
        return False  # リスナーを止める

    with pynkb.Listener(on_press=_on_press) as listener:
        listener.join(timeout)
    return result["key"]


def type_text_into_foreground(text):
    """クリップボード経由で、今フォーカスしているアプリに文字を貼り付ける。
    貼り付け後は元のクリップボードの中身に戻す。
    貼り付けキーはWindows/Linuxは Ctrl+V、MacはCmd+V。"""
    if not text:
        return
    prev_clip = None
    try:
        prev_clip = pyperclip.paste()
    except Exception:
        pass
    pyperclip.copy(text)
    time.sleep(0.05)
    modifier = pynkb.Key.cmd if IS_MAC else pynkb.Key.ctrl
    with _kb_controller.pressed(modifier):
        _kb_controller.press("v")
        _kb_controller.release("v")
    time.sleep(0.2)
    if prev_clip is not None:
        try:
            pyperclip.copy(prev_clip)
        except Exception:
            pass


class App:
    def __init__(self):
        self.cfg = load_config()
        self.recording = False
        self.transcribing = False
        self.waiting_hotkey_capture = False
        self.live_model = None
        self.final_model = None
        self.own_id = None
        self.level_history = collections.deque(maxlen=110)
        self.spectrum_history = collections.deque(maxlen=110)
        self.events = queue.Queue()
        self.recorder = AudioRecorder()
        self._last_toggle_time = 0.0
        self._stream_thread = None
        self._recording_session = 0
        # 今の録音が普通の入力("dictation")か翻訳("translate")か
        self.mode = "dictation"
        self.last_translation = ""
        # 確定処理(文字起こし・整形・翻訳・読み上げ)を1回ごとに識別するID。
        # ESC(強制停止)で打ち切った回のIDをここに入れておくことで、
        # 停止後にバックグラウンドの古いスレッドが処理を終えても、
        # 貼り付けや読み上げをせずに黙って捨てられるようにしている。
        self._finalize_id = 0
        self._cancelled_finalize_ids = set()
        # 読み上げ中のPowerShellプロセス(ESCで即座にterminateして黙らせる用)
        self._tts_proc = None
        self._capture_target = "hotkey"

        self.root = tk.Tk()
        self.root.title("NekoVoice Pixel")
        self.root.geometry("620x940")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self._build_ui()
        self.root.update_idletasks()
        # Macはウィンドウハンドルの概念が無いので、自プロセスのPIDで
        # 「フォーカスが自分自身のウィンドウか」を判定する
        self.own_id = os.getpid() if IS_MAC else self.root.winfo_id()

        self._start_model_load()
        self._register_hotkey()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._poll)

    # ---------- UI ----------

    def _build_ui(self):
        self.root.configure(bg=BG)
        self._setup_ttk_style()

        self._build_menubar()
        self._build_toolbar()
        self._build_wave_area()
        self._build_wave_footer()
        self._build_text_area()
        self._build_translate_row()
        self._build_bottom_row()

    def _make_toggle(self, parent, text, variable, command, bg):
        """ON/OFFがひと目で分かる自前のチェックボックス。
        tk標準のチェックボックスは背景が黒いとON/OFFの見分けがつかなかったため、
        「☑ 緑 / ☐ 灰色」を自分で描き分けている。"""
        btn = tk.Label(
            parent, font=("Meiryo", 9), bg=bg, anchor="w", cursor="hand2",
        )

        def render():
            on = bool(variable.get())
            btn.config(
                text=("☑ " if on else "☐ ") + text,
                fg=(NEON if on else TEXT_DIM),
            )

        def clicked(_event):
            variable.set(not variable.get())
            render()
            command()

        btn.bind("<Button-1>", clicked)
        render()
        btn._render = render
        return btn

    def _setup_ttk_style(self):
        """言語を選ぶドロップダウンも黒×緑に合わせる
        (何もしないとWindows標準の白いデザインで浮いてしまうため)。"""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Neon.TCombobox",
            fieldbackground=PANEL_DARK, background=PANEL_DARK,
            foreground=NEON, arrowcolor=NEON, bordercolor=NEON_DIM,
            lightcolor=PANEL_DARK, darkcolor=PANEL_DARK, selectbackground=PANEL_DARK,
            selectforeground=NEON,
        )
        style.map(
            "Neon.TCombobox",
            fieldbackground=[("readonly", PANEL_DARK)],
            foreground=[("readonly", NEON)],
            background=[("readonly", PANEL_DARK)],
        )
        # ドロップダウンで開くリスト側の色(styleでは変えられないので個別指定)
        self.root.option_add("*TCombobox*Listbox.background", PANEL_DARK)
        self.root.option_add("*TCombobox*Listbox.foreground", NEON)
        self.root.option_add("*TCombobox*Listbox.selectBackground", NEON_FAINT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", NEON)

    # --- メニューバー風の帯(見た目のための飾り) ---
    def _build_menubar(self):
        bar = tk.Frame(self.root, bg=PANEL, height=26)
        bar.pack(fill="x")
        for name in ("File", "Edit", "View", "Help"):
            tk.Label(
                bar, text=name, font=("Segoe UI", 9), bg=PANEL, fg=TEXT_DIM,
                padx=10, pady=4,
            ).pack(side="left")

    # --- 再生/一時停止/録音ボタンと音量メーターの帯 ---
    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg=PANEL, height=46)
        bar.pack(fill="x")

        left = tk.Frame(bar, bg=PANEL)
        left.pack(side="left", padx=(10, 0))
        for _ in range(2):
            c = tk.Canvas(left, width=22, height=26, bg=PANEL, highlightthickness=0)
            c.create_rectangle(4, 4, 18, 22, outline=TEXT_DIM)
            c.pack(side="left", padx=3)

        center = tk.Frame(bar, bg=PANEL)
        center.pack(side="left", expand=True)
        # 録音ボタンだけは実際に押せるようにしてある(マイクアイコンと同じ動作)
        self.play_btn = self._round_button(center, "▶", NEON, None)
        self.pause_btn = self._round_button(center, "❚❚", TEXT_DIM, None)
        self.rec_btn = self._round_button(
            center, "●", REC_RED, lambda e: self.toggle_recording()
        )
        # 強制停止(ESCキーと同じ動作)。録音中・確定処理中いつでも押せる
        self.stop_btn = self._round_button(
            center, "■", REC_RED, lambda e: self.force_stop()
        )

        right = tk.Frame(bar, bg=PANEL)
        right.pack(side="right", padx=(0, 12))
        tk.Label(
            right, text="🔊\ndB", font=("Segoe UI", 7), bg=PANEL, fg=TEXT_DIM,
            justify="right",
        ).pack(side="left", padx=(0, 4))
        self.vol_canvas = tk.Canvas(
            right, width=150, height=28, bg=PANEL, highlightthickness=0,
        )
        self.vol_canvas.pack(side="left")
        self._draw_volume_bars(0.0)

    def _round_button(self, parent, glyph, color, on_click):
        c = tk.Canvas(parent, width=38, height=38, bg=PANEL, highlightthickness=0)
        c.create_oval(3, 3, 35, 35, outline=color, width=2, fill=PANEL_DARK)
        c.create_text(19, 19, text=glyph, fill=color, font=("Segoe UI", 11))
        c.pack(side="left", padx=5, pady=4)
        if on_click:
            c.bind("<Button-1>", on_click)
        return c

    # --- 波形 + スペクトラム + 右のレベルメーター ---
    def _build_wave_area(self):
        area = tk.Frame(self.root, bg=BG)
        area.pack(fill="x", padx=10, pady=(8, 0))

        self.canvas = tk.Canvas(
            area, width=WAVE_W, height=WAVE_H, bg=PANEL_DARK,
            highlightthickness=1, highlightbackground=NEON_DIM,
        )
        self.canvas.pack(side="left")

        self.meter_canvas = tk.Canvas(
            area, width=METER_W, height=WAVE_H, bg=PANEL,
            highlightthickness=1, highlightbackground=NEON_DIM,
        )
        self.meter_canvas.pack(side="left", padx=(8, 0))

        self._draw_waveform()
        self._draw_meter(0.0)

    # --- 「CH 1-D」やズーム表示の帯(見た目のための飾り) ---
    def _build_wave_footer(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=10, pady=(4, 0))

        self.time_var = tk.StringVar(value="00:00:00")
        strip = tk.Frame(bar, bg=PANEL, highlightthickness=1,
                         highlightbackground=BORDER)
        strip.pack(side="left", fill="x", expand=True)

        tk.Label(strip, text="● CH 1-D", font=("Segoe UI", 8), bg=PANEL,
                 fg=NEON).pack(side="left", padx=10, pady=3)
        tk.Label(strip, text="🔍 100%", font=("Segoe UI", 8), bg=PANEL,
                 fg=TEXT_DIM).pack(side="left", padx=6)
        tk.Label(strip, textvariable=self.time_var, font=("Consolas", 10),
                 bg=PANEL, fg=NEON).pack(side="right", padx=10)

    # --- 音声テキスト欄 ---
    def _build_text_area(self):
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(fill="both", expand=True, padx=10, pady=(8, 0))

        self.status_var = tk.StringVar(value="モデル読み込み中…")
        head = tk.Frame(frame, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text="音声テキスト", font=("Meiryo", 10, "bold"),
                 bg=BG, fg=NEON).pack(side="left")
        tk.Label(head, textvariable=self.status_var, font=("Meiryo", 8),
                 bg=BG, fg=TEXT_DIM).pack(side="right")

        self.text_box = tk.Text(
            frame, height=6, wrap="word", font=("Meiryo", 11),
            bg=PANEL_DARK, fg=NEON, insertbackground=NEON,
            bd=0, highlightthickness=1, highlightbackground=NEON_DIM,
            highlightcolor=NEON, padx=8, pady=6,
        )
        self.text_box.pack(fill="both", expand=True, pady=(4, 0))

        # AI整形のON/OFF
        self.llm_var = tk.BooleanVar(
            value=bool(self.cfg.get("llm_cleanup_enabled", False))
        )
        self._make_toggle(
            frame, "AIで文章を整える", self.llm_var,
            self.toggle_llm_cleanup, BG,
        ).pack(anchor="w", pady=(6, 0))
        self.llm_hint_var = tk.StringVar()
        tk.Label(
            frame, textvariable=self.llm_hint_var, font=("Meiryo", 8),
            bg=BG, fg=TEXT_DIM, anchor="w", justify="left", wraplength=560,
        ).pack(anchor="w", padx=(20, 0))
        self._update_llm_hint()

        # 確定時の文字起こしエンジンをGoogleに切り替えるトグル(代替エンジン)
        self.google_stt_var = tk.BooleanVar(
            value=bool(self.cfg.get("use_google_stt", False))
        )
        self._make_toggle(
            frame, "Google音声認識を使う", self.google_stt_var,
            self.toggle_google_stt, BG,
        ).pack(anchor="w", pady=(4, 0))
        self.google_stt_hint_var = tk.StringVar()
        tk.Label(
            frame, textvariable=self.google_stt_hint_var, font=("Meiryo", 8),
            bg=BG, fg=TEXT_DIM, anchor="w", justify="left", wraplength=560,
        ).pack(anchor="w", padx=(20, 0))
        self._update_google_stt_hint()

    # --- 翻訳モードの操作列 ---
    def _build_translate_row(self):
        frame = tk.Frame(self.root, bg=PANEL, highlightthickness=1,
                         highlightbackground=BORDER)
        frame.pack(fill="x", padx=10, pady=(8, 0))

        row1 = tk.Frame(frame, bg=PANEL)
        row1.pack(fill="x", padx=8, pady=(6, 0))

        tk.Label(row1, text="翻訳", font=("Meiryo", 9, "bold"),
                 bg=PANEL, fg=NEON).pack(side="left")

        self.lang_var = tk.StringVar(
            value=self.cfg.get("target_language", "英語")
        )
        lang_box = ttk.Combobox(
            row1, textvariable=self.lang_var, values=list(LANGUAGES),
            state="readonly", width=8, font=("Meiryo", 9),
            style="Neon.TCombobox",
        )
        lang_box.pack(side="left", padx=(8, 0))
        lang_box.bind("<<ComboboxSelected>>", lambda e: self.on_language_change())

        # 翻訳モードのホットキー表示と変更ボタン
        self.tr_hotkey_var = tk.StringVar(
            value=self.cfg.get("translate_hotkey", "f7").upper()
        )
        tk.Label(row1, textvariable=self.tr_hotkey_var,
                 font=("Meiryo", 10, "bold"), bg=PANEL, fg=NEON,
                 ).pack(side="left", padx=(12, 0))
        tk.Button(
            row1, text="変更", font=("Meiryo", 8), bg=PANEL_DARK, fg=TEXT_MAIN,
            activebackground=NEON_FAINT, activeforeground=NEON,
            bd=0, highlightthickness=1, padx=6,
            command=lambda: self.start_hotkey_capture(translate=True),
        ).pack(side="left", padx=(6, 0))

        self.speak_btn = tk.Button(
            row1, text="🔊 音声を流す", font=("Meiryo", 9),
            bg=PANEL_DARK, fg=NEON, activebackground=NEON_FAINT,
            activeforeground=NEON, bd=0, highlightthickness=1, padx=8,
            command=self.speak_current,
        )
        self.speak_btn.pack(side="right")

        self.translate_ai_var = tk.BooleanVar(
            value=bool(self.cfg.get("translate_with_ai", False))
        )
        self._make_toggle(
            frame, "翻訳にAIを使う", self.translate_ai_var,
            self.toggle_translate_ai, PANEL,
        ).pack(anchor="w", padx=8, pady=(2, 0))

        self.tr_hint_var = tk.StringVar()
        tk.Label(
            frame, textvariable=self.tr_hint_var, font=("Meiryo", 8),
            bg=PANEL, fg=TEXT_DIM, anchor="w", justify="left", wraplength=560,
        ).pack(anchor="w", padx=(28, 8), pady=(0, 6))
        self._update_translate_hint()

    # --- 猫・マイク・ホットキー(元のレイアウトのまま) ---
    def _build_bottom_row(self):
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=16, pady=(8, 12))

        cat = tk.Canvas(bottom, width=76, height=64, bg=BG, highlightthickness=0)
        cat.pack(side="left")
        self._draw_cat(cat)

        right = tk.Frame(bottom, bg=BG)
        right.pack(side="right")

        self.mic_canvas = tk.Canvas(
            right, width=54, height=54, bg=BG, highlightthickness=0,
        )
        self.mic_canvas.pack()
        self._draw_mic(idle=True)
        self.mic_canvas.bind("<Button-1>", lambda e: self.toggle_recording())

        hk_frame = tk.Frame(right, bg=BG)
        hk_frame.pack(pady=(4, 0))
        self.hotkey_var = tk.StringVar(value=self.cfg["hotkey"].upper())
        tk.Label(
            hk_frame, textvariable=self.hotkey_var, font=("Meiryo", 10, "bold"),
            bg=BG, fg=NEON,
        ).pack(side="left")
        tk.Button(
            hk_frame, text="変更", font=("Meiryo", 8), bg=PANEL_DARK,
            fg=TEXT_MAIN, activebackground=NEON_FAINT, activeforeground=NEON,
            bd=0, highlightthickness=1, padx=6,
            command=self.start_hotkey_capture,
        ).pack(side="left", padx=(6, 0))

        stop_lbl = tk.Label(
            right, text="ESC: 強制停止(貼り付け・読み上げも中断)",
            font=("Meiryo", 8), bg=BG, fg=REC_RED, cursor="hand2",
        )
        stop_lbl.pack(pady=(4, 0))
        stop_lbl.bind("<Button-1>", lambda e: self.force_stop())

        if IS_MAC:
            # ホットキー・自動貼り付けはmacOSの「アクセシビリティ」権限が無いと
            # 一切反応しない(エラーも出ずに黙って効かないだけなので、
            # 初回起動時にここで案内しておく)
            tk.Label(
                right,
                text="初回のみ: システム設定→プライバシーとセキュリティ→\n"
                     "アクセシビリティ でこのアプリを許可してください",
                font=("Meiryo", 8), bg=BG, fg=TEXT_DIM, justify="right",
            ).pack(pady=(4, 0))

    def _draw_cat(self, c):
        """左下の猫マスコット(ネオングリーンの線画)"""
        n = NEON
        c.create_oval(20, 20, 56, 52, outline=n, width=2)          # 顔
        c.create_polygon(22, 26, 26, 10, 36, 22, outline=n, fill="", width=2)
        c.create_polygon(54, 26, 50, 10, 40, 22, outline=n, fill="", width=2)
        c.create_oval(30, 31, 34, 36, outline=n, width=2)          # 目
        c.create_oval(42, 31, 46, 36, outline=n, width=2)
        c.create_line(38, 39, 38, 42, fill=n, width=2)             # 鼻
        c.create_arc(32, 38, 38, 45, start=200, extent=140, style="arc",
                     outline=n, width=2)
        c.create_arc(38, 38, 44, 45, start=200, extent=140, style="arc",
                     outline=n, width=2)
        for y in (34, 38):                                          # ひげ
            c.create_line(8, y, 28, y + 2, fill=n, width=1)
            c.create_line(68, y, 48, y + 2, fill=n, width=1)
        c.create_arc(14, 40, 62, 64, start=200, extent=140, style="arc",
                     outline=n, width=2)                            # 体
        c.create_arc(56, 40, 74, 60, start=270, extent=180, style="arc",
                     outline=n, width=2)                            # しっぽ

    def _update_llm_hint(self):
        if self.llm_var.get():
            self.llm_hint_var.set(
                "ON: 誤字脱字も直して句読点・数字を整えます(確定まで数秒かかります)"
            )
        else:
            self.llm_hint_var.set(
                "OFF: 句読点と数字だけ整えます(すぐ確定します)"
            )

    def _update_google_stt_hint(self):
        if self.google_stt_var.get():
            self.google_stt_hint_var.set(
                "ON: 確定時にGoogle音声認識を使います"
                "(要ネット接続。音声がGoogleに送信されます。失敗時は自動でこのPC内の認識に戻ります)"
            )
        else:
            self.google_stt_hint_var.set(
                "OFF: 全てこのPC内で完結するfaster-whisperのみ使います"
            )

    def _update_translate_hint(self):
        lang = self.lang_var.get()
        key = self.cfg.get("translate_hotkey", "f7").upper()
        if self.translate_ai_var.get():
            self.tr_hint_var.set(
                f"ON: {key}を押して日本語で話すと{lang}に翻訳します"
                "(読み方も出ます。数秒かかります)"
            )
        else:
            self.tr_hint_var.set(
                f"OFF: AIを使わない軽い翻訳です。英語にしか訳せません"
                f"({lang}に訳すにはONにしてください)"
            )

    def toggle_llm_cleanup(self):
        """チェックボックスの切り替え。設定に即保存し、次の確定からすぐ効く
        (アプリの再起動は不要)。"""
        enabled = bool(self.llm_var.get())
        self.cfg["llm_cleanup_enabled"] = enabled
        try:
            save_config(self.cfg)
        except Exception as e:
            self.status_var.set(f"設定の保存に失敗: {e}")
            return
        self._update_llm_hint()

    def toggle_google_stt(self):
        self.cfg["use_google_stt"] = bool(self.google_stt_var.get())
        try:
            save_config(self.cfg)
        except Exception as e:
            self.status_var.set(f"設定の保存に失敗: {e}")
            return
        self._update_google_stt_hint()

    def toggle_translate_ai(self):
        self.cfg["translate_with_ai"] = bool(self.translate_ai_var.get())
        try:
            save_config(self.cfg)
        except Exception as e:
            self.status_var.set(f"設定の保存に失敗: {e}")
            return
        self._update_translate_hint()

    def on_language_change(self):
        self.cfg["target_language"] = self.lang_var.get()
        try:
            save_config(self.cfg)
        except Exception as e:
            self.status_var.set(f"設定の保存に失敗: {e}")
            return
        self._update_translate_hint()

    def speak_current(self):
        """「音声を流す」ボタン。翻訳結果(なければ表示中の文)を読み上げる。"""
        text = self.last_translation or self.text_box.get("1.0", "end").strip()
        if not text:
            return
        lang = LANGUAGES.get(self.lang_var.get(), LANGUAGES["英語"])
        culture = lang["tts_culture"] if self.last_translation else "ja"

        def _run():
            self.events.put(("status", "読み上げ中…"))
            ok, missing = speak_text(
                text, culture, on_start=lambda p: setattr(self, "_tts_proc", p)
            )
            if missing:
                self.events.put((
                    "status",
                    f"{self.lang_var.get()}の音声がこのPCに入っていないため"
                    "別の声で読み上げました",
                ))
            else:
                self.events.put(("status", self._idle_status_text()))
        threading.Thread(target=_run, daemon=True).start()

    def _draw_mic(self, idle=True):
        self.mic_canvas.delete("all")
        color = NEON if idle else REC_RED
        self.mic_canvas.create_oval(17, 6, 37, 34, outline=color, width=2)
        self.mic_canvas.create_line(11, 26, 11, 31, fill=color, width=2)
        self.mic_canvas.create_line(43, 26, 43, 31, fill=color, width=2)
        self.mic_canvas.create_arc(11, 20, 43, 44, start=200, extent=140,
                                   style="arc", outline=color, width=2)
        self.mic_canvas.create_line(27, 44, 27, 50, fill=color, width=2)
        self.mic_canvas.create_line(18, 50, 36, 50, fill=color, width=2)

    def _draw_volume_bars(self, level):
        c = self.vol_canvas
        c.delete("all")
        for row, y in ((0, 6), (1, 17)):
            c.create_rectangle(0, y, 150, y + 8, outline=BORDER, fill=PANEL_DARK)
            filled = int(150 * min(level * (1.0 if row == 0 else 0.85), 1.0))
            if filled > 2:
                c.create_rectangle(0, y, filled, y + 8, outline="", fill=NEON)

    def _draw_meter(self, level):
        """右側の縦型レベルメーター(dB目盛り付き)"""
        c = self.meter_canvas
        c.delete("all")
        top, bottom = 16, WAVE_H - 34
        for label, frac in (("0", 0.0), ("-5", 0.1), ("-10", 0.22),
                            ("-30", 0.42), ("-45", 0.6), ("-50", 0.72),
                            ("-60", 0.85), ("-70", 0.97)):
            y = top + (bottom - top) * frac
            c.create_text(METER_W - 12, y, text=label, fill=TEXT_DIM,
                          font=("Segoe UI", 6), anchor="e")
        for i, (x0, scale) in enumerate(((16, 1.0), (40, 0.86))):
            c.create_rectangle(x0, top, x0 + 16, bottom,
                               outline=BORDER, fill=PANEL_DARK)
            h = (bottom - top) * min(level * scale, 1.0)
            if h > 1:
                c.create_rectangle(x0, bottom - h, x0 + 16, bottom,
                                   outline="", fill=NEON)
            c.create_text(x0 + 8, bottom + 10, text=("L", "LB")[i],
                          fill=TEXT_DIM, font=("Segoe UI", 6))

    def _draw_waveform(self):
        c = self.canvas
        c.delete("all")
        w, h = WAVE_W, WAVE_H
        wave_h = h * 0.62          # 上側が波形
        spec_top = wave_h          # 下側がスペクトラム
        mid = wave_h / 2

        # 方眼のグリッド線
        for i in range(1, 8):
            x = w * i / 8
            c.create_line(x, 0, x, wave_h, fill=NEON_FAINT)
        for i in range(1, 5):
            y = wave_h * i / 5
            c.create_line(0, y, w, y, fill=NEON_FAINT)
        c.create_line(0, mid, w, mid, fill=NEON_DIM)

        n = len(self.level_history)
        if n:
            bar_w = w / max(n, 1)
            # 波形(左右対称の棒)
            for i, lvl in enumerate(self.level_history):
                amp = min(lvl * 16, 1.0) * (mid - 6)
                x = i * bar_w
                c.create_line(x, mid - amp, x, mid + amp, fill=NEON,
                              width=max(bar_w - 1, 1))
            # なめらかな包絡線(音の大きさの移り変わり)
            if n >= 3:
                pts = []
                for i, lvl in enumerate(self.level_history):
                    amp = min(lvl * 16, 1.0) * (mid - 10)
                    pts += [i * bar_w, mid - amp]
                c.create_line(*pts, fill="#b9ffcf", width=2, smooth=True)

        # スペクトラム(音の高さごとの強さを色の濃さで表す)
        cols = len(self.spectrum_history)
        if cols:
            col_w = w / max(cols, 1)
            band_h = (h - spec_top) / SPECTRUM_BANDS
            for xi, bands in enumerate(self.spectrum_history):
                for bi, v in enumerate(bands):
                    if v <= 0.0015:
                        continue
                    color = _spec_color(min(v * 26, 1.0))
                    y0 = h - (bi + 1) * band_h
                    c.create_rectangle(xi * col_w, y0, (xi + 1) * col_w,
                                       y0 + band_h, outline="", fill=color)

    # ---------- モデル読み込み ----------

    def _start_model_load(self):
        def _load_live():
            try:
                self.live_model = WhisperModel(
                    self.cfg.get("realtime_model_size", "small"), device="cpu", compute_type="int8",
                )
                self.events.put(("status", self._idle_status_text()))
            except Exception as e:
                # ここで失敗すると self.live_model が None のままになり、
                # ホットキーを押しても toggle_recording が何もせず戻ってしまう
                # (画面が「モデル読み込み中…」のまま反応しなくなって見える)。
                # 初回はモデルをネット経由でダウンロードするため、通信環境が
                # 悪い/無い場合にここで失敗しやすい。
                _log_exception("live_model load", e)
                self.events.put((
                    "status",
                    "音声認識モデルの読み込みに失敗しました。"
                    "インターネット接続を確認して再起動してください",
                ))

        def _load_final():
            try:
                self.final_model = WhisperModel(
                    self.cfg.get("final_model_size", "medium"), device="cpu", compute_type="int8",
                )
            except Exception as e:
                _log_exception("final_model load", e)
                # 確定用モデルが読めなくても、_finalize_workerはself.live_modelに
                # フォールバックするので致命的ではない。ステータスだけ出す。
                self.events.put(("status", "高精度モデルの読み込みに失敗しました(簡易モデルで動作します)"))

        # リアルタイム用(軽い)と確定用(精度重視)を並行して読み込む。
        # リアルタイム用が読み終わった時点でアプリは使い始められる。
        threading.Thread(target=_load_live, daemon=True).start()
        threading.Thread(target=_load_final, daemon=True).start()

    def _idle_status_text(self):
        return f"待機中(ホットキー: {self.cfg['hotkey'].upper()})"

    # ---------- 録音トグル ----------
    # F8を1回目に押す: それまでの音声テキストを消して録音開始。
    #                  話している間はリアルタイムに文字が表示され続ける。
    # F8を2回目に押す: 録音停止 → 高精度モデルで文字起こしし直す →
    #                  ローカルAIで文章を整形 → 今フォーカスしているアプリに貼り付け。

    def toggle_recording(self, mode="dictation"):
        if self.live_model is None or self.transcribing:
            return
        if not self.recording:
            self.mode = mode
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self.recording = True
        self._recording_session += 1
        session = self._recording_session
        self.level_history.clear()
        self.spectrum_history.clear()
        self.last_translation = ""
        self.events.put(("clear", None))
        play_start_beep()
        self.recorder.start()
        self._draw_mic(idle=False)
        if self.mode == "translate":
            self.events.put((
                "status", f"翻訳モードで録音中({self.lang_var.get()})…"
            ))
        else:
            self.events.put(("status", "録音中(リアルタイム認識)…"))
        self._stream_thread = threading.Thread(
            target=self._streaming_worker, args=(session,), daemon=True,
        )
        self._stream_thread.start()

    def _stop_recording(self):
        self.recording = False
        play_stop_beep()
        if self._stream_thread is not None:
            # session不一致で自然にループを抜けるので長くは待たない。
            # (万一時間がかかっても、古い結果はsessionチェックで無視される)
            self._stream_thread.join(timeout=2)
            self._stream_thread = None
        audio = self.recorder.stop()
        self._draw_mic(idle=True)
        self.events.put(("status", "文字起こし中(高精度)…"))
        self.transcribing = True
        self._finalize_id += 1
        finalize_id = self._finalize_id
        mode = self.mode
        foreground_id = get_foreground_id()
        threading.Thread(
            target=self._finalize_worker,
            args=(audio, foreground_id, finalize_id, mode), daemon=True,
        ).start()

    def force_stop(self):
        """ESC(強制停止)。録音中ならその場で中断して破棄する。
        確定処理(文字起こし・整形・翻訳・読み上げ)の途中なら、貼り付けや
        読み上げを一切せずに打ち切る。裏で動いているスレッドは即座には
        止められないが、finalize_idを「打ち切り済み」に登録しておくことで、
        後から結果が返ってきても黙って捨てられるようにしている。"""
        proc = self._tts_proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

        if self.recording:
            self.recording = False
            if self._stream_thread is not None:
                self._stream_thread.join(timeout=2)
                self._stream_thread = None
            try:
                self.recorder.stop()
            except Exception:
                pass
            self._draw_mic(idle=True)
            self.events.put(("clear", None))
            play_cancel_beep()
            self.events.put(("status", self._idle_status_text()))
            return

        if self.transcribing:
            self._cancelled_finalize_ids.add(self._finalize_id)
            self.transcribing = False
            play_cancel_beep()
            self.events.put(("status", self._idle_status_text()))

    def _streaming_worker(self, session):
        """録音中、少し話すたびにその場でざっくり文字起こしして表示欄に反映する。
        速さ優先の軽いモデル(realtime_model_size)を使い、確定はしない。
        sessionは「この録音が何回目のF8か」の通し番号。録音が切り替わった後に
        古いスレッドの結果が紛れ込んで表示を壊さないよう、poll側で照合する。"""
        last_len = 0
        min_new_samples = int(SAMPLE_RATE * 0.5)
        while self.recording and session == self._recording_session:
            audio = self.recorder.snapshot()
            if audio.size - last_len < min_new_samples:
                time.sleep(0.15)
                continue
            last_len = audio.size
            if audio.size < SAMPLE_RATE * 0.3:
                time.sleep(0.15)
                continue
            try:
                segments, _ = self.live_model.transcribe(
                    audio, language="ja", vad_filter=True, beam_size=1,
                )
                text = "".join(seg.text for seg in segments).strip()
            except Exception as e:
                _log_exception("streaming transcribe", e)
                text = None
            if text is not None:
                self.events.put(("partial", (session, text)))

    def _finalize_worker(self, audio, foreground_id, finalize_id, mode):
        model = self.final_model if self.final_model is not None else self.live_model
        raw_text = ""
        if self.cfg.get("use_google_stt", False):
            self.events.put(("status", "Google音声認識で確認しています…"))
            raw_text = transcribe_with_google(audio) or ""
            if not raw_text:
                # ネット無し・タイムアウト等で失敗した場合は、
                # 端末内で完結するfaster-whisperにフォールバックする
                raw_text = whisper_transcribe(model, audio)
        else:
            raw_text = whisper_transcribe(model, audio)

        if finalize_id in self._cancelled_finalize_ids:
            return

        if mode == "translate":
            self._finalize_translation(audio, model, raw_text, foreground_id, finalize_id)
            return

        final_text = raw_text
        if raw_text:
            if self.cfg.get("llm_cleanup_enabled", False):
                # ローカルAIあり: 誤字脱字まで直せるが数秒かかる
                self.events.put(("status", "AIが文章を整えています…"))
                final_text = cleanup_with_llm(raw_text, self.cfg)
            else:
                # ローカルAIなし: 句読点と数字だけ整える。一瞬で終わる
                self.events.put(("status", "文章を整えています…"))
                final_text = rule_based_cleanup(raw_text)

        if finalize_id in self._cancelled_finalize_ids:
            return

        self.transcribing = False
        if final_text and foreground_id != self.own_id:
            type_text_into_foreground(final_text)
        self.events.put(("final", final_text))
        self.events.put(("status", self._idle_status_text()))

    def _finalize_translation(self, audio, model, raw_text, foreground_id, finalize_id):
        """翻訳モードの確定処理。翻訳文と読み方を作って表示・貼り付けする。"""
        lang_key = self.cfg.get("target_language", "英語")
        translated, reading = "", ""

        if self.cfg.get("translate_with_ai", False):
            self.events.put(("status", f"AIが{lang_key}に翻訳しています…"))
            translated, reading = translate_with_ai(
                rule_based_cleanup(raw_text) or raw_text, self.cfg
            )
        else:
            # AIなしの軽い翻訳。Whisper内蔵の機能なので英語にしかできない
            self.events.put(("status", "翻訳しています(AIなし)…"))
            translated, reading = translate_with_whisper(audio, model)
            if LANGUAGES.get(lang_key, {}).get("code") != "en":
                self.events.put((
                    "status",
                    f"AIなしでは{lang_key}に翻訳できないため英語にしました",
                ))

        if finalize_id in self._cancelled_finalize_ids:
            return

        self.transcribing = False
        if not translated:
            self.events.put(("final", raw_text))
            self.events.put(("status", "翻訳できませんでした"))
            return

        self.last_translation = translated
        shown = f"{raw_text}\n→ {translated}"
        if reading:
            shown += f"\n[読み] {reading}"
        if foreground_id != self.own_id:
            type_text_into_foreground(translated)
        self.events.put(("final", shown))

        if finalize_id in self._cancelled_finalize_ids:
            return

        # 翻訳結果を自動でスピーカーから読み上げる(相手に直接聞こえるように)。
        # 「🔊 音声を流す」ボタンは、聞き逃した時にもう一度流すための手動再生用。
        culture = LANGUAGES.get(lang_key, LANGUAGES["英語"])["tts_culture"]
        self.events.put(("status", "読み上げ中…"))
        ok, missing = speak_text(
            translated, culture, on_start=lambda p: setattr(self, "_tts_proc", p)
        )
        if finalize_id in self._cancelled_finalize_ids:
            return
        if missing:
            self.events.put((
                "status",
                f"{lang_key}の音声がこのPCに入っていないため別の声で読み上げました",
            ))
        else:
            self.events.put(("status", self._idle_status_text()))

    def _set_transcript(self, text):
        self.text_box.delete("1.0", "end")
        if text:
            self.text_box.insert("end", text)
        self.text_box.see("end")

    # ---------- ホットキー(pynput、Windows/Mac両対応) ----------
    # keyboardライブラリはMacのグローバルホットキーにほぼ対応していないため、
    # pynputに統一した。仕組みがシンプルになる副作用として、ホットキーを
    # 登録し直す(unhook_all_hotkeys等)処理も不要になっている:
    # リスナーは1つだけ常駐させ、押されたキー名を毎回self.cfgと突き合わせる。
    # そのため変更後は次のキー入力からすぐ新しいキーで反応する。

    def _register_hotkey(self):
        self._kb_listener = pynkb.Listener(on_press=self._on_global_key)
        self._kb_listener.daemon = True
        self._kb_listener.start()

    def _on_global_key(self, key):
        # ホットキー変更待ちの間は、押しても録音/強制停止が発動しないようにする
        if self.waiting_hotkey_capture:
            return
        name = _pynput_key_name(key)
        if name is None:
            return
        if name == "esc":
            self.force_stop()
            return
        now = time.time()
        if name == self.cfg["hotkey"]:
            if now - self._last_toggle_time >= 0.4:
                self._last_toggle_time = now
                self.toggle_recording("dictation")
        elif name == self.cfg.get("translate_hotkey", "f7"):
            if now - self._last_toggle_time >= 0.4:
                self._last_toggle_time = now
                self.toggle_recording("translate")

    def start_hotkey_capture(self, translate=False):
        if self.waiting_hotkey_capture:
            return
        self.waiting_hotkey_capture = True
        self._capture_target = "translate_hotkey" if translate else "hotkey"
        (self.tr_hotkey_var if translate else self.hotkey_var).set("キー待機中…")

        def _capture():
            new_key = _capture_single_key(timeout=15)
            self.waiting_hotkey_capture = False
            if new_key:
                self.events.put(("hotkey_captured", new_key))
            else:
                self.events.put(("status", self._idle_status_text()))
                self._refresh_hotkey_labels()

        threading.Thread(target=_capture, daemon=True).start()

    def _refresh_hotkey_labels(self):
        self.hotkey_var.set(self.cfg["hotkey"].upper())
        self.tr_hotkey_var.set(self.cfg.get("translate_hotkey", "f7").upper())

    def _apply_new_hotkey(self, new_key):
        target = self._capture_target
        other = "hotkey" if target == "translate_hotkey" else "translate_hotkey"
        # escは強制停止専用として予約しているので、F8/F7には割り当てさせない
        if new_key == "esc":
            self.events.put(("status", "ESCは強制停止用に予約されています"))
            self._refresh_hotkey_labels()
            return
        # 2つのホットキーが同じキーになってしまうのを防ぐ
        if new_key == self.cfg.get(other):
            self.events.put((
                "status", f"{new_key.upper()}はもう一方で使用中です",
            ))
            self._refresh_hotkey_labels()
            return

        self.cfg[target] = new_key
        save_config(self.cfg)
        self._refresh_hotkey_labels()
        self._update_translate_hint()
        self.events.put(("status", self._idle_status_text()))

    # ---------- メインループ ----------

    def _poll(self):
        changed = False
        latest = 0.0
        while True:
            try:
                lvl = self.recorder.level_queue.get_nowait()
            except queue.Empty:
                break
            self.level_history.append(lvl)
            latest = max(latest, lvl)
            changed = True

        while True:
            try:
                bands = self.recorder.spectrum_queue.get_nowait()
            except queue.Empty:
                break
            self.spectrum_history.append(bands)
            changed = True

        if self.recording and changed:
            self._draw_waveform()
            level = min(latest * 16, 1.0)
            self._draw_meter(level)
            self._draw_volume_bars(level)
        elif not self.recording and changed:
            self._draw_meter(0.0)
            self._draw_volume_bars(0.0)

        self.time_var.set(time.strftime("%H:%M:%S"))

        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status_var.set(payload)
            elif kind == "clear":
                self._set_transcript("")
            elif kind == "partial":
                session, text = payload
                if session == self._recording_session:
                    self._set_transcript(text)
            elif kind == "final":
                self._set_transcript(payload)
            elif kind == "hotkey_captured":
                self._apply_new_hotkey(payload)

        self.root.after(50, self._poll)

    def _on_close(self):
        try:
            self._kb_listener.stop()
        except Exception:
            pass
        if self.recording:
            try:
                self.recorder.stop()
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def _thread_excepthook(args):
    """バックグラウンドスレッドで想定外の例外が起きた時、アプリを黙って
    固まらせるのではなくログに残す(--windowedビルドはコンソールが無いため、
    ここで拾わないとユーザーには何も見えないまま反応しなくなる)。"""
    _log_exception(f"unhandled in thread {args.thread.name}", args.exc_value)


if __name__ == "__main__":
    threading.excepthook = _thread_excepthook
    try:
        App().run()
    except Exception as e:
        _log_exception("main", e)
        raise
