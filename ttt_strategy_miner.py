#!/usr/bin/env python3
"""
TIC-TAC-TOE STRATEGY MINER
==========================
A data-mining project on tic-tac-toe.
Pure Python standard library (tkinter, csv, random, itertools, collections)
-> nothing to pip install.

Pipeline
  1. Generate : every legal game (255,168 games, 5,478 positions, 958 final boards)
  2. Mine     : outcome statistics, opening heat-maps, Apriori association rules
  3. Strategy : score every move by its win/loss record over all continuations
  4. Validate : compare mined strategies with minimax truth in a tournament

Run:  python ttt_strategy_miner.py
"""
import sys
import csv
import random
from collections import defaultdict, Counter
from itertools import combinations

# ----------------------------------------------------------------------------
# 1. GAME LOGIC + DATA MINING ENGINE  (no GUI code in this part)
# ----------------------------------------------------------------------------
EMPTY = "." * 9
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
CELL_NAMES = ["top-left", "top-middle", "top-right",
              "middle-left", "center", "middle-right",
              "bottom-left", "bottom-middle", "bottom-right"]
NAME2POS = {f"{p} {CELL_NAMES[i]}": (p, i) for p in "XO" for i in range(9)}
RESULT_NAME = {"X": "X wins", "O": "O wins", "D": "Draw"}
LEVELS = ["Rookie", "Miner", "Master"]


def win_line(b):
    for ln in LINES:
        i, j, k = ln
        if b[i] != "." and b[i] == b[j] == b[k]:
            return ln
    return None


def winner(b):
    ln = win_line(b)
    return b[ln[0]] if ln else None


def to_move(b):
    return "X" if b.count("X") == b.count("O") else "O"


def empties(b):
    return [i for i, c in enumerate(b) if c == "."]


def put(b, i, p):
    return b[:i] + p + b[i + 1:]


def is_over(b):
    return winner(b) is not None or "." not in b


class Miner:
    """Builds the full game dataset and mines it."""

    def __init__(self):
        self.stats = {}       # board -> (X-win games, O-win games, draw games) that continue from it
        self.value = {}       # board -> +1 X wins / 0 draw / -1 O wins with perfect play (minimax)
        self.paths = {}       # board -> number of distinct games passing through it
        self.terminals = {}   # final board -> (number of games, 'X'|'O'|'D')
        self.length = Counter()  # game length -> number of games
        self._count(EMPTY)
        self._flow()
        self.total = sum(self.stats[EMPTY])
        self.rules = self._mine_rules()

    # -- step 1: exhaustive game tree (memoised) ------------------------------
    def _count(self, b):
        if b in self.stats:
            return self.stats[b]
        w = winner(b)
        if w:
            res = (1, 0, 0) if w == "X" else (0, 1, 0)
            self.value[b] = 1 if w == "X" else -1
        elif "." not in b:
            res = (0, 0, 1)
            self.value[b] = 0
        else:
            p = to_move(b)
            x = o = d = 0
            vals = []
            for i in empties(b):
                nb = put(b, i, p)
                cx, co, cd = self._count(nb)
                x += cx
                o += co
                d += cd
                vals.append(self.value[nb])
            res = (x, o, d)
            self.value[b] = max(vals) if p == "X" else min(vals)
        self.stats[b] = res
        return res

    # -- step 2: how many games reach each position / end where ---------------
    def _flow(self):
        level = {EMPTY: 1}
        self.paths[EMPTY] = 1
        for depth in range(1, 10):
            nxt = defaultdict(int)
            for b, c in level.items():
                p = to_move(b)
                for i in empties(b):
                    nxt[put(b, i, p)] += c
            level = {}
            for nb, c in nxt.items():
                self.paths[nb] = c
                w = winner(nb)
                if w or "." not in nb:
                    self.terminals[nb] = (c, w if w else "D")
                    self.length[depth] += c
                else:
                    level[nb] = c

    # -- step 3: Apriori association rules  (pattern on final board -> result) -
    def _trivial(self, A):
        cells = {"X": set(), "O": set()}
        for it in A:
            p, i = NAME2POS[it]
            cells[p].add(i)
        return any(set(l) <= cells[p] for p in "XO" for l in LINES)

    def _mine_rules(self, min_support=0.03, max_len=3):
        trans = []
        for b, (w, res) in self.terminals.items():
            items = frozenset(f"{c} {CELL_NAMES[i]}" for i, c in enumerate(b) if c != ".")
            trans.append((items, w, res))
        total = sum(w for _, w, _ in trans)

        cnt = defaultdict(int)
        for items, w, _ in trans:
            for it in items:
                cnt[frozenset([it])] += w
        frequent = {s: c for s, c in cnt.items() if c / total >= min_support}
        prev = list(frequent)
        k = 2
        while prev and k <= max_len:
            cands = set()
            for a, b in combinations(prev, 2):
                u = a | b
                if len(u) == k:
                    cands.add(u)
            cands = [c for c in cands if all(frozenset(s) in frequent for s in combinations(c, k - 1))]
            cnt = defaultdict(int)
            for items, w, _ in trans:
                for c in cands:
                    if c <= items:
                        cnt[c] += w
            new = {s: c for s, c in cnt.items() if c / total >= min_support}
            frequent.update(new)
            prev = list(new)
            k += 1

        prior = Counter()
        for _, w, r in trans:
            prior[r] += w
        rules = []
        for A in frequent:
            if self._trivial(A):          # a full line of the same player is trivially a win
                continue
            res = Counter()
            for items, w, r in trans:
                if A <= items:
                    res[r] += w
            wa = sum(res.values())
            for r, wr in res.items():
                conf = wr / wa
                rules.append(dict(pattern=" + ".join(sorted(A)), result=r, support=wr / total,
                                  conf=conf, lift=conf / (prior[r] / total), games=wr, size=len(A)))
        return rules

    # -- step 4: strategy mining ---------------------------------------------
    def score(self, b, i):
        """Mover's (wins - losses) / games over every continuation after playing cell i."""
        p = to_move(b)
        x, o, d = self.stats[put(b, i, p)]
        w, l = (x, o) if p == "X" else (o, x)
        return (w - l) / (x + o + d)

    def best_value_moves(self, b):
        """Moves that keep the best possible result under perfect play (minimax)."""
        p = to_move(b)
        ms = empties(b)
        vals = {i: self.value[put(b, i, p)] for i in ms}
        target = max(vals.values()) if p == "X" else min(vals.values())
        return [i for i in ms if vals[i] == target]

    def candidates(self, b, level):
        ms = empties(b)
        if level == "Rookie":
            return ms
        pool = ms if level == "Miner" else self.best_value_moves(b)
        best = max(self.score(b, i) for i in pool)
        return [i for i in pool if abs(self.score(b, i) - best) < 1e-9]

    def choose(self, b, level):
        return random.choice(self.candidates(b, level))


# ===== UI =====
# ----------------------------------------------------------------------------
# 2. USER INTERFACE
# ----------------------------------------------------------------------------
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

BG = "#0d0a1f"
PANEL = "#181331"
PANEL2 = "#251c4d"
TEXT = "#f4f1ff"
MUTED = "#a79fd1"
PINK = "#ff4d8d"
CYAN = "#27e1ff"
YELLOW = "#ffd23f"
GREEN = "#5dff9a"
ORANGE = "#ff9f43"
PURPLE = "#b07cff"
GOLD = "#ffc933"
X_GLOW = "#7a1f45"
O_GLOW = "#0f6b7c"
RAINBOW = [PINK, ORANGE, YELLOW, GREEN, CYAN, PURPLE]
CELL_TINTS = ["#2b1650", "#14305a", "#3a1646", "#12423f", "#4a2a14", "#241a5e", "#4a1430", "#17394f", "#33204f"]
LEVEL_COL = {"Rookie": ORANGE, "Miner": PURPLE, "Master": GREEN}
HEAT_STOPS = [["#2a0f3d", "#8a1c6d", PINK, "#ffd1e3"],
              ["#0b1f45", "#106b9c", CYAN, "#d6faff"],
              ["#2a2a0b", "#9a7a0d", YELLOW, "#fff6c9"]]
OUTCOME_FILTER = {"All": None, "X wins": "X", "O wins": "O", "Draws": "D"}

FONT = "Segoe UI" if sys.platform.startswith("win") else "Helvetica"
CELL, GAP, PAD = 148, 8, 14           # play board
HM_CELL, HM_GAP, HM_PAD = 96, 8, 13   # heat-maps


def F(size=11, bold=False):
    return (FONT, size, "bold" if bold else "normal")


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(c1, c2, t):
    a, b = hex_to_rgb(c1), hex_to_rgb(c2)
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def lighten(c, t=0.25):
    return mix(c, "#ffffff", t)


def ramp(t, stops):
    t = max(0.0, min(1.0, t))
    n = len(stops) - 1
    k = min(int(t * n), n - 1)
    return mix(stops[k], stops[k + 1], t * n - k)


def button(parent, text, cmd, color, size=12):
    lb = tk.Label(parent, text=text, bg=color, fg=BG, font=F(size, True), padx=16, pady=8, cursor="hand2")
    lb.bind("<Button-1>", lambda e: cmd())
    lb.bind("<Enter>", lambda e: lb.config(bg=lighten(color)))
    lb.bind("<Leave>", lambda e: lb.config(bg=color))
    return lb


class Segmented(tk.Frame):
    """Colourful segmented control (replacement for radio buttons)."""

    def __init__(self, parent, options, command, initial=None, color=PINK, bg=PANEL):
        super().__init__(parent, bg=bg)
        self.command, self.color = command, color
        self.value = initial if initial is not None else options[0]
        self.labels = {}
        for opt in options:
            lb = tk.Label(self, text=opt, font=F(11, True), padx=13, pady=6, cursor="hand2")
            lb.pack(side="left", padx=2)
            lb.bind("<Button-1>", lambda e, o=opt: self.select(o))
            self.labels[opt] = lb
        self.refresh()

    def refresh(self):
        for o, lb in self.labels.items():
            lb.config(bg=self.color if o == self.value else PANEL2, fg=BG if o == self.value else TEXT)

    def select(self, o):
        self.value = o
        self.refresh()
        self.command(o)


def draw_piece(cv, ch, cx, cy, r, w=None):
    w = w or max(5, int(r / 3))
    if ch == "X":
        for col, ww in ((X_GLOW, w + 8), (PINK, w)):
            cv.create_line(cx - r, cy - r, cx + r, cy + r, width=ww, fill=col, capstyle="round")
            cv.create_line(cx - r, cy + r, cx + r, cy - r, width=ww, fill=col, capstyle="round")
    else:
        for col, ww in ((O_GLOW, w + 8), (CYAN, w)):
            cv.create_oval(cx - r, cy - r, cx + r, cy + r, width=ww, outline=col)


def draw_bars(cv, title, labels, values, colors, p=1.0):
    cv.delete("all")
    w, h = int(cv["width"]), int(cv["height"])
    cv.create_text(12, 16, text=title, anchor="w", fill=TEXT, font=F(12, True))
    left, right, top, bottom = 16, 16, 50, 30
    n = len(values)
    slot = (w - left - right) / n
    bw = slot * 0.62
    mx = max(values) or 1
    base = h - bottom
    cv.create_line(left, base, w - right, base, fill=MUTED)
    for k, (lab, v, col) in enumerate(zip(labels, values, colors)):
        x = left + k * slot + (slot - bw) / 2
        bh = (base - top) * (v / mx) * p
        cv.create_rectangle(x, base - bh, x + bw, base, fill=col, outline="")
        cv.create_rectangle(x, base - bh, x + bw, base - bh + min(6, bh), fill=lighten(col, 0.45), outline="")
        cv.create_text(x + bw / 2, base - bh - 11, text=f"{int(v * p):,}", fill=TEXT, font=F(10, True))
        cv.create_text(x + bw / 2, base + 14, text=lab, fill=MUTED, font=F(10))


def hm_cell_at(px, py):
    for i in range(9):
        r, c = divmod(i, 3)
        x0 = HM_PAD + c * (HM_CELL + HM_GAP)
        y0 = HM_PAD + r * (HM_CELL + HM_GAP)
        if x0 <= px <= x0 + HM_CELL and y0 <= py <= y0 + HM_CELL:
            return i
    return None


class App:
    def __init__(self, root, miner):
        self.root, self.m = root, miner
        self.board, self.human, self.level = EMPTY, "X", "Master"
        self.over, self.hover, self.hints = False, None, True
        self.gen, self.bot_busy = 0, False
        self.score = {"You": 0, "Draw": 0, "Bot": 0}
        self.metric, self.open_sel = 0, 4
        self.anim_gen = 0
        self.arena_res, self.arena_n = None, 100
        self.p_outcome, self.p_sort, self.shown_rules = None, "lift", []
        self.level_info = {
            "Rookie": "Plays completely at random - a warm-up opponent.",
            "Miner": f"Plays the move with the best win/loss record mined from all {miner.total:,} games. "
                     "Strong, but it can still be out-played.",
            "Master": "Mined statistics + minimax-verified safety. It never loses - the best you can do is draw.",
        }
        root.title("Tic-Tac-Toe Strategy Miner")
        root.configure(bg=BG)
        root.geometry("1260x830")
        root.minsize(1180, 760)
        self.setup_style()
        self.build_header()
        self.build_nav()
        self.container = tk.Frame(root, bg=BG)
        self.container.pack(fill="both", expand=True)
        self.pages = {}
        self.build_play()
        self.build_insights()
        self.build_patterns()
        self.build_arena()
        self.show("Play")
        self.new_game()

    # ------------------------------------------------------------------ chrome
    def setup_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                    rowheight=26, borderwidth=0, font=F(11))
        s.configure("Treeview.Heading", background=YELLOW, foreground=BG, font=F(11, True), relief="flat")
        s.map("Treeview", background=[("selected", PURPLE)], foreground=[("selected", TEXT)])
        s.map("Treeview.Heading", background=[("active", ORANGE)])
        s.configure("Vertical.TScrollbar", background=PURPLE, troughcolor=PANEL, bordercolor=PANEL, arrowcolor=TEXT)

    def build_header(self):
        cv = tk.Canvas(self.root, height=84, bg=BG, highlightthickness=0)
        cv.pack(fill="x")
        f = tkfont.Font(family=FONT, size=27, weight="bold")
        x = 24
        for k, ch in enumerate("TIC-TAC-TOE STRATEGY MINER"):
            cv.create_text(x, 34, text=ch, anchor="w", font=f, fill=RAINBOW[k % len(RAINBOW)])
            x += f.measure(ch)
        m = self.m
        cv.create_text(26, 66, anchor="w", fill=MUTED, font=F(11),
                       text=f"Data mining on {m.total:,} games   |   {len(m.stats):,} positions   |   "
                            f"{len(m.terminals)} final boards   |   pure Python, zero installs")

        def deco(e):
            cv.delete("deco")
            for k, ch in enumerate("XOX"):
                cx, cy = e.width - 190 + k * 58, 42
                if ch == "X":
                    cv.create_line(cx - 15, cy - 15, cx + 15, cy + 15, width=8, fill=PINK, capstyle="round", tags="deco")
                    cv.create_line(cx - 15, cy + 15, cx + 15, cy - 15, width=8, fill=PINK, capstyle="round", tags="deco")
                else:
                    cv.create_oval(cx - 16, cy - 16, cx + 16, cy + 16, width=8, outline=CYAN, tags="deco")
        cv.bind("<Configure>", deco)
        stripe = tk.Frame(self.root, bg=BG)
        stripe.pack(fill="x")
        for c in RAINBOW:
            tk.Frame(stripe, bg=c, height=4).pack(side="left", fill="x", expand=True)

    def build_nav(self):
        nav = tk.Frame(self.root, bg=BG)
        nav.pack(fill="x", padx=24, pady=(12, 0))
        self.tabs = {}
        for name, col in (("Play", PINK), ("Insights", CYAN), ("Patterns", YELLOW), ("Arena", GREEN)):
            lb = tk.Label(nav, text=name, font=F(13, True), bg=PANEL, fg=col, padx=22, pady=8, cursor="hand2")
            lb.pack(side="left", padx=(0, 8))
            lb.bind("<Button-1>", lambda e, n=name: self.show(n))
            self.tabs[name] = (lb, col)

    def show(self, name):
        for pg in self.pages.values():
            pg.pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        for n, (lb, col) in self.tabs.items():
            lb.config(bg=col if n == name else PANEL, fg=BG if n == name else col)
        if name == "Insights":
            self.draw_insights()
        elif name == "Arena":
            if self.arena_res is None:
                self.run_arena()
            else:
                self.animate(self.draw_arena)

    def animate(self, draw, frames=16, delay=16):
        self.anim_gen += 1
        g = self.anim_gen

        def step(k):
            if g != self.anim_gen:
                return
            t = k / frames
            draw(1 - (1 - t) ** 3)
            if k < frames:
                self.root.after(delay, lambda: step(k + 1))
        step(0)

    # -------------------------------------------------------------------- PLAY
    def build_play(self):
        page = tk.Frame(self.container, bg=BG)
        self.pages["Play"] = page
        left = tk.Frame(page, bg=BG)
        left.pack(side="left", padx=24, pady=14, anchor="n")
        size = 2 * PAD + 3 * CELL + 2 * GAP
        self.cv = tk.Canvas(left, width=size, height=size, bg=BG, highlightthickness=0)
        self.cv.pack()
        self.cv.bind("<Motion>", self.on_motion)
        self.cv.bind("<Leave>", self.on_leave)
        self.cv.bind("<Button-1>", self.on_click)

        right = tk.Frame(page, bg=PANEL)
        right.pack(side="left", fill="both", expand=True, padx=(0, 24), pady=14)
        box = tk.Frame(right, bg=PANEL)
        box.pack(fill="both", expand=True, padx=22, pady=14)

        def sec(text, color):
            tk.Label(box, text=text, bg=PANEL, fg=color, font=F(11, True)).pack(anchor="w", pady=(10, 4))

        sec("YOU PLAY AS", PINK)
        Segmented(box, ["X", "O"], self.on_side, "X", PINK).pack(anchor="w")
        sec("OPPONENT", PURPLE)
        Segmented(box, LEVELS, self.on_level, "Master", PURPLE).pack(anchor="w")
        self.level_lbl = tk.Label(box, text=self.level_info["Master"], bg=PANEL, fg=MUTED, font=F(10),
                                  wraplength=600, justify="left")
        self.level_lbl.pack(anchor="w", pady=4)
        row = tk.Frame(box, bg=PANEL)
        row.pack(anchor="w", pady=(8, 0))
        button(row, "New game", self.new_game, GREEN).pack(side="left", padx=(0, 10))
        self.hint_btn = button(row, "Hints: ON", self.toggle_hints, ORANGE)
        self.hint_btn.pack(side="left")

        sec("SCOREBOARD", YELLOW)
        sb = tk.Frame(box, bg=PANEL)
        sb.pack(fill="x")
        self.score_lbls = {}
        for key, col in (("You", GREEN), ("Draw", YELLOW), ("Bot", ORANGE)):
            f = tk.Frame(sb, bg=PANEL2)
            f.pack(side="left", expand=True, fill="x", padx=4, ipady=6)
            n = tk.Label(f, text="0", bg=PANEL2, fg=col, font=F(30, True))
            n.pack()
            tk.Label(f, text=key.upper(), bg=PANEL2, fg=MUTED, font=F(10, True)).pack()
            self.score_lbls[key] = n

        self.status = tk.Label(box, text="", bg=PANEL, fg=TEXT, font=F(18, True), anchor="w")
        self.status.pack(fill="x", pady=(16, 0))

        sec("POSITION INSIGHT  (mined from every possible continuation)", CYAN)
        self.ins_cv = tk.Canvas(box, width=560, height=30, bg=PANEL, highlightthickness=0)
        self.ins_cv.pack(anchor="w")
        lg = tk.Frame(box, bg=PANEL)
        lg.pack(anchor="w", pady=(4, 0))
        for t, c in (("[] X wins", PINK), ("[] Draw", YELLOW), ("[] O wins", CYAN)):
            tk.Label(lg, text=t.replace("[]", "\u25a0"), bg=PANEL, fg=c, font=F(10, True)).pack(side="left", padx=(0, 14))
        self.ins_lbl = tk.Label(box, text="", bg=PANEL, fg=TEXT, font=F(11), justify="left", anchor="w")
        self.ins_lbl.pack(anchor="w", pady=(6, 0))
        tk.Label(box, bg=PANEL, fg=MUTED, font=F(10), justify="left", wraplength=600,
                 text="Hints: the % on each square is your chance of winning after that move if both sides then play "
                      "at random.  \u2605 marks the best moves under perfect play.").pack(anchor="w", pady=(10, 0))

    def cell_box(self, i):
        r, c = divmod(i, 3)
        x0, y0 = PAD + c * (CELL + GAP), PAD + r * (CELL + GAP)
        return x0, y0, x0 + CELL, y0 + CELL

    def cell_at(self, px, py):
        for i in range(9):
            x0, y0, x1, y1 = self.cell_box(i)
            if x0 <= px <= x1 and y0 <= py <= y1:
                return i
        return None

    def can_human_move(self):
        return (not self.over) and (not self.bot_busy) and to_move(self.board) == self.human

    def draw_board(self, show_win=True):
        cv = self.cv
        cv.delete("all")
        hintable = self.hints and self.can_human_move()
        best = set(self.m.best_value_moves(self.board)) if hintable else set()
        for i in range(9):
            x0, y0, x1, y1 = self.cell_box(i)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            tint = CELL_TINTS[i]
            hot = (i == self.hover and self.board[i] == "." and self.can_human_move())
            cv.create_rectangle(x0, y0, x1, y1, fill=mix(tint, "#ffffff", 0.16) if hot else tint,
                                outline=mix(tint, "#ffffff", 0.35 if hot else 0.15), width=3 if hot else 2)
            ch = self.board[i]
            if ch != ".":
                draw_piece(cv, ch, cx, cy, 42, 11)
            elif hintable:
                xw, ow, dr = self.m.stats[put(self.board, i, self.human)]
                wins = xw if self.human == "X" else ow
                rate = wins / (xw + ow + dr)
                cv.create_text(cx, cy, text=f"{rate:.0%}", fill=ramp(rate, ["#ff5c7a", YELLOW, GREEN]), font=F(22, True))
                if i in best:
                    cv.create_text(x1 - 18, y0 + 18, text="\u2605", fill=GOLD, font=F(18, True))
        ln = win_line(self.board) if (self.over and show_win) else None
        if ln:
            a, b = self.cell_box(ln[0]), self.cell_box(ln[2])
            ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
            bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            cv.create_line(ax, ay, bx, by, width=30, fill=mix(GOLD, BG, 0.65), capstyle="round")
            cv.create_line(ax, ay, bx, by, width=12, fill=GOLD, capstyle="round")

    def on_motion(self, e):
        i = self.cell_at(e.x, e.y)
        if i != self.hover:
            self.hover = i
            self.draw_board()

    def on_leave(self, e):
        self.hover = None
        self.draw_board()

    def on_click(self, e):
        if not self.can_human_move():
            return
        i = self.cell_at(e.x, e.y)
        if i is None or self.board[i] != ".":
            return
        self.apply_move(i)

    def apply_move(self, i):
        p = to_move(self.board)
        self.board = put(self.board, i, p)
        if is_over(self.board):
            self.finish()
        elif to_move(self.board) != self.human:
            self.schedule_bot()
        self.refresh_play()
        if self.over and winner(self.board):
            self.flash(8, self.gen)

    def schedule_bot(self):
        self.bot_busy = True
        g = self.gen
        self.root.after(550, lambda: self.bot_move(g))

    def bot_move(self, g):
        if g != self.gen or self.over:
            return
        i = self.m.choose(self.board, self.level)
        self.bot_busy = False
        self.apply_move(i)

    def finish(self):
        self.over = True
        self.bot_busy = False
        w = winner(self.board)
        if w == self.human:
            self.score["You"] += 1
        elif w:
            self.score["Bot"] += 1
        else:
            self.score["Draw"] += 1

    def flash(self, n, gen):
        if gen != self.gen:
            return
        self.draw_board(show_win=(n % 2 == 0))
        if n > 0:
            self.root.after(170, lambda: self.flash(n - 1, gen))

    def new_game(self):
        self.gen += 1
        self.board, self.over, self.bot_busy, self.hover = EMPTY, False, False, None
        if to_move(self.board) != self.human:
            self.schedule_bot()
        self.refresh_play()

    def refresh_play(self):
        self.draw_board()
        self.update_insight()
        self.update_status()
        for k, lb in self.score_lbls.items():
            lb.config(text=str(self.score[k]))

    def update_status(self):
        if self.over:
            w = winner(self.board)
            if w == self.human:
                txt, col = "You win! Nice strategy.", GREEN
            elif w:
                txt, col = "Bot wins - try again!", ORANGE
            else:
                txt, col = "It's a draw.", YELLOW
        elif to_move(self.board) == self.human:
            txt, col = f"Your move - you are {self.human}", PINK if self.human == "X" else CYAN
        else:
            txt, col = "Bot is thinking...", PURPLE
        self.status.config(text=txt, fg=col)

    def update_insight(self):
        cv = self.ins_cv
        cv.delete("all")
        b = self.board
        x, o, d = self.m.stats[b]
        t = x + o + d
        W, H = int(cv["width"]), int(cv["height"])
        pos = 0
        for n, col in ((x, PINK), (d, YELLOW), (o, CYAN)):
            seg = W * n / t
            if seg > 0:
                cv.create_rectangle(pos, 0, pos + seg, H, fill=col, outline="")
                if seg > 46:
                    cv.create_text(pos + seg / 2, H / 2, text=f"{n / t:.0%}", fill=BG, font=F(11, True))
            pos += seg
        if is_over(b):
            verdict = "Game over"
        else:
            verdict = {1: "Perfect play from here: X wins", 0: "Perfect play from here: a draw",
                       -1: "Perfect play from here: O wins"}[self.m.value[b]]
        self.ins_lbl.config(text=f"{t:,} possible game(s) continue from this position\n{verdict}")

    def on_side(self, v):
        self.human = v
        self.new_game()

    def on_level(self, v):
        self.level = v
        self.level_lbl.config(text=self.level_info[v])
        self.new_game()

    def toggle_hints(self):
        self.hints = not self.hints
        self.hint_btn.config(text="Hints: ON" if self.hints else "Hints: OFF")
        self.draw_board()

    # ---------------------------------------------------------------- INSIGHTS
    def build_insights(self):
        page = tk.Frame(self.container, bg=BG)
        self.pages["Insights"] = page
        top = tk.Frame(page, bg=BG)
        top.pack(fill="x", padx=24, pady=(14, 6))
        tk.Label(top, text="Show chance of", bg=BG, fg=MUTED, font=F(12, True)).pack(side="left", padx=(0, 8))
        Segmented(top, ["X wins", "O wins", "Draws"], self.set_metric, "X wins", CYAN, BG).pack(side="left")
        tk.Label(top, text="   when the rest of the game is played at random", bg=BG, fg=MUTED,
                 font=F(11)).pack(side="left")
        body = tk.Frame(page, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=6)
        hm = 2 * HM_PAD + 3 * HM_CELL + 2 * HM_GAP

        c1 = tk.Frame(body, bg=PANEL)
        c1.pack(side="left", anchor="n", padx=(0, 14))
        tk.Label(c1, text="X's opening move  (click a square)", bg=PANEL, fg=PINK, font=F(12, True)).pack(padx=12, pady=(10, 4), anchor="w")
        self.hm1 = tk.Canvas(c1, width=hm, height=hm, bg=PANEL, highlightthickness=0)
        self.hm1.pack(padx=12)
        self.hm1.bind("<Button-1>", self.on_hm1_click)
        tk.Label(c1, bg=PANEL, fg=MUTED, font=F(10), wraplength=hm, justify="left",
                 text="With perfect play every opening is a draw, but the colours show how often each "
                      "opening wins when players are not perfect.").pack(padx=12, pady=(6, 12), anchor="w")

        c2 = tk.Frame(body, bg=PANEL)
        c2.pack(side="left", anchor="n", padx=(0, 14))
        self.hm2_title = tk.Label(c2, text="", bg=PANEL, fg=CYAN, font=F(12, True))
        self.hm2_title.pack(padx=12, pady=(10, 4), anchor="w")
        self.hm2 = tk.Canvas(c2, width=hm, height=hm, bg=PANEL, highlightthickness=0)
        self.hm2.pack(padx=12)
        tk.Label(c2, bg=PANEL, fg=MUTED, font=F(10), wraplength=hm, justify="left",
                 text="\u2605 = replies that keep O safe under perfect play. Every other reply hands X a forced win.").pack(
            padx=12, pady=(6, 12), anchor="w")

        c3 = tk.Frame(body, bg=PANEL)
        c3.pack(side="left", anchor="n")
        self.bars1 = tk.Canvas(c3, width=390, height=200, bg=PANEL, highlightthickness=0)
        self.bars1.pack(padx=8, pady=(6, 0))
        self.bars2 = tk.Canvas(c3, width=390, height=200, bg=PANEL, highlightthickness=0)
        self.bars2.pack(padx=8, pady=(0, 6))

    def set_metric(self, v):
        self.metric = {"X wins": 0, "O wins": 1, "Draws": 2}[v]
        self.animate(lambda p: (self.draw_hm1(p), self.draw_hm2(p)))

    def on_hm1_click(self, e):
        i = hm_cell_at(e.x, e.y)
        if i is not None:
            self.open_sel = i
            self.animate(lambda p: (self.draw_hm1(p), self.draw_hm2(p)))

    def rate(self, b):
        s = self.m.stats[b]
        return s[self.metric] / sum(s), s[self.metric]

    def stars_for(self, b):
        best = self.m.best_value_moves(b)
        return set(best) if len(best) < len(empties(b)) else set()

    def draw_hm1(self, p=1.0):
        items = []
        for i in range(9):
            r, cnt = self.rate(put(EMPTY, i, "X"))
            items.append(("val", r, cnt))
        self.draw_heat(self.hm1, items, self.stars_for(EMPTY), self.open_sel, p)

    def draw_hm2(self, p=1.0):
        s = self.open_sel
        b1 = put(EMPTY, s, "X")
        items = []
        for j in range(9):
            if j == s:
                items.append(("piece", "X"))
            else:
                r, cnt = self.rate(put(b1, j, "O"))
                items.append(("val", r, cnt))
        self.draw_heat(self.hm2, items, self.stars_for(b1), None, p)
        self.hm2_title.config(text=f"O's reply after X opens {CELL_NAMES[s]}")

    def draw_heat(self, cv, items, stars, selected, p):
        cv.delete("all")
        vals = [it[1] for it in items if it[0] == "val"]
        lo, hi = min(vals), max(vals)
        stops = HEAT_STOPS[self.metric]
        for i, it in enumerate(items):
            r, c = divmod(i, 3)
            x0 = HM_PAD + c * (HM_CELL + HM_GAP)
            y0 = HM_PAD + r * (HM_CELL + HM_GAP)
            x1, y1 = x0 + HM_CELL, y0 + HM_CELL
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if it[0] == "piece":
                cv.create_rectangle(x0, y0, x1, y1, fill=PANEL2, outline="")
                draw_piece(cv, it[1], cx, cy, 26)
                continue
            _, rate, cnt = it
            t = (rate - lo) / (hi - lo) if hi > lo else 0.5
            col = mix(PANEL2, ramp(t, stops), p)
            sel = (i == selected)
            cv.create_rectangle(x0, y0, x1, y1, fill=col, outline=GOLD if sel else col, width=4 if sel else 1)
            fg = BG if t > 0.55 else TEXT
            if p > 0.5:
                cv.create_text(cx, cy - 8, text=f"{rate:.1%}", fill=fg, font=F(16, True))
                cv.create_text(cx, cy + 15, text=f"{cnt:,} games", fill=fg, font=F(9))
                if i in stars:
                    cv.create_text(x1 - 14, y0 + 14, text="\u2605", fill=fg, font=F(14, True))

    def draw_insights(self):
        m = self.m

        def frame(p):
            self.draw_hm1(p)
            self.draw_hm2(p)
            draw_bars(self.bars1, "Outcome of all games", ["X wins", "O wins", "Draws"],
                      list(m.stats[EMPTY]), [PINK, CYAN, YELLOW], p)
            draw_bars(self.bars2, "Game length (moves played)", [str(k) for k in range(5, 10)],
                      [m.length[k] for k in range(5, 10)],
                      [ramp(k / 4, [YELLOW, ORANGE, PINK, PURPLE]) for k in range(5)], p)
        self.animate(frame)

    # ---------------------------------------------------------------- PATTERNS
    def build_patterns(self):
        page = tk.Frame(self.container, bg=BG)
        self.pages["Patterns"] = page
        head = tk.Frame(page, bg=BG)
        head.pack(fill="x", padx=24, pady=(14, 6))

        def lab(t):
            tk.Label(head, text=t, bg=BG, fg=MUTED, font=F(12, True)).pack(side="left", padx=(0, 8))

        lab("Outcome")
        Segmented(head, list(OUTCOME_FILTER), self.on_outcome, "All", YELLOW, BG).pack(side="left", padx=(0, 22))
        lab("Min confidence")
        self.conf = tk.Scale(head, from_=0.4, to=1.0, resolution=0.05, orient="horizontal", length=160,
                             bg=BG, fg=TEXT, troughcolor=PANEL2, highlightthickness=0, bd=0,
                             activebackground=YELLOW, command=lambda v: self.refresh_rules())
        self.conf.pack(side="left", padx=(0, 22))
        lab("Sort by")
        Segmented(head, ["Lift", "Confidence", "Support"], self.on_sort, "Lift", ORANGE, BG).pack(side="left")

        card = tk.Frame(page, bg=PANEL)
        card.pack(fill="x", padx=24, pady=6)
        tk.Label(card, text="HEADLINE FINDINGS  (Apriori on the 958 final boards)", bg=PANEL, fg=YELLOW,
                 font=F(11, True)).pack(anchor="w", padx=16, pady=(10, 2))
        for r, col in (("X", PINK), ("O", CYAN), ("D", YELLOW)):
            txt = self.finding(r)
            if txt:
                tk.Label(card, text=txt, bg=PANEL, fg=col, font=F(12), anchor="w").pack(anchor="w", padx=16)
        tk.Frame(card, bg=PANEL, height=8).pack()

        foot = tk.Frame(page, bg=BG)
        foot.pack(side="bottom", fill="x", padx=24, pady=(2, 12))
        tk.Label(foot, bg=BG, fg=MUTED, font=F(10), justify="left",
                 text="Support = share of all games ending with pattern + outcome   |   Confidence = P(outcome | pattern)\n"
                      "Lift > 1 means the pattern makes that outcome more likely than average.").pack(side="left")
        button(foot, "Export strategy table", self.export_strategy, PURPLE, 11).pack(side="right", padx=(8, 0))
        button(foot, "Export final boards", self.export_boards, CYAN, 11).pack(side="right", padx=(8, 0))
        button(foot, "Export rules", self.export_rules, GREEN, 11).pack(side="right", padx=(8, 0))

        wrap = tk.Frame(page, bg=PANEL)
        wrap.pack(fill="both", expand=True, padx=24, pady=6)
        cols = ("pattern", "outcome", "support", "conf", "lift", "games")
        self.tree = ttk.Treeview(wrap, columns=cols, show="headings", selectmode="browse")
        for c, t, w, a in (("pattern", "Pattern on the final board", 470, "w"), ("outcome", "Outcome", 100, "center"),
                           ("support", "Support", 100, "center"), ("conf", "Confidence", 110, "center"),
                           ("lift", "Lift", 90, "center"), ("games", "Games", 120, "e")):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor=a)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        sb.pack(side="right", fill="y", pady=8, padx=(0, 8))
        for r, col in (("X", PINK), ("O", CYAN), ("D", YELLOW)):
            self.tree.tag_configure(r, foreground=col)
        self.conf.set(0.7)

    def finding(self, r):
        c = [x for x in self.m.rules if x["result"] == r and x["support"] >= 0.02 and x["conf"] >= 0.5]
        if not c:
            return ""
        b = max(c, key=lambda x: (x["lift"], x["support"]))
        return (f"{RESULT_NAME[r]}:  {b['pattern']}   ->   confidence {b['conf']:.0%},  "
                f"lift {b['lift']:.2f},  support {b['support']:.1%}")

    def on_outcome(self, v):
        self.p_outcome = OUTCOME_FILTER[v]
        self.refresh_rules()

    def on_sort(self, v):
        self.p_sort = {"Lift": "lift", "Confidence": "conf", "Support": "support"}[v]
        self.refresh_rules()

    def refresh_rules(self):
        if not hasattr(self, "tree"):
            return
        mc = float(self.conf.get())
        rows = [r for r in self.m.rules
                if (self.p_outcome is None or r["result"] == self.p_outcome) and r["conf"] >= mc - 1e-9]
        rows.sort(key=lambda r: (r[self.p_sort], r["support"]), reverse=True)
        self.shown_rules = rows
        self.tree.delete(*self.tree.get_children())
        for r in rows[:300]:
            self.tree.insert("", "end", tags=(r["result"],),
                             values=(r["pattern"], RESULT_NAME[r["result"]], f"{r['support']:.1%}",
                                     f"{r['conf']:.1%}", f"{r['lift']:.2f}", f"{r['games']:,}"))

    def _save(self, name):
        return filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile=name)

    def export_rules(self):
        path = self._save("ttt_rules.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pattern", "outcome", "support", "confidence", "lift", "games"])
            for r in self.shown_rules:
                w.writerow([r["pattern"], RESULT_NAME[r["result"]], round(r["support"], 5),
                            round(r["conf"], 5), round(r["lift"], 4), r["games"]])
        messagebox.showinfo("Exported", f"Saved {len(self.shown_rules)} rules to\n{path}")

    def export_boards(self):
        path = self._save("ttt_final_boards.csv")
        if not path:
            return
        sym = {"X": "x", "O": "o", ".": "b"}
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CELL_NAMES + ["result", "games"])
            for b, (g, res) in sorted(self.m.terminals.items()):
                w.writerow([sym[c] for c in b] + [RESULT_NAME[res], g])
        messagebox.showinfo("Exported", f"Saved {len(self.m.terminals)} final boards to\n{path}")

    def export_strategy(self):
        path = self._save("ttt_strategy_table.csv")
        if not path:
            return
        n = 0
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["board", "to_move", "x_win_games", "o_win_games", "draw_games",
                        "perfect_play_value", "miner_move", "master_move"])
            for b, (x, o, d) in sorted(self.m.stats.items()):
                if is_over(b):
                    continue
                names = lambda lv: "/".join(CELL_NAMES[i] for i in self.m.candidates(b, lv))
                w.writerow([b, to_move(b), x, o, d, {1: "X wins", 0: "draw", -1: "O wins"}[self.m.value[b]],
                            names("Miner"), names("Master")])
                n += 1
        messagebox.showinfo("Exported", f"Saved {n} positions to\n{path}")

    # ------------------------------------------------------------------- ARENA
    def build_arena(self):
        page = tk.Frame(self.container, bg=BG)
        self.pages["Arena"] = page
        head = tk.Frame(page, bg=BG)
        head.pack(fill="x", padx=24, pady=(14, 6))
        tk.Label(head, text="Games per matchup", bg=BG, fg=MUTED, font=F(12, True)).pack(side="left", padx=(0, 8))
        Segmented(head, ["100", "300", "1000"], lambda v: setattr(self, "arena_n", int(v)), "100", GREEN, BG).pack(side="left", padx=(0, 18))
        button(head, "Run tournament", self.run_arena, GREEN).pack(side="left")
        tk.Label(head, bg=BG, fg=MUTED, font=F(10), text="   Rookie = random,  Miner = mined win-rate strategy,  "
                 "Master = mined + minimax-verified").pack(side="left")
        self.arena_cv = tk.Canvas(page, width=1160, height=545, bg=PANEL, highlightthickness=0)
        self.arena_cv.pack(padx=24, pady=8, anchor="w")

    def play_game(self, xl, ol):
        b = EMPTY
        while not is_over(b):
            p = to_move(b)
            b = put(b, self.m.choose(b, xl if p == "X" else ol), p)
        return winner(b) or "D"

    def run_arena(self):
        n = self.arena_n
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        res = {}
        for xl in LEVELS:
            for ol in LEVELS:
                c = Counter(self.play_game(xl, ol) for _ in range(n))
                res[(xl, ol)] = (c["X"], c["O"], c["D"], n)
        self.arena_res = res
        self.root.config(cursor="")
        self.animate(self.draw_arena)

    def overall(self, lv):
        pts = []
        for other in LEVELS:
            xw, ow, d, n = self.arena_res[(lv, other)]
            pts.append((xw + 0.5 * d) / n)
            xw, ow, d, n = self.arena_res[(other, lv)]
            pts.append((ow + 0.5 * d) / n)
        return sum(pts) / len(pts)

    def draw_arena(self, p=1.0):
        cv = self.arena_cv
        cv.delete("all")
        if not self.arena_res:
            return
        W = int(cv["width"])
        LEFT, TOP, CW, CH, G = 150, 70, 326, 100, 8
        cv.create_text(14, 22, anchor="w", fill=MUTED, font=F(11),
                       text="X (rows) vs O (columns) - bars show how often X wins / draws / O wins")
        for c, ol in enumerate(LEVELS):
            cv.create_text(LEFT + c * (CW + G) + CW / 2, TOP - 16, text=f"O: {ol}", fill=LEVEL_COL[ol], font=F(13, True))
        for r, xl in enumerate(LEVELS):
            cv.create_text(LEFT - 12, TOP + r * (CH + G) + CH / 2, anchor="e", text=f"X: {xl}",
                           fill=LEVEL_COL[xl], font=F(13, True))
            for c, ol in enumerate(LEVELS):
                xw, ow, d, n = self.arena_res[(xl, ol)]
                x0, y0 = LEFT + c * (CW + G), TOP + r * (CH + G)
                cv.create_rectangle(x0, y0, x0 + CW, y0 + CH, fill=PANEL2, outline="")
                bx, by, bw = x0 + 16, y0 + 24, CW - 32
                cv.create_rectangle(bx, by, bx + bw, by + 28, fill=BG, outline="")
                pos = bx
                for cnt, col in ((xw, PINK), (d, YELLOW), (ow, CYAN)):
                    seg = bw * cnt / n * p
                    if seg > 0:
                        cv.create_rectangle(pos, by, pos + seg, by + 28, fill=col, outline="")
                    pos += seg
                cv.create_text(bx, y0 + 72, anchor="w", text=f"X {xw / n:.0%}", fill=PINK, font=F(11, True))
                cv.create_text(bx + bw / 2, y0 + 72, text=f"Draw {d / n:.0%}", fill=YELLOW, font=F(11, True))
                cv.create_text(bx + bw, y0 + 72, anchor="e", text=f"O {ow / n:.0%}", fill=CYAN, font=F(11, True))
        base = TOP + 3 * (CH + G) + 18
        cv.create_text(14, base, anchor="w", fill=TEXT, font=F(12, True),
                       text="Overall score  (win = 1 point, draw = 1/2, averaged over playing both X and O)")
        for k, lv in enumerate(LEVELS):
            sc = self.overall(lv)
            y = base + 24 + k * 30
            cv.create_text(LEFT - 12, y + 10, anchor="e", text=lv, fill=LEVEL_COL[lv], font=F(12, True))
            bw = (W - LEFT - 110) * sc * p
            cv.create_rectangle(LEFT, y, LEFT + bw, y + 20, fill=LEVEL_COL[lv], outline="")
            cv.create_text(LEFT + bw + 8, y + 10, anchor="w", text=f"{sc * p:.2f}", fill=TEXT, font=F(11, True))


def main():
    miner = Miner()
    root = tk.Tk()
    App(root, miner)
    root.mainloop()


if __name__ == "__main__":
    main()
