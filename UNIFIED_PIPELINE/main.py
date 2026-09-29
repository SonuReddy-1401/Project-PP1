#!/usr/bin/env python3
"""
Interactive Desktop GUI Launcher for Unified Pro Football Analytics Pipeline
Redesigned with high-contrast obsidian styling, crisp input cursors, and UTF-8 logging.
"""
import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Reconfigure stdout/stderr to UTF-8 on Windows to eliminate charmap codec errors
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))

class ProPipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pro Tactical Football Analytics — Setup & Ingestion Center")
        self.root.geometry("840x820")
        self.root.minsize(760, 650)
        self.root.configure(bg="#0A0D14")
        self.root.resizable(True, True)

        self.selected_file_path = tk.StringVar(value="")
        self.team_name_var = tk.StringVar(value="")
        self.opponent_name_var = tk.StringVar(value="")
        self.team_color_var = tk.StringVar(value="sky_blue")
        self.attacking_dir_var = tk.StringVar(value="left_to_right")
        self.match_start_var = tk.StringVar(value="00:00")
        self.port_var = tk.StringVar(value="8090")

        self.setup_styles()
        self.build_ui()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Base obsidian color palette
        BG_APP = "#0A0D14"
        BG_CARD = "#111622"
        BG_INPUT = "#1E293B"
        TEXT_PRIMARY = "#F8FAFC"
        TEXT_SECONDARY = "#94A3B8"
        ACCENT_AZURE = "#0080FF"

        self.style.configure(".", background=BG_APP, foreground=TEXT_PRIMARY)
        self.style.configure("TFrame", background=BG_APP)
        self.style.configure("Card.TFrame", background=BG_CARD, relief="flat", borderwidth=0)

        self.style.configure("Header.TLabel", font=("Outfit", 18, "bold"), foreground=TEXT_PRIMARY, background=BG_APP)
        self.style.configure("SubHeader.TLabel", font=("Inter", 10), foreground=TEXT_SECONDARY, background=BG_APP)
        self.style.configure("CardTitle.TLabel", font=("Outfit", 11, "bold"), foreground=ACCENT_AZURE, background=BG_CARD)
        self.style.configure("FormLabel.TLabel", font=("Inter", 10, "bold"), foreground="#CBD5E1", background=BG_CARD)

        # Combobox styling with high contrast text
        self.root.option_add('*TCombobox*Listbox.font', ('Inter', 10))
        self.root.option_add('*TCombobox*Listbox.background', '#1E293B')
        self.root.option_add('*TCombobox*Listbox.foreground', '#F8FAFC')
        self.root.option_add('*TCombobox*Listbox.selectBackground', '#0080FF')
        self.root.option_add('*TCombobox*Listbox.selectForeground', '#FFFFFF')

        self.style.configure("TCombobox", fieldbackground=BG_INPUT, background=BG_INPUT, foreground=TEXT_PRIMARY, borderwidth=0)
        self.style.map("TCombobox", fieldbackground=[("readonly", BG_INPUT)], foreground=[("readonly", TEXT_PRIMARY)])

        # Action Buttons
        self.style.configure("Action.TButton", font=("Outfit", 12, "bold"), background=ACCENT_AZURE, foreground="#FFFFFF", borderwidth=0)
        self.style.map("Action.TButton", background=[("active", "#0066CC"), ("pressed", "#0052A3")])

        self.style.configure("Browse.TButton", font=("Inter", 9, "bold"), background="#334155", foreground="#FFFFFF", borderwidth=0)
        self.style.map("Browse.TButton", background=[("active", "#475569")])

    def create_input_entry(self, parent, variable, width=None):
        """Creates a high-contrast Entry widget with visible cyan cursor and bright white text."""
        entry = tk.Entry(
            parent,
            textvariable=variable,
            font=("Inter", 10, "bold"),
            bg="#1E293B",
            fg="#FFFFFF",
            insertbackground="#38BDF8",  # Glowing cyan insertion cursor
            selectbackground="#0080FF",
            selectforeground="#FFFFFF",
            relief="flat",
            bd=6,
            highlightthickness=1,
            highlightbackground="#334155",
            highlightcolor="#0080FF"
        )
        if width:
            entry.config(width=width)
        return entry

    def build_ui(self):
        # Header Banner
        header_frame = ttk.Frame(self.root, padding="24 20 24 10")
        header_frame.pack(fill="x")

        lbl_title = ttk.Label(header_frame, text="PRO TACTICAL FOOTBALL ANALYTICS", style="Header.TLabel")
        lbl_title.pack(anchor="w")

        lbl_sub = ttk.Label(
            header_frame,
            text="Flexible Broadcast Video & Position Ingestion Command Center",
            style="SubHeader.TLabel"
        )
        lbl_sub.pack(anchor="w", pady=(3, 0))

        # Main Form Container Card
        form_card = ttk.Frame(self.root, style="Card.TFrame", padding="20")
        form_card.pack(fill="x", padx=24, pady=10)

        # Section 1: File Explorer Picker
        lbl_sec1 = ttk.Label(form_card, text="1. MATCH VIDEO / DATASET FILE", style="CardTitle.TLabel")
        lbl_sec1.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        lbl_file = ttk.Label(form_card, text="Input Path:", style="FormLabel.TLabel")
        lbl_file.grid(row=1, column=0, sticky="w", pady=6)

        entry_file = self.create_input_entry(form_card, self.selected_file_path)
        entry_file.grid(row=1, column=1, sticky="ew", padx=(10, 8), pady=6)

        btn_browse = ttk.Button(form_card, text="Browse File...", style="Browse.TButton", command=self.browse_file)
        btn_browse.grid(row=1, column=2, sticky="e", pady=6)

        form_card.columnconfigure(1, weight=1)

        # Divider
        ttk.Separator(form_card, orient="horizontal").grid(row=2, column=0, columnspan=3, sticky="ew", pady=14)

        # Section 2: Match & Team Details
        lbl_sec2 = ttk.Label(form_card, text="2. MATCH METADATA & KIT PALETTE", style="CardTitle.TLabel")
        lbl_sec2.grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 10))

        # Team Name
        lbl_team = ttk.Label(form_card, text="Target Team Name:", style="FormLabel.TLabel")
        lbl_team.grid(row=4, column=0, sticky="w", pady=6)
        entry_team = self.create_input_entry(form_card, self.team_name_var)
        entry_team.grid(row=4, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=6)

        # Opponent Name
        lbl_opp = ttk.Label(form_card, text="Opponent Name:", style="FormLabel.TLabel")
        lbl_opp.grid(row=5, column=0, sticky="w", pady=6)
        entry_opp = self.create_input_entry(form_card, self.opponent_name_var)
        entry_opp.grid(row=5, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=6)

        # Kit Color Theme Dropdown
        lbl_color = ttk.Label(form_card, text="Target Kit Color:", style="FormLabel.TLabel")
        lbl_color.grid(row=6, column=0, sticky="w", pady=6)

        color_combo = ttk.Combobox(
            form_card,
            textvariable=self.team_color_var,
            values=["sky_blue", "red", "white", "yellow", "emerald", "navy"],
            state="normal",
            font=("Inter", 10, "bold")
        )
        color_combo.grid(row=6, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=6)

        # Attacking Direction Dropdown
        lbl_atk = ttk.Label(form_card, text="Attacking Direction:", style="FormLabel.TLabel")
        lbl_atk.grid(row=7, column=0, sticky="w", pady=6)

        atk_combo = ttk.Combobox(
            form_card,
            textvariable=self.attacking_dir_var,
            values=["left_to_right", "right_to_left"],
            state="readonly",
            font=("Inter", 10, "bold")
        )
        atk_combo.grid(row=7, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=6)

        # Match Start Clock Offset
        lbl_clock = ttk.Label(form_card, text="Match Clock Start (MM:SS):", style="FormLabel.TLabel")
        lbl_clock.grid(row=8, column=0, sticky="w", pady=6)
        entry_clock = self.create_input_entry(form_card, self.match_start_var)
        entry_clock.grid(row=8, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=6)

        # Dashboard Port
        lbl_port = ttk.Label(form_card, text="Dashboard HTTP Port:", style="FormLabel.TLabel")
        lbl_port.grid(row=9, column=0, sticky="w", pady=6)
        entry_port = self.create_input_entry(form_card, self.port_var)
        entry_port.grid(row=9, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=6)

        # Section 3: Execution Output Console
        console_card = ttk.Frame(self.root, style="Card.TFrame", padding="16")
        console_card.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        lbl_console = ttk.Label(console_card, text="PIPELINE LOG CONSOLE", style="CardTitle.TLabel")
        lbl_console.pack(anchor="w", pady=(0, 6))

        self.txt_console = tk.Text(
            console_card,
            height=6,
            bg="#0A0D14",
            fg="#34D399",  # Mint green visible console log
            insertbackground="#FFFFFF",
            font=("Consolas", 9, "bold"),
            relief="flat",
            bd=0
        )
        self.txt_console.pack(fill="both", expand=True)

        # Action Button Footer
        footer_frame = ttk.Frame(self.root, padding="24 0 24 20")
        footer_frame.pack(fill="x")

        self.btn_run = ttk.Button(
            footer_frame,
            text="START TACTICAL ANALYSIS & LAUNCH DASHBOARD",
            style="Action.TButton",
            command=self.run_pipeline_thread
        )
        self.btn_run.pack(fill="x", ipady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Match Video or Position Dataset",
            filetypes=[
                ("Supported Match Files (*.mp4, *.avi, *.mov, *.json)", "*.mp4 *.avi *.mov *.mkv *.json"),
                ("Video Files (*.mp4, *.avi, *.mov, *.mkv)", "*.mp4 *.avi *.mov *.mkv"),
                ("JSON Dataset (*.json)", "*.json"),
                ("All Files (*.*)", "*.*")
            ]
        )
        if file_path:
            self.selected_file_path.set(file_path)
            self.log(f"Selected input file: {file_path}")

    def log(self, text):
        self.txt_console.insert(tk.END, text + "\n")
        self.txt_console.see(tk.END)

    def run_pipeline_thread(self):
        t = threading.Thread(target=self.execute_pipeline, daemon=True)
        t.start()

    def execute_pipeline(self):
        file_path = self.selected_file_path.get().strip()
        team_name = self.team_name_var.get().strip()
        opponent_name = self.opponent_name_var.get().strip()
        team_color = self.team_color_var.get().strip()
        attacking_dir = self.attacking_dir_var.get().strip()
        match_start = self.match_start_var.get().strip()
        port = self.port_var.get().strip()

        self.log("==========================================")
        self.log("Initiating Pro Tactical Pipeline Execution...")
        self.log(f"   * Target Squad: {team_name}")
        self.log(f"   * Opponent Squad: {opponent_name}")
        self.log(f"   * Kit Color Theme: {team_color}")
        self.log(f"   * Attacking Direction: {attacking_dir}")
        self.log(f"   * Match Start Clock: {match_start}")
        self.log(f"   * Dashboard Port: {port}")
        if file_path:
            self.log(f"   * Input File: {file_path}")
        else:
            self.log("   * Input File: Default positions dataset")
        self.log("==========================================")

        py_executable = sys.executable
        venv_py = os.path.abspath(os.path.join(current_dir, "..", ".venv", "Scripts", "python.exe"))
        if os.path.exists(venv_py):
            py_executable = venv_py

        cmd = [
            py_executable,
            os.path.join(current_dir, "run_pipeline.py"),
            "--team_name", team_name,
            "--opponent_name", opponent_name,
            "--team_color", team_color,
            "--attacking_dir", attacking_dir,
            "--match_start", match_start,
            "--port", port
        ]

        if file_path:
            if file_path.endswith(".json"):
                cmd.extend(["--dataset", file_path])
            else:
                cmd.extend(["--video", file_path])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                cwd=current_dir
            )

            for line in proc.stdout:
                self.log(line.strip())

        except Exception as e:
            self.log(f"Execution Error: {str(e)}")

def main():
    root = tk.Tk()
    app = ProPipelineGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
