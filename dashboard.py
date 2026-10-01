"""
Módulo do Dashboard de Análise de Treino e Simulação de Plumas de Odor.

Fornece uma interface gráfica em Tkinter (estilo claro de engenharia)
para visualização de métricas de aprendizagem (Q-Learning), inspeção
interativa de trajetórias episódio a episódio, exploração da Q-Table
e edição direta dos parâmetros de simulação em config.json.
"""

import os
import glob
import pickle
import json
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class DashboardApp:
    """
    Aplicação gráfica principal do Dashboard de controlo e análise.
    Integra gráficos Matplotlib dinâmicos, tabelas de exploração e
    um gestor visual de rotas gravadas.
    """

    def __init__(self, root):
        """
        Inicializa a janela principal do Dashboard, os estados internos e o tema visual.
        """
        self.root = root
        self.root.title("Dashboard Q-Learning - Análise de Pluma")
        self.root.geometry("1100x880")
        self.root.minsize(980, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.fechar_aplicacao)

        self.pasta_modelos = "modelos"
        self.dados_modelo_atual = None
        self.chaves_episodios_gravados = []
        self.indice_episodio_atual = 0  # 0 corresponde à Visão Global de todos os episódios

        # Contentores para integração dos gráficos do Matplotlib no Tkinter
        self.canvas_metricas = None
        self.canvas_mapa = None

        # Configuração do esquema de cores claro e profissional
        self.configurar_tema()

        # Construção da arquitetura de janelas e separadores
        self.construir_layout_principal()

        # Leitura inicial dos modelos .pkl disponíveis na pasta de modelos
        self.listar_modelos()

    def fechar_aplicacao(self):
        """Liberta recursos do Matplotlib e fecha ordenadamente a aplicação."""
        plt.close("all")
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    # =========================================================================
    # TEMA E ESTILIZAÇÃO VISUAL CLARA
    # =========================================================================
    def configurar_tema(self):
        """
        Configura o tema claro ('clam') e define a paleta de cores para os controlos TTK,
        garantindo elevado contraste e legibilidade em todas as caixas de texto e seletores.
        """
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Paleta de Cores
        self.BG_MAIN = "#f1f5f9"        # Fundo geral cinzento suave
        self.BG_CARD = "#ffffff"        # Fundo de cartões, painéis e abas
        self.BORDER_COLOR = "#cbd5e1"   # Borda cinzenta suave
        self.TEXT_PRIMARY = "#0f172a"   # Texto escuro nítido
        self.TEXT_MUTED = "#64748b"     # Texto secundário ou de apoio
        self.COLOR_PRIMARY = "#0284c7"  # Azul de destaque para ações principais
        self.COLOR_SUCCESS = "#16a34a"  # Verde para indicação de sucesso
        self.COLOR_WARN = "#d97706"     # Âmbar para alertas e metas
        self.COLOR_DANGER = "#dc2626"   # Vermelho para falhas ou interrupções

        self.root.configure(bg=self.BG_MAIN)

        # Regras de contraste global para caixas de seleção suspensas (Combobox)
        self.root.option_add("*TCombobox*Listbox.background", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.foreground", "#0f172a")
        self.root.option_add("*TCombobox*Listbox.selectBackground", self.COLOR_PRIMARY)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 9))

        # Estilos base TTK
        self.style.configure(".", background=self.BG_MAIN, foreground=self.TEXT_PRIMARY)
        self.style.configure("TFrame", background=self.BG_MAIN)
        self.style.configure("Card.TFrame", background=self.BG_CARD)

        # Rótulos de texto
        self.style.configure("TLabel", background=self.BG_MAIN, foreground=self.TEXT_PRIMARY, font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", background=self.BG_CARD, foreground=self.TEXT_PRIMARY, font=("Segoe UI", 10, "bold"))
        self.style.configure("Muted.TLabel", background=self.BG_CARD, foreground=self.TEXT_MUTED, font=("Segoe UI", 8))

        # Caixas de seleção suspensa (Combobox)
        self.style.configure(
            "TCombobox",
            fieldbackground="#ffffff",
            foreground="#0f172a",
            background="#ffffff",
            selectbackground="#e2e8f0",
            selectforeground="#0f172a",
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#ffffff"), ("!disabled", "#ffffff")],
            foreground=[("readonly", "#0f172a"), ("!disabled", "#0f172a")],
            selectbackground=[("readonly", "#e2e8f0")],
            selectforeground=[("readonly", "#0f172a")],
        )

        # Campos de entrada de texto (Entry)
        self.style.configure(
            "TEntry",
            fieldbackground="#ffffff",
            foreground="#0f172a",
            insertcolor="#0f172a",
        )
        self.style.map(
            "TEntry",
            fieldbackground=[("!disabled", "#ffffff"), ("disabled", "#f1f5f9")],
            foreground=[("!disabled", "#0f172a"), ("disabled", "#94a3b8")],
        )

        # Botões com estilo consistente
        self.style.configure(
            "Primary.TButton",
            background=self.COLOR_PRIMARY,
            foreground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padding=(10, 5),
            borderwidth=0,
        )
        self.style.map("Primary.TButton", background=[("active", "#0369a1")])

        self.style.configure(
            "Secondary.TButton",
            background="#e2e8f0",
            foreground=self.TEXT_PRIMARY,
            font=("Segoe UI", 9),
            padding=(8, 4),
            borderwidth=1,
        )
        self.style.map("Secondary.TButton", background=[("active", "#cbd5e1")])

        # Separadores (Tabs / Notebook)
        self.style.configure(
            "TNotebook",
            background=self.BG_MAIN,
            borderwidth=0,
            tabmargins=[10, 4, 2, 0],
        )
        self.style.configure(
            "TNotebook.Tab",
            background="#e2e8f0",
            foreground=self.TEXT_MUTED,
            font=("Segoe UI", 9, "bold"),
            padding=[14, 6],
            borderwidth=0,
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", self.BG_CARD), ("active", "#cbd5e1")],
            foreground=[("selected", self.COLOR_PRIMARY), ("active", self.TEXT_PRIMARY)],
        )

        # Grelha de exploração tabular (Treeview)
        self.style.configure(
            "Treeview",
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground=self.TEXT_PRIMARY,
            rowheight=23,
            font=("Segoe UI", 9),
        )
        self.style.configure(
            "Treeview.Heading",
            background="#e2e8f0",
            foreground=self.TEXT_PRIMARY,
            font=("Segoe UI", 9, "bold"),
        )
        self.style.map("Treeview", background=[("selected", "#0284c7")], foreground=[("selected", "#ffffff")])

    # =========================================================================
    # CONSTRUÇÃO DO LAYOUT PRINCIPAL
    # =========================================================================
    def construir_layout_principal(self):
        """Constrói os contentores fundamentais: barra superior, cartões KPI e separadores."""
        # 1. Barra superior para seleção do ficheiro de modelo
        top_bar = tk.Frame(self.root, bg=self.BG_CARD, bd=1, relief="solid", padx=14, pady=10)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(10, 4))

        tk.Label(
            top_bar,
            text="Modelo Selecionado:",
            font=("Segoe UI", 10, "bold"),
            bg=self.BG_CARD,
            fg=self.TEXT_PRIMARY,
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.combo_modelos = ttk.Combobox(
            top_bar,
            state="readonly",
            width=48,
            font=("Segoe UI", 10),
        )
        self.combo_modelos.pack(side=tk.LEFT, padx=4)
        self.combo_modelos.bind("<<ComboboxSelected>>", self.carregar_dados)

        self.btn_atualizar = ttk.Button(
            top_bar,
            text="Atualizar Lista",
            command=self.listar_modelos,
            style="Secondary.TButton",
        )
        self.btn_atualizar.pack(side=tk.LEFT, padx=10)

        # 2. Painel horizontal de 5 cartões de indicadores chave (KPI Cards)
        self.cards_frame = tk.Frame(self.root, bg=self.BG_MAIN)
        self.cards_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=4)

        self.card_widgets = {}
        cards_info = [
            ("eps", "Episódios Totais", "--", "Treinos gravados"),
            ("sucesso", "Taxa de Sucesso", "--%", "Últimos 100 eps"),
            ("passos", "Duração Média", "-- passos", "Eficiência final"),
            ("qtable", "Estados Q-Table", "--", "Espaço explorado"),
            ("epsilon", "Taxa Epsilon", "--", "Exploração final"),
        ]

        for key, title, val_def, sub_def in cards_info:
            card = tk.Frame(self.cards_frame, bg=self.BG_CARD, bd=1, relief="solid", padx=10, pady=6)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)

            lbl_title = tk.Label(card, text=title, font=("Segoe UI", 8), bg=self.BG_CARD, fg=self.TEXT_MUTED)
            lbl_title.pack(anchor="w")

            lbl_val = tk.Label(card, text=val_def, font=("Segoe UI", 13, "bold"), bg=self.BG_CARD, fg=self.TEXT_PRIMARY)
            lbl_val.pack(anchor="w", pady=(1, 0))

            lbl_sub = tk.Label(card, text=sub_def, font=("Segoe UI", 8), bg=self.BG_CARD, fg=self.COLOR_PRIMARY)
            lbl_sub.pack(anchor="w")

            self.card_widgets[key] = (lbl_val, lbl_sub)

        # 3. Contentor em abas (Notebook)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(4, 10))

        self.tab_metricas = tk.Frame(self.notebook, bg=self.BG_CARD)
        self.tab_mapa = tk.Frame(self.notebook, bg=self.BG_CARD)
        self.tab_config = tk.Frame(self.notebook, bg=self.BG_CARD)
        self.tab_qtable = tk.Frame(self.notebook, bg=self.BG_CARD)

        self.notebook.add(self.tab_metricas, text="Métricas de Treino")
        self.notebook.add(self.tab_mapa, text="Mapa de Rotas")
        self.notebook.add(self.tab_config, text="Configurações")
        self.notebook.add(self.tab_qtable, text="Visualizador Q-Table")

        # Construção do conteúdo interno de cada aba
        self.construir_aba_mapa()
        self.construir_aba_qtable()
        self.variaveis_form = {}
        self.construir_aba_configuracoes()

    # =========================================================================
    # ABA MAPA DE ROTAS (INSPEÇÃO EPISÓDIO A EPISÓDIO)
    # =========================================================================
    def construir_aba_mapa(self):
        """Constrói a barra de controlo e o espaço de renderização do mapa de rotas."""
        control_mapa = tk.Frame(self.tab_mapa, bg=self.BG_MAIN, bd=1, relief="solid", padx=10, pady=6)
        control_mapa.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))

        tk.Label(
            control_mapa,
            text="Inspecionar Episódio:",
            font=("Segoe UI", 9, "bold"),
            bg=self.BG_MAIN,
            fg=self.TEXT_PRIMARY,
        ).pack(side=tk.LEFT, padx=(0, 6))

        # Seletor do episódio individual ou visão global
        self.combo_episodio_mapa = ttk.Combobox(
            control_mapa,
            state="readonly",
            width=42,
            font=("Segoe UI", 9),
        )
        self.combo_episodio_mapa.pack(side=tk.LEFT, padx=4)
        self.combo_episodio_mapa.bind("<<ComboboxSelected>>", self.ao_mudar_episodio_combo)

        # Botões de navegação rápida
        self.btn_ant_ep = ttk.Button(
            control_mapa, text="Anterior", command=self.episodio_anterior, style="Secondary.TButton"
        )
        self.btn_ant_ep.pack(side=tk.LEFT, padx=3)

        self.btn_prox_ep = ttk.Button(
            control_mapa, text="Próximo", command=self.episodio_proximo, style="Secondary.TButton"
        )
        self.btn_prox_ep.pack(side=tk.LEFT, padx=3)

        self.btn_ver_todos = ttk.Button(
            control_mapa, text="Ver Todos", command=self.ver_todos_episodios, style="Primary.TButton"
        )
        self.btn_ver_todos.pack(side=tk.LEFT, padx=6)

        # Rótulo de telemetria do episódio selecionado
        self.lbl_info_episodio = tk.Label(
            control_mapa,
            text="Selecione um modelo para visualizar as trajetórias.",
            font=("Segoe UI", 9),
            bg=self.BG_MAIN,
            fg=self.TEXT_PRIMARY,
        )
        self.lbl_info_episodio.pack(side=tk.RIGHT, padx=6)

        self.frame_canvas_mapa = tk.Frame(self.tab_mapa, bg=self.BG_CARD)
        self.frame_canvas_mapa.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # =========================================================================
    # ABA VISUALIZADOR Q-TABLE
    # =========================================================================
    def construir_aba_qtable(self):
        """Constrói a grelha de consulta interativa da Q-Table com campo de filtragem."""
        frame_tabela = tk.Frame(self.tab_qtable, bg=self.BG_CARD, padx=8, pady=8)
        frame_tabela.pack(fill=tk.BOTH, expand=True)

        filtro_frame = tk.Frame(frame_tabela, bg=self.BG_MAIN, padx=8, pady=6, bd=1, relief="solid")
        filtro_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        tk.Label(
            filtro_frame,
            text="Pesquisar estado:",
            font=("Segoe UI", 9),
            bg=self.BG_MAIN,
            fg=self.TEXT_PRIMARY,
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.entry_filtro_qtable = ttk.Entry(filtro_frame, width=30)
        self.entry_filtro_qtable.pack(side=tk.LEFT, padx=4)
        self.entry_filtro_qtable.bind("<KeyRelease>", self.filtrar_qtable)

        self.lbl_count_qtable = tk.Label(
            filtro_frame,
            text="0 estados",
            font=("Segoe UI", 9, "bold"),
            bg=self.BG_MAIN,
            fg=self.COLOR_PRIMARY,
        )
        self.lbl_count_qtable.pack(side=tk.RIGHT, padx=8)

        # Colunas com os valores Q por ação e ação recomendada
        colunas = ("estado", "q_up", "q_down", "q_left", "q_right", "melhor_acao")
        self.tree_qtable = ttk.Treeview(frame_tabela, columns=colunas, show="headings", height=24)

        self.tree_qtable.heading("estado", text="Estado Discreto (X, Y, Conc, Vento, Missão, Bat)")
        self.tree_qtable.heading("q_up", text="Q(Cima)")
        self.tree_qtable.heading("q_down", text="Q(Baixo)")
        self.tree_qtable.heading("q_left", text="Q(Esquerda)")
        self.tree_qtable.heading("q_right", text="Q(Direita)")
        self.tree_qtable.heading("melhor_acao", text="Melhor Ação (Greedy)")

        self.tree_qtable.column("estado", width=360, anchor=tk.W)
        for col in colunas[1:]:
            self.tree_qtable.column(col, width=110, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(frame_tabela, orient=tk.VERTICAL, command=self.tree_qtable.yview)
        self.tree_qtable.configure(yscroll=scrollbar.set)

        self.tree_qtable.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def atualizar_qtable_view(self, q_table):
        """Atualiza a cópia em memória da Q-Table e desencadeia a filtragem da vista."""
        self.tabela_q_cache = q_table
        self.filtrar_qtable()

    def filtrar_qtable(self, event=None):
        """Filtra e preenche a grelha com os estados que correspondam ao termo pesquisado."""
        if not hasattr(self, "tabela_q_cache"):
            return

        termo = self.entry_filtro_qtable.get().strip().lower()
        for item in self.tree_qtable.get_children():
            self.tree_qtable.delete(item)

        acoes = ["Cima", "Baixo", "Esquerda", "Direita"]
        inseridos = 0
        for estado, q_vals in self.tabela_q_cache.items():
            est_str = str(estado)
            if termo and termo not in est_str.lower():
                continue

            melhor_idx = int(np.argmax(q_vals))
            melhor = acoes[melhor_idx] if np.any(q_vals != 0) else "Neutro"

            self.tree_qtable.insert(
                "",
                tk.END,
                values=(
                    est_str,
                    f"{q_vals[0]:.2f}",
                    f"{q_vals[1]:.2f}",
                    f"{q_vals[2]:.2f}",
                    f"{q_vals[3]:.2f}",
                    melhor,
                ),
            )
            inseridos += 1
            if inseridos >= 800:  # Limite para salvaguardar a reatividade da interface
                break

        self.lbl_count_qtable.config(text=f"A mostrar {inseridos} de {len(self.tabela_q_cache)} estados")

    # =========================================================================
    # ABA CONFIGURAÇÕES (ALTA VISIBILIDADE DOS PARÂMETROS)
    # =========================================================================
    def construir_aba_configuracoes(self):
        """Constrói o formulário de edição dos parâmetros do config.json com alto contraste."""
        config_path = "config.json"
        if not os.path.exists(config_path):
            tk.Label(
                self.tab_config,
                text="O ficheiro config.json ainda não existe.\nInicie o main.py pelo menos uma vez para o gerar.",
                font=("Segoe UI", 11),
                bg=self.BG_CARD,
                fg=self.TEXT_MUTED,
            ).pack(pady=40)
            return

        with open(config_path, "r", encoding="utf-8") as f:
            self.config_atual = json.load(f)

        frame_scroll = tk.Frame(self.tab_config, bg=self.BG_CARD)
        frame_scroll.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        frame_principal = tk.Frame(frame_scroll, bg=self.BG_CARD)
        frame_principal.pack(fill=tk.BOTH, expand=True)

        # Coluna da Física da Pluma
        col_esq = tk.LabelFrame(
            frame_principal,
            text=" Física da Pluma ",
            bg=self.BG_CARD,
            fg=self.TEXT_PRIMARY,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
        )
        col_esq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        # Coluna do Agente e Ambiente
        col_dir = tk.LabelFrame(
            frame_principal,
            text=" Agente e Ambiente ",
            bg=self.BG_CARD,
            fg=self.TEXT_PRIMARY,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
        )
        col_dir.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6)

        def criar_campo(parent, label, key_pai, key_filho=None, tipo=float):
            """Cria um campo de entrada com texto preto sobre fundo branco de alto contraste."""
            frame = tk.Frame(parent, bg=self.BG_CARD)
            frame.pack(fill=tk.X, pady=3)

            tk.Label(
                frame,
                text=label,
                width=28,
                anchor="w",
                font=("Segoe UI", 9),
                bg=self.BG_CARD,
                fg=self.TEXT_PRIMARY,
            ).pack(side=tk.LEFT)

            if key_filho:
                val = self.config_atual.get(key_pai, {}).get(key_filho, 0)
            else:
                val = self.config_atual.get(key_pai, 0)

            if isinstance(val, list) and key_pai == "mean_wind_velocity":
                val = val[0]

            var_tk = tk.StringVar(value=str(val))
            entry = tk.Entry(
                frame,
                textvariable=var_tk,
                width=14,
                bg="#ffffff",
                fg="#0f172a",
                insertbackground="#0f172a",
                bd=1,
                relief="solid",
                font=("Segoe UI", 9),
            )
            entry.pack(side=tk.RIGHT)
            self.variaveis_form[(key_pai, key_filho, tipo)] = var_tk
            return entry

        # Parâmetros da Física da Pluma
        criar_campo(col_esq, "Taxa de Emissão (Puffs/s):", "emission_rate", tipo=int)
        criar_campo(col_esq, "Massa Inicial (Puff):", "initial_puff_mass")
        criar_campo(col_esq, "Velocidade do Vento (X):", "mean_wind_velocity")
        criar_campo(col_esq, "Taxa de Expansão:", "puff_expansion_rate")
        criar_campo(col_esq, "Taxa de Decaimento:", "puff_decay_rate")
        criar_campo(col_esq, "Intensidade Turbulência:", "turbulence_intensity")

        # Seleção do Modelo de Pluma (Sinusoidal vs Linear)
        frame_tipo = tk.Frame(col_esq, bg=self.BG_CARD)
        frame_tipo.pack(fill=tk.X, pady=5)
        tk.Label(
            frame_tipo,
            text="Modelo de Pluma:",
            width=28,
            anchor="w",
            font=("Segoe UI", 9),
            bg=self.BG_CARD,
            fg=self.TEXT_PRIMARY,
        ).pack(side=tk.LEFT)

        val_atual = self.config_atual.get("meander_model", "ou")
        opcao_inicial = "Sinusoidal" if val_atual.lower() in ["sum_of_sinusoids", "sinusoid", "sinusoidal"] else "Linear"

        self.var_tipo_pluma = tk.StringVar(value=opcao_inicial)
        self.combo_tipo_pluma = ttk.Combobox(
            frame_tipo,
            textvariable=self.var_tipo_pluma,
            values=["Sinusoidal", "Linear"],
            state="readonly",
            width=12,
        )
        self.combo_tipo_pluma.pack(side=tk.RIGHT)

        # Campos específicos da pluma sinusoidal
        self.entry_amplitude = criar_campo(col_esq, "Amplitude da Onda (v_var):", "sinusoid_meander_config", "v_var")
        self.entry_comprimento = criar_campo(col_esq, "Comprimento da Onda (L):", "sinusoid_meander_config", "integral_length_scale_L")
        self.entry_velocidade = criar_campo(col_esq, "Velocidade Curva (Freq):", "sinusoid_meander_config", "frequency")
        self.entry_espessura = criar_campo(col_esq, "Espessura Pluma (Exp):", "sinusoid_meander_config", "puff_expansion_rate")

        def atualizar_estado_campos(event=None):
            """Habilita ou desabilita campos dependendo do modelo de pluma selecionado."""
            estado = "normal" if self.var_tipo_pluma.get() == "Sinusoidal" else "disabled"
            bg_c = "#ffffff" if estado == "normal" else "#f1f5f9"
            fg_c = "#0f172a" if estado == "normal" else "#94a3b8"
            for ent in [self.entry_amplitude, self.entry_comprimento, self.entry_velocidade, self.entry_espessura]:
                ent.config(state=estado, bg=bg_c, fg=fg_c)

        self.combo_tipo_pluma.bind("<<ComboboxSelected>>", atualizar_estado_campos)
        atualizar_estado_campos()

        # Parâmetros do Veículo e Ambiente
        criar_campo(col_dir, "Duração do Episódio (s):", "max_time")
        criar_campo(col_dir, "Velocidade do Agente (m/s):", "agent_config", "agent_speed")
        criar_campo(col_dir, "Raio do Agente (m):", "agent_config", "agent_radius")
        criar_campo(col_dir, "Capacidade da Bateria:", "battery_capacity")
        criar_campo(col_dir, "Decaimento Base Bateria:", "battery_base_decay")
        criar_campo(col_dir, "Fator Vento Bateria:", "battery_wind_factor")
        criar_campo(col_dir, "Limiar Deteção Odor:", "agent_config", "agent_concentration_threshold")
        criar_campo(col_dir, "Distância de Sucesso (m):", "agent_config", "success_distance")

        btn_salvar = ttk.Button(
            self.tab_config,
            text="Guardar Configurações",
            command=self.salvar_configuracoes,
            style="Primary.TButton",
        )
        btn_salvar.pack(side=tk.BOTTOM, pady=12)

    def salvar_configuracoes(self):
        """Valida os valores numéricos e guarda as alterações no ficheiro config.json."""
        for (key_pai, key_filho, tipo), var_tk in self.variaveis_form.items():
            try:
                valor_novo = tipo(var_tk.get())
                if key_pai == "mean_wind_velocity":
                    self.config_atual[key_pai] = [valor_novo, 0.0]
                elif key_filho:
                    self.config_atual[key_pai][key_filho] = valor_novo
                else:
                    self.config_atual[key_pai] = valor_novo
            except ValueError:
                messagebox.showerror("Erro de Valor", f"O valor para '{key_pai}' é inválido. Utilize apenas números.")
                return

        tipo_pluma = self.var_tipo_pluma.get()
        self.config_atual["meander_model"] = "sum_of_sinusoids" if tipo_pluma == "Sinusoidal" else "linear"

        if "sinusoid_meander_config" in self.config_atual and "v_var" in self.config_atual["sinusoid_meander_config"]:
            self.config_atual["sinusoid_meander_config"]["amplitude"] = self.config_atual["sinusoid_meander_config"]["v_var"]

        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(self.config_atual, f, indent=4)

        messagebox.showinfo("Sucesso", "Configurações guardadas com sucesso no config.json.")

    # =========================================================================
    # CARREGAMENTO E ATUALIZAÇÃO DE DADOS DO MODELO
    # =========================================================================
    def listar_modelos(self):
        """Pesquisa todos os ficheiros .pkl na pasta 'modelos' e atualiza o seletor."""
        if not os.path.exists(self.pasta_modelos):
            messagebox.showerror("Erro", f"Pasta '{self.pasta_modelos}' não encontrada.")
            return

        modelos = glob.glob(os.path.join(self.pasta_modelos, "*.pkl"))
        modelos.sort(key=os.path.getmtime, reverse=True)
        self.modelos_dict = {os.path.basename(m): m for m in modelos}

        if self.modelos_dict:
            nomes = list(self.modelos_dict.keys())
            self.combo_modelos["values"] = nomes
            self.combo_modelos.current(0)
            self.carregar_dados()
        else:
            self.combo_modelos.set("")
            messagebox.showinfo("Aviso", "Nenhum modelo .pkl encontrado na pasta 'modelos'.")

    def carregar_dados(self, event=None):
        """Lê o ficheiro .pkl selecionado, atualiza os cartões KPI e reconstrói os gráficos."""
        selecao = self.combo_modelos.get()
        if not selecao:
            return

        caminho_completo = self.modelos_dict[selecao]

        try:
            with open(caminho_completo, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Erro ao carregar", f"Não foi possível ler o ficheiro:\n{e}")
            return

        stats = data.get("stats", {})
        if not stats or "rewards" not in stats:
            messagebox.showwarning("Aviso", "Este ficheiro não contém um histórico de treino válido.")
            return

        self.dados_modelo_atual = data

        # Extração de métricas dos ensaios
        sucessos = stats.get("successes", [])
        tamanhos = stats.get("lengths", [])
        total_eps = len(stats.get("rewards", []))

        ultimos_100_sucesso = (np.mean(sucessos[-100:]) * 100) if len(sucessos) >= 100 else (np.mean(sucessos) * 100 if sucessos else 0.0)
        ultimos_100_passos = np.mean(tamanhos[-100:]) if len(tamanhos) >= 100 else (np.mean(tamanhos) if tamanhos else 0.0)
        qtable_len = len(data.get("q_table", {}))
        epsilon_val = data.get("epsilon", 0.0)

        # Atualização dos cartões KPI no topo
        self.card_widgets["eps"][0].config(text=f"{total_eps:,}".replace(",", " "))
        cor_sucesso = self.COLOR_SUCCESS if ultimos_100_sucesso >= 70 else (self.COLOR_WARN if ultimos_100_sucesso >= 40 else self.COLOR_DANGER)
        self.card_widgets["sucesso"][0].config(text=f"{ultimos_100_sucesso:.1f}%", fg=cor_sucesso)
        self.card_widgets["passos"][0].config(text=f"{ultimos_100_passos:.1f}")
        self.card_widgets["qtable"][0].config(text=f"{qtable_len:,}".replace(",", " "))
        self.card_widgets["epsilon"][0].config(text=f"{epsilon_val:.4f}")

        # Atualização da tabela e dos gráficos
        self.atualizar_qtable_view(data.get("q_table", {}))
        self.desenhar_graficos_metricas(data)
        self.preparar_mapa_rotas(data)

    # =========================================================================
    # GRÁFICOS DE MÉTRICAS (TEMA CLARO E ELEGANTE)
    # =========================================================================
    def desenhar_graficos_metricas(self, data):
        """Gera os 3 subgráficos de convergência: Taxa de Sucesso, Recompensas e Duração em passos."""
        stats = data.get("stats", {})
        recompensas = stats.get("rewards", [])
        tamanhos = stats.get("lengths", [])
        sucessos = stats.get("successes", [])

        if self.canvas_metricas:
            self.canvas_metricas.get_tk_widget().destroy()

        if not recompensas:
            return

        n_eps = len(recompensas)
        janela = max(5, min(100, n_eps // 30))

        fig, axs = plt.subplots(3, 1, figsize=(9.5, 8.0), dpi=100, facecolor="#ffffff")
        fig.subplots_adjust(hspace=0.40, top=0.94, bottom=0.08, left=0.08, right=0.95)

        # 1. Taxa de Sucesso (%)
        ax1 = axs[0]
        ax1.set_facecolor("#ffffff")

        if len(sucessos) >= janela:
            media_sucessos = np.convolve(sucessos, np.ones(janela) / janela, mode="valid") * 100
            x_janela = np.arange(janela - 1, len(sucessos))
            ax1.plot(x_janela, media_sucessos, color=self.COLOR_SUCCESS, linewidth=2.2, label=f"Média Móvel ({janela} eps)")
            ax1.fill_between(x_janela, 0, media_sucessos, color="#dcfce7", alpha=0.6)

        sucesso_acumulado = (np.cumsum(sucessos) / np.arange(1, len(sucessos) + 1)) * 100
        ax1.plot(sucesso_acumulado, color=self.COLOR_PRIMARY, linewidth=1.4, linestyle="--", label="Taxa Acumulada")
        ax1.axhline(80, color=self.COLOR_WARN, linestyle=":", linewidth=1.3, label="Meta (80%)")

        ax1.set_title("Taxa de Sucesso (%) - Eficácia na Busca da Fonte", color="#0f172a", fontsize=10, fontweight="bold", pad=5)
        ax1.set_ylabel("Sucesso (%)", color="#475569", fontsize=8)
        ax1.set_ylim(-2, 105)
        ax1.grid(True, color="#e2e8f0", linestyle="--", alpha=0.8)
        ax1.tick_params(colors="#475569", labelsize=8)
        ax1.legend(loc="lower right", facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)

        # 2. Recompensas Acumuladas
        ax2 = axs[1]
        ax2.set_facecolor("#ffffff")

        ax2.plot(recompensas, color="#94a3b8", alpha=0.35, linewidth=0.8, label="Recompensa por Episódio")
        if len(recompensas) >= janela:
            media_recompensas = np.convolve(recompensas, np.ones(janela) / janela, mode="valid")
            ax2.plot(np.arange(janela - 1, len(recompensas)), media_recompensas, color="#2563eb", linewidth=2.0, label=f"Tendência Suavizada ({janela} eps)")

        ax2.axhline(0, color="#94a3b8", linestyle=":", linewidth=1.0)
        ax2.set_title("Recompensa Acumulada por Episódio", color="#0f172a", fontsize=10, fontweight="bold", pad=5)
        ax2.set_ylabel("Recompensa", color="#475569", fontsize=8)
        ax2.grid(True, color="#e2e8f0", linestyle="--", alpha=0.8)
        ax2.tick_params(colors="#475569", labelsize=8)
        ax2.legend(loc="upper left", facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)

        # 3. Duração da Busca em Passos
        ax3 = axs[2]
        ax3.set_facecolor("#ffffff")

        ax3.plot(tamanhos, color="#fdba74", alpha=0.4, linewidth=0.8, label="Passos por Episódio")
        if len(tamanhos) >= janela:
            media_passos = np.convolve(tamanhos, np.ones(janela) / janela, mode="valid")
            ax3.plot(np.arange(janela - 1, len(tamanhos)), media_passos, color="#ea580c", linewidth=2.0, label=f"Passos Médios ({janela} eps)")

        ax3.set_title("Duração da Busca em Passos", color="#0f172a", fontsize=10, fontweight="bold", pad=5)
        ax3.set_xlabel("Episódio", color="#0f172a", fontsize=8, labelpad=4)
        ax3.set_ylabel("Passos", color="#475569", fontsize=8)
        ax3.grid(True, color="#e2e8f0", linestyle="--", alpha=0.8)
        ax3.tick_params(colors="#475569", labelsize=8)
        ax3.legend(loc="upper right", facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)

        self.canvas_metricas = FigureCanvasTkAgg(fig, master=self.tab_metricas)
        self.canvas_metricas.draw()
        self.canvas_metricas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

    # =========================================================================
    # MAPA DE ROTAS (TEMA CLARO E NAVEGAÇÃO DE ROTAS)
    # =========================================================================
    def preparar_mapa_rotas(self, data):
        """Popula a lista de episódios gravados no seletor do mapa de rotas."""
        stats = data.get("stats", {})
        paths = stats.get("paths", {})
        sucessos = stats.get("successes", [])
        tamanhos = stats.get("lengths", [])

        self.chaves_episodios_gravados = sorted(paths.keys())

        if not self.chaves_episodios_gravados:
            self.combo_episodio_mapa["values"] = ["Sem trajetórias gravadas"]
            self.combo_episodio_mapa.current(0)
            self.lbl_info_episodio.config(
                text="Este modelo não contém trajetórias salvas.", fg=self.COLOR_WARN
            )
            self.desenhar_mapa_vazio()
            return

        opcoes = ["Todos os Episódios (Visão Global)"]
        for ep in self.chaves_episodios_gravados:
            num_passos = tamanhos[ep] if ep < len(tamanhos) else len(paths[ep])
            teve_sucesso = (sucessos[ep] == 1) if ep < len(sucessos) else False
            status_txt = "Sucesso" if teve_sucesso else "Falha"
            opcoes.append(f"Episódio #{ep} - {num_passos} passos - {status_txt}")

        self.combo_episodio_mapa["values"] = opcoes
        self.indice_episodio_atual = 0
        self.combo_episodio_mapa.current(0)

        self.desenhar_mapa_selecionado()

    def ao_mudar_episodio_combo(self, event=None):
        """Callback acionado pela alteração da seleção na caixa de episódios."""
        self.indice_episodio_atual = self.combo_episodio_mapa.current()
        self.desenhar_mapa_selecionado()

    def episodio_anterior(self):
        """Navega para o episódio anterior da lista de trajetórias gravadas."""
        if not self.chaves_episodios_gravados:
            return
        if self.indice_episodio_atual > 0:
            self.indice_episodio_atual -= 1
        else:
            self.indice_episodio_atual = len(self.chaves_episodios_gravados)
        self.combo_episodio_mapa.current(self.indice_episodio_atual)
        self.desenhar_mapa_selecionado()

    def episodio_proximo(self):
        """Navega para o episódio seguinte da lista de trajetórias gravadas."""
        if not self.chaves_episodios_gravados:
            return
        if self.indice_episodio_atual < len(self.chaves_episodios_gravados):
            self.indice_episodio_atual += 1
        else:
            self.indice_episodio_atual = 0
        self.combo_episodio_mapa.current(self.indice_episodio_atual)
        self.desenhar_mapa_selecionado()

    def ver_todos_episodios(self):
        """Repõe o seletor no modo de Visão Global com todas as rotas sobrepostas."""
        self.indice_episodio_atual = 0
        self.combo_episodio_mapa.current(0)
        self.desenhar_mapa_selecionado()

    def converter_para_metros(self, coords, world_size=(100.0, 50.0)):
        """Converte coordenadas normalizadas [0, 1] para a escala física real em metros."""
        p = np.array(coords)
        if p.ndim != 2 or p.shape[0] == 0:
            return p
        max_x = np.max(p[:, 0]) if p.shape[0] > 0 else 0
        max_y = np.max(p[:, 1]) if p.shape[0] > 0 else 0
        scale_x = world_size[0] if max_x <= 1.5 else 1.0
        scale_y = world_size[1] if max_y <= 1.5 else 1.0
        return np.column_stack([p[:, 0] * scale_x, p[:, 1] * scale_y])

    def desenhar_mapa_vazio(self):
        """Desenha uma mensagem informativa caso o modelo não contenha trajetórias gravadas."""
        if self.canvas_mapa:
            self.canvas_mapa.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=100, facecolor="#ffffff")
        ax.set_facecolor("#f8fafc")
        ax.text(
            50,
            25,
            "Sem dados de trajetórias gravadas neste modelo.",
            ha="center",
            va="center",
            color=self.TEXT_MUTED,
            fontsize=11,
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 50)
        ax.grid(True, color="#e2e8f0", linestyle="--", alpha=0.8)

        self.canvas_mapa = FigureCanvasTkAgg(fig, master=self.frame_canvas_mapa)
        self.canvas_mapa.draw()
        self.canvas_mapa.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

    def desenhar_mapa_selecionado(self):
        """Renderiza a rota individual ou o mapa global de dispersão e navegação."""
        if not self.dados_modelo_atual:
            return

        stats = self.dados_modelo_atual.get("stats", {})
        paths = stats.get("paths", {})
        sucessos = stats.get("successes", [])
        tamanhos = stats.get("lengths", [])
        recompensas = stats.get("rewards", [])

        source_pos = self.dados_modelo_atual.get("source_pos", (20.0, 25.0))
        world_size = self.dados_modelo_atual.get("world_size", (100.0, 50.0))

        if self.canvas_mapa:
            self.canvas_mapa.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=100, facecolor="#ffffff")
        ax.set_facecolor("#f8fafc")

        # Representação gráfica do cone de dispersão da pluma a jusante da corrente
        x_cone = np.linspace(source_pos[0], world_size[0], 60)
        y_top = np.clip(source_pos[1] + 0.22 * (x_cone - source_pos[0]) + 2.5, 0, world_size[1])
        y_bot = np.clip(source_pos[1] - 0.22 * (x_cone - source_pos[0]) - 2.5, 0, world_size[1])
        ax.fill_between(
            x_cone,
            y_bot,
            y_top,
            color="#e0f2fe",
            edgecolor="#bae6fd",
            alpha=0.6,
            label="Área de Dispersão da Pluma",
            zorder=1,
        )

        # Linha de eixo central
        ax.axhline(source_pos[1], color="#94a3b8", linestyle=":", alpha=0.4, zorder=2)

        # Origem da Pluma (Estrela Dourada)
        ax.scatter(
            source_pos[0],
            source_pos[1],
            marker="*",
            s=260,
            color="#eab308",
            edgecolor="#0f172a",
            linewidth=1.4,
            zorder=10,
            label=f"Origem da Pluma ({source_pos[0]:.0f}m, {source_pos[1]:.0f}m)",
        )

        # Vetor ilustrativo da corrente / vento principal
        ax.annotate(
            "Vento -> (Direção Principal)",
            xy=(35, 47),
            xytext=(15, 47),
            arrowprops=dict(arrowstyle="->", color=self.COLOR_PRIMARY, lw=1.5),
            color=self.COLOR_PRIMARY,
            fontsize=8,
            fontweight="bold",
            va="center",
            zorder=8,
        )

        # CASO 1: Visão Global com todas as rotas do treino
        if self.indice_episodio_atual == 0:
            eps_gravados = self.chaves_episodios_gravados
            self.lbl_info_episodio.config(
                text=f"Visão Global: {len(eps_gravados)} trajetórias gravadas",
                fg=self.COLOR_PRIMARY,
            )

            cmap = plt.get_cmap("viridis")
            norm = mcolors.Normalize(vmin=min(eps_gravados), vmax=max(eps_gravados))

            for ep in eps_gravados:
                p = self.converter_para_metros(paths[ep], world_size)
                cor = cmap(norm(ep))
                alpha_line = 0.6 if ep == eps_gravados[-1] else 0.35
                lw = 2.0 if ep == eps_gravados[-1] else 1.1
                ax.plot(p[:, 0], p[:, 1], color=cor, alpha=alpha_line, linewidth=lw, zorder=3)

            p_last = self.converter_para_metros(paths[eps_gravados[-1]], world_size)
            ax.scatter(p_last[0, 0], p_last[0, 1], marker="o", s=80, color=self.COLOR_SUCCESS, edgecolor="#0f172a", zorder=7, label=f"Início Último Ep (#{eps_gravados[-1]})")
            ax.scatter(p_last[-1, 0], p_last[-1, 1], marker="s", s=80, color=self.COLOR_PRIMARY, edgecolor="#0f172a", zorder=7, label=f"Fim Último Ep (#{eps_gravados[-1]})")

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
            cbar.set_label("Número do Episódio", color=self.TEXT_PRIMARY, fontsize=8)
            cbar.ax.tick_params(colors=self.TEXT_MUTED, labelsize=8)

            ax.set_title("Evolução Global das Rotas de Navegação", color="#0f172a", fontsize=10, fontweight="bold", pad=6)

        # CASO 2: Inspeção Detalhada da Trajetória Individual
        else:
            ep_idx = self.indice_episodio_atual - 1
            ep_real = self.chaves_episodios_gravados[ep_idx]
            p = self.converter_para_metros(paths[ep_real], world_size)

            teve_sucesso = (sucessos[ep_real] == 1) if ep_real < len(sucessos) else False
            num_passos = tamanhos[ep_real] if ep_real < len(tamanhos) else len(p)
            recompensa_ep = recompensas[ep_real] if ep_real < len(recompensas) else 0.0

            diffs = np.diff(p, axis=0)
            dist_total = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1)))) if len(p) > 1 else 0.0

            status_cor = self.COLOR_SUCCESS if teve_sucesso else self.COLOR_DANGER
            status_nome = "Sucesso" if teve_sucesso else "Falha"
            self.lbl_info_episodio.config(
                text=f"Ep #{ep_real}  |  {num_passos} passos  |  Distância: {dist_total:.1f} m  |  Recompensa: {recompensa_ep:+.1f}  |  {status_nome}",
                fg=status_cor,
            )

            cor_rota = self.COLOR_SUCCESS if teve_sucesso else self.COLOR_PRIMARY

            ax.plot(
                p[:, 0],
                p[:, 1],
                color=cor_rota,
                linewidth=2.2,
                alpha=0.9,
                label=f"Trajetória Ep #{ep_real} ({status_nome})",
                zorder=4,
            )

            # Setas de orientação direcional ao longo do caminho
            num_pontos = len(p)
            if num_pontos >= 4:
                num_setas = min(12, max(4, num_pontos // 15))
                indices_setas = np.linspace(0, num_pontos - 2, num_setas, dtype=int)
                for idx in indices_setas:
                    dx = p[idx + 1, 0] - p[idx, 0]
                    dy = p[idx + 1, 1] - p[idx, 1]
                    if np.hypot(dx, dy) > 1e-4:
                        ax.annotate(
                            "",
                            xy=(p[idx + 1, 0], p[idx + 1, 1]),
                            xytext=(p[idx, 0], p[idx, 1]),
                            arrowprops=dict(
                                arrowstyle="->",
                                color="#0f172a",
                                lw=1.5,
                                mutation_scale=12,
                            ),
                            zorder=6,
                        )

            # Marcador de Início (Círculo Verde)
            ax.scatter(
                p[0, 0],
                p[0, 1],
                marker="o",
                s=140,
                color=self.COLOR_SUCCESS,
                edgecolor="#0f172a",
                linewidth=1.5,
                zorder=7,
                label=f"Início ({p[0, 0]:.1f}m, {p[0, 1]:.1f}m)",
            )

            # Marcador de Fim (Quadrado Verde em sucesso, Cruz Vermelha em falha)
            cor_fim = self.COLOR_SUCCESS if teve_sucesso else self.COLOR_DANGER
            icone_fim = "s" if teve_sucesso else "X"
            label_fim = f"Fim - Sucesso ({p[-1, 0]:.1f}m, {p[-1, 1]:.1f}m)" if teve_sucesso else f"Fim - Interrupção ({p[-1, 0]:.1f}m, {p[-1, 1]:.1f}m)"
            ax.scatter(
                p[-1, 0],
                p[-1, 1],
                marker=icone_fim,
                s=140,
                color=cor_fim,
                edgecolor="#0f172a",
                linewidth=1.5,
                zorder=7,
                label=label_fim,
            )

            ax.set_title(
                f"Trajetória do Episódio #{ep_real} ({status_nome})",
                color="#0f172a",
                fontsize=10,
                fontweight="bold",
                pad=6,
            )

        ax.set_xlim(0, world_size[0])
        ax.set_ylim(0, world_size[1])
        ax.set_xlabel("Distância Longitudinal - Eixo X (metros)", color="#0f172a", fontsize=8, labelpad=4)
        ax.set_ylabel("Distância Transversal - Eixo Y (metros)", color="#0f172a", fontsize=8, labelpad=4)
        ax.grid(True, color="#e2e8f0", linestyle="--", alpha=0.8)
        ax.tick_params(colors="#475569", labelsize=8)

        ax.legend(
            loc="lower right",
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            fontsize=8,
            framealpha=0.95,
        )

        fig.tight_layout()
        self.canvas_mapa = FigureCanvasTkAgg(fig, master=self.frame_canvas_mapa)
        self.canvas_mapa.draw()
        self.canvas_mapa.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()
