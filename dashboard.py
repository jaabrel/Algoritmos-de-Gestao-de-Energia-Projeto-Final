import os
import glob
import pickle
import json
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dashboard Q-Learning")
        self.root.geometry("1000x850")
        self.root.protocol("WM_DELETE_WINDOW", self.fechar_aplicacao)

        self.pasta_modelos = "modelos"

        # --- PAINEL SUPERIOR (Controlos) ---
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Selecione o Modelo:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=5)

        self.combo_modelos = ttk.Combobox(control_frame, state="readonly", width=45, font=("Arial", 10))
        self.combo_modelos.pack(side=tk.LEFT, padx=5)
        self.combo_modelos.bind("<<ComboboxSelected>>", self.carregar_dados)

        self.btn_atualizar = ttk.Button(
            control_frame, text="Atualizar Lista", command=self.listar_modelos
        )
        self.btn_atualizar.pack(side=tk.LEFT, padx=10)

        # --- PAINEL DE ESTATÍSTICAS ---
        stats_frame = ttk.Frame(root, padding="5")
        stats_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_stats = ttk.Label(
            stats_frame,
            text="A carregar...",
            font=("Consolas", 11),
            foreground="#333333",
        )
        self.lbl_stats.pack()
        
        # --- SISTEMA DE PÁGINAS (Tabs) ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Criar os frames de cada página ---
        self.tab_metricas = ttk.Frame(self.notebook)
        self.tab_mapa = ttk.Frame(self.notebook)
        self.tab_config = ttk.Frame(self.notebook)
        self.tab_qtable = ttk.Frame(self.notebook)

        # --- Adicionar as páginas ao Notebook ---
        self.notebook.add(self.tab_metricas, text="Métricas de Treino")
        self.notebook.add(self.tab_mapa, text="Mapa de Rotas")
        self.notebook.add(self.tab_config, text="Configurações (JSON)")
        self.notebook.add(self.tab_qtable, text="Visualizador Q-Table")
        
        self.construir_aba_qtable()
        self.variaveis_form = {}
        self.construir_aba_configuracoes()

        # --- Controlos dos Gráficos ----
        self.canvas_metricas = None
        self.canvas_mapa = None

        self.listar_modelos()

    def fechar_aplicacao(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    def construir_aba_qtable(self):
        """Cria uma grelha de consulta da Q-Table"""
        frame_tabela = ttk.Frame(self.tab_qtable, padding="10")
        frame_tabela.pack(fill=tk.BOTH, expand=True)

        colunas = ("estado", "q_up", "q_down", "q_left", "q_right", "melhor_acao")
        self.tree_qtable = ttk.Treeview(frame_tabela, columns=colunas, show="headings", height=25)

        self.tree_qtable.heading("estado", text="Estado (X, Y, Conc, Vento, Missão, Bat)")
        self.tree_qtable.heading("q_up", text="Q(Cima)")
        self.tree_qtable.heading("q_down", text="Q(Baixo)")
        self.tree_qtable.heading("q_left", text="Q(Esquerda)")
        self.tree_qtable.heading("q_right", text="Q(Direita)")
        self.tree_qtable.heading("melhor_acao", text="Melhor Ação (Greedy)")
        self.tree_qtable.column("estado", width=260, anchor=tk.CENTER)
        for col in colunas[1:]:
            self.tree_qtable.column(col, width=100, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(frame_tabela, orient=tk.VERTICAL, command=self.tree_qtable.yview)
        self.tree_qtable.configure(yscroll=scrollbar.set)

        self.tree_qtable.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    

    def atualizar_qtable_view(self, q_table):
        """Limpa e preenche a grelha com os dados do modelo selecionado."""
        for item in self.tree_qtable.get_children():
            self.tree_qtable.delete(item)

        acoes = ["Cima", "Baixo", "Esquerda", "Direita"]
        for estado, q_vals in q_table.items():
            melhor_idx = int(np.argmax(q_vals))
            melhor = acoes[melhor_idx] if np.any(q_vals != 0) else "Neutro"
            
            self.tree_qtable.insert("", tk.END, values=(
                str(estado),
                f"{q_vals[0]:.2f}",
                f"{q_vals[1]:.2f}",
                f"{q_vals[2]:.2f}",
                f"{q_vals[3]:.2f}",
                melhor
            ))

    def listar_modelos(self):
        """Procura todos os ficheiros .pkl e atualiza o dropdown."""
        if not os.path.exists(self.pasta_modelos):
            messagebox.showerror(
                "Erro",
                f"Pasta '{self.pasta_modelos}' não encontrada. Treina o agente primeiro.",
            )
            return

        modelos = glob.glob(os.path.join(self.pasta_modelos, "qlearning_plume_*.pkl"))
        modelos.sort(key=os.path.getmtime, reverse=True)
        self.modelos_dict = {os.path.basename(m): m for m in modelos}

        if self.modelos_dict:
            self.combo_modelos["values"] = list(self.modelos_dict.keys())
            self.combo_modelos.current(0)  # Seleciona o mais recente por defeito
            self.carregar_dados()
        else:
            self.combo_modelos.set("")
            self.lbl_stats.config(
                text="Nenhum modelo encontrado. Inicia um treino primeiro."
            )
            if self.canvas_widget:
                self.canvas_widget.get_tk_widget().destroy()

    def carregar_dados(self, event=None):
        """Carrega os dados do ficheiro selecionado e atualiza a interface."""
        selecao = self.combo_modelos.get()
        if not selecao:
            return

        caminho_completo = self.modelos_dict[selecao]

        try:
            with open(caminho_completo, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            messagebox.showerror(
                "Erro ao carregar", f"Não foi possível ler o ficheiro:\n{e}"
            )
            return

        stats = data.get("stats", {})
        if not stats or "rewards" not in stats:
            messagebox.showwarning(
                "Aviso", "Este ficheiro não contém um histórico de treino válido."
            )
            return

        # Calcular estatísticas finais
        sucessos = stats["successes"]
        tamanhos = stats["lengths"]

        ultimos_100_sucesso = (
            np.mean(sucessos[-100:]) * 100
            if len(sucessos) >= 100
            else np.mean(sucessos) * 100
        )
        ultimos_100_passos = (
            np.mean(tamanhos[-100:]) if len(tamanhos) >= 100 else np.mean(tamanhos)
        )

        texto_stats = (
            f"Episódios Totais: {len(stats['rewards'])}   |   "
            f"Tabela Q: {len(data.get('q_table', {}))} estados   |   "
            f"Sucesso Final: {ultimos_100_sucesso:.1f}%   |   "
            f"Passos (Média): {ultimos_100_passos:.1f}   |   "
            f"Exploração Final: {data.get('epsilon', 0.0):.4f}"
        )
        self.lbl_stats.config(text=texto_stats)

        self.desenhar_graficos(data)
        self.atualizar_qtable_view(data.get("q_table", {}))
    

    def construir_aba_configuracoes(self):
        """Constrói a interface para editar o ficheiro config.json"""
        config_path = "config.json"
        
        if not os.path.exists(config_path):
            ttk.Label(self.tab_config, text="O ficheiro config.json ainda não existe.\nCorre o main.py pelo menos uma vez para o gerar.", font=("Arial", 12)).pack(pady=20)
            return

        with open(config_path, "r") as f:
            self.config_atual = json.load(f)

        # Criar um layout de duas colunas (Esquerda: Física, Direita: Agente)
        frame_principal = ttk.Frame(self.tab_config, padding="20")
        frame_principal.pack(fill=tk.BOTH, expand=True)
        
        col_esq = ttk.LabelFrame(frame_principal, text="Física da Pluma", padding="10")
        col_esq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        col_dir = ttk.LabelFrame(frame_principal, text="Agente e Ambiente", padding="10")
        col_dir.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Função auxiliar para criar os campos de input
        def criar_campo(parent, label, key_pai, key_filho=None, tipo=float):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=5)
            ttk.Label(frame, text=label, width=25).pack(side=tk.LEFT)
            
            # Vai buscar o valor atual
            if key_filho:
                val = self.config_atual.get(key_pai, {}).get(key_filho, 0)
            else:
                val = self.config_atual.get(key_pai, 0)
                
            # Se for uma lista/tupla (como o vento), lidamos com o primeiro valor (X)
            if isinstance(val, list) and key_pai == "mean_wind_velocity":
                val = val[0]
                
            var_tk = tk.StringVar(value=str(val))
            entry = ttk.Entry(frame, textvariable=var_tk, width=15)
            entry.pack(side=tk.RIGHT)
            
            # Guarda a referência para podermos salvar depois
            self.variaveis_form[(key_pai, key_filho, tipo)] = var_tk
            return entry

        # --- CAMPOS DA FÍSICA (Coluna Esquerda) ---
        criar_campo(col_esq, "Taxa de Emissão (Puffs/s):", "emission_rate", tipo=int)
        criar_campo(col_esq, "Massa Inicial (Tinta):", "initial_puff_mass")
        criar_campo(col_esq, "Velocidade do Vento (X):", "mean_wind_velocity")
        criar_campo(col_esq, "Taxa de Expansão (Volume):", "puff_expansion_rate")
        criar_campo(col_esq, "Taxa de Decaimento (Decaída):", "puff_decay_rate")
        criar_campo(col_esq, "Intensidade Turbulência:", "turbulence_intensity")
        
        # Modelo de Pluma
        frame_tipo = ttk.Frame(col_esq)
        frame_tipo.pack(fill=tk.X, pady=5)
        ttk.Label(frame_tipo, text="Modelo de Pluma:", width=25).pack(side=tk.LEFT)
        
        val_atual = self.config_atual.get("meander_model", "ou")
        if val_atual.lower() == "sum_of_sinusoids":
            opcao_inicial = "Sinusoidal"
        else:
            opcao_inicial = "Linear"
            
        self.var_tipo_pluma = tk.StringVar(value=opcao_inicial)
        self.combo_tipo_pluma = ttk.Combobox(
            frame_tipo, 
            textvariable=self.var_tipo_pluma, 
            values=["Sinusoidal", "Linear"], 
            state="readonly", 
            width=12
        )
        self.combo_tipo_pluma.pack(side=tk.RIGHT)

        self.entry_amplitude = criar_campo(col_esq, "Amplitude da Onda (v_var):", "sinusoid_meander_config", "v_var")
        self.entry_comprimento = criar_campo(col_esq, "Comprimento da Onda (L):", "sinusoid_meander_config", "integral_length_scale_L")

        def atualizar_estado_campos(event=None):
            if self.var_tipo_pluma.get() == "Sinusoidal":
                self.entry_amplitude.config(state="normal")
                self.entry_comprimento.config(state="normal")
            else:
                self.entry_amplitude.config(state="disabled")
                self.entry_comprimento.config(state="disabled")

        self.combo_tipo_pluma.bind("<<ComboboxSelected>>", atualizar_estado_campos)
        atualizar_estado_campos()

        # --- CAMPOS DO AGENTE (Coluna Direita) ---
        criar_campo(col_dir, "Duração do Episódio:", "max_time")
        criar_campo(col_dir, "Velocidade do Agente:", "agent_config", "agent_speed")
        criar_campo(col_dir, "Raio do Agente:", "agent_config", "agent_radius")
        criar_campo(col_dir, "Capacidade da Bateria:", "battery_capacity")
        criar_campo(col_dir, "Decaimento Base Bateria:", "battery_base_decay")
        criar_campo(col_dir, "Fator Vento Bateria:", "battery_wind_factor")
        criar_campo(col_dir, "Limiar Deteção Odor:", "agent_config", "agent_concentration_threshold")
        criar_campo(col_dir, "Distância de Sucesso:", "agent_config", "success_distance")
        
        # Botão de Salvar
        btn_salvar = ttk.Button(self.tab_config, text="💾 Guardar Configurações", command=self.salvar_configuracoes)
        btn_salvar.pack(side=tk.BOTTOM, pady=20)

    def salvar_configuracoes(self):
        
        for (key_pai, key_filho, tipo), var_tk in self.variaveis_form.items():
            try:
                # Converte o texto da caixa para número (int ou float)
                valor_novo = tipo(var_tk.get())
                
                # Tratamento especial para o vento (que é uma lista de 2 valores)
                if key_pai == "mean_wind_velocity":
                    self.config_atual[key_pai] = [valor_novo, 0.0]
                elif key_filho:
                    self.config_atual[key_pai][key_filho] = valor_novo
                else:
                    self.config_atual[key_pai] = valor_novo
                    
            except ValueError:
                messagebox.showerror("Erro de Valor", f"O valor para '{key_pai}' é inválido. Usa apenas números.")
                return

        # Atualiza o modelo de pluma com base na seleção do dropdown
        tipo_pluma = self.var_tipo_pluma.get()
        if tipo_pluma == "Sinusoidal":
            self.config_atual["meander_model"] = "sum_of_sinusoids"
        else:
            self.config_atual["meander_model"] = "linear"

        # Guarda no ficheiro
        with open("config.json", "w") as f:
            json.dump(self.config_atual, f, indent=4)
            
        messagebox.showinfo("Sucesso", "Configurações guardadas com sucesso no config.json!\nO teu próximo treino no main.py já vai usar estas regras.")
    
    def desenhar_graficos(self, data):
        stats = data.get("stats", {})
        recompensas = stats["rewards"]
        tamanhos = stats["lengths"]
        sucessos = stats["successes"]
        paths = stats.get("paths", {})

        window = max(1, len(recompensas) // 50)
        
        # ==========================================
        # PÁGINA 1: MÉTRICAS DE TREINO
        # ==========================================
        if self.canvas_metricas:
            self.canvas_metricas.get_tk_widget().destroy()
            
        fig_met, axs_met = plt.subplots(3, 1, figsize=(8, 8), dpi=100)
        
        axs_met[0].plot(recompensas, alpha=0.3, color='royalblue')
        axs_met[0].plot(np.convolve(recompensas, np.ones(window)/window, mode='valid'), color='navy', linewidth=2)
        axs_met[0].set_title('Recompensas Acumuladas', fontsize=11)
        axs_met[0].grid(True, alpha=0.3)
        
        axs_met[1].plot(tamanhos, alpha=0.3, color='darkorange')
        axs_met[1].plot(np.convolve(tamanhos, np.ones(window)/window, mode='valid'), color='saddlebrown', linewidth=2)
        axs_met[1].set_title('Duração da Busca (Passos)', fontsize=11)
        axs_met[1].grid(True, alpha=0.3)
        
        sucesso_acumulado = np.cumsum(sucessos) / np.arange(1, len(sucessos) + 1) * 100
        axs_met[2].plot(sucesso_acumulado, color='forestgreen', linewidth=2)
        axs_met[2].set_title('Taxa de Sucesso Acumulada (%)', fontsize=11)
        axs_met[2].set_xlabel('Episódio', fontsize=10)
        axs_met[2].grid(True, alpha=0.3)
        
        fig_met.tight_layout()
        self.canvas_metricas = FigureCanvasTkAgg(fig_met, master=self.tab_metricas)
        self.canvas_metricas.draw()
        self.canvas_metricas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ==========================================
        # PÁGINA 2: MAPA DE ROTAS
        # ==========================================
        if self.canvas_mapa:
            self.canvas_mapa.get_tk_widget().destroy()
            
        fig_map, ax_map = plt.subplots(figsize=(8, 8), dpi=100)
        ax_map.set_title('Evolução das Rotas de Navegação', fontsize=12)
        ax_map.set_xlim(0, 1)
        ax_map.set_ylim(0, 1)
        
        if paths:
            eps_gravados = sorted(paths.keys())
            cmap = plt.get_cmap('RdYlGn') 
            norm = mcolors.Normalize(vmin=min(eps_gravados), vmax=max(eps_gravados))
            
            for ep in eps_gravados:
                p = np.array(paths[ep])
                cor = cmap(norm(ep))
                ax_map.plot(p[:, 0], p[:, 1], color=cor, alpha=0.5, linewidth=1.5)
            
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig_map.colorbar(sm, ax=ax_map)
            cbar.set_label('Número do Episódio', fontsize=10)
            
            # --- DESENHAR A ESTRELA ---
            # Vai buscar os valores salvos (se não existirem usa os teus defaults de backup)
            source_pos = data.get("source_pos", (20.0, 25.0))
            world_size = data.get("world_size", (100.0, 50.0))
            
            sx_norm = source_pos[0] / world_size[0]
            sy_norm = source_pos[1] / world_size[1]
            
            # Desenha a estrela com zorder=5 para ficar sempre por cima das linhas
            ax_map.plot(sx_norm, sy_norm, marker='*', color='gold', markersize=22, 
                        markeredgecolor='black', label='Origem da Pluma (Alvo)', zorder=5)
            
            ax_map.legend(loc='lower right', framealpha=0.9, fontsize=10)
            # --------------------------
            
        else:
            ax_map.text(0.5, 0.5, "Sem dados de rotas.\nTreina um modelo novo!", 
                        ha='center', va='center', color='gray', fontsize=14)
            
        ax_map.grid(True, alpha=0.3)
        fig_map.tight_layout()
        
        self.canvas_mapa = FigureCanvasTkAgg(fig_map, master=self.tab_mapa)
        self.canvas_mapa.draw()
        self.canvas_mapa.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    app = DashboardApp(root)
    root.mainloop()

        
