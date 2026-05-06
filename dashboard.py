import os
import glob
import pickle
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

        # --- Adicionar as páginas ao Notebook ---
        self.notebook.add(self.tab_metricas, text="Métricas de Treino")
        self.notebook.add(self.tab_mapa, text="Mapa de Rotas")

        # --- Controlos dos Gráficos ----
        self.canvas_metricas = None
        self.canvas_mapa = None

        self.listar_modelos()

    def fechar_aplicacao(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

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

        
