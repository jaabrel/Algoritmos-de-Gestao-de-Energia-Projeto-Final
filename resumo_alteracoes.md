# Resumo das Alterações — Configuração, Volume e Longevidade da Pluma

Este documento descreve todas as modificações realizadas no código para suportar a seleção de plumas (Sinusoidal vs. Linear), aumentar o seu volume físico e melhorar a sua longevidade (velocidade e taxa de decaimento).

---

## 1. Modificações no Simulador (`main.py`)

### A. Novo Modelo de Pluma Linear
* **O que foi feito**: Criou-se a classe `Linear_MeanderModel` que retorna uma velocidade de meandro vertical constante e igual a `0.0`. Isto garante que a pluma viaje em linha reta (linear) de acordo com o vento.
* **Onde está no código**: [main.py (Linhas ~120-130)](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py#L120-L130).

### B. Integração na Física da Pluma (`FilamentPlume`)
* **O que foi feito**: 
  1. Atualizou-se o construtor da classe `FilamentPlume` para reconhecer `"linear"` e instanciar o modelo linear.
  2. Atualizou-se o método `update` para passar o novo parâmetro `"puff_expansion_rate"` (Taxa de Expansão) para a função Numba.
* **Onde está no código**: 
  * Construtor: [main.py (Linhas ~270-290)](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py#L270-L290).
  * Atualização da Física: [main.py (Linhas ~342-350)](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py#L342-L350).

### C. Atualização da Função JIT-compilada (`update_puffs_numba`)
* **O que foi feito**: Adicionou-se o parâmetro `puff_expansion_rate` à assinatura da função otimizada por Numba. O cálculo do raio do puff (`puffs_array[i, 2]`) passou a usar este parâmetro em vez do valor fixo de `0.2`.
* **Onde está no código**: [main.py (Linhas ~20-47)](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py#L20-L47).

### D. Novos Valores por Defeito (`DEFAULT_CONFIG`)
* **O que foi feito**: Ajustou-se as configurações padrão da pluma para torná-la maior e mais duradoura:
  * `"puff_expansion_rate"`: `0.35` (aumenta a largura/volume da pluma).
  * `"puff_decay_rate"`: `0.02` (reduz a decaída, fazendo com que as partículas durem mais).
  * `"mean_wind_velocity"`: `(1.5, 0)` (aumenta a velocidade do vento no eixo X).
* **Onde está no código**: [main.py (Linhas ~1665-1715)](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py#L1665-L1715).

---

## 2. Modificações no Dashboard (`dashboard.py`)

### A. Retorno do Elemento na Função `criar_campo`
* **O que foi feito**: A função utilitária `criar_campo` foi ajustada para retornar a referência do widget `Entry` de Tkinter. Isto permitiu controlar o seu estado (ativo/inativo) dinamicamente.
* **Onde está no código**: [dashboard.py (Linhas ~174-196)](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/dashboard.py#L174-L196).

### B. Adição de Novos Campos no Separador "Configurações (JSON)"
* **O que foi feito**: Foram criados novos campos para ajustar a física da pluma a partir do painel visual:
  1. **Modelo de Pluma** (Combobox): Permite selecionar entre `"Sinusoidal"` e `"Linear"`.
  2. **Taxa de Expansão (Volume)** (`puff_expansion_rate`).
  3. **Taxa de Decaimento (Decaída)** (`puff_decay_rate`).
* **Onde está no código**: [dashboard.py (Linhas ~198-225)](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/dashboard.py#L198-L225).

### C. Comportamento Dinâmico dos Campos
* **O que foi feito**: Adicionou-se a função interna `atualizar_estado_campos`. Quando o utilizador seleciona o modelo **Linear**, os campos de *Amplitude da Onda* e *Comprimento da Onda* ficam automaticamente desativados (cinzentos), voltando a ficar ativos se for selecionada a opção **Sinusoidal**.
* **Onde está no código**: [dashboard.py (Linhas ~227-235)](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/dashboard.py#L227-L235).

### D. Gravação das Configurações
* **O que foi feito**: O método `salvar_configuracoes` mapeia a escolha visual do utilizador para a chave JSON correspondente (`"sum_of_sinusoids"` ou `"linear"`) e salva-a diretamente no `config.json`.
* **Onde está no código**: [dashboard.py (Linhas ~240-265)](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/dashboard.py#L240-L265).

---

## 3. Configurações Atualizadas (`config.json`)

* **O que foi feito**: Atualizou-se o ficheiro de configuração central com os novos parâmetros otimizados para uma pluma de maior volume, maior alcance e maior longevidade:
  * `"puff_decay_rate"`: `0.02`
  * `"puff_expansion_rate"`: `0.35`
  * `"mean_wind_velocity"`: `[1.5, 0.0]`
* **Onde está no código**: [config.json](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/config.json).
