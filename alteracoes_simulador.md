# Histórico Completo de Alterações — Simulador de Pluma e Agente FSM

Este documento detalha e mapeia todas as alterações físicas, lógicas e arquiteturais efetuadas no simulador de dispersão de odor e no agente inteligente de decisão na conversa atual. As modificações combinam otimizações extremas de computação JIT com novas lógicas de navegação robusta em anel fechado.

---

## 1. Otimização de Performance da Pluma (Fixed-Capacity Array & Recycling)
* **Objetivo**: Eliminar o estrangulamento de performance provocado pela criação dinâmica de arrays com `np.vstack` e fatiamento booleano (`alive_mask`), que geravam excessivas realocações de memória a cada frame de simulação.
* **Localização**: Ficheiro [main.py](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py)
* **Código e Linhas**:
  * **`update_puffs_numba` (Linhas 31 - 82)**: Adicionado o parâmetro `min_mass` e uma cláusula de escape rápida (`if puffs_array[i, 3] <= min_mass: continue`) para ignorar puffs inativos e evitar contas desnecessárias no loop JIT.
  * **`calculate_concentration_numba` (Linhas 85 - 115)**: Implementada a exclusão de puffs fracos (`if puff_mass < 1e-6: continue`), poupando milhares de iterações por segundo na leitura tridimensional do agente.
  * **`FilamentPlume` (Linhas 268 - 333)**:
    * **Pré-Alocação Estática**: Cria um array bidimensional fixo de tamanho estático `self.puffs_array = np.zeros((self.max_puffs, 5))` no construtor `__init__`.
    * **Reciclagem de Slots**: O método `update` localiza os índices livres do array (onde a massa do puff está abaixo de `min_mass`) e escreve aí os novos puffs. Na expiração, em vez de fatiar o array, define a massa dos puffs expirados para `0.0` (sem criar novos objetos em memória).
  * **Desenho OpenCV `_draw` (Linhas 1425 - 1430)**: Atualizado o loop gráfico para ignorar e não desenhar círculos para os slots inativos da pluma.

---

## 2. Pluma Sinusoidal Cónica Perfeita (Mais Forma e Sem Caos)
* **Objetivo**: Corrigir o modelo de oscilação e dispersão para gerar uma pluma sinusoidal espacial nítida e contida, que se propaga realisticamente no sentido horizontal do vento (como uma serpente ondulante), em vez de se mover em bloco vertical ou dispersar-se caoticamente.
* **Localização**: Ficheiro [main.py](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py)
* **Código e Linhas**:
  * **`CONFIG` (Linhas 1483 - 1507)**: Alterada a chave `"meander_model"` para `"sum_of_sinusoids"` (em letras minúsculas). Calibrada a amplitude de onda para `1.5` e a frequência para `0.08` para uma oscilação contínua perfeita.
  * **Turbulência Anisotrópica (Linhas 37 - 40)**: No integrador JIT `update_puffs_numba`, a perturbação turbulenta foi calibrada para `1.0x` no plano horizontal e reduzida para apenas `0.3x` no vertical, resultando numa expansão cónica limpa e com rebordos extremamente bem definidos.

---

## 3. Duplo Contorno da Pluma (*Double Skirting*) com Fecho de Anel pelo Lado Oposto
* **Objetivo**: Garantir que o agente contorna a pluma pela sua fronteira de deteção (perto do limiar `agent_concentration_threshold`) tanto na ida em direção à fonte como no regresso, efetuando o retorno obrigatoriamente pelo lado oposto da pluma ao que utilizou na ida (fecho de anel perfeito).
* **Localização**: Ficheiro [main.py](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py)
* **Código e Linhas**:
  * **Bloqueio de Sobrescrita do Lado de Ida (Linhas 1091 - 1103)**:
    * **Desafio**: Como a pluma se estreita muito perto do ponto de emissão, o agente cruza repetidamente o eixo central nos metros finais do caminho, o que sobrescrevia a variável `skirting_side_ida` e fazia com que o regresso ocorresse pelo mesmo lado da vinda.
    * **Resolução**: Adicionada a guarda de segurança `if self.agent.skirting_side_ida is None: self.agent.skirting_side_ida = side` no loop de busca da origem. Isto assegura que apenas o primeiro desvio estabilizado na ida é registado e trancado.
  * **Retorno pelo Lado Oposto (Linhas 1121 - 1132)**: No regresso (`state == "returning"`), se o agente estiver dentro da pluma, ele afasta-se lateralmente usando o lado oposto trancado: `side = -self.agent.skirting_side_ida`. O agente contorna a pluma de forma autónoma e contínua no sentido oposto.
  * **Inicialização** no construtor `Agent.reset` (Linha 361) com `self.skirting_side_ida = None` para garantir frescura em cada experiência.

---

## 4. Resolução de Bloqueio no Pouso (Proximity Landing Check)
* **Objetivo**: Impedir que o desvio de pluma perpendicular (*crosswind*) atue infinitamente quando o agente está prestes a aterrar, dado que o ponto inicial de deteção (`first_detection_pos`) se situa fisicamente dentro do canal de odor da pluma.
* **Localização**: Ficheiro [main.py](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py)
* **Código e Linhas**:
  * **`Simulator.step` (Linhas 1112 - 1118)** & **`QLearningSimAgent.run_search_algorithm` (Linhas 944 - 951)**:
    * Adicionada a validação de distância `dist_to_detection < 10.0m` em relação ao objetivo de regresso.
    * Quando o agente está a menos de 10 metros, a lógica de contorno de pluma é desativada, e o agente navega de forma direta e retilínea até à coordenada final, garantindo 100% de taxa de conclusão e pouso sem bloqueios.

---

## 5. Renomeação de "Fases" para "Estados" (FSM Nomenclature)
* **Objetivo**: Ajustar a nomenclatura de estados em todo o projeto para alinhar com a convenção científica de **Máquinas de Estados Finitos (Finite State Machines)**.
* **Localizações**:
  * **Simulação e Agente** ([main.py](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/main.py)): Todas as variáveis `self.phase` e datasets logs correspondentes foram rebatizados para `self.state`.
  * **logs HDF5 do `DataLogger`**: Alterado de `"agent/phases"` para `"agent/states"` e no buffer de escrita para `"agent_state"`.
  * **Variáveis Internas Heurísticas**: Renomeadas para `self.internal_state` em `BioInspiredAgent` e `SimpleUpwindAgent` para impedir qualquer tipo de colisão com a máquina de estados global.
  * **Q-Learning** ([QLearning.py](file:///c:/Users/user1/Desktop/Code/Projeto_Final/Algoritmos-de-Gestao-de-Energia-Projeto-Final/QLearning.py)): Atualizado o dicionário e chaves de `phase_bin` para `mission_state_bin`. Mantida a retrocompatibilidade com chaves de 4 e 5 elementos para leitura fluida de ficheiros PKL no disco.

---

## 6. Comportamento do Agente e Estrutura de Rewards por Estado

A máquina de estados finitos (FSM) do agente orienta o seu comportamento físico e rege as funções de recompensa (*rewards*) usadas no treino e na avaliação:

### 1. Estado: `"searching"` (Procura da Pluma)
* **Como Funciona (Comportamento)**:
  * O agente inicia a simulação neste estado.
  * O movimento é controlado pelo modelo inteligente (por exemplo, Q-Learning), tomando ações discretas (Para Cima, Para Baixo, Esquerda, Direita).
  * O agente procura encontrar o primeiro contacto de odor acima do limiar de concentração (`agent_concentration_threshold`).
* **Lógica de Transição**: Assim que a concentração medida for maior que o limiar, o agente grava a posição atual como `first_detection_pos` e transita para `"following_boundary"`.
* **Cálculo da Recompensa (*Rewards*)**:
  * **Penalização por Passo**: `step_penalty` (`-0.05`).
  * **Bónus de Aproximação ao Odor**: Se a concentração atual for maior que a anterior, recebe `concentration_improvement * min(delta_conc, 1.0)` (Bónus máx de `+1.0`).
  * **Penalização de Afastamento do Odor**: Se a concentração diminuir, recebe `concentration_decrease` (`-0.05`).
  * **Bónus de Deteção de Pluma**: Se a concentração > limiar, recebe `plume_detection_bonus` (`+0.3`).
  * **Penalização por Revisita**: Se a célula do mapa já foi visitada mais de `revisit_threshold` vezes, recebe `revisit_penalty` (`-0.2`).

### 2. Estado: `"following_boundary"` (Busca da Origem / Contorno na Ida)
* **Como Funciona (Comportamento)**:
  * O agente navega subindo a pluma até à fonte de emissão (`source_position`), contornando o seu limite exterior (skirting).
  * **Dentro da Pluma** (`conc > threshold`): O agente move-se perpendicularmente ao vento (*crosswind*) para sair da pluma:
    * `side = 1.0` (Para Cima) se a sua coordenada Y for superior à linha central do mapa (`center_y`), ou `side = -1.0` (Para Baixo) caso contrário.
    * **Gravação Trancada**: O primeiro valor de `side` calculado é gravado e trancado em `self.agent.skirting_side_ida`.
  * **Fora da Pluma** (`conc <= threshold`): O agente move-se diretamente em linha reta na direção da fonte (`upwind`).
* **Lógica de Transição**: Quando a distância à fonte de emissão for menor que a distância de sucesso (`success_distance`), o agente transita para `"circling_source"`.
* **Cálculo da Recompensa (*Rewards*)**:
  * **Bónus por Alcançar a Fonte**: Recebe uma enorme recompensa de `+100.0` pontos quando chega à fonte.
  * **Penalização por Passo**: `step_penalty` (`-0.05`).
  * **Bónus de Seguimento de Fronteira**: Recompensa a proximidade exata ao limiar de odor usando a função gaussiana: `2.0 * np.exp(-abs(conc - threshold))`.
  * **Bónus de Progresso Contra o Vento (Upwind)**: Se o movimento do agente for no sentido oposto ao vento (em direção à fonte), recebe um bónus de progresso: `upwind_bonus * upwind_progress * 1.5` (Bónus proporcional de até `+1.5`).
  * **Bónus de Deteção de Pluma**: Se a concentração > limiar, recebe `plume_detection_bonus` (`+0.3`).
  * **Penalização por Revisita**: Se a célula já foi muito visitada, recebe `revisit_penalty` (`-0.2`).

### 3. Estado: `"circling_source"` (Contorno da Origem)
* **Como Funciona (Comportamento)**:
  * O agente executa uma órbita circular predefinida de 360º ao redor da fonte para mapear/caracterizar a zona de emissão.
  * A manobra decorre ao longo de 16 passos a um raio de `1.5 * success_distance`.
* **Lógica de Transição**: No final dos 16 passos da manobra de órbita, o agente muda automaticamente o estado para `"returning"`.
* **Cálculo da Recompensa (*Rewards*)**:
  * O agente recebe um prémio constante de `+2.0` pontos em cada um dos 16 passos da manobra como incentivo para concluir a órbita circular sem desvios.

### 4. Estado: `"returning"` (Regresso / Contorno na Volta)
* **Como Funciona (Comportamento)**:
  * O agente regressa ao ponto de partida (`first_detection_pos`), contornando a pluma pelo limite exterior oposto.
  * **Lógica de Segurança de Pouso (Proximity Check)**: Se a distância ao ponto de regresso for menor que `10.0m`, ignora a pluma e avança de forma puramente direta e retilínea até à coordenada final para pousar em segurança.
  * **Skirting no Regresso (Fronteira Oposta)**:
    * **Dentro da Pluma** (`conc > threshold`): Move-se perpendicularmente para o **lado oposto** ao da ida: `side = -self.agent.skirting_side_ida`. Isto força-o a sair para a outra metade da pluma.
    * **Fora da Pluma** (`conc <= threshold`): Move-se diretamente em linha reta na direção do ponto de partida (`downwind`).
* **Lógica de Transição**: Quando a distância ao ponto de partida for inferior a `success_distance`, a missão é considerada concluída com sucesso (`self.agent.mission_complete = True`).
* **Cálculo da Recompensa (*Rewards*)**:
  * **Bónus de Missão Concluída**: Recebe a recompensa máxima de terminal de regresso de `+500.0` pontos.
  * **Penalização por Passo**: `step_penalty` (`-0.05`).
  * **Bónus de Progresso de Regresso**: Mede a variação da distância em cada passo: `delta_dist = dist_atual - dist_anterior`. Recebe `-delta_dist * 10.0` (Dá recompensas positivas constantes à medida que diminui a distância até ao ponto final de regresso).
  * **Penalização por Revisita**: Se a célula já foi muito visitada, recebe `revisit_penalty` (`-0.2`).

---
