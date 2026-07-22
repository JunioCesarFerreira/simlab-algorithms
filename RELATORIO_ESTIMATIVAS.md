# Relatório — Como as populações foram estimadas e o quanto isso desviou da proposta original

**Instância de referência:** `ind2` (N = 64 candidatos, M = 4 móveis, R = 50, T = 181, sink na origem).
**α = 0,05** em todos os métodos.

---

## 1. A proposta original

O ponto de partida (documentado em [estimando-população-p2.md](estimando-população-p2.md)) era a
fórmula de **sizing por ruína do jogador (Harik et al.)** aplicada por *building block*:

$$
\widehat{n}_i \;\approx\; -\ln(\alpha_i)\,2^{k_i-1}\,
\frac{\widehat{\sigma}_{BB,i}\sqrt{2m}}{\widehat{d}_i},
\qquad
\widehat{n} = \max_i \lceil \widehat{n}_i \rceil .
$$

Ingredientes exigidos pela fórmula:

| Símbolo | Significado | Como se obtém |
|---|---|---|
| $k_i$ | tamanho do bloco construtivo $i$ | partição semântica do cromossomo |
| $\widehat{d}_i$ | **sinal**: vantagem média de fitness da instância correta do bloco sobre a competidora enganosa | diferenças inter-amostrais $F^\*-F_{local}$ |
| $\widehat{\sigma}_{BB,i}$ | **ruído** de fitness do bloco | desvio das mesmas diferenças |
| $2^{k_i-1}$ | fator de deriva genética (pior caso) | direto de $k_i$ |
| $m$ | nº de blocos construtivos | da partição |

A pergunta deixada em aberto no documento era exatamente **"o que é um bloco construtivo no P2?"**.

### Esta fórmula *foi* implementada — e continua no repositório

Ela está fielmente codificada em
[pop-estimator/p2_population_estimator/estimator.py](pop-estimator/p2_population_estimator/estimator.py)
(`estimate_uniform` / `estimate_bernoulli`), alimentada por
[statistics.py](pop-estimator/p2_population_estimator/statistics.py) (`d_hat`, `sigma_BB_hat`) e por uma
partição em 8 blocos via grade + heurísticas `H*` (greedy estrutural) vs `H_local` (competidor enganoso).

**Resultado dela em `ind2`:** $\widehat{n} = \mathbf{2839}$ (bloco mais difícil: block 4, k=9, σ=0,123, d=0,141).
Veja [results/method1/gambler_ruin_baseline/population_estimate_result.json](results/method1/gambler_ruin_baseline/population_estimate_result.json).

**O ponto central deste relatório:** essa estimativa fiel à proposta **não é** o resultado do trabalho.
Ela foi rebaixada à condição de *"baseline de pior caso"* e descrita no artigo como "duas ordens de
grandeza grande demais". O artigo e o diretório [methods/](methods/) construíram **quatro estimadores
inteiramente novos** que **não usam a fórmula proposta**.

---

## 2. O que efetivamente foi feito — os quatro métodos

Todos vivem em [methods/](methods/) e produzem um registro `NHat(n_hat, sigma, ...)`. Nenhum deles usa
$2^{k_i-1}$, $\widehat{d}_i$ ou $\widehat{\sigma}_{BB,i}$ no sentido da proposta.

### Método 1 — "inferência por retornos decrescentes" → $\widehat N_1 = 22$
Arquivo: [methods/method1_inference.py](methods/method1_inference.py).

1. Roda o GA num *ladder* de populações $n\in\{4,8,16,32,64,128,256\}$ × 5 seeds e mede o fitness
   convergido $F^\*(n)$ (de [results/ga_runs/ga_summary.csv](results/ga_runs/ga_summary.csv)).
2. Ajusta a curva de saturação $F^\*(n) = F_\infty - A\,e^{-n/\tau}$ (grid em $\tau$ + mínimos quadrados).
   Obteve $F_\infty=0{,}970$, $A=0{,}013$, $\tau=7{,}2$.
3. Estimador: **$\widehat N_1 = \lceil \tau\ln(1/\varepsilon)\rceil$** com $\varepsilon=0{,}05$ → **22** (IC 95% [15, 73]).

> **Desvio:** troca-se a teoria de blocos por um *fit* fenomenológico de "quando o ganho marginal de
> qualidade fica abaixo de 5%". $\sigma_{BB}$, $d$, $k$ e o $2^{k-1}$ desaparecem. Único resíduo da proposta:
> a fórmula gambler-ruin é carregada e impressa **ao lado**, apenas como baseline (2839).

### Método 2 — "dinâmica temporal da matriz de adjacência" → $\widehat N_2 = 24$
Arquivo: [methods/method2_adjacency.py](methods/method2_adjacency.py).

1. Monta $A(t)[u,v]=1$ se $\lVert p_u-p_v\rVert\le R$, acumula $A_{total}=\sum_t A(t)$.
2. Score de cobertura por candidato $\mathrm{cov}(u)=\sum_{m}A_{total}[u,m]$; conta os "indispensáveis"
   $C_\theta=\#\{u:\mathrm{cov}(u)\ge\theta\,TM\}$. Em $\theta=0{,}10$: $C_\theta=16$.
3. Estimador **inventado**: **$\widehat N_2 = \lceil C_\theta\,(1/\alpha)^{1/\bar k}\rceil$**, com $\bar k = N/8 = 8$ → **24**.

> **Desvio:** total. Não há ruína do jogador aqui. O fator $(1/\alpha)^{1/\bar k}$ é uma heurística *ad hoc*
> (substitui $-\ln\alpha\,2^{k-1}\sigma\sqrt{2m}/d$ por um multiplicador de uma única região crítica).
> "Bloco" sobrevive só como o número $\bar k$ dentro desse expoente; $\sigma_{BB}$ e $d$ não entram.

### Método 3 — "variante da matriz de roteamento" → $\widehat N_3 = 14$
Arquivo: [methods/method3_routing.py](methods/method3_routing.py).

Idêntico ao M2, mas troca *alcance* por *uso*: BFS a partir do sink em cada $t$, conta nós em caminhos
mínimos móvel→sink. $\mathrm{route}(u)=\text{node\_acc}[u]/(TM)$; críticos $|R_\phi|$ com $\phi=0{,}05$ → 9.
Mesmo estimador: **$\widehat N_3=\lceil |R_\phi|\,(1/\alpha)^{1/\bar k}\rceil$** → **14**.

> **Desvio:** mesmo do M2.

### Método 4 — "calibração via MILP" → $\widehat N_4 = 48$
Arquivo: [methods/method4_milp.py](methods/method4_milp.py).

1. Toma o ótimo de referência do sweep MILP: $r_{opt}=18$ relés.
2. Mede o gap do GA $\mathrm{gap}(n)=(r_{GA}(n)-r_{opt})/r_{opt}$ e ajusta $\mathrm{gap}(n)=g_\infty+G\,e^{-n/\tau}$
   ($g_\infty=0{,}076$, $G=0{,}40$, $\tau=16{,}8$).
3. Estimador: **$\widehat N_4=\lceil \tau\ln(G/(\tau_{gap}-g_\infty))\rceil$** com alvo $\tau_{gap}=0{,}10$ → **48** (IC [32, 72]).

> **Desvio:** total. É outra teoria (calibração contra ótimo exato), sem nenhum elemento da fórmula proposta.

### Combinação → consenso 25, envelope conservador 48
Arquivo: [methods/combined_estimator.py](methods/combined_estimator.py).

- Envelope conservador $\max_i\widehat N_i = 48$.
- Média ponderada por precisão ($w_i=1/\sigma_i^2$) = **25**.
- Fusão bayesiana gaussiana = **25** (CrI [14, 35]).

---

## 3. Quadro comparativo: proposta vs. o que foi entregue

| | Proposta original | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| Base teórica | ruína do jogador (Harik) | fit saturação de qualidade | contagem de cobertura | contagem de roteamento | calibração MILP |
| Usa $2^{k-1}$? | **sim** | não | não | não | não |
| Usa $\widehat\sigma_{BB}$ e $\widehat d$? | **sim** | não | não | não | não |
| Usa $\alpha$? | sim ($-\ln\alpha$) | não (usa $\varepsilon$) | sim, mas como $(1/\alpha)^{1/\bar k}$ | idem M2 | não |
| Conceito de "bloco" | central, semântico | ausente | só $\bar k=N/8$ no expoente | idem | ausente |
| $\widehat n$ em `ind2` | **2839** | 22 | 24 | 14 | 48 |

**Distância da proposta:** a estimativa fiel à fórmula dá **2839**; tudo o que o artigo apresenta como
resultado fica em **14–48** (fusão ≈ 25). Ou seja, o trabalho entregue está cerca de **100× abaixo** da
proposta original, e a fórmula citada por você só aparece como um número descartado ("baseline grande demais").

---

## 4. Estado no GitHub (o que está pendente)

`git status`: o ramo está **1 commit à frente de origin/main** e **todo o aparato novo está sem versionar**:

- não rastreados: `methods/`, `ga/`, `results/`, `paper/`, `requirements.txt`, `run_all.sh`, `REPO_MAP.md`;
- modificado: `README.md`.

Ou seja, **os quatro métodos divergentes e o artigo `paper/main.tex` ainda não foram commitados/enviados** —
são exatamente a "mudança de rota" em relação ao gambler-ruin que já estava no repositório (`pop-estimator/`).

---

## 5. Conclusão e opções

1. **A fórmula proposta não foi usada como estimador** — foi substituída por quatro heurísticas novas e
   relegada a baseline (2839). Os números do artigo (14–48) vêm de teorias diferentes.
2. O único elo remanescente é o fator $(1/\alpha)^{1/\bar k}$ de M2/M3, que **não** é a expressão da ruína do
   jogador, e o uso da partição em 8 blocos (herdada do `pop-estimator`).
3. A pergunta em aberto do documento original — "o que é um bloco construtivo no P2?" — foi **contornada**, não
   respondida: o `pop-estimator` a respondeu com grade+heurísticas (→2839); o artigo a abandonou.

**Se o objetivo é voltar à proposta**, o caminho é: definir blocos construtivos semânticos para P2 (p.ex.
regiões críticas de cobertura/roteamento como blocos, em vez de grade arbitrária), recomputar $\widehat d_i$
e $\widehat\sigma_{BB,i}$ por bloco e aplicar a fórmula — possivelmente investigando por que $2^{k-1}$ infla
tanto (blocos grandes, $k=9$) e se uma definição de bloco mais justa reduz o 2839 para a faixa estrutural.

---

## 6. Resolução (implementada)

A proposta foi recolocada como estimador comparável — **Método 5**
([methods/method5_building_blocks.py](methods/method5_building_blocks.py)). Resultado:

- **Definição de bloco construtivo em P2:** *ordem 1*. Cada candidato indispensável (crítico por cobertura
  $\cov\ge\theta TM$ **ou** por roteamento $\route\ge\phi$) é um bloco $\{u\}$ que deve permanecer ligado.
  P2 **não tem** engano Trap-$k$; sua dificuldade são muitos genes obrigatórios espalhados, não um bloco
  grande. Filtro de significância (teste unilateral 95%) descarta genes substituíveis ($d\approx 0$).
- **Mesma fórmula, ordem correta:** com $k_u=1$, $2^{k-1}=1$ e
  $\widehat n_u = -\ln(\alpha)\,\widehat\sigma_{BB,u}\sqrt{2m}/\widehat d_u$. Sobre $m=18$ blocos vinculantes
  (de 22 críticos): **$\widehat N_5 = 75$**, faixa $[18,75]$ — **mesma ordem de grandeza** de M1–M4.
- **Reconciliação:** o **2839** (grade) e o blow-up de componentes ($k{=}12\Rightarrow 6{,}3\times10^4$) são
  artefatos do $2^{k-1}$ sob ordem incorreta — não do sinal/ruído. A discrepância de ~100× era a hipótese de
  bloco, não a fórmula.
- **Fusão com 5 métodos:** média ponderada **29** (IC [19,38]), envelope conservador **75**.

O novo artigo em português, com o devido rigor matemático (dedução da ruína do jogador, definição de bloco,
os cinco métodos e a fusão), está em [paper/artigo_pt.tex](paper/artigo_pt.tex). Pipeline reprodutível
atualizado em [run_all.sh](run_all.sh) (passo 5b).
