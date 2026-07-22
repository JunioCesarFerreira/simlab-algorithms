# A fórmula de ruína do jogador e seus elementos em cada método

Após a unificação, **todos os métodos usam a mesma fórmula** como estimador. Cada
método 1–5 serve apenas como uma **base para direcionar os BBs** (qual conjunto de
genes é bloco e qual é o alelo correto $H^*$); o cálculo da população é sempre

$$
\widehat{n}_u \;=\; -\ln(\alpha)\,2^{k_u-1}\,
\frac{\widehat{\sigma}_{BB,u}\sqrt{2m}}{\widehat{d}_u},
\qquad k_u=1,\qquad
\widehat N=\max_u\lceil \widehat n_u\rceil .
$$

Núcleo único: [methods/bb_core.py](methods/bb_core.py).

O **ganho marginal** de cada gene $u$ na amostra $r$ é

$$
\Delta_u^{(r)} = F(H_u^*,\,x_{-u}^{(r)}) - F(H_u^L,\,x_{-u}^{(r)}),
$$

onde $H_u^*$ é o alelo correto do gene $u$, $H_u^L$ seu complemento, e
$x_{-u}^{(r)}\sim\text{Bernoulli}(0{,}5)$ é um contexto aleatório ($R=60$
amostras). O sinal $\widehat d_u$ e o ruído $\widehat\sigma_{BB,u}$ da fórmula
são, respectivamente, a média e o desvio-padrão de $\{\Delta_u^{(r)}\}$.

## Elementos (idênticos em todos os métodos)

| Símbolo | Significado | Como se obtém |
|---|---|---|
| $\alpha$ | prob. de fracasso aceita | fixo, $0{,}05$ |
| $k_u$ | **ordem** do bloco | $1$ — cada gene indispensável é um bloco (P2 não tem engano Trap-$k$) ⟹ $2^{k_u-1}=1$ |
| $m$ | nº de blocos **vinculantes** | nº de genes que passam no teste de significância |
| $\widehat\sigma_{BB,u}$ | **ruído** | $\operatorname{sd}_r(\Delta_u^{(r)})$ |
| $\widehat d_u$ | **sinal** | $\operatorname{mean}_r(\Delta_u^{(r)})$ |

com $\Delta_u^{(r)} = F(H_u^*,x_{-u}^{(r)}) - F(H_u^L,x_{-u}^{(r)})$, onde $H_u^*$ é o
alelo correto do gene, $H_u^L$ seu complemento, e $x_{-u}^{(r)}\sim\text{Bernoulli}(0{,}5)$
($R=60$ complementos). Gene é **vinculante** (vira bloco) só se
$\widehat d_u > 1{,}645\,\widehat\sigma_{BB,u}/\sqrt R$ (teste unilateral 95%);
genes substituíveis ($d\approx0$) são descartados.

## O que muda entre os métodos: **só a fonte dos BBs**

($m$ = blocos vinculantes, após o filtro de significância.)

| Método | Fonte que direciona os BBs (gene\_set) | $H_u^*$ | $k_u$ | $m$ | $\widehat N$ |
|---|---|---|:---:|---:|---:|
| **M1** inferência | genes ligados nas **elites do AG** (top 5% por $F$, freq.\ $\ge0{,}8$); 18 nomeados | ON | 1 | 18 | **56** |
| **M2** adjacência | candidatos com **cobertura** $\text{cov}(u)\ge\theta TM$; 16 nomeados | ON | 1 | 10 | **51** |
| **M3** roteamento | candidatos **críticos de roteamento** $\text{route}(u)\ge\phi$; 9 nomeados | ON | 1 | 9 | **26** |
| **M4** MILP | genes **instalados no ótimo MILP** (mín. relés); 18 nomeados | ON (alelo MILP) | 1 | 14 | **73** |
| **M5** união | $\text{cov}\!\ge\!\theta TM \ \vee\ \text{route}\!\ge\!\phi$; 22 nomeados | ON | 1 | 18 | **75** |

Cada método é, portanto, um **detector de BBs**; o sinal/ruído ($d,\sigma_{BB}$) e a
fórmula são os mesmos. As diferenças de $\widehat N$ vêm só de *quais* genes cada
fonte aponta (e, portanto, de qual gene tem o pior $\sigma_{BB}/d$).

## Fusão (5 fontes)

Combinando as cinco fontes ([methods/combined_estimator.py](methods/combined_estimator.py)):
média ponderada por precisão **37** (IC [29, 44]); envelope conservador **75**.

> Cálculo das fontes: M1 dinâmica do AG ([methods/method1_inference.py](methods/method1_inference.py));
> M2 cobertura ([methods/method2_adjacency.py](methods/method2_adjacency.py));
> M3 roteamento ([methods/method3_routing.py](methods/method3_routing.py));
> M4 ótimo MILP ([methods/method4_milp.py](methods/method4_milp.py));
> M5 união ([methods/method5_building_blocks.py](methods/method5_building_blocks.py)).
