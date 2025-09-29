# Pork-AI-Agents — monitor.py

Monitoramento de uma aplicação (Spring Boot Actuator) usando um time de agentes IA com CrewAI e um LLM via RouteLLM (Abacus.AI).

Este repositório contém um único script principal, `monitor.py`, que cria três agentes:

- Coletor: consulta endpoints do Actuator e agrupa os dados em um JSON.
- Analista: interpreta as métricas, aplica thresholds e gera um diagnóstico estruturado (JSON).
- Notificador: formata uma mensagem curta e acionável a partir da análise.

O LLM usado no script é configurado para falar com RouteLLM (Abacus.AI) via cliente compatível com OpenAI.

---

## Conteúdo

- `monitor.py` — script principal.
- `monitoring_logs.json` — (gerado) arquivo de logs locais em JSON com execuções do monitor.

---

## Requisitos

- Python 3.9+ recomendado
- Pacotes Python (exemplos): `crewai`, `requests`, `python-dotenv`, `openai` (usado como client HTTP para RouteLLM).

Instale com pip (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install crewai requests python-dotenv openai
```

Se preferir, crie um `requirements.txt` com essas dependências.

---

## Variáveis de ambiente (.env)

O `monitor.py` espera um arquivo `.env` com, pelo menos, as variáveis abaixo:

- OPENAI_API_KEY — chave de API usada pelo cliente OpenAI/RouteLLM (pode ser a chave do provedor que você estiver usando via RouteLLM)
- API_BASE_URL — base URL da API que contém os endpoints do Actuator (ex.: `https://minha-api.example.com`)
- MONITORING_API_KEY — chave usada para autenticação nas rotas de monitoramento da API (header `X-API-KEY`)

Opções configuráveis (com valores padrão mostrados no código):

- MEMORY_ALERT_MB — (default 700) limite em MB para gerar alerta de memória
- CPU_ALERT_PCT — (default 85.0) limite em % para CPU
- ERROR_RATE_SLO_PCT — (default 1.0) limite % de erro aceitável
- LATENCY_P95_SLO_MS — (default 800) limite em ms para p95 de latência

Exemplo de `.env`:

```text
OPENAI_API_KEY=s2_xxx-your-route-llm-key-xxx
API_BASE_URL=https://minha-api.example.com
MONITORING_API_KEY=xxxxxxxx-xxxx-xxxx
MEMORY_ALERT_MB=700
CPU_ALERT_PCT=85.0
ERROR_RATE_SLO_PCT=1.0
LATENCY_P95_SLO_MS=800
```

Observação importante sobre chaves OpenAI / litellm / variáveis de ambiente

- Se você tiver variáveis como `OPENAI_API_KEY` definidas que apontam para a OpenAI (ou outro serviço), algumas bibliotecas (litellm, openai, langchain) podem tentar autenticar automaticamente ao serem importadas. Isso causa erros como:

```
litellm.AuthenticationError: AuthenticationError: OpenAIException - Incorrect API key provided: s2_5aaac... You can find your API key at https://platform.openai.com/account/api-keys.
```

Para evitar conflitos e garantir que o script use explicitamente seu RouteLLM/Abacus endpoint:

- Verifique o `.env` e remova/ajuste chaves que não deveriam ser lidas (por exemplo, remova chaves OpenAI que pertençam a outro provedor).
- Se precisar manter múltiplas chaves no mesmo ambiente, carregue/limpe variáveis antes de importar clientes que usam essas variáveis. No `monitor.py` já carregamos `load_dotenv()` no topo, mas se você integrar esse script em um projeto maior, garanta que a limpeza de variáveis ocorra antes de qualquer `import` que inicialize clients.

---

## Como rodar

No PowerShell, a sequência mínima:

```powershell
# ative o virtualenv (se criado anteriormente)
.\.venv\Scripts\Activate.ps1

# coloque seu .env ao lado de monitor.py
python monitor.py
```

Saída esperada:

- Logs no console exibindo o progresso dos agentes.
- Um arquivo `monitoring_logs.json` com um registro da execução.

---

## Comportamento e limites

- O coletor tenta acessar vários endpoints do Spring Boot Actuator. Se um endpoint não existir, o coletor preenche `null` para aquele campo no JSON.
- O analisador converte métricas (bytes -> MB, frações -> percentuais) e compara com os thresholds definidos pelas variáveis de ambiente.
- A notificação condensa a saída em uma mensagem curta e acionável.

---

## Depuração do erro litellm/OpenAI que você encontrou

Causa provável:

- Uma variável de ambiente OPENAI_API_KEY (ou similar) contém uma chave inválida ou da OpenAI e está sendo lida por `openai` / `litellm` / `langchain` em algum ponto da inicialização, provocando a exceção.

Soluções práticas:

1. Garanta que o `.env` contenha a chave correta para RouteLLM (caso a operação passe por um proxy) e que não exista uma outra `OPENAI_API_KEY` no ambiente do sistema (ex.: variáveis do Windows). No PowerShell, verifique com:

```powershell
Get-ChildItem Env:OPENAI_API_KEY
```

Se estiver definida e for de outro provedor, remova temporariamente antes de executar:

```powershell
Remove-Item Env:OPENAI_API_KEY
# then run the script
python monitor.py
```

2. Se você precisa trabalhar com múltiplas chaves, modifique o código para limpar variáveis indesejadas antes de importar bibliotecas que criam clientes automaticamente. Exemplos de variáveis a revisar/remover: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_API_BASE`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `LITELLM_BASE_URL`.

3. Forçar o cliente do RouteLLM explicitamente (o `monitor.py` já cria `OpenAI(base_url=..., api_key=...)`). Contudo, se alguma outra importação (por bibliotecas de terceiros) já inicializar um cliente OpenAI usando a variável de ambiente, isso causará o erro. Solução: mover/limpar env vars antes dessas importações.

---

## Melhorias sugeridas

- Migrar logs para `logging` com rotação (`logging.handlers.RotatingFileHandler`).
- Tornar thresholds configuráveis por argumentos de linha de comando além do `.env`.
- Adicionar testes unitários para `ApiMonitoringTool` com `responses` ou `requests-mock`.
- Usar `requests.Session` com retry/backoff centralizado (o script já tem um retry simples na ferramenta).

---

## Próximos passos

Se quiser, posso:

- Gerar um `requirements.txt` e um `setup` rápido para execução.
- Aplicar as correções no `monitor.py` para limpar variáveis de ambiente antes das importações que inicializam clientes (evita o erro do litellm/OpenAI).
- Adicionar testes pytest para o `ApiMonitoringTool`.

Informe qual dessas opções prefere que eu implemente a seguir.

---

© Projeto Pork-AI-Agents — ferramentas de monitoramento com agentes de IA
