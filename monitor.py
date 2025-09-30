import os
import json
import time
import math
import requests
import schedule
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# ====
# Carrega variáveis de ambiente
# ====
load_dotenv()

# Variáveis essenciais
required_vars = ["OPENAI_API_KEY", "API_BASE_URL", "MONITORING_API_KEY"]
missing = [k for k in required_vars if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Variáveis ausentes no .env: {', '.join(missing)}")

# Configurações do Telegram (opcionais)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Variável para controlar updates já processados
last_update_id = 0

# Thresholds com defaults razoáveis (você pode ajustar no .env)
MEMORY_ALERT_MB = int(os.getenv("MEMORY_ALERT_MB", "700"))
CPU_ALERT_PCT = float(os.getenv("CPU_ALERT_PCT", "85.0"))    # %
ERROR_RATE_SLO_PCT = float(os.getenv("ERROR_RATE_SLO_PCT", "1.0"))  # %
LATENCY_P95_SLO_MS = int(os.getenv("LATENCY_P95_SLO_MS", "800"))

# ====
# Configuração RouteLLM (Abacus.AI)
# ====
client = OpenAI(
    base_url="https://routellm.abacus.ai/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

class RouteLLMWrapper:
    """
    Wrapper para usar RouteLLM com CrewAI.
    Simula um objeto LLM compatível com CrewAI/LangChain.
    """
    def __init__(self, model="gpt-4o-mini", temperature=0):
        self.model = model
        self.temperature = temperature

    def __call__(self, messages, **kwargs):
        return self._generate(messages, **kwargs)

    def invoke(self, messages, **kwargs):
        return self._generate(messages, **kwargs)

    def _generate(self, messages, **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", 1500),
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro na chamada LLM: {str(e)}"

llm = RouteLLMWrapper(model="gpt-4o-mini", temperature=0)

# ====
# Persistência de logs
# ====
LOG_FILE = "monitoring_logs.json"

def persist_data(entry: dict):
    try:
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        with open(LOG_FILE, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.truncate()
    except Exception as e:
        print(f"⚠️ Erro ao salvar log: {e}")

# ====
# Notificação via Telegram
# ====
def send_telegram_notification(message: str, analysis_data: dict = None):
    """
    Envia notificação via Telegram com métricas importantes.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram não configurado. Pulando notificação.")
        return False
    
    try:
        # Monta mensagem completa com métricas
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        telegram_message = f"🤖 Monitor API - {timestamp}\n\n"
        telegram_message += message
        
        # Adiciona métricas importantes se disponível
        if analysis_data and isinstance(analysis_data, dict):
            numbers = analysis_data.get("numbers", {})
            if numbers:
                telegram_message += "\n\n📊 Métricas Importantes:\n"
                
                if numbers.get("memory_used_mb"):
                    memory_pct = numbers.get("memory_used_pct", 0)
                    telegram_message += f"• Memória: {numbers['memory_used_mb']}MB ({memory_pct:.1f}%)\n"
                
                if numbers.get("cpu_usage_pct"):
                    telegram_message += f"• CPU: {numbers['cpu_usage_pct']:.1f}%\n"
                
                if numbers.get("http_error_rate_pct") is not None:
                    telegram_message += f"• Taxa de Erro: {numbers['http_error_rate_pct']:.2f}%\n"
        
        # Envia via API do Telegram (sem parse_mode para evitar erros)
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": telegram_message,
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        
        print("✅ Notificação enviada via Telegram")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao enviar Telegram: {e}")
        return False

# ====
# Escuta de comandos do Telegram
# ====
def listen_for_commands():
    """
    Faz polling no Telegram para ouvir comandos enviados pelo usuário.
    """
    global last_update_id
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"offset": last_update_id + 1, "timeout": 1}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if "result" in data and data["result"]:
            for update in data["result"]:
                last_update_id = update["update_id"]
                
                if "message" in update:
                    chat_id = update["message"]["chat"]["id"]
                    text = update["message"].get("text", "").strip()
                    username = update["message"]["from"].get("username", "Usuário")

                    if str(chat_id) != str(TELEGRAM_CHAT_ID):
                        continue  # ignora outros chats

                    print(f"📱 Comando recebido de @{username}: {text}")

                    # Processa os comandos
                    if text == "/status":
                        send_telegram_notification("✅ Bot está rodando normalmente!")
                    
                    elif text == "/run":
                        send_telegram_notification("🚀 Executando monitoramento sob comando...")
                        run_monitoring()
                    
                    elif text == "/help":
                        help_msg = (
                            "📖 Comandos disponíveis:\n\n"
                            "/status - Ver se o bot está online\n"
                            "/run - Executar monitoramento agora\n"
                            "/logs - Ver últimos logs\n"
                            "/config - Ver configurações atuais\n"
                            "/help - Mostrar esta ajuda"
                        )
                        send_telegram_notification(help_msg)
                    
                    elif text == "/logs":
                        try:
                            if os.path.exists(LOG_FILE):
                                with open(LOG_FILE, "r", encoding="utf-8") as f:
                                    logs = json.load(f)
                                if logs:
                                    last_log = logs[-1]
                                    timestamp = last_log.get("timestamp", "N/A")
                                    log_msg = f"📋 Último log:\n\n⏰ {timestamp}\n\n"
                                    if "error" in last_log:
                                        log_msg += f"❌ Erro: {last_log['error']}"
                                    else:
                                        log_msg += "✅ Execução bem-sucedida"
                                    send_telegram_notification(log_msg)
                                else:
                                    send_telegram_notification("📋 Nenhum log encontrado ainda.")
                            else:
                                send_telegram_notification("📋 Arquivo de log não existe ainda.")
                        except Exception as e:
                            send_telegram_notification(f"❌ Erro ao ler logs: {e}")
                    
                    elif text == "/config":
                        config_msg = (
                            f"⚙️ Configurações atuais:\n\n"
                            f"• Memória Alert: {MEMORY_ALERT_MB}MB\n"
                            f"• CPU Alert: {CPU_ALERT_PCT}%\n"
                            f"• Error Rate SLO: {ERROR_RATE_SLO_PCT}%\n"
                            f"• Latency SLO: {LATENCY_P95_SLO_MS}ms\n"
                            f"• Telegram: {'✅ Configurado' if TELEGRAM_BOT_TOKEN else '❌ Não configurado'}"
                        )
                        send_telegram_notification(config_msg)
                    
                    else:
                        send_telegram_notification(
                            f"❓ Comando não reconhecido: {text}\n\n"
                            "Digite /help para ver os comandos disponíveis."
                        )

    except Exception as e:
        print(f"⚠️ Erro ao ouvir comandos: {e}")

# ====
# Ferramenta de coleta com retry/backoff
# ====
class ApiMonitoringTool(BaseTool):
    """
    Ferramenta para requisições aos endpoints de monitoramento da API.
    Usa API_BASE_URL e MONITORING_API_KEY do .env.
    """
    name: str = "API Monitoring Tool"
    description: str = "Faz GET em endpoints de monitoramento e retorna o texto da resposta."

    def _run(self, endpoint: str) -> str:
    return self._get_with_retry(endpoint)

    def _get_with_retry(self, endpoint: str, retries: int = 3, timeout: int = 15) -> str:
        api_base_url = os.getenv("API_BASE_URL")
        api_key = os.getenv("MONITORING_API_KEY")

        if not api_base_url or not api_key:
            return "Erro: Variáveis API_BASE_URL ou MONITORING_API_KEY não definidas."

        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key
        }

        url = f"{api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        last_error = None
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                try:
                    return json.dumps(response.json(), ensure_ascii=False)
                except Exception:
                    return response.text
            except requests.exceptions.RequestException as e:
                last_error = e
                # backoff exponencial simples
                sleep_secs = 2 ** attempt
                time.sleep(sleep_secs)
        return f"Erro ao acessar {endpoint}: {last_error}"

api_tool = ApiMonitoringTool()

# ====
# Agentes aprimorados
# ====

# Agente 1: Coletor - agora coleta múltiplos endpoints e retorna JSON estrito
collector_prompt = f"""
Você é um coletor de métricas de uma aplicação Spring Boot (Actuator).
Use a ferramenta "API Monitoring Tool" para buscar e AGRUPAR os dados dos seguintes endpoints, retornando EXATAMENTE um JSON único, sem texto adicional:

- /actuator/health
- /actuator/info
- /actuator/loggers
- /actuator/httptrace (ou /actuator/httpexchanges, se o seu ambiente usar esse nome)
- /actuator/metrics/jvm.memory.used
- /actuator/metrics/jvm.memory.max
- /actuator/metrics/process.cpu.usage
- /actuator/metrics/http.server.requests

Regras:
- Caso algum endpoint não exista, coloque null no respectivo campo.
- Não escreva nenhum texto fora do JSON final.
- O JSON final deve ter este formato (exemplo de chaves; preencha com os dados reais obtidos):
{{
  "health": <obj ou null>,
  "info": <obj ou null>,
  "loggers": <obj ou null>,
  "httptrace": <obj ou null>,
  "metrics": {{
    "jvm.memory.used": <obj ou null>,
    "jvm.memory.max": <obj ou null>,
    "process.cpu.usage": <obj ou null>,
    "http.server.requests": <obj ou null>
  }}
}}
"""

data_collector_agent = Agent(
    role='Coletor de Métricas da API',
    goal='Coletar dados vitais dos endpoints de saúde, info, loggers, httptrace e métricas.',
    backstory='Especialista em requisições HTTP e normalização de respostas.',
    tools=[api_tool],
    verbose=True,
    llm=llm,
    allow_delegation=False
)

collect_data_task = Task(
    description=collector_prompt + '\nExecute as chamadas e retorne somente o JSON final.',
    expected_output='JSON único com as chaves: health, info, loggers, httptrace, metrics{...}',
    agent=data_collector_agent
)

# Agente 2: Analisador - aplica thresholds, calcula percentuais/MB e sugere ações
analyzer_prompt = f"""
Você é um analista de observabilidade. Receberá um JSON bruto com as respostas do Actuator e deve produzir um diagnóstico estruturado.
Siga as regras:

1) Converta quando possível:
   - jvm.memory.used.value (bytes) -> MB (arredonde para inteiro) e percentual de jvm.memory.max (se disponível).
   - process.cpu.usage (0..1) -> % (duas casas).
   - Em http.server.requests, se houver contadores por status, estime taxa de erro (5xx / total) em %.

2) Compare com thresholds (do ambiente):
   - MEMORY_ALERT_MB = {MEMORY_ALERT_MB} MB
   - CPU_ALERT_PCT = {CPU_ALERT_PCT} %
   - ERROR_RATE_SLO_PCT = {ERROR_RATE_SLO_PCT} %
   - LATENCY_P95_SLO_MS = {LATENCY_P95_SLO_MS} ms (se houver latências ou max disponíveis)

3) Produza saída EXATAMENTE neste formato JSON (sem texto fora do JSON):
{{
  "summary": "texto curto",
  "severity": "INFO|WARN|ALERT|CRITICAL",
  "findings": [
    {{
    "area": "HEALTH|MEMORY|CPU|HTTP|DB|LOGGERS|TRACE|RELEASE",
    "status": "OK|WARN|ALERT",
    "metric": "nome_da_metrica",
    "value": "valor_legível",
    "threshold": "limite_legível",
    "details": "explicação curta"
    }}
  ],
  "actions": [
    "ação recomendada 1",
    "ação recomendada 2"
  ],
  "numbers": {{
    "memory_used_mb": <int ou null>,
    "memory_max_mb": <int ou null>,
    "memory_used_pct": <float ou null>,
    "cpu_usage_pct": <float ou null>,
    "http_error_rate_pct": <float ou null>
  }}
}}

4) Priorize problemas reais (ex.: memória acima do limite, CPU alta, erro 5xx alto).
5) Se alguma métrica não existir, preencha campos numéricos com null e ajuste findings/status conforme.
"""

data_analyzer_agent = Agent(
    role='Analista de Saúde da Aplicação',
    goal='Interpretar dados coletados, detectar anomalias e recomendar ações.',
    backstory='Engenheiro de confiabilidade com foco em SLOs e performance.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

analyze_data_task = Task(
    description=analyzer_prompt + "\nEntrada: use o JSON retornado pelo coletor.",
    expected_output='JSON estruturado com summary, severity, findings, actions, numbers',
    agent=data_analyzer_agent,
    context=[collect_data_task]
)

# Agente 3: Notificador - mensagem curta e acionável
notifier_prompt = """
Você é um gerador de alertas. A partir do JSON de análise (summary, severity, findings, actions),
gere uma notificação curta, objetiva e acionável para a equipe.

Regras:
- Título curto com o status (ex.: [ALERT] Memória acima do limite).
- 1 a 2 parágrafos no máximo, em português, mencionando números-chave (memória/CPU/erros).
- Lista de 2 a 4 ações objetivas (bullets curtos).
- Sem verborragia. Texto final limpo (sem markdown pesado).
"""

notification_agent = Agent(
    role='Gerador de Alertas',
    goal='Converter a análise técnica em mensagem clara e acionável.',
    backstory='Especialista em comunicação técnica para times de engenharia.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

notify_task = Task(
    description=notifier_prompt + "\nEntrada: use o JSON estruturado do analisador.",
    expected_output='Mensagem final curta e acionável para a equipe.',
    agent=notification_agent,
    context=[analyze_data_task]
)

# ====
# Orquestração
# ====
api_monitoring_crew = Crew(
    agents=[data_collector_agent, data_analyzer_agent, notification_agent],
    tasks=[collect_data_task, analyze_data_task, notify_task],
    process=Process.sequential,
    verbose=True
)

# ====
# Função principal de monitoramento
# ====
def run_monitoring():
    """
    Executa o monitoramento completo e envia notificação via Telegram.
    """
    start_time = time.time()
    
    try:
        print("🚀 Iniciando monitoramento da API...")

        # Teste rápido da conexão com RouteLLM
        try:
            _ = llm("teste de conexão")
            print("✅ Conexão com RouteLLM OK")
        except Exception as e:
            print(f"⚠️ Aviso: Problema na conexão RouteLLM: {e}")

        # Executa o monitoramento
        result = api_monitoring_crew.kickoff()
        
        # Calcula tempo de resposta
        response_time = time.time() - start_time
        
        # Tenta extrair dados de análise para métricas detalhadas
        analysis_data = None
        try:
            # Tenta parsear o resultado do analisador se estiver em formato JSON
            if hasattr(result, 'tasks_output') and len(result.tasks_output) >= 2:
                analysis_output = str(result.tasks_output[1])
                analysis_data = json.loads(analysis_output)
        except:
            pass

        # Salva log com tempo de resposta
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_time_seconds": round(response_time, 2),
            "result": str(result),
            "analysis_data": analysis_data
        }
        persist_data(log_entry)

        print(f"\n\n####")
        print(f"## Resultado Final do Monitoramento:")
        print(f"## Tempo de resposta: {response_time:.2f}s")
        print(f"####\n")
        print(result)
        print("📁 Salvo em monitoring_logs.json")

        # Envia notificação via Telegram
        notification_message = str(result)
        if response_time > 30:
            notification_message = f"⚡ Tempo de resposta alto: {response_time:.1f}s\n\n{notification_message}"
        else:
            notification_message = f"⚡ Tempo de resposta: {response_time:.1f}s\n\n{notification_message}"
        
        send_telegram_notification(notification_message, analysis_data)

    except Exception as e:
        error_msg = f"❌ Erro durante execução: {str(e)}"
        print(error_msg)
        
        # Salva erro no log
        persist_data({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "response_time_seconds": round(time.time() - start_time, 2)
        })
        
        # Notifica erro via Telegram
        send_telegram_notification(f"🚨 ERRO no Monitor\n\n{error_msg}")

# ====
# Agendamento e execução
# ====
if __name__ == "__main__":
    print("🤖 Monitor de API iniciado!")
    print("📅 Agendado para rodar diariamente às 12:00")
    
    # Verifica configuração do Telegram
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print("✅ Telegram configurado - comandos disponíveis")
        print("💬 Envie /help no Telegram para ver os comandos")
    else:
        print("⚠️ Telegram não configurado - apenas logs locais")
    
    # Agenda execução diária às 12:00
    schedule.every().day.at("12:00").do(run_monitoring)
    
    # Executa uma vez imediatamente para teste (opcional)
    print("\n🧪 Executando teste inicial...")
    run_monitoring()
    
    print(f"\n⏰ Aguardando próxima execução às 12:00...")
    print("💡 Pressione Ctrl+C para parar")
    
    # Loop principal do agendador + escuta de comandos
    try:
        while True:
            schedule.run_pending()
            listen_for_commands()  # escuta comandos do Telegram
            time.sleep(5)  # verifica a cada 5 segundos
    except KeyboardInterrupt:
        print("\n👋 Monitor interrompido pelo usuário")