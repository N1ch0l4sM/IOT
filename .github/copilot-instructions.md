<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Pipeline IoT para Previsão de Chuvas - Instruções para Copilot

## Contexto do Projeto
Este é um projeto de pipeline IoT completo para previsão de chuvas usando dados climáticos. O sistema inclui:
- Coleta e processamento ETL com Apache Airflow
- Armazenamento em PostgreSQL e MinIO
- Machine Learning com Scikit-learn
- Dashboard interativo com Streamlit

## Diretrizes de Código

### Estrutura e Organização
- Use a estrutura de pastas definida no projeto
- Mantenha separação clara entre ETL, ML e visualização
- Siga o padrão de nomenclatura snake_case para Python

### Processamento de Dados
- Use Pandas para manipulação de dados
- Implemente validação de dados em todas as etapas
- Documente transformações e limpezas aplicadas
- Use logging para rastreamento do pipeline

### Machine Learning
- Implemente validação cruzada nos modelos
- Salve métricas de performance
- Use pipelines do Scikit-learn para preprocessamento
- Implemente versionamento de modelos

### Airflow DAGs
- Use decorators (@dag, @task) quando possível
- Implemente tratamento de erros robusto
- Configure retry e timeout apropriados
- Use XComs para comunicação entre tasks

### Dashboard Streamlit
- Implemente cache para queries pesadas
- Use session state para manter estado
- Crie visualizações interativas e responsivas
- Implemente validação de inputs do usuário

### Banco de Dados
- Use SQLAlchemy para ORM
- Implemente conexões com pool
- Use transações para operações críticas
- Documente schema e relacionamentos

### Boas Práticas
- Implemente testes unitários com pytest
- Use type hints em todas as funções
- Documente funções com docstrings
- Use variáveis de ambiente para configurações
- Implemente logging estruturado

### Segurança
- Nunca commit credenciais no código
- Use .env para configurações sensíveis
- Implemente validação de entrada
- Use conexões seguras para banco de dados
