# Detecção de anexos maliciosos de e-mail

O projeto consiste no teste de modelos tradicionais de aprendizado de máquina para a detecção de anomalias em um contexto de e-mails com anexos maliciosos no formato `.pdf`. O propósito do projeto é realizar uma prova de conceito, adquirir prática com modelos de aprendizado de máquina e analisar o desempenho de diferentes modelos, bem como suas vantagens e desvantagens.

As diferentes abordagens de modelagem utilizadas foram:

* Estatística com Isolation Forest
* Agrupamento com K-Means e DBSCAN
* Rede neural com Autoencoder

A principal métrica utilizada foi o F0.5, que é uma variante do F1 que dá maior ênfase à precisão. Essa foi uma decisão com fins utilitários, visando evitar falsos positivos. Dependendo da aplicação e do contexto de negócio, novos treinamentos utilizando uma métrica diferente podem ser realizados para reduzir a ênfase nos falsos positivos e obter uma maior cobertura.

<img width="900" height="500" alt="Comparação F0.5, Precision, Recall e AUC-ROC" src="https://github.com/user-attachments/assets/ea6bd44d-ca8c-4d9d-8c7d-4657b9c42f25" />

<img width="700" height="600" alt="Comparação das curvas ROC" src="https://github.com/user-attachments/assets/4e530bbe-3318-4aff-99fe-8180de076f7a" />

### Possíveis aplicações

O classificador pode ser integrado a sistemas de segurança que processam anexos de e-mail. Um caso de uso primário seria um Secure Email Gateway, no qual os anexos PDF podem ser analisados antes de serem entregues aos usuários.

O classificador também poderia servir como um mecanismo de triagem em um primeiro estágio de um pipeline de análise de malware. Arquivos considerados suspeitos poderiam ser encaminhados para uma sandbox ou outra ferramenta de análise para uma investigação mais aprofundada.

Uma possível aplicação seria a integração com um pipeline de SOC/SIEM, no qual os resultados da classificação poderiam ser convertidos em alertas e correlacionados com outros eventos de segurança.

De toda forma, o modelo é projetado para atuar como um componente de detecção, e não como uma solução completa de segurança de e-mail. Um deploy em produção tipicamente o combinaria com outros controles de segurança.

---

### Fonte dos datasets utilizados na modelagem

Nejati, N. et al.
"A Comprehensive Multi-Format Malicious Attachment Dataset for Email Threat Detection."
Canadian Institute for Cybersecurity (CIC), University of New Brunswick, 2025.
