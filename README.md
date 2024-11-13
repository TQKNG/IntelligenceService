# Flows
## High-Level Request Flow
Query -> API Endpoint -> Multi-agent Service -> API Response -> End

## Multi-agent Service Flow
Multi-agent Service -> Agents -> Graph -> Invoke/Stream -> End

## Supervisor Agent and Agent Chaining Flow
Agents -> Supervisor Agent -> Prompts -> LLM (structured output) -> Chain -> Next Agent /Finish -> End

## Other Agents, Tools, and Execution Flow
Other Agents -> Prompts -> LLM -> Tools -> React Agent Creation -> Bind Tools to Agent -> Chain/Agent Executor -> Agent Response -> End

## Graph Execution Flow
Graph -> Agent States -> Agent Nodes / Tool Nodes -> Agent Edges -> Conditional Edges -> Compile Graph -> Stream Graph -> End


# Project Folder Structure
```plaintext
├── main.py
├── routes
│   └── routes.py
├── controllers
│   └── controllers.py
├── services
│   └── multi_agent_service.py
├── agents
│   └── agent_factory.py
│   └── base_agent.py
├── chains
│   └── graph_builder.py
│   └── agent1_chain.py
├── tools
├── utils
└── prompts




    
