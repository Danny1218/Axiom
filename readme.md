# Axiom Engine

Axiom is an experimental domain-specific programming language and neural compiler. It bridges deterministic software engineering with probabilistic deep learning by directly compiling high-level logic into a dynamically evolving, Differentiable Neural Architecture Search (DNAS) graph.

Instead of writing text prompts for massive, static, black-box LLMs, Axiom allows developers to define the explicit logical constraints of a domain (e.g., genetics, physics engines, rule-based systems) and natively compiles that logic into an ultra-sparse neural topology. 

## Core Architecture

The Axiom compiler replaces the standard Python/CUDA deep learning stack with a multi-paradigm execution engine:

* **Latent Supernet (TT-LoRA):** Axiom bypasses hardware VRAM constraints by utilizing a shared frozen trunk equipped with thousands of unmasked Tensor-Train LoRA adapters. The model evolves its architecture locally without crashing memory limits.
* **Sinkhorn Topological Routing:** Standard control flow (`if/else`) is compiled into continuous mathematical logic using Sinkhorn-Knopp optimal transport, ensuring perfectly balanced token routing and preventing Expert collapse in the MoE graph.
* **Runtime Meta-Compilation:** The execution graph monitors its own epistemic uncertainty. When it encounters high-variance anomalies, it synthesizes new Intermediate Representation (IR) bytecode, unmasks latent experts, and trains them in an isolated "Shadow Mode" before integrating them into the core network.
* **Liquid-KAN State Memory:** Axiom abandons static KV-caches. Sequential logic (loops) are compiled into continuous-depth Liquid Kolmogorov-Arnold Networks, maintaining long-term memory via continuous probability distributions.

## The Compiler Pipeline

1. **Parser:** Parses `.ax` syntax into a strict Abstract Syntax Tree.
2. **IR Bridge:** Translates the AST into deterministic IR Bytecode.
3. **Topology Mapper:** Wires the IR into a directed acyclic graph (NetworkX) of PyTorch modules.
4. **Execution:** Runs data through the probabilistic graph, self-modifying the topology based on loss and variance.

## Getting Started

From the repo root: `pip install -e .` (installs the **`axiom-engine`** package and the **`axiom`** CLI). Train: `axiom train train.ax --epochs 10 --out axiom_bundle`, or built-ins `axiom train examples/titanic.ax --dataset titanic` / `axiom train examples/sequence.ax --dataset sine`. Inference: `axiom train --mode inference --out axiom_bundle`. Glass Box: `axiom inspect`. Optional: `pip install -r requirements.txt` mirrors core deps.

## Philosophy

Current AI scales by brute force. Axiom scales by structural elegance. By giving a model a hardcoded algorithmic skeleton and allowing it to mathematically evolve its own musculature, we can achieve state-of-the-art domain reasoning on local consumer hardware.
