export type Difficulty = "Easy" | "Medium" | "Hard";

export const ALL_DIFFICULTIES = ["All", "Easy", "Medium", "Hard"] as const;
export type DifficultyFilter = (typeof ALL_DIFFICULTIES)[number];

export interface FeedItem {
  id: string;
  category: string;
  title: string;
  summary: string;
  codeSnippet?: string;
  codeLang?: string;
  difficulty: Difficulty;
  source: string;
  sourceUrl: string;
  readTime: string;
  tags: string[];
}

const dsContent: FeedItem[] = [
  {
    id: "ds-001",
    category: "Data Structures",
    title: "Two Sum - Optimal HashMap Approach",
    summary: "Given an array of integers and a target, return indices of two numbers that add up to target. The hash map approach achieves O(n) time by storing complements during a single pass.",
    codeSnippet: `def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []`,
    codeLang: "python",
    difficulty: "Easy",
    source: "LeetCode #1",
    sourceUrl: "https://leetcode.com/problems/two-sum/",
    readTime: "2 min",
    tags: ["HashMap", "Array", "O(n)"],
  },
  {
    id: "ds-002",
    category: "Data Structures",
    title: "LRU Cache - OrderedDict Implementation",
    summary: "Design a data structure that follows LRU eviction policy. Use OrderedDict to maintain insertion order with O(1) get and put operations.",
    codeSnippet: `from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.cap = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)`,
    codeLang: "python",
    difficulty: "Medium",
    source: "LeetCode #146",
    sourceUrl: "https://leetcode.com/problems/lru-cache/",
    readTime: "4 min",
    tags: ["Design", "OrderedDict", "Cache"],
  },
  {
    id: "ds-003",
    category: "Data Structures",
    title: "Merge K Sorted Lists - Heap Approach",
    summary: "Merge k sorted linked lists using a min-heap. Push the head of each list, pop the smallest, and push its next node. O(N log k) time complexity.",
    codeSnippet: `import heapq

def merge_k_lists(lists):
    heap = []
    for i, l in enumerate(lists):
        if l:
            heapq.heappush(heap, (l.val, i, l))
    dummy = curr = ListNode(0)
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, 
                (node.next.val, i, node.next))
    return dummy.next`,
    codeLang: "python",
    difficulty: "Hard",
    source: "LeetCode #23",
    sourceUrl: "https://leetcode.com/problems/merge-k-sorted-lists/",
    readTime: "3 min",
    tags: ["Heap", "LinkedList", "Divide & Conquer"],
  },
  {
    id: "ds-004",
    category: "Data Structures",
    title: "Sliding Window Maximum - Deque Pattern",
    summary: "Find the maximum in each sliding window of size k. Use a monotonic decreasing deque to maintain candidates in O(n) time.",
    codeSnippet: `from collections import deque

def max_sliding_window(nums, k):
    dq, result = deque(), []
    for i, num in enumerate(nums):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < num:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result`,
    codeLang: "python",
    difficulty: "Hard",
    source: "LeetCode #239",
    sourceUrl: "https://leetcode.com/problems/sliding-window-maximum/",
    readTime: "3 min",
    tags: ["Deque", "Sliding Window", "Monotonic"],
  },
  {
    id: "ds-005",
    category: "Data Structures",
    title: "Trie - Prefix Tree Implementation",
    summary: "Implement a trie for efficient prefix-based search. Each node stores children and an end-of-word flag. Insert, search, and startsWith in O(m) time.",
    codeSnippet: `class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        node = self._traverse(word)
        return node is not None and node.is_end

    def _traverse(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node`,
    codeLang: "python",
    difficulty: "Medium",
    source: "LeetCode #208",
    sourceUrl: "https://leetcode.com/problems/implement-trie-prefix-tree/",
    readTime: "4 min",
    tags: ["Trie", "String", "Design"],
  },
  {
    id: "ds-006",
    category: "Data Structures",
    title: "Binary Tree Level Order Traversal - BFS",
    summary: "Traverse a binary tree level by level using BFS with a queue. Process each level's nodes before moving to the next depth.",
    codeSnippet: `from collections import deque

def level_order(root):
    if not root:
        return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result`,
    codeLang: "python",
    difficulty: "Medium",
    source: "LeetCode #102",
    sourceUrl: "https://leetcode.com/problems/binary-tree-level-order-traversal/",
    readTime: "2 min",
    tags: ["BFS", "Binary Tree", "Queue"],
  },
  {
    id: "ds-007",
    category: "Data Structures",
    title: "Topological Sort - Kahn's Algorithm",
    summary: "Order DAG vertices so every directed edge u->v has u before v. Use in-degree counting with BFS for cycle detection and ordering.",
    codeSnippet: `from collections import deque, defaultdict

def topological_sort(n, edges):
    graph = defaultdict(list)
    in_degree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    queue = deque(
        i for i in range(n) if in_degree[i] == 0
    )
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nei in graph[node]:
            in_degree[nei] -= 1
            if in_degree[nei] == 0:
                queue.append(nei)
    return order if len(order) == n else []`,
    codeLang: "python",
    difficulty: "Medium",
    source: "LeetCode #210",
    sourceUrl: "https://leetcode.com/problems/course-schedule-ii/",
    readTime: "3 min",
    tags: ["Graph", "BFS", "DAG"],
  },
  {
    id: "ds-008",
    category: "Data Structures",
    title: "Union-Find with Path Compression",
    summary: "Disjoint Set Union for connected components. Path compression + union by rank achieves near O(1) amortized per operation.",
    codeSnippet: `class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True`,
    codeLang: "python",
    difficulty: "Medium",
    source: "LeetCode #684",
    sourceUrl: "https://leetcode.com/problems/redundant-connection/",
    readTime: "3 min",
    tags: ["Union-Find", "Graph", "Amortized"],
  },
];

const systemDesignContent: FeedItem[] = [
  {
    id: "sd-001",
    category: "System Design",
    title: "Designing a URL Shortener at Scale",
    summary: "Key decisions: Base62 encoding vs hash-based IDs, read-heavy (100:1 ratio), horizontal partitioning by hash prefix, caching hot URLs with Redis, 301 vs 302 redirects for analytics.",
    source: "System Design Primer",
    sourceUrl: "https://github.com/donnemartin/system-design-primer",
    readTime: "5 min",
    difficulty: "Medium",
    tags: ["Hashing", "Caching", "Scale"],
  },
  {
    id: "sd-002",
    category: "System Design",
    title: "Rate Limiter - Token Bucket Algorithm",
    summary: "Protect services from abuse. Token bucket adds tokens at fixed rate, requests consume tokens. Sliding window log alternative tracks timestamps for precise limiting.",
    codeSnippet: `class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()

    def allow_request(self):
        now = time.time()
        elapsed = now - self.last_time
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_time = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False`,
    codeLang: "python",
    difficulty: "Medium",
    source: "ByteByteGo",
    sourceUrl: "https://bytebytego.com/courses/system-design-interview/design-a-rate-limiter",
    readTime: "4 min",
    tags: ["Rate Limiting", "API Gateway", "Throttling"],
  },
  {
    id: "sd-003",
    category: "System Design",
    title: "Consistent Hashing for Distributed Systems",
    summary: "Map both servers and keys to a hash ring. Adding/removing nodes only remaps K/N keys on average. Virtual nodes improve load distribution across the ring.",
    source: "Martin Kleppmann",
    sourceUrl: "https://dataintensive.net/",
    difficulty: "Hard",
    readTime: "5 min",
    tags: ["Distributed", "Hashing", "Load Balancing"],
  },
  {
    id: "sd-004",
    category: "System Design",
    title: "Event-Driven Architecture with Kafka",
    summary: "Decouple producers from consumers using topics and partitions. Consumer groups enable parallel processing. Exactly-once semantics via idempotent producers and transactional writes.",
    source: "Confluent",
    sourceUrl: "https://developer.confluent.io/patterns/",
    difficulty: "Hard",
    readTime: "5 min",
    tags: ["Kafka", "Event Sourcing", "CQRS"],
  },
  {
    id: "sd-005",
    category: "System Design",
    title: "Database Sharding Strategies",
    summary: "Horizontal partitioning: range-based (hotspots risk), hash-based (even distribution, hard range queries), directory-based (flexible, single point of failure). Resharding via consistent hashing.",
    source: "System Design Primer",
    sourceUrl: "https://github.com/donnemartin/system-design-primer#sharding",
    difficulty: "Medium",
    readTime: "4 min",
    tags: ["Database", "Sharding", "Partitioning"],
  },
];

const mlDlContent: FeedItem[] = [
  {
    id: "ml-001",
    category: "ML / DL",
    title: "Transformer Self-Attention Mechanism",
    summary: "Q, K, V matrices from input embeddings. Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V. Multi-head attention runs h parallel attention functions, concatenates outputs.",
    codeSnippet: `import torch
import torch.nn.functional as F

def scaled_dot_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(
            mask == 0, float('-inf')
        )
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)`,
    codeLang: "python",
    difficulty: "Hard",
    source: "Attention Is All You Need",
    sourceUrl: "https://arxiv.org/abs/1706.03762",
    readTime: "5 min",
    tags: ["Transformer", "Attention", "NLP"],
  },
  {
    id: "ml-002",
    category: "ML / DL",
    title: "Gradient Descent Variants Compared",
    summary: "SGD: noisy but fast. Adam: adaptive learning rates per parameter with momentum. AdamW: decoupled weight decay for better generalization. Learning rate warmup prevents early divergence.",
    source: "Deep Learning Book",
    sourceUrl: "https://www.deeplearningbook.org/",
    difficulty: "Medium",
    readTime: "4 min",
    tags: ["Optimization", "Adam", "SGD"],
  },
  {
    id: "ml-003",
    category: "ML / DL",
    title: "LoRA - Low-Rank Adaptation for LLMs",
    summary: "Fine-tune large models by freezing pretrained weights and injecting trainable rank decomposition matrices. Reduces trainable parameters by 10,000x while maintaining quality.",
    codeSnippet: `# LoRA pseudo-code concept
# W_original: frozen pretrained weights
# A, B: trainable low-rank matrices
# r: rank (typically 4-64)

# Forward pass:
# h = W_original @ x + (B @ A) @ x
# Only A and B are trained
# Params: r * (d_in + d_out) << d_in * d_out`,
    codeLang: "python",
    difficulty: "Medium",
    source: "Microsoft Research",
    sourceUrl: "https://arxiv.org/abs/2106.09685",
    readTime: "4 min",
    tags: ["Fine-tuning", "LoRA", "Parameter Efficient"],
  },
  {
    id: "ml-004",
    category: "ML / DL",
    title: "Batch Norm vs Layer Norm vs RMSNorm",
    summary: "BatchNorm: normalizes across batch dimension, unstable with small batches. LayerNorm: normalizes across features, standard in transformers. RMSNorm: simplified LayerNorm without mean centering, used in LLaMA.",
    source: "Papers With Code",
    sourceUrl: "https://paperswithcode.com/method/layer-normalization",
    difficulty: "Easy",
    readTime: "3 min",
    tags: ["Normalization", "Training", "Architecture"],
  },
];

const genAiContent: FeedItem[] = [
  {
    id: "gen-001",
    category: "GenAI",
    title: "RAG Pipeline Architecture",
    summary: "Retrieval-Augmented Generation: chunk documents, embed with sentence transformers, store in vector DB (Pinecone/Weaviate), retrieve top-k similar chunks, inject into LLM context window.",
    codeSnippet: `# RAG Pipeline pseudo-code
def rag_query(question, vector_db, llm):
    # 1. Embed the question
    q_embedding = embed_model.encode(question)
    
    # 2. Retrieve relevant chunks
    chunks = vector_db.similarity_search(
        q_embedding, top_k=5
    )
    
    # 3. Build augmented prompt
    context = "\\n".join(
        c.text for c in chunks
    )
    prompt = f"""Context: {context}
    
Question: {question}
Answer based on the context above:"""
    
    # 4. Generate with LLM
    return llm.generate(prompt)`,
    codeLang: "python",
    difficulty: "Medium",
    source: "LangChain Docs",
    sourceUrl: "https://python.langchain.com/docs/tutorials/rag/",
    readTime: "5 min",
    tags: ["RAG", "Vector DB", "Embeddings"],
  },
  {
    id: "gen-002",
    category: "GenAI",
    title: "Prompt Engineering - Chain of Thought",
    summary: "CoT prompting improves reasoning by asking models to show work. Zero-shot CoT: append 'Let's think step by step'. Few-shot CoT: provide exemplar reasoning chains. Tree-of-Thought for complex branching.",
    source: "Google Research",
    sourceUrl: "https://arxiv.org/abs/2201.11903",
    difficulty: "Easy",
    readTime: "3 min",
    tags: ["Prompting", "CoT", "Reasoning"],
  },
  {
    id: "gen-003",
    category: "GenAI",
    title: "LLM Evaluation Metrics & Benchmarks",
    summary: "MMLU for broad knowledge, HumanEval for code, MT-Bench for conversation quality. Custom evals: use LLM-as-judge with rubrics. Track latency, cost per token, and hallucination rate alongside quality.",
    source: "Hugging Face",
    sourceUrl: "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard",
    difficulty: "Medium",
    readTime: "4 min",
    tags: ["Evaluation", "Benchmarks", "MMLU"],
  },
  {
    id: "gen-004",
    category: "GenAI",
    title: "Structured Output with Function Calling",
    summary: "Force LLMs to return valid JSON matching a schema. OpenAI function calling, Anthropic tool use, or constrained decoding with outlines/guidance libraries for reliable structured generation.",
    codeSnippet: `# OpenAI Function Calling
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", 
               "content": query}],
    tools=[{
        "type": "function",
        "function": {
            "name": "extract_entity",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["name", "category"]
            }
        }
    }]
)`,
    codeLang: "python",
    difficulty: "Medium",
    source: "OpenAI Docs",
    sourceUrl: "https://platform.openai.com/docs/guides/function-calling",
    readTime: "4 min",
    tags: ["Function Calling", "Structured Output", "JSON"],
  },
];

const agenticAiContent: FeedItem[] = [
  {
    id: "ag-001",
    category: "Agentic AI",
    title: "ReAct Pattern - Reasoning + Acting",
    summary: "Agents alternate between reasoning (thinking about what to do) and acting (calling tools). The loop: Observe -> Think -> Act -> Observe. Enables multi-step problem solving with tool use.",
    codeSnippet: `# ReAct Agent Loop
def react_agent(query, tools, llm):
    history = [f"Question: {query}"]
    for step in range(max_steps):
        # Think
        thought = llm.generate(
            "\\n".join(history) + 
            "\\nThought:"
        )
        history.append(f"Thought: {thought}")
        
        # Act
        action = llm.generate(
            "\\n".join(history) + 
            "\\nAction:"
        )
        
        # Observe
        result = tools.execute(action)
        history.append(f"Observation: {result}")
        
        if "Final Answer" in thought:
            return thought
    return "Max steps reached"`,
    codeLang: "python",
    difficulty: "Medium",
    source: "LangChain Agents",
    sourceUrl: "https://python.langchain.com/docs/concepts/agents/",
    readTime: "5 min",
    tags: ["ReAct", "Tool Use", "Agent Loop"],
  },
  {
    id: "ag-002",
    category: "Agentic AI",
    title: "Multi-Agent Orchestration Patterns",
    summary: "Supervisor pattern: one agent delegates to specialists. Debate pattern: agents argue and reach consensus. Hierarchical: tree of agent teams. CrewAI, AutoGen, and LangGraph implementations.",
    source: "Microsoft AutoGen",
    sourceUrl: "https://microsoft.github.io/autogen/",
    difficulty: "Hard",
    readTime: "5 min",
    tags: ["Multi-Agent", "Orchestration", "CrewAI"],
  },
  {
    id: "ag-003",
    category: "Agentic AI",
    title: "Tool Calling & MCP Protocol",
    summary: "Model Context Protocol standardizes tool integration. Servers expose tools via JSON-RPC, clients connect agents to any MCP server. Enables composable, reusable tool ecosystems across AI frameworks.",
    source: "Anthropic MCP",
    sourceUrl: "https://modelcontextprotocol.io/",
    difficulty: "Medium",
    readTime: "4 min",
    tags: ["MCP", "Tool Calling", "Protocol"],
  },
  {
    id: "ag-004",
    category: "Agentic AI",
    title: "LangGraph - Stateful Agent Workflows",
    summary: "Build agents as directed graphs. Nodes are functions, edges are conditional transitions. Built-in persistence, human-in-the-loop checkpoints, and streaming support for complex agentic workflows.",
    source: "LangGraph Docs",
    sourceUrl: "https://langchain-ai.github.io/langgraph/",
    difficulty: "Medium",
    readTime: "4 min",
    tags: ["LangGraph", "State Machine", "Workflow"],
  },
];

const aiOpsContent: FeedItem[] = [
  {
    id: "aio-001",
    category: "AIOps",
    title: "ML Model Monitoring in Production",
    summary: "Track data drift (PSI, KS test), concept drift (performance degradation), and feature drift. Set up automated alerts when distributions shift beyond thresholds. Tools: Evidently AI, WhyLabs, Arize.",
    source: "Evidently AI",
    sourceUrl: "https://www.evidentlyai.com/",
    difficulty: "Medium",
    readTime: "4 min",
    tags: ["Monitoring", "Data Drift", "Observability"],
  },
  {
    id: "aio-002",
    category: "AIOps",
    title: "Anomaly Detection for Infrastructure",
    summary: "Use statistical methods (Z-score, IQR) for simple metrics. Isolation Forest for multivariate anomalies. LSTM autoencoders for time-series patterns. Alert correlation to reduce noise.",
    codeSnippet: `from sklearn.ensemble import IsolationForest

def detect_anomalies(metrics_df):
    clf = IsolationForest(
        contamination=0.05,
        random_state=42
    )
    predictions = clf.fit_predict(metrics_df)
    # -1 = anomaly, 1 = normal
    anomalies = metrics_df[predictions == -1]
    return anomalies`,
    codeLang: "python",
    difficulty: "Medium",
    source: "Google SRE Book",
    sourceUrl: "https://sre.google/sre-book/table-of-contents/",
    readTime: "4 min",
    tags: ["Anomaly Detection", "SRE", "Alerting"],
  },
  {
    id: "aio-003",
    category: "AIOps",
    title: "Automated Incident Response with AI",
    summary: "LLM-powered runbook automation: parse alerts, correlate with historical incidents, suggest remediation steps. PagerDuty + LLM integration for intelligent triage and auto-remediation of known issues.",
    source: "PagerDuty",
    sourceUrl: "https://www.pagerduty.com/resources/learn/what-is-aiops/",
    difficulty: "Hard",
    readTime: "5 min",
    tags: ["Incident Response", "Automation", "Runbooks"],
  },
];

const agenticOpsContent: FeedItem[] = [
  {
    id: "ago-001",
    category: "Agentic Ops",
    title: "LLM Observability with LangSmith",
    summary: "Trace every LLM call, tool invocation, and chain step. Track latency, token usage, and cost per trace. Debug agent failures by replaying execution traces. A/B test prompt variants in production.",
    source: "LangSmith",
    sourceUrl: "https://smith.langchain.com/",
    difficulty: "Medium",
    readTime: "4 min",
    tags: ["Observability", "Tracing", "LangSmith"],
  },
  {
    id: "ago-002",
    category: "Agentic Ops",
    title: "Agent Guardrails & Safety Layers",
    summary: "Input validation: content filtering, prompt injection detection. Output validation: fact-checking, PII redaction, toxicity scoring. Circuit breakers for runaway agents. NeMo Guardrails for policy enforcement.",
    source: "NVIDIA NeMo",
    sourceUrl: "https://github.com/NVIDIA/NeMo-Guardrails",
    difficulty: "Hard",
    readTime: "5 min",
    tags: ["Guardrails", "Safety", "Prompt Injection"],
  },
  {
    id: "ago-003",
    category: "Agentic Ops",
    title: "Cost Optimization for LLM Pipelines",
    summary: "Semantic caching with embeddings similarity. Prompt compression to reduce token count. Model routing: use cheaper models for simple tasks, expensive ones for complex. Batch API for non-real-time workloads.",
    source: "OpenAI Cookbook",
    sourceUrl: "https://cookbook.openai.com/",
    difficulty: "Medium",
    readTime: "4 min",
    tags: ["Cost", "Caching", "Model Routing"],
  },
];

const deploymentContent: FeedItem[] = [
  {
    id: "dep-001",
    category: "Deployment",
    title: "Blue-Green vs Canary Deployments",
    summary: "Blue-green: two identical environments, instant switch. Canary: gradual traffic shift (1% -> 5% -> 25% -> 100%) with automated rollback on error rate spikes. Feature flags for granular control.",
    source: "Martin Fowler",
    sourceUrl: "https://martinfowler.com/bliki/BlueGreenDeployment.html",
    difficulty: "Medium",
    readTime: "4 min",
    tags: ["CI/CD", "Canary", "Blue-Green"],
  },
  {
    id: "dep-002",
    category: "Deployment",
    title: "Kubernetes Pod Autoscaling Strategies",
    summary: "HPA: scale on CPU/memory metrics or custom metrics. VPA: right-size resource requests. KEDA: event-driven scaling on queue depth, HTTP requests. Cluster autoscaler adds/removes nodes.",
    codeSnippet: `# HPA with custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: Pods
    pods:
      metric:
        name: inference_queue_depth
      target:
        type: AverageValue
        averageValue: "10"`,
    codeLang: "yaml",
    difficulty: "Hard",
    source: "Kubernetes Docs",
    sourceUrl: "https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/",
    readTime: "5 min",
    tags: ["K8s", "Autoscaling", "HPA"],
  },
  {
    id: "dep-003",
    category: "Deployment",
    title: "ML Model Serving - TorchServe vs Triton",
    summary: "TorchServe: PyTorch-native, easy setup, model archiver. Triton: multi-framework (TF, PyTorch, ONNX), dynamic batching, model ensembles. vLLM for LLM-specific serving with PagedAttention.",
    source: "NVIDIA Triton",
    sourceUrl: "https://developer.nvidia.com/triton-inference-server",
    difficulty: "Hard",
    readTime: "5 min",
    tags: ["Model Serving", "Triton", "vLLM"],
  },
  {
    id: "dep-004",
    category: "Deployment",
    title: "GitOps with ArgoCD for ML Pipelines",
    summary: "Declarative ML deployments: model configs in Git, ArgoCD syncs to cluster. Automatic rollback on health check failures. Kustomize overlays for dev/staging/prod environment configs.",
    source: "ArgoCD Docs",
    sourceUrl: "https://argo-cd.readthedocs.io/",
    difficulty: "Medium",
    readTime: "4 min",
    tags: ["GitOps", "ArgoCD", "ML Pipeline"],
  },
];

const techStackContent: FeedItem[] = [
  {
    id: "ts-001",
    category: "Tech Stacks",
    title: "Modern ML Platform Stack 2025",
    summary: "Data: Spark + Delta Lake. Feature Store: Feast/Tecton. Training: PyTorch + Lightning. Experiment Tracking: MLflow/W&B. Serving: vLLM/Triton. Orchestration: Airflow/Dagster. Monitoring: Evidently AI.",
    source: "ML Ops Community",
    sourceUrl: "https://mlops.community/",
    difficulty: "Medium",
    readTime: "5 min",
    tags: ["MLOps", "Platform", "Architecture"],
  },
  {
    id: "ts-002",
    category: "Tech Stacks",
    title: "Vector Database Comparison 2025",
    summary: "Pinecone: managed, serverless, easy to start. Weaviate: hybrid search, multi-modal. Qdrant: Rust-based, fast. Milvus: distributed, GPU acceleration. ChromaDB: lightweight, embedded. pgvector: if you already use Postgres.",
    source: "AI Engineer",
    sourceUrl: "https://www.latent.space/",
    difficulty: "Easy",
    readTime: "4 min",
    tags: ["Vector DB", "RAG", "Comparison"],
  },
  {
    id: "ts-003",
    category: "Tech Stacks",
    title: "LLM Framework Landscape - LangChain vs LlamaIndex vs DSPy",
    summary: "LangChain: general-purpose, large ecosystem, chains + agents. LlamaIndex: data-focused, best for RAG. DSPy: programmatic prompt optimization, compiler approach. Haystack: production-focused pipelines.",
    source: "The AI Engineer",
    sourceUrl: "https://www.latent.space/p/2024-frameworks",
    difficulty: "Medium",
    readTime: "5 min",
    tags: ["LangChain", "LlamaIndex", "Frameworks"],
  },
  {
    id: "ts-004",
    category: "Tech Stacks",
    title: "GPU Cloud Providers - Cost Comparison",
    summary: "AWS (p5 instances): enterprise, expensive. GCP (A3): TPU alternative. Lambda Labs: cheapest H100s. RunPod: serverless GPU. Together AI: inference API. Modal: serverless compute with cold start optimization.",
    source: "GPU Benchmarks",
    sourceUrl: "https://fullstackdeeplearning.com/cloud-gpus/",
    difficulty: "Easy",
    readTime: "3 min",
    tags: ["GPU", "Cloud", "Cost"],
  },
];

export const ALL_CATEGORIES = [
  "All",
  "Data Structures",
  "System Design",
  "ML / DL",
  "GenAI",
  "Agentic AI",
  "AIOps",
  "Agentic Ops",
  "Deployment",
  "Tech Stacks",
] as const;

export type Category = (typeof ALL_CATEGORIES)[number];

const ALL_CONTENT: FeedItem[] = [
  ...dsContent,
  ...systemDesignContent,
  ...mlDlContent,
  ...genAiContent,
  ...agenticAiContent,
  ...aiOpsContent,
  ...agenticOpsContent,
  ...deploymentContent,
  ...techStackContent,
];

function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

export function getFeedPage(
  page: number,
  pageSize: number,
  category: Category = "All",
  difficultyFilter: DifficultyFilter = "All",
): FeedItem[] {
  let filtered =
    category === "All"
      ? ALL_CONTENT
      : ALL_CONTENT.filter((item) => item.category === category);

  if (difficultyFilter !== "All") {
    filtered = filtered.filter((item) => item.difficulty === difficultyFilter);
  }

  const seed = page * 7 + 13;
  const shuffled = shuffleArray(filtered);

  const rotated = [
    ...shuffled.slice(seed % shuffled.length),
    ...shuffled.slice(0, seed % shuffled.length),
  ];

  const start = (page % Math.ceil(rotated.length / pageSize)) * pageSize;
  const items = rotated.slice(start, start + pageSize);

  return items.map((item) => ({
    ...item,
    id: `${item.id}-page${page}-${Date.now().toString(36).slice(-4)}${Math.random().toString(36).slice(2, 6)}`,
  }));
}

export function getTotalItems(category: Category = "All"): number {
  if (category === "All") return ALL_CONTENT.length;
  return ALL_CONTENT.filter((item) => item.category === category).length;
}
