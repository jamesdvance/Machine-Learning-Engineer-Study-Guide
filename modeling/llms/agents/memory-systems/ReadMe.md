# Memory Systems for LLM Agents

## Summary

Memory systems enable LLM agents to retain and utilize information across interactions. Since LLMs have fixed context windows and no inherent persistent state, external memory architectures are essential for agents that need to remember past interactions, learn from experience, or work with large knowledge bases. Memory systems range from simple conversation buffers to sophisticated retrieval-augmented architectures.

Key points to remember:

- Short-term memory: Conversation history within context window
- Long-term memory: Persistent storage across sessions (vector DBs, knowledge graphs)
- Working memory: Scratchpad for current task reasoning
- Episodic memory: Records of past experiences and outcomes
- Semantic memory: Factual knowledge and learned concepts
- Memory compression and summarization extend effective context
- Vector retrieval is the primary mechanism for long-term memory access
- Memory management (what to store, when to forget) is critical for scalability

## Memory Types

### Short-Term Memory (Context Window)

The most basic memory is the conversation history within the LLM's context window:

```python
class ConversationMemory:
    def __init__(self, max_tokens=4000):
        self.messages = []
        self.max_tokens = max_tokens

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._trim_to_limit()

    def _trim_to_limit(self):
        while self._count_tokens() > self.max_tokens:
            # Remove oldest messages (keep system prompt)
            if len(self.messages) > 1:
                self.messages.pop(1)

    def get_messages(self):
        return self.messages
```

### Working Memory (Scratchpad)

Temporary storage for current task reasoning:

```python
class WorkingMemory:
    def __init__(self):
        self.scratchpad = {}
        self.current_plan = None
        self.intermediate_results = []

    def store(self, key, value):
        self.scratchpad[key] = value

    def retrieve(self, key):
        return self.scratchpad.get(key)

    def add_result(self, step, result):
        self.intermediate_results.append({
            "step": step,
            "result": result,
            "timestamp": datetime.now()
        })

    def get_context(self):
        return {
            "plan": self.current_plan,
            "results": self.intermediate_results,
            "scratchpad": self.scratchpad
        }
```

### Long-Term Memory (Vector Store)

Persistent memory using embedding-based retrieval:

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class LongTermMemory:
    def __init__(self, collection_name="agent_memory"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./memory_db"
        )

    def store(self, content, metadata=None):
        """Store a memory with optional metadata."""
        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[metadata or {}]
        )

    def retrieve(self, query, k=5):
        """Retrieve relevant memories."""
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def retrieve_with_scores(self, query, k=5, threshold=0.7):
        """Retrieve memories above similarity threshold."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results
                if score >= threshold]
```

### Episodic Memory

Records of past experiences for learning:

```python
class EpisodicMemory:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def record_episode(self, task, actions, outcome, reflection=None):
        """Record a complete episode."""
        episode = {
            "task": task,
            "actions": actions,
            "outcome": outcome,
            "reflection": reflection,
            "timestamp": datetime.now().isoformat(),
            "success": outcome.get("success", False)
        }

        # Store as searchable text
        episode_text = self._format_episode(episode)
        self.vectorstore.add_texts(
            texts=[episode_text],
            metadatas=[{"type": "episode", **episode}]
        )

    def recall_similar_episodes(self, current_task, k=3):
        """Find similar past experiences."""
        results = self.vectorstore.similarity_search(
            current_task,
            k=k,
            filter={"type": "episode"}
        )
        return results

    def _format_episode(self, episode):
        return f"""
        Task: {episode['task']}
        Actions taken: {episode['actions']}
        Outcome: {episode['outcome']}
        Reflection: {episode.get('reflection', 'None')}
        """
```

### Semantic Memory

Structured knowledge and facts:

```python
class SemanticMemory:
    def __init__(self, vectorstore, graph_db=None):
        self.vectorstore = vectorstore
        self.graph_db = graph_db  # Optional knowledge graph

    def store_fact(self, fact, source=None, confidence=1.0):
        """Store a factual statement."""
        self.vectorstore.add_texts(
            texts=[fact],
            metadatas=[{
                "type": "fact",
                "source": source,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }]
        )

    def store_concept(self, concept, definition, related_concepts=None):
        """Store a concept with its definition."""
        text = f"{concept}: {definition}"
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[{
                "type": "concept",
                "name": concept,
                "related": related_concepts or []
            }]
        )

        # Optionally add to knowledge graph
        if self.graph_db:
            self._add_to_graph(concept, definition, related_concepts)

    def query_knowledge(self, query, k=5):
        """Query semantic knowledge."""
        return self.vectorstore.similarity_search(query, k=k)
```

## Memory Architectures

### Unified Memory System

```python
class UnifiedMemory:
    def __init__(self):
        self.short_term = ConversationMemory()
        self.working = WorkingMemory()
        self.long_term = LongTermMemory()
        self.episodic = EpisodicMemory(self.long_term.vectorstore)
        self.semantic = SemanticMemory(self.long_term.vectorstore)

    def get_context(self, query):
        """Build context from all memory types."""
        context = {
            "conversation": self.short_term.get_messages()[-5:],
            "working": self.working.get_context(),
            "relevant_memories": self.long_term.retrieve(query),
            "similar_episodes": self.episodic.recall_similar_episodes(query),
            "knowledge": self.semantic.query_knowledge(query)
        }
        return self._format_context(context)

    def _format_context(self, context):
        """Format memory context for LLM."""
        sections = []

        if context["relevant_memories"]:
            sections.append("Relevant memories:\n" +
                          "\n".join(f"- {m}" for m in context["relevant_memories"]))

        if context["similar_episodes"]:
            sections.append("Similar past experiences:\n" +
                          "\n".join(str(e) for e in context["similar_episodes"]))

        return "\n\n".join(sections)
```

### Hierarchical Memory

```python
class HierarchicalMemory:
    def __init__(self):
        self.levels = {
            "immediate": [],           # Last few turns
            "session": [],             # Current session
            "persistent": VectorStore() # Cross-session
        }
        self.summarizer = SummarizerLLM()

    def add(self, content, level="immediate"):
        self.levels[level].append(content)
        self._maybe_consolidate()

    def _maybe_consolidate(self):
        """Move and summarize memories up the hierarchy."""
        # Immediate -> Session (summarize every 10 items)
        if len(self.levels["immediate"]) > 10:
            summary = self.summarizer.summarize(
                self.levels["immediate"][:10]
            )
            self.levels["session"].append(summary)
            self.levels["immediate"] = self.levels["immediate"][10:]

        # Session -> Persistent (summarize every 5 summaries)
        if len(self.levels["session"]) > 5:
            meta_summary = self.summarizer.summarize(
                self.levels["session"][:5]
            )
            self.levels["persistent"].add(meta_summary)
            self.levels["session"] = self.levels["session"][5:]
```

## Memory Compression

### Conversation Summarization

```python
class SummarizingMemory:
    def __init__(self, llm, max_tokens=2000):
        self.llm = llm
        self.max_tokens = max_tokens
        self.messages = []
        self.summary = ""

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})

        if self._count_tokens() > self.max_tokens:
            self._summarize_and_trim()

    def _summarize_and_trim(self):
        # Keep last few messages, summarize the rest
        to_summarize = self.messages[:-3]
        to_keep = self.messages[-3:]

        if to_summarize:
            new_summary = self.llm.generate(f"""
            Summarize this conversation, preserving key information:

            Previous summary: {self.summary}

            New messages:
            {self._format_messages(to_summarize)}
            """)

            self.summary = new_summary
            self.messages = to_keep

    def get_context(self):
        context = []
        if self.summary:
            context.append({"role": "system",
                          "content": f"Conversation summary: {self.summary}"})
        context.extend(self.messages)
        return context
```

### Entity Extraction and Tracking

```python
class EntityMemory:
    def __init__(self, llm):
        self.llm = llm
        self.entities = {}  # name -> {type, attributes, last_mentioned}

    def extract_and_update(self, text):
        """Extract entities from text and update memory."""
        extraction_prompt = f"""
        Extract entities and their attributes from this text:
        {text}

        Format: entity_name | entity_type | attributes
        """

        response = self.llm.generate(extraction_prompt)
        new_entities = self._parse_entities(response)

        for name, info in new_entities.items():
            if name in self.entities:
                # Merge with existing
                self.entities[name]["attributes"].update(info["attributes"])
                self.entities[name]["last_mentioned"] = datetime.now()
            else:
                self.entities[name] = info

    def get_entity_context(self, query):
        """Get relevant entity information for a query."""
        relevant = []
        for name, info in self.entities.items():
            if name.lower() in query.lower():
                relevant.append(f"{name} ({info['type']}): {info['attributes']}")
        return "\n".join(relevant)
```

## Retrieval Strategies

### Hybrid Retrieval

```python
class HybridRetriever:
    def __init__(self, vectorstore, keyword_index):
        self.vectorstore = vectorstore
        self.keyword_index = keyword_index

    def retrieve(self, query, k=5, alpha=0.5):
        """Combine vector and keyword search."""
        # Vector search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k*2)

        # Keyword search
        keyword_results = self.keyword_index.search(query, k=k*2)

        # Combine and rerank
        combined = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            alpha=alpha
        )

        return combined[:k]

    def _reciprocal_rank_fusion(self, results1, results2, alpha=0.5):
        scores = {}

        for rank, (doc, score) in enumerate(results1):
            doc_id = doc.metadata.get("id", hash(doc.page_content))
            scores[doc_id] = alpha * (1 / (rank + 1))

        for rank, (doc, score) in enumerate(results2):
            doc_id = doc.metadata.get("id", hash(doc.page_content))
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * (1 / (rank + 1))

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Time-Weighted Retrieval

```python
class TimeWeightedRetriever:
    def __init__(self, vectorstore, decay_rate=0.01):
        self.vectorstore = vectorstore
        self.decay_rate = decay_rate

    def retrieve(self, query, k=5):
        # Get more results than needed
        results = self.vectorstore.similarity_search_with_score(query, k=k*3)

        # Apply time decay
        now = datetime.now()
        weighted_results = []

        for doc, similarity_score in results:
            timestamp = datetime.fromisoformat(doc.metadata.get("timestamp", now.isoformat()))
            hours_ago = (now - timestamp).total_seconds() / 3600
            time_weight = math.exp(-self.decay_rate * hours_ago)

            combined_score = similarity_score * time_weight
            weighted_results.append((doc, combined_score))

        # Sort by combined score
        weighted_results.sort(key=lambda x: x[1], reverse=True)

        return weighted_results[:k]
```

### Importance-Based Retrieval

```python
class ImportanceWeightedMemory:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def add_with_importance(self, content):
        """Score importance and store."""
        importance = self._score_importance(content)

        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[{
                "importance": importance,
                "timestamp": datetime.now().isoformat()
            }]
        )

    def _score_importance(self, content):
        prompt = f"""Rate the importance of remembering this information (1-10):

        Content: {content}

        Consider: Is this a key fact? A user preference? A critical decision?
        Return only a number 1-10."""

        response = self.llm.generate(prompt)
        return int(response.strip())

    def retrieve(self, query, k=5):
        results = self.vectorstore.similarity_search_with_score(query, k=k*2)

        # Weight by importance
        weighted = []
        for doc, sim_score in results:
            importance = doc.metadata.get("importance", 5) / 10
            combined = sim_score * (0.7 + 0.3 * importance)
            weighted.append((doc, combined))

        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted[:k]
```

## Memory in Agent Loops

### Integration Pattern

```python
class MemoryAugmentedAgent:
    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def run(self, task):
        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve(task)

        # Build prompt with memory context
        prompt = self._build_prompt(task, relevant_memories)

        # Execute agent loop
        result = self._execute(prompt)

        # Store experience
        self.memory.store_episode(task, result)

        return result

    def _build_prompt(self, task, memories):
        return f"""You are an AI assistant with access to past memories.

        Relevant memories:
        {self._format_memories(memories)}

        Current task: {task}

        Use your memories to inform your response when relevant."""
```

### Memory-Guided Planning

```python
class MemoryGuidedPlanner:
    def __init__(self, llm, episodic_memory):
        self.llm = llm
        self.memory = episodic_memory

    def plan(self, task):
        # Recall similar past tasks
        similar_episodes = self.memory.recall_similar_episodes(task)

        # Extract lessons learned
        lessons = self._extract_lessons(similar_episodes)

        # Generate plan informed by past experience
        prompt = f"""Create a plan for: {task}

        Lessons from similar past tasks:
        {lessons}

        Avoid past mistakes and apply successful strategies."""

        return self.llm.generate(prompt)

    def _extract_lessons(self, episodes):
        lessons = []
        for ep in episodes:
            if ep.metadata.get("success"):
                lessons.append(f"SUCCESS: {ep.metadata.get('reflection')}")
            else:
                lessons.append(f"AVOID: {ep.metadata.get('reflection')}")
        return "\n".join(lessons)
```

## Memory Management

### Forgetting Strategies

```python
class ManagedMemory:
    def __init__(self, vectorstore, max_memories=10000):
        self.vectorstore = vectorstore
        self.max_memories = max_memories

    def cleanup(self):
        """Remove old or low-importance memories."""
        all_memories = self.vectorstore.get_all()

        if len(all_memories) > self.max_memories:
            # Score each memory
            scored = []
            for mem in all_memories:
                score = self._compute_retention_score(mem)
                scored.append((mem, score))

            # Keep top memories
            scored.sort(key=lambda x: x[1], reverse=True)
            to_remove = scored[self.max_memories:]

            for mem, _ in to_remove:
                self.vectorstore.delete(mem.id)

    def _compute_retention_score(self, memory):
        """Score based on recency, importance, and access frequency."""
        recency = self._recency_score(memory.metadata.get("timestamp"))
        importance = memory.metadata.get("importance", 5) / 10
        access_freq = memory.metadata.get("access_count", 0) / 100

        return 0.4 * recency + 0.4 * importance + 0.2 * access_freq
```

## Key Takeaways

1. **Layer memory types**: Short-term for conversation, working for current task, long-term for persistence.

2. **Vector retrieval is core**: Embedding-based similarity search powers most long-term memory.

3. **Compress strategically**: Summarization and entity extraction extend effective memory.

4. **Time and importance matter**: Weight retrieval by recency and importance, not just similarity.

5. **Learn from episodes**: Store and retrieve past experiences to improve future performance.

6. **Manage growth**: Implement forgetting strategies to prevent unbounded memory growth.

7. **Integrate into agent loop**: Memory should inform planning, execution, and reflection.
