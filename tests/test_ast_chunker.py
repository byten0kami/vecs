from vecs.ast_chunker import chunk_code_file_ast


CS_CODE = """using System;

namespace MyApp
{
    public class Player
    {
        public int Health { get; set; }

        public void TakeDamage(int amount)
        {
            Health -= amount;
            if (Health < 0) Health = 0;
        }

        public void Heal(int amount)
        {
            Health += amount;
        }
    }

    public class Enemy
    {
        public void Attack(Player target)
        {
            target.TakeDamage(10);
        }
    }
}
"""

TS_CODE = """
export function greet(name: string): string {
    return `Hello, ${name}!`;
}

export class UserService {
    private users: Map<string, User> = new Map();

    addUser(user: User): void {
        this.users.set(user.id, user);
    }

    getUser(id: string): User | undefined {
        return this.users.get(id);
    }
}

interface User {
    id: string;
    name: string;
}
"""


def test_cs_chunks_at_class_boundaries():
    """C# file is chunked at class boundaries."""
    chunks = chunk_code_file_ast(CS_CODE, "Player.cs")
    # Should have chunks for Player and Enemy classes
    assert len(chunks) >= 2
    texts = [c["text"] for c in chunks]
    assert any("Player" in t and "TakeDamage" in t for t in texts)
    assert any("Enemy" in t and "Attack" in t for t in texts)


def test_ts_chunks_at_boundaries():
    """TypeScript file is chunked at function/class boundaries."""
    chunks = chunk_code_file_ast(TS_CODE, "user.ts")
    assert len(chunks) >= 2
    texts = [c["text"] for c in chunks]
    assert any("greet" in t for t in texts)
    assert any("UserService" in t for t in texts)


def test_metadata_has_file_path():
    """Chunks carry file_path and line numbers in metadata."""
    chunks = chunk_code_file_ast(CS_CODE, "Player.cs")
    for c in chunks:
        assert c["metadata"]["file_path"] == "Player.cs"
        assert "start_line" in c["metadata"]
        assert "end_line" in c["metadata"]
        assert "chunk_index" in c["metadata"]


def test_unknown_extension_falls_back():
    """Unsupported file extensions fall back to line-based chunking."""
    content = "\n".join(f"line {i}" for i in range(300))
    chunks = chunk_code_file_ast(content, "data.shader", chunk_lines=200, overlap=50)
    assert len(chunks) >= 2  # 300 lines / 200 = at least 2 chunks


def test_empty_file_returns_empty():
    """Empty file returns no chunks."""
    chunks = chunk_code_file_ast("", "Empty.cs")
    assert chunks == []


def test_large_class_is_split():
    """A class exceeding max_chunk_lines is split into sub-chunks."""
    # Generate a class with many methods (>500 lines)
    methods = []
    for i in range(60):
        methods.append(f"""
        public void Method{i}()
        {{
            var x = {i};
            var y = x + 1;
            var z = y * 2;
            Console.WriteLine(z);
            // padding line
            // more padding
        }}""")
    content = f"public class BigClass\n{{\n{''.join(methods)}\n}}"
    chunks = chunk_code_file_ast(content, "Big.cs", max_chunk_lines=500)
    assert len(chunks) >= 2
