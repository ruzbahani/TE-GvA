import argparse
import ast
import io
import sys
import tokenize
from pathlib import Path

import yaml

DOC_PARENTS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstrings(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, DOC_PARENTS) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                out.append((node, first))
    return out


def _without_docstrings(tree):
    for parent, expr in _docstrings(tree):
        parent.body.remove(expr)
        if not parent.body and not isinstance(parent, ast.Module):
            parent.body.append(ast.Pass())
    return tree


def strip_python(src):
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop, cut, replace, protected = set(), {}, {}, set()
    for parent, expr in _docstrings(tree):
        s, e = expr.lineno - 1, expr.end_lineno - 1
        indent = lines[s][:expr.col_offset]
        tail = lines[e][expr.end_col_offset:].strip()
        if indent.strip() or (tail and not tail.startswith("#")):
            raise ValueError(f"docstring at line {s + 1} shares its lines with code")
        if len(parent.body) == 1 and not isinstance(parent, ast.Module):
            replace[s] = indent + "pass\n"
            drop.update(range(s + 1, e + 1))
        else:
            drop.update(range(s, e + 1))
    fstring_start, fstring_end, open_fstrings = getattr(tokenize, "FSTRING_START", None), \
        getattr(tokenize, "FSTRING_END", None), []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            r, c = tok.start[0] - 1, tok.start[1]
            if lines[r][:c].strip():
                cut[r] = min(c, cut.get(r, c))
            else:
                drop.add(r)
        elif tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
            protected.update(range(tok.start[0] - 1, tok.end[0]))
        elif fstring_start is not None and tok.type == fstring_start:
            open_fstrings.append(tok.start[0])
        elif fstring_end is not None and tok.type == fstring_end:
            first = open_fstrings.pop()
            if tok.end[0] > first:
                protected.update(range(first - 1, tok.end[0]))
    out = []
    for i, line in enumerate(lines):
        if i in replace:
            new = replace[i]
        elif i in drop:
            continue
        elif i in cut:
            new = lines[i][:cut[i]].rstrip() + "\n"
        else:
            new = line
        blank = not new.strip() and i not in protected
        if blank and (not out or out[-1][1]):
            continue
        if blank and out and out[-1][0].rstrip().endswith(":") and not out[-1][2]:
            continue
        out.append((new, blank, i in protected))
    while out and out[-1][1]:
        out.pop()
    final = []
    for line, blank, prot in out:
        top = not prot and line[:1] in ("d", "c", "a", "@") and line.startswith(("def ", "class ", "async def ", "@"))
        if top and final:
            prev = next((x for x in reversed(final) if not x[1]), None)
            if prev is not None and not (prev[0].startswith("@") and not prev[2]):
                while final and final[-1][1]:
                    final.pop()
                final.extend([("\n", True, False)] * 2)
        final.append((line, blank, prot))
    result = "".join(line for line, _, _ in final)
    if result and not result.endswith("\n"):
        result += "\n"
    expected = ast.dump(_without_docstrings(ast.parse(src)))
    if ast.dump(ast.parse(result)) != expected:
        raise AssertionError("stripping changed the program")
    if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(result).readline)):
        raise AssertionError("a comment survived")
    if _docstrings(ast.parse(result)):
        raise AssertionError("a docstring survived")
    return result


def strip_yaml(text):
    spans = []
    for tok in yaml.scan(text, Loader=yaml.SafeLoader):
        if isinstance(tok, yaml.ScalarToken):
            spans.append((tok.start_mark.index, tok.end_mark.index))
    inside = bytearray(len(text) + 1)
    for a, b in spans:
        inside[a:b] = b"\x01" * (b - a)
    out, pos = [], 0
    for line in text.splitlines(keepends=True):
        end = len(line.rstrip("\r\n"))
        code = line[:end]
        for j, ch in enumerate(code):
            if ch == "#" and not inside[pos + j] and (j == 0 or code[j - 1] in " \t"):
                code = code[:j]
                break
        stripped = code.rstrip()
        if stripped or line.strip() == "":
            out.append(stripped)
        pos += len(line)
    while out and not out[0]:
        out.pop(0)
    tidy = []
    for line in out:
        if not line and (not tidy or not tidy[-1]):
            continue
        tidy.append(line)
    while tidy and not tidy[-1]:
        tidy.pop()
    result = "\n".join(tidy) + "\n"
    if yaml.safe_load(result) != yaml.safe_load(text):
        raise AssertionError("stripping changed the YAML document")
    if any("#" in line and line.lstrip().startswith("#") for line in tidy):
        raise AssertionError("a YAML comment survived")
    return result


def strip_toml(text):
    import tomllib
    out = []
    for line in text.splitlines():
        if '"""' in line or "'''" in line:
            raise ValueError("multi-line TOML strings are not handled")
        quote, j, cut = None, 0, len(line)
        while j < len(line):
            ch = line[j]
            if quote == '"' and ch == "\\":
                j += 2
                continue
            if quote:
                if ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == "#":
                cut = j
                break
            j += 1
        code = line[:cut].rstrip()
        if code or not line.strip():
            out.append(code)
    tidy = []
    for line in out:
        if not line and (not tidy or not tidy[-1]):
            continue
        tidy.append(line)
    while tidy and not tidy[-1]:
        tidy.pop()
    result = "\n".join(tidy) + "\n"
    if tomllib.loads(result) != tomllib.loads(text):
        raise AssertionError("stripping changed the TOML document")
    return result


def strip_tree(root):
    counts = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        fn = {".py": strip_python, ".yaml": strip_yaml, ".yml": strip_yaml, ".toml": strip_toml}.get(path.suffix)
        if fn is None:
            continue
        text = path.read_text(encoding="utf-8")
        new = fn(text)
        if new != text:
            path.write_text(new, encoding="utf-8", newline="\n")
            counts[path.suffix] = counts.get(path.suffix, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    args = parser.parse_args()
    print(strip_tree(args.directory))

if __name__ == "__main__":
    sys.exit(main())
