"""
src/core/emulation/stream_parser.py — Sliding-window lookahead parser for streaming SSE chunks
and non-streaming response transformation.
Handles:
  1. Real-time interception and suppression of <tool_call> tags
  2. Conversion into standard OpenAI delta.tool_calls SSE chunks
  3. Stop sequence truncation and max_tokens enforcement
  4. Non-streaming response parsing with malformed JSON recovery
"""
from __future__ import annotations

import html
import json
import logging
import re
import time
import uuid
from typing import Any, AsyncGenerator

log = logging.getLogger(__name__)

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"
DIRECT_TOOL_TAGS = ("update", "edit", "write", "bash", "read", "glob", "grep", "ls")
ALL_TOOL_TAGS = ("tool_call", "invoke", "update", "edit", "write", "bash", "read", "glob", "grep", "ls")
OPEN_TAGS = [f"<{t}" for t in ALL_TOOL_TAGS]

AUTOLINK_URL_PATTERN = re.compile(
    r"\[(https?://[^\s\[\]\(\)\"']+?)\]\(https?://[^\s\[\]\(\)\"']+?\)"
)

GENUI_TAG_PATTERN = re.compile(
    r'(```[\s\S]*?```|`[^`\n]*`)|<([A-Z][a-zA-Z0-9_]*)\b([^>]*?)(?:\/>|>(?:(.*?)(?:<\/\2>))|>)',
    re.DOTALL
)

GENUI_OPEN_PATTERN = re.compile(r'<([A-Z][a-zA-Z0-9_]*)\b([^>]*?)(?:\/>|>(?:(.*?)(?:<\/\1>))|>)', re.DOTALL)


def unwrap_markdown_autolinks(text: str) -> str:
    """Unwrap Gemini's automatic [https://...](https://...) markdown link injection inside code/strings."""
    if not text or "[http" not in text:
        return text
    return AUTOLINK_URL_PATTERN.sub(r"\1", text)


def _unwrap_autolinks_in_obj(obj: Any) -> Any:
    """Recursively unwrap [https://...](https://...) markdown auto-links inside parsed tool arguments."""
    if isinstance(obj, str):
        return unwrap_markdown_autolinks(obj)
    if isinstance(obj, dict):
        return {k: _unwrap_autolinks_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap_autolinks_in_obj(v) for v in obj]
    return obj


def sanitize_genui_tags(text: str) -> str:
    """Sanitize GenUI React widget tags into human-readable text."""
    if not text:
        return ""

    framework_attrs = {
        'mode', 'duration', 'locale', 'key', 'resolutionid', '__resolutionid',
        'version', 'id', 'ref', 'classname', 'class', 'style', 'time_format', 'format',
        'lang', 'interactive', 'disabled'
    }

    def replacer(m):
        code_block = m.group(1)
        if code_block:
            return code_block
        attrs_str = m.group(3) or ""
        inner_content = m.group(4)
        if inner_content is not None and inner_content.strip():
            return inner_content.strip()

        attr_regex = re.compile(r'([a-zA-Z0-9_-]+)=(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))')
        attrs = {}
        for am in attr_regex.finditer(attrs_str):
            key = am.group(1)
            val = (
                am.group(2) if am.group(2) is not None
                else (am.group(3) if am.group(3) is not None else (am.group(4) or ""))
            ).strip()
            if val and not val.startswith('{') and key.lower() not in framework_attrs:
                attrs[key] = val

        if not attrs:
            return ""

        primary_keys = ['location', 'city', 'title', 'name', 'label', 'symbol', 'query', 'flight']
        primary_key = next((k for k in primary_keys if k in attrs), next(iter(attrs)))
        primary_val = attrs.pop(primary_key)

        remaining = []
        for k, v in attrs.items():
            if k in ('tz_name', 'timezone'):
                remaining.append(v)
            else:
                clean_key = k.replace('_', ' ').title()
                remaining.append(f"{clean_key}: {v}")

        if remaining:
            return f"**{primary_val}** ({', '.join(remaining)})"
        return f"**{primary_val}**"

    return GENUI_TAG_PATTERN.sub(replacer, text)


def _escape_control_chars_in_json_strings(raw: str) -> str:
    """Escape raw newlines, carriage returns, and tabs only inside double-quoted JSON string literals."""
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
        out.append(ch)
    return "".join(out)


def _extract_balanced_braces(text: str, start_idx: int) -> str | None:
    """Extract a balanced {...} substring starting at start_idx, respecting string quotes and escapes."""
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]
    return None


def _sanitize_json_string(raw: str) -> str:
    """Tolerant cleaner for LLM JSON outputs (unwraps auto-links, strips fences, fixes trailing commas)."""
    clean = unwrap_markdown_autolinks(raw).strip()
    # Remove markdown code fences if wrapped
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()
    # Remove trailing commas before closing braces
    clean = re.sub(r",\s*(\}|\])", r"\1", clean)
    return clean


def normalize_tool_name(tool_name: str, params: dict | None = None) -> str:
    """Normalize common LLM tool aliases to standard Claude Code tool names."""
    if not tool_name:
        return "unknown"
    tn = str(tool_name).strip()
    tn_lower = tn.lower()
    params_dict = params if isinstance(params, dict) else {}

    if tn_lower in ("update", "update_file"):
        if "content" in params_dict and "old_string" not in params_dict:
            return "Write"
        return "Edit"
    if tn_lower in ("edit_file", "str_replace_editor", "str_replace_edit", "replace"):
        return "Edit"
    if tn_lower in ("write_file", "create_file"):
        return "Write"
    if tn_lower in ("read_file", "view_file", "view"):
        return "Read"
    if tn_lower in ("execute_command", "run_command", "shell", "sh", "terminal"):
        return "Bash"
    if tn_lower in ("list_directory", "ls", "dir"):
        return "LS"
    if tn_lower in ("search_files", "find_files"):
        return "Glob"
    if tn_lower in ("grep_search",):
        return "Grep"
    return tn


STRING_PARAM_NAMES: set[str] = {
    "file_path", "path", "file", "filename", "filepath",
    "content", "old_string", "new_string", "old_str", "new_str",
    "command", "cmd", "code", "script",
    "pattern", "regex", "search", "query",
    "description", "prompt", "explanation", "url", "dir", "directory", "folder",
}


def normalize_tool_parameters(tool_name: str, params: dict) -> dict:
    """Normalize parameter aliases, enforce strict schemas for code tools, unescape commands, and strip trailing whitespace on code lines."""
    if not isinstance(params, dict):
        return {}

    tl = str(tool_name or "").strip().lower()

    # 1. Map parameter aliases
    if tl in ("edit", "update", "update_file", "edit_file", "write_file", "create_file", "read_file", "view_file") or tool_name in ("Edit", "Write", "Read"):
        if "file_path" not in params:
            for alt in ("path", "file", "filename", "filepath"):
                if alt in params:
                    params["file_path"] = params.pop(alt)
                    break

    if tl in ("edit", "update", "update_file", "edit_file"):
        if "old_string" not in params and "old_str" in params:
            params["old_string"] = params.pop("old_str")
        if "new_string" not in params and "new_str" in params:
            params["new_string"] = params.pop("new_str")

    if tl in ("bash", "sh", "execute_command", "run_command", "command"):
        if "command" not in params:
            for alt in ("cmd", "code", "script"):
                if alt in params:
                    params["command"] = params.pop(alt)
                    break

    if tl in ("glob", "grep", "ls", "list_dir", "list_directory"):
        if "path" not in params:
            for alt in ("dir", "directory", "folder", "file_path"):
                if alt in params:
                    params["path"] = params.pop(alt)
                    break

    if tl in ("grep", "search"):
        if "pattern" not in params:
            for alt in ("query", "regex", "search"):
                if alt in params:
                    params["pattern"] = params.pop(alt)
                    break

    # 2. Enforce strict parameter types and clean line whitespace
    if "content" in params:
        val = params["content"]
        if isinstance(val, (dict, list)):
            params["content"] = json.dumps(val, indent=2)
        elif not isinstance(val, str):
            params["content"] = "" if val is None else str(val)
        params["content"] = re.sub(r"[ \t]+(?=\r?\n)", "", params["content"])

    for key in ("old_string", "new_string"):
        if key in params:
            val = params[key]
            if isinstance(val, bool):
                params[key] = "true" if val else "false"
            elif isinstance(val, (int, float)):
                params[key] = str(val)
            elif isinstance(val, (dict, list)):
                params[key] = json.dumps(val, indent=2)
            elif not isinstance(val, str):
                params[key] = "" if val is None else str(val)
            params[key] = re.sub(r"[ \t]+(?=\r?\n)", "", params[key])

    if tl == "multiedit" and isinstance(params.get("edits"), list):
        for edit_item in params["edits"]:
            if isinstance(edit_item, dict):
                for key in ("old_string", "new_string"):
                    if key in edit_item:
                        val = edit_item[key]
                        if isinstance(val, bool):
                            edit_item[key] = "true" if val else "false"
                        elif isinstance(val, (int, float)):
                            edit_item[key] = str(val)
                        elif isinstance(val, (dict, list)):
                            edit_item[key] = json.dumps(val, indent=2)
                        elif not isinstance(val, str):
                            edit_item[key] = "" if val is None else str(val)
                        edit_item[key] = re.sub(r"[ \t]+(?=\r?\n)", "", edit_item[key])

    if "file_path" in params and not isinstance(params["file_path"], str):
        params["file_path"] = str(params["file_path"])

    if "path" in params and not isinstance(params["path"], str):
        params["path"] = str(params["path"])

    if "pattern" in params and not isinstance(params["pattern"], str):
        params["pattern"] = str(params["pattern"])

    if "command" in params:
        if not isinstance(params["command"], str):
            params["command"] = str(params["command"])
        if "&" in params["command"]:
            params["command"] = html.unescape(params["command"])

    if "replace_all" in params:
        val = params["replace_all"]
        if isinstance(val, str):
            params["replace_all"] = val.strip().lower() in ("true", "1", "yes")
        elif not isinstance(val, bool):
            params["replace_all"] = bool(val)

    for int_key in ("offset", "limit", "timeout"):
        if int_key in params and not isinstance(params[int_key], int):
            try:
                params[int_key] = int(params[int_key])
            except (ValueError, TypeError):
                pass

    return params


def parse_parameter_xml_value(p_name: str, raw_val: str) -> Any:
    """Parse parameter text from XML, respecting string vs JSON/primitive data types."""
    if p_name in ("content", "old_string", "new_string", "old_str", "new_str"):
        trimmed = re.sub(r"^\r?\n|\r?\n$", "", raw_val)
    else:
        trimmed = raw_val.strip()

    unwrapped = unwrap_markdown_autolinks(trimmed)
    if "&" in unwrapped:
        unwrapped = html.unescape(unwrapped)

    p_name_lower = p_name.lower()
    if p_name_lower in STRING_PARAM_NAMES:
        # Expected to be a string parameter. Only unwrap explicit outer JSON quotes if present.
        if len(unwrapped) >= 2 and unwrapped.startswith('"') and unwrapped.endswith('"'):
            try:
                parsed = json.loads(unwrapped)
                val = parsed if isinstance(parsed, str) else unwrapped
            except Exception:
                val = unwrapped[1:-1]
        elif len(unwrapped) >= 2 and unwrapped.startswith("'") and unwrapped.endswith("'"):
            val = unwrapped[1:-1]
        else:
            val = unwrapped
    else:
        try:
            val = json.loads(unwrapped)
        except Exception:
            val = unwrapped

    if isinstance(val, str) and "&" in val:
        val = html.unescape(val)

    return _unwrap_autolinks_in_obj(val)


def _normalize_parsed_tool_dict(data: dict) -> dict | None:
    """Normalize a parsed tool dict with 'name'/'tool'/'function' and 'arguments'/'input'/'parameters'."""
    name = data.get("name") or data.get("tool") or data.get("function")
    if not isinstance(name, str) or not name.strip():
        return None
    
    if "arguments" in data:
        args = data["arguments"]
    elif "input" in data:
        args = data["input"]
    elif "parameters" in data:
        args = data["parameters"]
    elif "args" in data:
        args = data["args"]
    else:
        # Support flat JSON parameter payloads: {"name": "Bash", "command": "..."}
        reserved = {"name", "tool", "function", "id", "type"}
        flat_args = {k: v for k, v in data.items() if k not in reserved}
        args = flat_args if flat_args else {}

    if isinstance(args, str):
        args_clean = _sanitize_json_string(args)
        try:
            args = json.loads(args_clean)
        except Exception:
            try:
                args = json.loads(_escape_control_chars_in_json_strings(args_clean))
            except Exception:
                pass
    if not isinstance(args, dict):
        args = {}

    norm_name = normalize_tool_name(name.strip(), args)
    cleaned_args = normalize_tool_parameters(norm_name, _unwrap_autolinks_in_obj(args))
    return {"name": norm_name, "arguments": cleaned_args}


def _fallback_extract_arguments(func_name: str, raw_text: str) -> dict:
    """Robust fallback parameter extractor for LLM JSON outputs containing raw code, unescaped quotes, or newlines."""
    fn_lower = func_name.lower()
    args: dict = {}

    if fn_lower in ("write", "write_file"):
        fp_match = re.search(r'"(?:file_path|path|file)"\s*:\s*"([^"]+)"', raw_text)
        if fp_match:
            args["file_path"] = fp_match.group(1).strip()
        c_match = re.search(r'"content"\s*:\s*', raw_text)
        if c_match:
            c_raw = raw_text[c_match.end() :].strip()
            if c_raw.startswith('"'):
                c_raw = c_raw[1:]
            c_clean = re.sub(r'"?\s*\}*\s*$', "", c_raw)
            c_clean = c_clean.replace(r'\"', '"')
            args["content"] = c_clean

    elif fn_lower in ("bash", "execute_command", "sh"):
        cmd_match = re.search(r'"(?:command|cmd)"\s*:\s*"([^"]+)"', raw_text)
        if cmd_match:
            args["command"] = cmd_match.group(1).strip()
        else:
            c_match = re.search(r'"(?:command|cmd)"\s*:\s*', raw_text)
            if c_match:
                c_raw = raw_text[c_match.end() :].strip()
                if c_raw.startswith('"'):
                    c_raw = c_raw[1:]
                c_clean = re.sub(r'"?\s*\}*\s*$', "", c_raw)
                c_clean = c_clean.replace(r'\"', '"')
                args["command"] = c_clean

    elif fn_lower in ("edit", "edit_file", "update", "update_file"):
        fp_match = re.search(r'"(?:file_path|path|file)"\s*:\s*"([^"]+)"', raw_text)
        if fp_match:
            args["file_path"] = fp_match.group(1).strip()
        old_match = re.search(r'"(?:old_string|old_str)"\s*:\s*', raw_text)
        new_match = re.search(r'"(?:new_string|new_str)"\s*:\s*', raw_text)
        if old_match and new_match:
            if old_match.start() < new_match.start():
                old_raw = raw_text[old_match.end() : new_match.start()].strip()
                old_clean = re.sub(r'[\s",]+$', "", old_raw.lstrip('"'))
                new_raw = raw_text[new_match.end() :].strip()
                new_clean = re.sub(r'"?\s*\}*\s*$', "", new_raw.lstrip('"'))
                args["old_string"] = old_clean.replace(r'\"', '"')
                args["new_string"] = new_clean.replace(r'\"', '"')
        elif old_match:
            old_raw = raw_text[old_match.end() :].strip()
            old_clean = re.sub(r'"?\s*\}*\s*$', "", old_raw.lstrip('"'))
            args["old_string"] = old_clean.replace(r'\"', '"')
        c_match = re.search(r'"content"\s*:\s*', raw_text)
        if c_match and "old_string" not in args:
            c_raw = raw_text[c_match.end() :].strip()
            if c_raw.startswith('"'):
                c_raw = c_raw[1:]
            c_clean = re.sub(r'"?\s*\}*\s*$', "", c_raw)
            c_clean = c_clean.replace(r'\"', '"')
            args["content"] = c_clean

    elif fn_lower in ("read", "read_file"):
        fp_match = re.search(r'"(?:file_path|path|file)"\s*:\s*"([^"]+)"', raw_text)
        if fp_match:
            args["file_path"] = fp_match.group(1).strip()
        for int_field in ("offset", "limit"):
            m = re.search(rf'"{int_field}"\s*:\s*(\d+)', raw_text)
            if m:
                args[int_field] = int(m.group(1))

    elif fn_lower in ("glob", "grep"):
        pat_match = re.search(r'"pattern"\s*:\s*"([^"]+)"', raw_text)
        if pat_match:
            args["pattern"] = pat_match.group(1).strip()
        path_match = re.search(r'"(?:path|dir|directory)"\s*:\s*"([^"]+)"', raw_text)
        if path_match:
            args["path"] = path_match.group(1).strip()

    return args


def parse_tool_payload(raw_payload: str) -> dict | None:
    """Safely parse a string payload (JSON, <invoke> XML, or direct tool XML tags) into a tool call dict with 'name' and 'arguments'."""
    if not raw_payload or not raw_payload.strip():
        return None

    unwrapped = unwrap_markdown_autolinks(raw_payload).strip()
    if "<tool_call>" in unwrapped or "<tool_call " in unwrapped or unwrapped.startswith("<tool_call"):
        unwrapped = re.sub(r"</?tool_call[^>]*>", "", unwrapped).strip()
        unwrapped = re.sub(r"^>+", "", unwrapped).strip()
        unwrapped = re.sub(r"[ \t]*>$", "", unwrapped).strip()

    # 1. Support direct tool XML tags e.g. <update>...</update>, <edit>...</edit>, <write>...</write>, <bash>...</bash>
    tag_m = re.match(r"^\s*<([a-zA-Z0-9_-]+)", unwrapped)
    if tag_m and tag_m.group(1).lower() in DIRECT_TOOL_TAGS:
        cand_tag = tag_m.group(1).lower()
        params: dict = {}
        for dk in (
            "file_path", "path", "file", "filename", "filepath",
            "old_string", "new_string", "old_str", "new_str",
            "content", "command", "cmd", "pattern", "replace_all"
        ):
            dk_match = re.search(rf'<{dk}\b[^>]*>(.*?)</{dk}>', unwrapped, re.DOTALL)
            if dk_match:
                val = dk_match.group(1)
                if dk in ("content", "old_string", "new_string", "old_str", "new_str"):
                    val = re.sub(r"^\r?\n|\r?\n$", "", val)
                else:
                    val = val.strip()
                val = unwrap_markdown_autolinks(val)
                if "&" in val:
                    val = html.unescape(val)
                params[dk] = val

        open_tag_match = re.match(r"^\s*<[^>]+>", unwrapped)
        if open_tag_match:
            open_tag_str = open_tag_match.group(0)
            attr_matches = re.finditer(r'([a-zA-Z0-9_-]+)=["\']([^"\']*)["\']', open_tag_str)
            for am in attr_matches:
                k, v = am.group(1), am.group(2).strip()
                if k not in params:
                    params[k] = v

        # Parameter name normalization
        if "file_path" not in params:
            for alt in ("path", "file", "filename", "filepath"):
                if alt in params:
                    params["file_path"] = params.pop(alt)
                    break
        if "old_string" not in params and "old_str" in params:
            params["old_string"] = params.pop("old_str")
        if "new_string" not in params and "new_str" in params:
            params["new_string"] = params.pop("new_str")
        if "command" not in params:
            for alt in ("cmd", "code", "script"):
                if alt in params:
                    params["command"] = params.pop(alt)
                    break

        norm_name = normalize_tool_name(cand_tag, params)
        return {"name": norm_name, "arguments": normalize_tool_parameters(norm_name, _unwrap_autolinks_in_obj(params))}

    # 2. Support <invoke name="...">...</invoke> XML format if present
    if "<parameter" in unwrapped or unwrapped.startswith("<invoke") or 'name="' in unwrapped[:60] or "name='" in unwrapped[:60]:
        m_name = re.search(r'name=["\']([^"\']+)["\']', unwrapped)
        if m_name and ("<parameter" in unwrapped or "</invoke>" in unwrapped or unwrapped.startswith("<invoke")):
            tool_name = m_name.group(1).strip()
            params = {}
            param_matches = list(
                re.finditer(r'<parameter\s+name=["\']([^"\']+)["\']\s*>(.*?)</parameter>', unwrapped, re.DOTALL)
            )
            if param_matches:
                for match in param_matches:
                    p_name = match.group(1)
                    params[p_name] = parse_parameter_xml_value(p_name, match.group(2))
                norm_name = normalize_tool_name(tool_name, params)
                return {"name": norm_name, "arguments": normalize_tool_parameters(norm_name, params)}
            # Check for raw JSON inside <invoke ...>{...}</invoke>
            brace_idx = unwrapped.find("{")
            if brace_idx != -1:
                balanced = _extract_balanced_braces(unwrapped, brace_idx)
                if balanced:
                    for candidate in (balanced, _escape_control_chars_in_json_strings(balanced)):
                        try:
                            parsed_inner = json.loads(re.sub(r",\s*(\}|\])", r"\1", candidate))
                            if isinstance(parsed_inner, dict):
                                norm_name = normalize_tool_name(tool_name, parsed_inner)
                                return {"name": norm_name, "arguments": normalize_tool_parameters(norm_name, _unwrap_autolinks_in_obj(parsed_inner))}
                        except Exception:
                            pass
            norm_name = normalize_tool_name(tool_name, params)
            return {"name": norm_name, "arguments": normalize_tool_parameters(norm_name, params)}

    clean = _sanitize_json_string(unwrapped)
    decoder = json.JSONDecoder()

    # 2. Try direct json.loads or raw_decode from the first '{' (handles trailing ']' or extra text)
    first_brace = clean.find("{")
    candidates = [clean]
    if first_brace > 0:
        candidates.append(clean[first_brace:])
    escaped_clean = _escape_control_chars_in_json_strings(clean)
    if escaped_clean != clean:
        candidates.append(escaped_clean)
        if first_brace > 0:
            candidates.append(escaped_clean[ escaped_clean.find("{") :])

    for cand in candidates:
        if not cand.startswith("{"):
            continue
        try:
            data, _ = decoder.raw_decode(cand)
            if isinstance(data, dict):
                norm = _normalize_parsed_tool_dict(data)
                if norm:
                    return norm
        except Exception:
            pass

    # 3. Fallback: extract "name" and balanced "{...}" for "arguments" / "input" / "parameters"
    name_match = re.search(r'"(?:name|tool|function)"\s*:\s*"([^"]+)"', clean)
    if name_match:
        func_name = name_match.group(1).strip()
        key_match = re.search(r'"(?:arguments|input|parameters|args)"\s*:\s*', clean)
        if key_match:
            after_key = clean[key_match.end() :].lstrip()
            if after_key.startswith("{"):
                balanced = _extract_balanced_braces(after_key, 0)
                target_str = balanced if balanced is not None else after_key
                for cand in (target_str, _escape_control_chars_in_json_strings(target_str)):
                    cand_clean = re.sub(r",\s*(\}|\])", r"\1", cand)
                    try:
                        args_obj, _ = decoder.raw_decode(cand_clean)
                        if isinstance(args_obj, dict):
                            norm_name = normalize_tool_name(func_name, args_obj)
                            return {"name": norm_name, "arguments": normalize_tool_parameters(norm_name, _unwrap_autolinks_in_obj(args_obj))}
                    except Exception:
                        pass
            elif after_key.startswith('"'):
                try:
                    str_val, _ = decoder.raw_decode(after_key)
                    if isinstance(str_val, str):
                        args_obj = json.loads(_sanitize_json_string(str_val))
                        if isinstance(args_obj, dict):
                            norm_name = normalize_tool_name(func_name, args_obj)
                            return {"name": norm_name, "arguments": normalize_tool_parameters(norm_name, _unwrap_autolinks_in_obj(args_obj))}
                except Exception:
                    pass

        # Robust parameter recovery for unescaped code / malformed JSON
        fallback_args = _fallback_extract_arguments(func_name, clean)
        if fallback_args:
            norm_name = normalize_tool_name(func_name, fallback_args)
            return {"name": norm_name, "arguments": normalize_tool_parameters(norm_name, _unwrap_autolinks_in_obj(fallback_args))}

        norm_name = normalize_tool_name(func_name)
        return {"name": norm_name, "arguments": {}}

    return None


class StreamEmulationProcessor:
    """
    Sliding-window stream processor that buffers potential tag boundaries
    and transforms emulated tool calls into standard OpenAI SSE chunks.
    """

    def __init__(self, context: dict):
        self.context = context
        self.model = context.get("model", "auto")
        self.stop_sequences: list[str] = context.get("stop", [])
        self.max_tokens: int | None = context.get("max_tokens")
        self.is_synthesis_phase: bool = context.get("is_synthesis_phase", False)
        self.executed_tools: set[str] = set(context.get("executed_tool_names", []))

        self.chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())

        self.buffer = ""
        self.in_tool_tag = False
        self.active_close_tag = CLOSE_TAG
        self.tool_accumulator = ""
        self.tool_calls_emitted: list[dict] = []
        self.tool_call_index = 0
        self.conversation_id: str | None = None
        self.parent_message_id: str | None = None

        self.total_tokens_estimated = 0
        self.stop_triggered = False
        self.length_triggered = False
        self.first_chunk_sent = False

        self.system_fingerprint = context.get("system_fingerprint") or f"fp_ghost_{uuid.uuid4().hex[:10]}"
        self.stream_options = context.get("stream_options") or {}
        self.include_usage = bool(
            isinstance(self.stream_options, dict)
            and self.stream_options.get("include_usage")
        )
        self.prompt_tokens = context.get("prompt_tokens", 0)
        self.expose_session_metadata_in_chunks = context.get("expose_session_metadata_in_chunks", False)

    async def process_stream(
        self, raw_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """Consumes provider SSE lines and yields emulated OpenAI SSE strings."""
        async for line in raw_stream:
            if self.stop_triggered or self.length_triggered:
                break

            line_str = line.strip()
            if not line_str.startswith("data: "):
                continue

            payload = line_str[6:].strip()
            if payload == "[DONE]":
                break

            try:
                chunk_dict = json.loads(payload)
            except Exception:
                continue

            if not isinstance(chunk_dict, dict):
                continue

            # Capture chunk_id and model from upstream if available
            if "id" in chunk_dict and chunk_dict["id"]:
                self.chunk_id = chunk_dict["id"]
            if "model" in chunk_dict and chunk_dict["model"]:
                self.model = chunk_dict["model"]
            if "conversation_id" in chunk_dict and chunk_dict["conversation_id"]:
                self.conversation_id = chunk_dict["conversation_id"]
            if "parent_message_id" in chunk_dict and chunk_dict["parent_message_id"]:
                self.parent_message_id = chunk_dict["parent_message_id"]

            choices = chunk_dict.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            text_delta = delta.get("content", "")

            # If there's content text, process it through the state machine
            if text_delta:
                async for sse_chunk in self._feed_text(text_delta):
                    yield sse_chunk
                    if self.stop_triggered or self.length_triggered:
                        break

        # Stream ended or truncated: flush buffers and emit final chunks
        async for sse_chunk in self._finalize():
            yield sse_chunk

        if self.include_usage:
            usage_chunk = {
                "id": self.chunk_id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "system_fingerprint": self.system_fingerprint,
                "choices": [],
                "usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": max(1, self.total_tokens_estimated),
                    "total_tokens": self.prompt_tokens + max(1, self.total_tokens_estimated),
                },
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    async def _feed_text(self, text: str) -> AsyncGenerator[str, None]:
        """Processes incoming text fragment with lookahead sliding window."""
        cleaned = text.replace("<escape>", "").replace("</escape>", "")
        if not cleaned:
            return
        self.buffer += cleaned

        while self.buffer:
            if not self.in_tool_tag:
                # Look for earliest opening tool tag (<tool_call> or <invoke)
                open_idx = -1
                matched_open_tag = None
                for candidate_tag in OPEN_TAGS:
                    idx = self.buffer.find(candidate_tag)
                    if idx != -1 and (open_idx == -1 or idx < open_idx):
                        open_idx = idx
                        matched_open_tag = candidate_tag

                if open_idx != -1 and matched_open_tag is not None:
                    # Check if this tag is already complete and self-closing, e.g. <read path="..."/>
                    remainder = self.buffer[open_idx:]
                    self_close_m = re.match(r"^<[a-zA-Z0-9_-]+\b[^>]*?\/>", remainder)
                    if self_close_m:
                        pre_text = self.buffer[:open_idx]
                        pre_text = re.sub(r"https?://googleusercontent\.com/card_content/\d+\s*", "", pre_text)
                        if pre_text and pre_text.strip():
                            async for c in self._emit_content(pre_text):
                                yield c
                                if self.stop_triggered or self.length_triggered:
                                    return
                        full_tag = self_close_m.group(0)
                        self.buffer = self.buffer[open_idx + len(full_tag):]
                        async for c in self._emit_tool_call(full_tag):
                            yield c
                        continue

                    # Emit everything before the open tag as normal content
                    pre_text = self.buffer[:open_idx]
                    pre_text = re.sub(r"https?://googleusercontent\.com/card_content/\d+\s*", "", pre_text)
                    if pre_text and pre_text.strip():
                        async for c in self._emit_content(pre_text):
                            yield c
                            if self.stop_triggered or self.length_triggered:
                                return

                    # Transition to in_tool_tag
                    self.in_tool_tag = True
                    tag_name = matched_open_tag[1:].lower()
                    if tag_name == "tool_call":
                        self.active_close_tag = CLOSE_TAG
                        self.tool_accumulator = ""
                    else:
                        self.active_close_tag = f"</{tag_name}>"
                        self.tool_accumulator = matched_open_tag
                    self.buffer = self.buffer[open_idx + len(matched_open_tag):]
                    continue

                # Not found: check if buffer contains a GenUI component tag or incomplete GenUI tag
                genui_match = re.search(r'<[A-Z]', self.buffer)
                if genui_match:
                    start_idx = genui_match.start()
                    if start_idx > 0:
                        safe_text = self.buffer[:start_idx]
                        self.buffer = self.buffer[start_idx:]
                        if safe_text:
                            async for c in self._emit_content(safe_text):
                                yield c
                                if self.stop_triggered or self.length_triggered:
                                    return
                        continue

                    # Buffer starts with <[A-Z]; check if the tag is complete
                    gm = GENUI_OPEN_PATTERN.match(self.buffer)
                    if gm:
                        full_tag = gm.group(0)
                        sanitized = sanitize_genui_tags(full_tag)
                        self.buffer = self.buffer[gm.end():]
                        if sanitized:
                            async for c in self._emit_content(sanitized):
                                yield c
                                if self.stop_triggered or self.length_triggered:
                                    return
                        continue
                    else:
                        # Incomplete GenUI tag; wait for more stream chunks
                        break

                # Check if buffer ends with a potential prefix of any open tag
                match_prefix_len = 0
                for candidate_tag in OPEN_TAGS:
                    for length in range(min(len(candidate_tag) - 1, len(self.buffer)), 0, -1):
                        if candidate_tag.startswith(self.buffer[-length:]):
                            if length > match_prefix_len:
                                match_prefix_len = length
                            break

                if match_prefix_len > 0:
                    # Safe text to emit is before the prefix
                    safe_text = self.buffer[:-match_prefix_len]
                    if safe_text:
                        async for c in self._emit_content(safe_text):
                            yield c
                            if self.stop_triggered or self.length_triggered:
                                return
                    self.buffer = self.buffer[-match_prefix_len:]
                    break  # Wait for more input to resolve tag prefix
                else:
                    # Entire buffer is safe content
                    safe_text = self.buffer
                    self.buffer = ""
                    async for c in self._emit_content(safe_text):
                        yield c
                        if self.stop_triggered or self.length_triggered:
                            return
                    break

            else:
                # Inside tool tag: check for self-closing tag finish or active_close_tag
                if not self.tool_accumulator.endswith(">"):
                    m_self = re.search(r'^[^>]*?\/>', self.buffer)
                    if m_self:
                        self.tool_accumulator += self.buffer[:m_self.end()]
                        self.buffer = self.buffer[m_self.end():]
                        self.in_tool_tag = False
                        async for c in self._emit_tool_call(self.tool_accumulator):
                            yield c
                        self.tool_accumulator = ""
                        continue

                close_tag = self.active_close_tag
                close_idx = self.buffer.find(close_tag)
                if close_idx != -1:
                    self.tool_accumulator += self.buffer[:close_idx]
                    if close_tag != CLOSE_TAG:
                        self.tool_accumulator += close_tag
                    self.buffer = self.buffer[close_idx + len(close_tag):]
                    self.in_tool_tag = False

                    # Emit complete tool call chunk
                    async for c in self._emit_tool_call(self.tool_accumulator):
                        yield c
                    self.tool_accumulator = ""
                    continue

                # Not closed yet: check for potential prefix of close_tag
                match_prefix_len = 0
                for length in range(min(len(close_tag) - 1, len(self.buffer)), 0, -1):
                    if close_tag.startswith(self.buffer[-length:]):
                        match_prefix_len = length
                        break

                if match_prefix_len > 0:
                    safe_acc = self.buffer[:-match_prefix_len]
                    self.tool_accumulator += safe_acc
                    self.buffer = self.buffer[-match_prefix_len:]
                    break
                else:
                    self.tool_accumulator += self.buffer
                    self.buffer = ""
                    break

    async def _emit_content(self, text: str) -> AsyncGenerator[str, None]:
        """Emits standard OpenAI delta.content chunk after stop and max_tokens checks."""
        # Strip internal google card tokens if any leak through
        text = re.sub(r"https?://googleusercontent\.com/card_content/\d+\s*", "", text)
        text = unwrap_markdown_autolinks(text)
        if not text:
            return

        # Stop sequence checking
        if self.stop_sequences:
            for s in self.stop_sequences:
                if s in text:
                    idx = text.find(s)
                    truncated_text = text[:idx]
                    if truncated_text:
                        yield self._build_chunk(delta={"content": truncated_text}, finish_reason=None)
                    self.stop_triggered = True
                    yield self._build_chunk(delta={}, finish_reason="stop")
                    return

        # Max tokens checking
        if self.max_tokens is not None:
            # Approximate 4 characters per token
            est_tokens = max(1, len(text) // 4)
            if self.total_tokens_estimated + est_tokens >= self.max_tokens:
                remaining_tokens = max(0, self.max_tokens - self.total_tokens_estimated)
                safe_chars = remaining_tokens * 4
                truncated_text = text[:safe_chars]
                if truncated_text:
                    yield self._build_chunk(delta={"content": truncated_text}, finish_reason=None)
                self.length_triggered = True
                yield self._build_chunk(delta={}, finish_reason="length")
                return
            self.total_tokens_estimated += est_tokens

        role_attr = {"role": "assistant"} if not self.first_chunk_sent else {}
        self.first_chunk_sent = True
        yield self._build_chunk(
            delta={"content": text, **role_attr},
            finish_reason=None,
        )

    async def _emit_tool_call(self, raw_tool_str: str) -> AsyncGenerator[str, None]:
        """Emits delta.tool_calls SSE chunk."""
        parsed = parse_tool_payload(raw_tool_str)
        if not parsed:
            # If payload couldn't be parsed, fallback to flushing raw text
            fallback_text = f"{OPEN_TAG}{raw_tool_str}{CLOSE_TAG}"
            async for c in self._emit_content(fallback_text):
                yield c
            return

        # Suppress re-invocation of tools whose results were already provided in synthesis phase
        if self.is_synthesis_phase and parsed["name"] in self.executed_tools:
            return

        call_id = f"call_{uuid.uuid4().hex[:12]}"
        func_name = normalize_tool_name(parsed["name"], parsed["arguments"])
        arguments_str = (
            json.dumps(parsed["arguments"])
            if not isinstance(parsed["arguments"], str)
            else parsed["arguments"]
        )

        tool_call_delta = {
            "index": self.tool_call_index,
            "id": call_id,
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": arguments_str,
            },
        }
        self.tool_calls_emitted.append(tool_call_delta)
        self.tool_call_index += 1

        role_attr = {"role": "assistant"} if not self.first_chunk_sent else {}
        self.first_chunk_sent = True

        yield self._build_chunk(
            delta={"tool_calls": [tool_call_delta], **role_attr},
            finish_reason=None,
        )

    async def _finalize(self) -> AsyncGenerator[str, None]:
        """Flushes remaining buffers and emits the final finish_reason chunk."""
        if self.stop_triggered or self.length_triggered:
            return

        # If stream closed while inside tool tag, attempt to parse what we got
        if self.in_tool_tag:
            full_tool_payload = self.tool_accumulator + self.buffer
            self.buffer = ""
            self.in_tool_tag = False
            if full_tool_payload.strip():
                async for c in self._emit_tool_call(full_tool_payload):
                    yield c
        elif self.buffer:
            # Remaining buffered text (like unmatched prefix '<' or partial GenUI tag)
            cleaned_remaining = sanitize_genui_tags(self.buffer)
            if not self.tool_calls_emitted or cleaned_remaining.strip():
                async for c in self._emit_content(cleaned_remaining):
                    yield c
            self.buffer = ""

        # Determine finish reason
        if self.tool_calls_emitted:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        yield self._build_chunk(delta={}, finish_reason=finish_reason)

    def _build_chunk(self, delta: dict, finish_reason: str | None) -> str:
        """Serializes dictionary into SSE line."""
        chunk = {
            "id": self.chunk_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if self.expose_session_metadata_in_chunks:
            if self.conversation_id:
                chunk["conversation_id"] = self.conversation_id
            if self.parent_message_id:
                chunk["parent_message_id"] = self.parent_message_id
        return f"data: {json.dumps(chunk)}\n\n"


def process_non_streaming_response(raw_resp: dict, context: dict) -> dict:
    """
    Parses full response dictionary for non-streaming completions:
    extracts tool calls, enforces stop/max_tokens, and formats standard message.tool_calls.
    """
    resp = dict(raw_resp)
    choices = resp.get("choices", [])
    if not choices:
        return resp

    choice = dict(choices[0])
    message = dict(choice.get("message", {}))
    raw_content = message.get("content") or ""

    def _finalize_response(r: dict, ch: dict, msg: dict) -> dict:
        ch["message"] = msg
        r["choices"] = [ch]
        r["object"] = "chat.completion"
        r.setdefault("id", f"chatcmpl-{uuid.uuid4().hex[:24]}")
        r.setdefault("created", int(time.time()))
        r.setdefault("model", context.get("model", "auto"))
        r.setdefault("system_fingerprint", f"fp_ghost_{uuid.uuid4().hex[:10]}")
        ch.setdefault("index", 0)
        ch.setdefault("logprobs", None)
        ch.setdefault("finish_reason", "stop")

        prompt_tokens = context.get("prompt_tokens", 0)
        final_text = msg.get("content") or ""
        completion_tokens = max(1, len(final_text) // 4) if final_text else 1
        r["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        if not context.get("expose_session_metadata_in_chunks", False):
            if "conversation_id" in r:
                context["captured_conversation_id"] = r.pop("conversation_id")
            if "parent_message_id" in r:
                context["captured_parent_message_id"] = r.pop("parent_message_id")
        return r

    stop_sequences: list[str] = context.get("stop", [])
    max_tokens: int | None = context.get("max_tokens")
    is_synthesis_phase: bool = context.get("is_synthesis_phase", False)
    executed_tools: set[str] = set(context.get("executed_tool_names", []))

    raw_content = raw_content.replace("<escape>", "").replace("</escape>", "")
    raw_content = re.sub(r"https?://googleusercontent\.com/card_content/\d+\s*", "", raw_content)
    raw_content = unwrap_markdown_autolinks(raw_content)
    raw_content = sanitize_genui_tags(raw_content)

    # 1. Enforce stop sequences
    if stop_sequences:
        for s in stop_sequences:
            if s in raw_content:
                raw_content = raw_content[: raw_content.find(s)]
                choice["finish_reason"] = "stop"

    # 2. Extract <tool_call>, <invoke>, and direct tool XML blocks
    pattern = re.compile(
        r"(<tool_call\b[^>]*>.*?(?:</tool_call>|\}[ \t]*>))|"
        r"(<invoke\b.*?</invoke>)|"
        r"(<(?:update|edit|write|bash|read|glob|grep|ls)\b[^>]*?(?:\/>|>.*?</(?:update|edit|write|bash|read|glob|grep|ls)>))",
        re.DOTALL | re.IGNORECASE,
    )
    matches = [m.group(0) for m in pattern.finditer(raw_content)]

    if matches:
        tool_calls = []
        for i, match in enumerate(matches):
            parsed = parse_tool_payload(match)
            if parsed:
                if is_synthesis_phase and parsed["name"] in executed_tools:
                    continue

                call_id = f"call_{uuid.uuid4().hex[:12]}"
                args_str = (
                    json.dumps(parsed["arguments"])
                    if not isinstance(parsed["arguments"], str)
                    else parsed["arguments"]
                )
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": parsed["name"],
                        "arguments": args_str,
                    },
                })

        clean_content = pattern.sub("", raw_content).strip()

        if tool_calls:
            message["content"] = clean_content if clean_content else None
            message["tool_calls"] = tool_calls
            choice["finish_reason"] = "tool_calls"
            return _finalize_response(resp, choice, message)
        elif clean_content:
            # All tool calls were suppressed echoes; emit synthesized content
            message["content"] = clean_content
            message.pop("tool_calls", None)
            choice["finish_reason"] = "stop"
            return _finalize_response(resp, choice, message)

    # 3. Enforce max_tokens if no tool calls
    if max_tokens is not None:
        est_tokens = max(1, len(raw_content) // 4)
        if est_tokens > max_tokens:
            raw_content = raw_content[: max_tokens * 4]
            choice["finish_reason"] = "length"

    message["content"] = raw_content
    return _finalize_response(resp, choice, message)
