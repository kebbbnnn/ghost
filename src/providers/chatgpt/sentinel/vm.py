"""
vm.py — Python implementation of the Sentinel SDK Turnstile VM (jt / _n function).

The VM:
  1. XOR-decrypts the dx bytecode with the gAAAAAC requirements token (p_token)
  2. Parses the result as a JSON instruction array
  3. Resolves the opcode alias map (via COPY chain from instruction 0)
  4. Executes data-collection instructions
  5. Resolves to a base64-encoded token via RESOLVE
"""
import base64
import json
import random
import time
import re

from .browser_env import BrowserEnv

# ---------------------------------------------------------------------------
# Opcode IDs (as defined in the deobfuscated SDK)
# ---------------------------------------------------------------------------
OP_LOAD_PROGRAM   = 0
OP_XOR_REG        = 1
OP_SET            = 2
OP_RESOLVE        = 3
OP_REJECT         = 4
OP_ADD            = 5
OP_GET_INDEX      = 6
OP_CALL           = 7
OP_COPY           = 8
OP_IP             = 9
OP_WINDOW         = 10
OP_SCRIPT_MATCH   = 11
OP_GET_VM         = 12
OP_TRY_CALL       = 13
OP_JSON_PARSE     = 14
OP_JSON_STRINGIFY = 15
OP_KEY            = 16
OP_TRY_CALL_RES   = 17
OP_ATOB           = 18
OP_BTOA           = 19
OP_IF_EQUAL       = 20
OP_IF_NOT_CLOSE   = 21
OP_SUBROUTINE     = 22
OP_IF_DEFINED     = 23
OP_BIND           = 24
OP_SUB            = 27
OP_LESS_THAN      = 29
OP_DEFINE_FUNC    = 30
OP_MULTIPLY       = 33
OP_AWAIT          = 34
OP_DIVIDE         = 35

# ---------------------------------------------------------------------------
# Simulated browser/window environment
# ---------------------------------------------------------------------------

class _Env:
    """Simulated browser globals."""

    WINDOW_PROPS = {
        "Reflect":            {"set": {"_method": "set"}},
        "Object":             {"keys": {"_method": "keys"}, "create": {"_method": "create"}, "getOwnPropertyNames": {"_method": "getOwnPropertyNames"}},
        "Math":               {"random": {"_method": "random"}},
        "performance":        {"now": {"_method": "now"}, "timeOrigin": BrowserEnv.time_origin(), "memory": {"jsHeapSizeLimit": 4294967296}},
        "navigator":          {
            "userAgent":             BrowserEnv.user_agent,
            "language":              BrowserEnv.language,
            "languages":             BrowserEnv.languages,
            "hardwareConcurrency":   BrowserEnv.hardware_concurrency,
            "deviceMemory":          BrowserEnv.device_memory,
            "vendor":                BrowserEnv.vendor,
            "platform":              BrowserEnv.platform,
            "maxTouchPoints":        BrowserEnv.max_touch_points,
            "storage":               {"estimate": {"_method": "estimate"}, "_quota": 296630877388, "_usage": 913230},
        },
        "screen": {
            "width":           BrowserEnv.screen_width,
            "height":          BrowserEnv.screen_height,
            "availWidth":      BrowserEnv.screen_width,
            "availHeight":     BrowserEnv.screen_height - 23,
            "availLeft":       0,
            "availTop":        0,
            "colorDepth":      24,
            "pixelDepth":      24,
        },
        "document":           {"cookie": "", "referrer": "", "body": {"style": {}}},
        "localStorage":       {
            "d2bd098fc8793ef1": "0",
            "statsig.session_id.444584300": "0",
            "statsig.last_modified_time.evaluations": "0",
            "statsig.stable_id.444584300": "0",
            "6fbbfe1cd1015f3d": "0",
            "statsig.cached.evaluations.3523433505": "0"
        },
        "sessionStorage":     {},
        "history":            {"length": BrowserEnv.history_length},
        "location":           {"href": "https://chatgpt.com/"},
        # Top-level navigator.storage is also accessed as window.storage by some paths
        "storage":            {"estimate": {"_method": "estimate"}, "_quota": 296630877388, "_usage": 913230},
        "length":             0,
    }

    @staticmethod
    def call(fn_name: str, args: list):
        """Simulate a browser function call and return its result."""
        # performance.now()
        if fn_name == "now" or "performance" in str(fn_name) and "now" in str(fn_name):
            return BrowserEnv.performance_now()

        # Math.random()
        if fn_name == "random":
            return random.random()

        # Date.now()
        if fn_name == "getTime" or "Date" in str(fn_name):
            return int(time.time() * 1000)

        # getTimezoneOffset()
        if "TimezoneOffset" in str(fn_name) or fn_name == "getTimezoneOffset":
            return BrowserEnv.timezone_offset

        # toLocaleString()
        if "toLocaleString" in str(fn_name):
            return time.strftime("%-m/%-d/%Y, %H:%M:%S %p")

        # Object.keys / Object.getOwnPropertyNames
        if fn_name in ("keys", "getOwnPropertyNames"):
            obj = args[0] if args else {}
            if isinstance(obj, dict):
                return list(obj.keys())
            return ["length"]

        # Object.create
        if fn_name == "create":
            return {}

        # WebGL
        if fn_name == "getContext" and args and args[0] == "webgl":
            return {"_type": "WebGLRenderingContext"}
        if fn_name == "getExtension" and args and "debug_renderer" in str(args[0]):
            return {"_type": "WEBGL_debug_renderer_info", "UNMASKED_VENDOR_WEBGL": 37445, "UNMASKED_RENDERER_WEBGL": 37446}
        if fn_name == "getParameter":
            param = args[0] if args else 0
            if param == 37445:
                return BrowserEnv.gpu_vendor
            if param == 37446:
                return BrowserEnv.gpu_renderer
            return 0

        # getBoundingClientRect
        if "BoundingClientRect" in str(fn_name):
            return {"x": 0, "y": 744.6875, "width": 19.640625, "height": 14,
                    "top": 744.6875, "right": 19.640625, "bottom": 758.6875, "left": 0}

        # Storage estimate (navigator.storage.estimate())
        if fn_name == "estimate":
            # JS expects a Promise. We return a mock object where `.then`
            # contains the resolved value, which `_call_fn` will pass to the callback.
            return {
                "then": {
                    "_type": "promise_then",
                    "_value": {"quota": 296630877388, "usage": 913230}
                }
            }

        # Reflect.set — spoof screen properties
        if fn_name == "set":
            obj, key, val = (args + [None, None, None])[:3]
            if isinstance(obj, dict) and key is not None:
                # Use our spoofed screen values
                screen_spoof = {
                    "width": BrowserEnv.screen_width,
                    "height": BrowserEnv.screen_height,
                    "availWidth": BrowserEnv.screen_width,
                    "availHeight": BrowserEnv.screen_height - 23,
                    "colorDepth": 24,
                    "pixelDepth": 24,
                }
                obj[key] = screen_spoof.get(key, val)
            return True

        if fn_name == "random":
            return 0.25

        if fn_name == "now":
            return BrowserEnv.performance_now()

        if fn_name == "timeOrigin":
            return BrowserEnv.time_origin()

        return None


# ---------------------------------------------------------------------------
# VM Executor
# ---------------------------------------------------------------------------

class _VM:
    """
    Sentinel Turnstile VM executor.
    
    Registers are keyed by arbitrary floats/strings (not just ints).
    The opcode alias map is built from the COPY chain at the start.
    """

    def __init__(self):
        self.reg: dict = {}
        self.result: str | None = None
        self._opcode_aliases: dict = {}  # reg_id → canonical opcode int

        # Seed registers with opcode handlers so dynamic dispatch (e.g., IF_DEFINED) works
        def make_handler(op_code):
            # fn_args is a list of register IDs passed directly by IF_DEF/IF_EQ/etc.
            return lambda args: self._dispatch(op_code, args)

        for op in [
            OP_LOAD_PROGRAM, OP_XOR_REG, OP_SET,
            OP_ADD, OP_GET_INDEX, OP_CALL, OP_COPY,
            OP_TRY_CALL, OP_JSON_PARSE,
            OP_JSON_STRINGIFY, OP_KEY, OP_TRY_CALL_RES, OP_ATOB, OP_BTOA,
            OP_IF_EQUAL, OP_IF_NOT_CLOSE, OP_SUBROUTINE, OP_IF_DEFINED,
            OP_BIND, OP_SUB, OP_LESS_THAN, OP_DEFINE_FUNC, OP_MULTIPLY,
            OP_AWAIT, OP_DIVIDE
        ]:
            self.reg[op] = make_handler(op)
            self.reg[float(op)] = make_handler(op)

        # Set OP_WINDOW to the simulated global object dictionary
        self.reg[OP_WINDOW] = _Env.WINDOW_PROPS
        self.reg[float(OP_WINDOW)] = _Env.WINDOW_PROPS

        # RESOLVE (3) and REJECT (4) are called with literal values, not register IDs
        def _resolve(val=None, *_):
            try:
                self.result = base64.b64encode(str(val or "").encode("latin-1")).decode("latin-1")
            except Exception:
                self.result = base64.b64encode(str(val or "").encode("utf-8", errors="replace")).decode("latin-1")

        def _reject(val=None, *_):
            self.result = ""

        self.reg[OP_RESOLVE] = _resolve
        self.reg[float(OP_RESOLVE)] = _resolve
        self.reg[OP_REJECT] = _reject
        self.reg[float(OP_REJECT)] = _reject

    # ------------------------------------------------------------------
    # Phase 1: Build opcode alias map from COPY chain
    # ------------------------------------------------------------------

    def _build_alias_map(self, instructions: list):
        """
        The first instruction is always [8, X, 8] defining the COPY register.
        Subsequent [X, target, source_opcode] define all other opcode aliases.
        """
        if not instructions:
            return
        first = instructions[0]
        if len(first) < 3 or first[0] != OP_COPY:
            return

        copy_reg = first[1]
        self._opcode_aliases[copy_reg] = OP_COPY

        for instr in instructions[1:]:
            if (isinstance(instr, list) and len(instr) == 3
                    and instr[0] == copy_reg):
                target, source = instr[1], instr[2]
                if isinstance(source, (int, float)):
                    # source might itself be an alias already resolved
                    self._opcode_aliases[target] = self._opcode_aliases.get(
                        source, int(source)
                    )

    def _resolve_op(self, raw_op) -> int:
        """Resolve a raw opcode (possibly a float alias) to a canonical int."""
        if isinstance(raw_op, float) and raw_op in self._opcode_aliases:
            return self._opcode_aliases[raw_op]
        return int(raw_op)

    # ------------------------------------------------------------------
    # Phase 2: Register access
    # ------------------------------------------------------------------

    def _get(self, key):
        return self.reg.get(key)

    def _set(self, key, value):
        self.reg[key] = value

    # ------------------------------------------------------------------
    # Phase 3: Opcode handlers
    # ------------------------------------------------------------------

    def _handle_set(self, args):
        """SET dest, literal_value"""
        if len(args) >= 2:
            self._set(args[0], args[1])

    def _handle_copy(self, args):
        """COPY dest, src_reg"""
        if len(args) >= 2:
            self._set(args[0], self._get(args[1]))

    def _handle_xor(self, args):
        """XOR_REG dest, key_reg  → dest = xor(dest, key)"""
        if len(args) < 2:
            return
        val = self._get(args[0])
        key = self._get(args[1])
        if val is None or key is None:
            return
        val_s, key_s = str(val), str(key)
        result = "".join(chr(ord(v) ^ ord(key_s[i % len(key_s)])) for i, v in enumerate(val_s))
        self._set(args[0], result)

    def _handle_add(self, args):
        """ADD dest, src_reg"""
        if len(args) < 2:
            return
        cur = self._get(args[0])
        val = self._get(args[1])
        if isinstance(cur, list):
            cur.append(val)
        elif isinstance(cur, (int, float)) and isinstance(val, (int, float)):
            self._set(args[0], cur + val)
        else:
            self._set(args[0], str(cur or "") + str(val or ""))

    def _handle_sub(self, args):
        """SUB dest, src_reg"""
        if len(args) < 2:
            return
        cur = self._get(args[0])
        val = self._get(args[1])
        if isinstance(cur, list):
            try:
                cur.remove(val)
            except ValueError:
                pass
        elif isinstance(cur, (int, float)) and isinstance(val, (int, float)):
            self._set(args[0], cur - val)

    def _handle_multiply(self, args):
        if len(args) >= 3:
            a, b = self._get(args[1]), self._get(args[2])
            self._set(args[0], float(a or 0) * float(b or 0))

    def _handle_divide(self, args):
        if len(args) >= 3:
            a, b = float(self._get(args[1]) or 0), float(self._get(args[2]) or 0)
            self._set(args[0], 0 if b == 0 else a / b)

    def _handle_get_index(self, args):
        """GET_INDEX dest, obj, prop"""
        if len(args) < 3:
            return
        obj = self._get(args[1])
        prop = self._get(args[2])
        if isinstance(obj, dict):
            self._set(args[0], obj.get(prop))
        elif hasattr(obj, "__getitem__") and not isinstance(obj, str):
            try:
                self._set(args[0], obj[prop])
            except (KeyError, IndexError, TypeError):
                self._set(args[0], None)
        else:
            self._set(args[0], None)

    def _handle_bind(self, args):
        """BIND dest, obj_reg, method_name_reg → store bound method reference"""
        if len(args) < 3:
            return
        obj = self._get(args[1])
        method = self._get(args[2])
        self._set(args[0], {"_obj": obj, "_method": str(method or "")})

    def _handle_window(self, args):
        """WINDOW dest  → set dest to the 'window' object (our env dict)"""
        if args:
            self._set(args[0], _Env.WINDOW_PROPS)

    def _handle_call(self, args):
        """CALL fn_reg, *arg_regs"""
        if not args:
            return
        fn = self._get(args[0])
        call_args = [self._get(r) for r in args[1:]]
        result = self._call_fn(fn, call_args)
        return result

    def _handle_try_call(self, args):
        """TRY_CALL dest, fn_reg, *arg_regs — stores result in dest"""
        if len(args) < 2:
            return
        fn = self._get(args[1])
        call_args = [self._get(r) for r in args[2:]]
        try:
            result = self._call_fn(fn, call_args)
            self._set(args[0], result)
        except Exception as exc:
            self._set(args[0], str(exc))

    def _handle_try_call_res(self, args):
        """TRY_CALL_RES — same as TRY_CALL for our purposes"""
        self._handle_try_call(args)

    def _call_fn(self, fn, call_args: list):
        fn_name = "unknown"
        if isinstance(fn, dict) and "_method" in fn:
            fn_name = fn["_method"]
            args = call_args[1:]
        elif isinstance(fn, str):
            fn_name = fn
            args = call_args[1:]
        else:
            args = call_args

        if fn is None:
            # INTERCEPT: If the VM calls a missing global function with a long string,
            # this is usually the computed Turnstile/Sentinel payload.
            for arg in call_args:
                if isinstance(arg, str) and len(arg) > 100:
                    self.result = arg
                    return None
            return None

        # Bound method reference from BIND
        if isinstance(fn, dict) and "_method" in fn:
            method_name = fn["_method"]
            obj = fn.get("_obj")
            
            # Special cases
            if method_name == "set" and isinstance(obj, dict) and "_type" not in obj:
                # Reflect.set(target, key, value)
                return _Env.call("set", call_args)
            
            # Get property from object then call it
            if isinstance(obj, dict):
                actual_method = obj.get(method_name)
                if callable(actual_method):
                    return actual_method(*call_args)
                # Handle synchronous Promise resolution for mocked JS Promises
                if isinstance(actual_method, dict) and actual_method.get("_type") == "promise_then":
                    callback = call_args[0] if call_args else None
                    if callable(callback):
                        return callback(actual_method.get("_value"))
                    return None
            
            return _Env.call(method_name, call_args)

        if isinstance(fn, str):
            return _Env.call(fn, call_args)

        if callable(fn):
            return fn(*call_args)

        return None

    def _handle_json_parse(self, args):
        if len(args) >= 2:
            raw = str(self._get(args[1]) or "null")
            try:
                self._set(args[0], json.loads(raw))
            except Exception:
                self._set(args[0], None)

    def _handle_json_stringify(self, args):
        if len(args) >= 2:
            val = self._get(args[1])
            try:
                self._set(args[0], json.dumps(val, separators=(",", ":")))
            except Exception:
                self._set(args[0], str(val))

    def _handle_atob(self, args):
        if args:
            val = self._get(args[0])
            try:
                self._set(args[0], base64.b64decode(str(val or "")).decode("latin-1"))
            except Exception:
                pass

    def _handle_btoa(self, args):
        if args:
            val = self._get(args[0])
            try:
                self._set(args[0], base64.b64encode(str(val or "").encode("latin-1")).decode("latin-1"))
            except Exception:
                pass

    def _handle_resolve(self, args):
        """RESOLVE val_reg — encodes register value as base64 and stores as result."""
        val = self._get(args[0]) if args else None
        if val is None:
            val = ""
        try:
            self.result = base64.b64encode(str(val).encode("latin-1")).decode("latin-1")
        except Exception:
            self.result = base64.b64encode(str(val).encode("utf-8", errors="replace")).decode("latin-1")

    def _handle_if_defined(self, args):
        """IF_DEFINED check_reg, fn_reg, *fn_args — call fn if check_reg is not None."""
        if len(args) < 2:
            return
        val = self._get(args[0])
        if val is not None:
            fn_reg = args[1]
            fn_args = list(args[2:])
            fn = self._get(fn_reg)
            if fn is not None:
                if callable(fn):
                    # Opcode handlers expect a list of register IDs as their first arg
                    fn(fn_args)

    def _handle_if_equal(self, args):
        """IF_EQUAL a, b, fn_reg, *fn_args — call fn if reg[a] == reg[b]."""
        if len(args) < 3:
            return
        if self._get(args[0]) == self._get(args[1]):
            fn = self._get(args[2])
            fn_args = list(args[3:])
            if callable(fn):
                fn(fn_args)

    def _handle_if_not_close(self, args):
        """IF_NOT_CLOSE a, b, threshold_reg, fn_reg, *fn_args"""
        if len(args) < 4:
            return
        try:
            va = float(self._get(args[0]) or 0)
            vb = float(self._get(args[1]) or 0)
            threshold = float(self._get(args[2]) or 0)
            if abs(va - vb) > threshold:
                fn = self._get(args[3])
                fn_args = list(args[4:])
                if callable(fn):
                    fn(fn_args)
        except Exception:
            pass

    def _handle_script_match(self, args):
        """SCRIPT_MATCH dest, pattern_reg — match document script srcs."""
        if len(args) >= 2:
            pattern = self._get(args[1])
            try:
                if re.search(str(pattern), BrowserEnv.sdk_script_url):
                    self._set(args[0], BrowserEnv.sdk_script_url)
                    return
            except Exception:
                pass
            self._set(args[0], None)

    def _handle_await(self, args):
        """AWAIT dest, src_reg — synchronous await (just copy)."""
        if len(args) >= 2:
            self._set(args[0], self._get(args[1]))

    # ------------------------------------------------------------------
    # Phase 4: Main execution loop
    # ------------------------------------------------------------------

    def _dispatch(self, op: int, args: list):
        handlers = {
            OP_SET:            self._handle_set,
            OP_COPY:           self._handle_copy,
            OP_XOR_REG:        self._handle_xor,
            OP_ADD:            self._handle_add,
            OP_SUB:            self._handle_sub,
            OP_MULTIPLY:       self._handle_multiply,
            OP_DIVIDE:         self._handle_divide,
            OP_GET_INDEX:      self._handle_get_index,
            OP_BIND:           self._handle_bind,
            OP_WINDOW:         self._handle_window,
            OP_CALL:           self._handle_call,
            OP_TRY_CALL:       self._handle_try_call,
            OP_TRY_CALL_RES:   self._handle_try_call_res,
            OP_JSON_PARSE:     self._handle_json_parse,
            OP_JSON_STRINGIFY: self._handle_json_stringify,
            OP_ATOB:           self._handle_atob,
            OP_BTOA:           self._handle_btoa,
            OP_RESOLVE:        self._handle_resolve,
            OP_SCRIPT_MATCH:   self._handle_script_match,
            OP_AWAIT:          self._handle_await,
            OP_IF_DEFINED:     self._handle_if_defined,
            OP_IF_EQUAL:       self._handle_if_equal,
            OP_IF_NOT_CLOSE:   self._handle_if_not_close,
        }
        handler = handlers.get(op)
        if handler:
            handler(args)

    def execute(self, instructions: list) -> str | None:
        """Execute the instruction list and return the resolved token."""
        # Step 1: Build the opcode alias map
        self._build_alias_map(instructions)

        # Step 2: Initialize IP (register 9) with initial instructions
        self._set(OP_IP, instructions)

        # Step 3: Execute using an instruction pointer
        ip = 0
        step = 0
        while True:
            # The instruction array might have been replaced (e.g. by JSON_PARSE into OP_IP)
            current_instructions = self._get(OP_IP)
            if not isinstance(current_instructions, list) or ip >= len(current_instructions):
                break

            instr = current_instructions[ip]
            ip += 1

            if not isinstance(instr, list) or len(instr) < 1:
                continue

            raw_op = instr[0]
            op = self._resolve_op(raw_op)
            args = instr[1:]

            # If the instruction pointer array is replaced during this instruction
            # (e.g. via IF_DEFINED -> JSON_PARSE into OP_IP), reset ip to 0.
            old_instructions_id = id(self._get(OP_IP))

            try:
                self._dispatch(op, args)
            except Exception:
                pass  # VM is resilient

            if self.result is not None:
                break  # RESOLVE was called — we're done

            if id(self._get(OP_IP)) != old_instructions_id:
                ip = 0  # Restart execution from the new instruction array
            
            step += 1
            if step > 2000:
                print("VM execution loop limit reached!")
                break
        
        return self.result


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _xor_decrypt(data: str, key: str) -> str:
    return "".join(chr(ord(d) ^ ord(key[i % len(key)])) for i, d in enumerate(data))


def execute_turnstile_vm(dx: str, p_token: str) -> str | None:
    """
    Decrypt and execute the Turnstile VM bytecode.

    Args:
        dx:      Base64-encoded encrypted bytecode (from chat-requirements)
        p_token: The gAAAAAC requirements token (XOR decryption key)

    Returns:
        The resolved base64 token string, or None on failure.
    """
    try:
        raw = base64.b64decode(dx).decode("latin-1")
        decrypted = _xor_decrypt(raw, p_token)
        instructions = json.loads(decrypted)

        vm = _VM()
        return vm.execute(instructions)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"VM Execution failed: {e}")
        return None
