"""Small, data-only Python interpreter for generated selection programs.

This deliberately does not use exec/eval or expose Python objects/callables to
generated code. Programs can assign variables and evaluate comprehensions over
JSON values. They cannot import, access attributes, write files or run commands.
"""
import ast
import math
import operator
import time


class SelectionCodeError(ValueError):
    pass


class SelectionProgram:
    FUNCTIONS = {"len", "min", "max", "sum", "abs", "round", "sorted", "any", "all"}
    BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
    COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
               ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
               ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
               ast.Is: operator.is_, ast.IsNot: operator.is_not}

    def __init__(self, cancelled=lambda: False, max_steps=3000000, seconds=12):
        self.cancelled = cancelled
        self.steps = max_steps
        self.deadline = time.monotonic() + seconds

    def tick(self):
        self.steps -= 1
        if self.cancelled():
            raise SelectionCodeError("已停止筛选。")
        if self.steps < 0 or time.monotonic() > self.deadline:
            raise SelectionCodeError("筛选计算超出限制，请简化条件或缩小范围。")

    def bounded(self, value):
        if isinstance(value, (list, tuple, dict, str)) and len(value) > 200000:
            raise SelectionCodeError("筛选中间结果过大。")
        if isinstance(value, (int, float)) and (not math.isfinite(value) or abs(value) > 1e18):
            raise SelectionCodeError("筛选数值超出范围。")
        return value

    def evaluate(self, node, env):
        self.tick()
        ev = lambda n: self.evaluate(n, env)
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
            return self.bounded(node.value)
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return self.bounded([ev(n) for n in node.elts])
        if isinstance(node, ast.Dict) and all(k is not None for k in node.keys):
            return {ev(k): ev(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Subscript):
            return ev(node.value)[ev(node.slice)]
        if isinstance(node, ast.Slice):
            return slice(ev(node.lower) if node.lower else None,
                         ev(node.upper) if node.upper else None,
                         ev(node.step) if node.step else None)
        if isinstance(node, ast.UnaryOp):
            value = ev(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            if type(value) in (int, float) and isinstance(node.op, (ast.USub, ast.UAdd)):
                return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in self.BINARY:
            left, right = ev(node.left), ev(node.right)
            # No sequence multiplication or concatenation (unbounded allocation).
            if type(left) not in (int, float) or type(right) not in (int, float):
                raise SelectionCodeError("算术运算只接受数值。")
            return self.bounded(self.BINARY[type(node.op)](left, right))
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                result = ev(value)
                if isinstance(node.op, ast.And) and not result:
                    return result
                if isinstance(node.op, ast.Or) and result:
                    return result
            return result
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, other in zip(node.ops, node.comparators):
                right = ev(other)
                if type(op) not in self.COMPARE or not self.COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return ev(node.body if ev(node.test) else node.orelse)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            output = []

            def collect(index, scope):
                self.tick()
                if index == len(node.generators):
                    output.append(self.evaluate(node.elt, scope))
                    self.bounded(output)
                    return
                clause = node.generators[index]
                if clause.is_async or not isinstance(clause.target, ast.Name):
                    raise SelectionCodeError("只支持简单变量的列表推导。")
                values = self.evaluate(clause.iter, scope)
                if not isinstance(values, (list, tuple, dict)):
                    raise SelectionCodeError("只能遍历数据列表。")
                for value in values:
                    self.tick()
                    local = dict(scope, **{clause.target.id: value})
                    if all(self.evaluate(condition, local) for condition in clause.ifs):
                        collect(index + 1, local)

            collect(0, env)
            return output
        if isinstance(node, ast.Call):
            if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
                raise SelectionCodeError("不支持参数展开。")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                value = ev(node.func.value)
                args = [ev(a) for a in node.args]
                if type(value) is dict and 1 <= len(args) <= 2 and not node.keywords:
                    return value.get(*args)
            if isinstance(node.func, ast.Name) and node.func.id in self.FUNCTIONS:
                name = node.func.id
                args = [ev(a) for a in node.args]
                kwargs = {}
                for kw in node.keywords:
                    if name == "sorted" and kw.arg == "key" and isinstance(kw.value, ast.Lambda):
                        lam = kw.value
                        if (len(lam.args.args) != 1 or lam.args.posonlyargs or lam.args.kwonlyargs
                                or lam.args.defaults or lam.args.vararg or lam.args.kwarg):
                            raise SelectionCodeError("排序 key 需要单参数 lambda。")
                        kwargs["key"] = lambda v: self.evaluate(lam.body, dict(env, **{lam.args.args[0].arg: v}))
                    else:
                        kwargs[kw.arg] = ev(kw.value)
                allowed = {"sorted": {"key", "reverse"}, "min": {"default"}, "max": {"default"}}
                if set(kwargs) - allowed.get(name, set()):
                    raise SelectionCodeError("不支持的函数参数。")
                functions = {"len": len, "min": min, "max": max, "sum": sum, "abs": abs,
                             "round": round, "sorted": sorted, "any": any, "all": all}
                return self.bounded(functions[name](*args, **kwargs))
        raise SelectionCodeError("不支持的筛选语法：" + type(node).__name__)

    def run(self, code, records, groups):
        if not isinstance(code, str) or len(code) > 12000:
            raise SelectionCodeError("筛选代码为空或过长。")
        try:
            tree = ast.parse(code)
            if len(list(ast.walk(tree))) > 2000:
                raise SelectionCodeError("筛选代码过于复杂。")
            env = {"records": records, "groups": groups}
            for statement in tree.body:
                if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                        or not isinstance(statement.targets[0], ast.Name)):
                    raise SelectionCodeError("筛选程序只支持变量赋值及列表推导。")
                name = statement.targets[0].id
                if name.startswith("_") or name in {"records", "groups"} | self.FUNCTIONS:
                    raise SelectionCodeError("不能覆盖输入或保留名称。")
                env[name] = self.evaluate(statement.value, env)
            result = env.get("result")
            if not isinstance(result, list) or any(type(v) is not str for v in result):
                raise SelectionCodeError("程序必须生成 result 图片 ID 列表。")
            known = {r["id"] for r in records}
            if any(v not in known for v in result):
                raise SelectionCodeError("程序返回了当前工程以外或不可用的图片 ID。")
            return list(dict.fromkeys(result))
        except SelectionCodeError:
            raise
        except Exception as exc:
            raise SelectionCodeError("筛选代码无法执行：" + str(exc)) from None
